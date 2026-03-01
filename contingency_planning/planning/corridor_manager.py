"""
Corridor Manager - Multi-hypothesis corridor generation

Generates N corridor hypotheses for the planner:
1. Map-aligned nominal corridor
2. Perception-shifted corridors (cones, barrels)
3. Conservative corridor (tight boundary + lower speed)

Each corridor provides:
- SDF representation of drivable area
- Speed limit constraints
- Keepout zones
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class CorridorType(Enum):
    NOMINAL = "nominal"
    PERCEPTION_SHIFTED = "perception_shifted"
    CONSERVATIVE = "conservative"
    CONSTRUCTION = "construction"


@dataclass
class CorridorConfig:
    """Configuration for corridor generation"""
    n_corridors: int = 4
    horizon: float = 150.0  # meters
    lateral_width: float = 3.5  # meters per lane
    n_lanes: int = 3
    temporal_filter_alpha: float = 0.3  # EMA smoothing


@dataclass
class Corridor:
    """Represents a driving corridor hypothesis"""
    corridor_id: int
    corridor_type: CorridorType
    confidence: float  # 0-1
    
    # Centerline in Frenet coordinates (s, l_center)
    centerline_s: np.ndarray
    centerline_l: np.ndarray
    
    # Lateral bounds (left, right)
    left_bound: np.ndarray
    right
    
    # Speed limit along corridor (m/s_bound: np.ndarray)
    speed_limit: np.ndarray
    
    # Keepout zones (static obstacles)
    keepout_regions: List[Dict]
    
    # Validity flags
    is_valid: bool = True
    validity_reason: str = ""


class CorridorManager:
    """
    Manages generation and tracking of corridor hypotheses.
    
    Key responsibilities:
    - Generate N corridor hypotheses with confidence scores
    - Temporal filtering to avoid flicker
    - ID tracking for stability
    """
    
    def __init__(self, config: Optional[CorridorConfig] = None):
        self.config = config or CorridorConfig()
        
        # State for temporal filtering
        self.previous_corridors: List[Corridor] = []
        self.corridor_ids: Dict[int, int] = {}  # corridor_id -> count
        
    def update(self, 
               ego_state: Dict,
               map_data: Optional[Dict] = None,
               perception_data: Optional[Dict] = None) -> List[Corridor]:
        """
        Update corridor hypotheses based on current state.
        
        Args:
            ego_state: {'x', 'y', 'heading', 'speed'}
            map_data: Optional map information
            perception_data: Optional perception obstacles
            
        Returns:
            List of N Corridor hypotheses
        """
        ego_x = ego_state.get('x', 0)
        ego_y = ego_state.get('y', 0)
        ego_heading = ego_state.get('heading', 0)
        ego_speed = ego_state.get('speed', 0)
        
        corridors = []
        
        # 1. Nominal corridor (map-aligned)
        nominal = self._create_nominal_corridor(
            ego_x, ego_y, ego_heading, ego_speed
        )
        corridors.append(nominal)
        
        # 2. Perception-shifted corridors (if perception data available)
        if perception_data and perception_data.get('obstacles'):
            shifted = self._create_perception_shifted_corridors(
                ego_x, ego_y, ego_heading, ego_speed,
                perception_data['obstacles']
            )
            corridors.extend(shifted)
        else:
            # Default shifted corridors
            for offset in [-3.5, 3.5]:  # Left/right lane
                shifted = self._create_shifted_corridor(
                    ego_x, ego_y, ego_heading, ego_speed, offset
                )
                corridors.append(shifted)
        
        # 3. Conservative corridor (tight + lower speed)
        conservative = self._create_conservative_corridor(
            ego_x, ego_y, ego_heading, ego_speed
        )
        corridors.append(conservative)
        
        # Apply temporal filtering
        corridors = self._apply_temporal_filter(corridors)
        
        self.previous_corridors = corridors
        return corridors
    
    def _create_nominal_corridor(self, x, y, heading, speed) -> Corridor:
        """Create map-aligned nominal corridor"""
        s = np.linspace(0, self.config.horizon, 50)
        l_center = np.zeros_like(s)
        
        # Full lane width
        left = np.full_like(s, self.config.lateral_width * self.config.n_lanes / 2)
        right = -left
        
        # Speed limit based on current speed
        speed_limit = np.full_like(s, min(speed + 5, 30))  # Target speed
        
        return Corridor(
            corridor_id=0,
            corridor_type=CorridorType.NOMINAL,
            confidence=0.9,
            centerline_s=s,
            centerline_l=l_center,
            left_bound=left,
            right_bound=right,
            speed_limit=speed_limit,
            keepout_regions=[],
            is_valid=True
        )
    
    def _create_perception_shifted_corridors(self, x, y, heading, speed, obstacles) -> List[Corridor]:
        """Create corridors that avoid perceived obstacles"""
        corridors = []
        
        # Find obstacle lateral positions
        obstacle_l_positions = []
        for obs in obstacles:
            # Simplified: assume obstacles have lateral position
            obs_l = obs.get('l', 0)
            if abs(obs_l) < 10:  # Within range
                obstacle_l_positions.append(obs_l)
        
        if not obstacle_l_positions:
            return [self._create_shifted_corridor(x, y, heading, speed, 3.5)]
        
        # Create corridor avoiding obstacles
        for obs_l in obstacle_l_positions[:2]:  # Max 2
            offset = 3.5 if obs_l > 0 else -3.5
            
            s = np.linspace(0, self.config.horizon, 50)
            l_center = np.full_like(s, offset)
            
            # Restrict lateral bounds to avoid obstacle
            left = np.full_like(s, self.config.lateral_width)
            right = -left
            
            speed_limit = np.full_like(s, min(speed + 2, 25))
            
            corridors.append(Corridor(
                corridor_id=len(corridors) + 1,
                corridor_type=CorridorType.PERCEPTION_SHIFTED,
                confidence=0.6,
                centerline_s=s,
                centerline_l=l_center,
                left_bound=left,
                right_bound=right,
                speed_limit=speed_limit,
                keepout_regions=[{'s': obs.get('s', 0), 'l': obs_l, 'width': 2, 'length': 4.5}],
                is_valid=True
            ))
        
        return corridors
    
    def _create_shifted_corridor(self, x, y, heading, speed, lateral_offset: float) -> Corridor:
        """Create a laterally shifted corridor"""
        s = np.linspace(0, self.config.horizon, 50)
        l_center = np.full_like(s, lateral_offset)
        
        left = l_center + self.config.lateral_width / 2
        right = l_center - self.config.lateral_width / 2
        
        speed_limit = np.full_like(s, min(speed + 3, 28))
        
        return Corridor(
            corridor_id=hash(str(lateral_offset)) % 1000,
            corridor_type=CorridorType.PERCEPTION_SHIFTED,
            confidence=0.7,
            centerline_s=s,
            centerline_l=l_center,
            left_bound=left,
            right_bound=right,
            speed_limit=speed_limit,
            keepout_regions=[]
        )
    
    def _create_conservative_corridor(self, x, y, heading, speed) -> Corridor:
        """Create conservative corridor with tight bounds and lower speed"""
        s = np.linspace(0, self.config.horizon, 50)
        l_center = np.zeros_like(s)
        
        # Tight lateral bounds (single lane)
        left = np.full_like(s, 1.5)  # Half lane width
        right = -left
        
        # Lower speed limit
        speed_limit = np.full_like(s, min(speed, 15))  # Cap at 15 m/s
        
        return Corridor(
            corridor_id=999,
            corridor_type=CorridorType.CONSERVATIVE,
            confidence=0.5,
            centerline_s=s,
            centerline_l=l_center,
            left_bound=left,
            right_bound=right,
            speed_limit=speed_limit,
            keepout_regions=[],
            is_valid=True
        )
    
    def _apply_temporal_filter(self, corridors: List[Corridor]) -> List[Corridor]:
        """Apply temporal filtering for stability"""
        # Simple hysteresis: prefer keeping same corridor IDs
        if not self.previous_corridors:
            return corridors
        
        # Count corridor types in previous
        prev_types = {c.corridor_type for c in self.previous_corridors}
        
        for corr in corridors:
            if corr.corridor_type in prev_types:
                # Boost confidence for stability
                corr.confidence = min(corr.confidence * 1.1, 1.0)
        
        return corridors
    
    def get_sdf_handles(self, corridor: Corridor) -> Dict:
        """
        Get SDF (Signed Distance Field) handles for a corridor.
        
        These handles are used by the GPU evaluator for fast collision checking.
        """
        return {
            'sdf_drivable': self._compute_sdf_drivable(corridor),
            'sdf_keepout': self._compute_sdf_keepout(corridor),
            'speed_limit': corridor.speed_limit,
            'centerline': (corridor.centerline_s, corridor.centerline_l),
            'bounds': (corridor.left_bound, corridor.right_bound)
        }
    
    def _compute_sdf_drivable(self, corridor: Corridor) -> np.ndarray:
        """Compute SDF to drivable boundary"""
        # For each s position, compute distance to closest boundary
        l = corridor.centerline_l
        left = corridor.left_bound
        right = corridor.right_bound
        
        # Distance to left and right bounds
        dist_left = left - l
        dist_right = l - right
        
        # SDF: positive inside, negative outside
        sdf = np.minimum(dist_left, dist_right)
        return sdf
    
    def _compute_sdf_keepout(self, corridor: Corridor) -> np.ndarray:
        """Compute SDF to keepout zones"""
        sdf = np.full_like(corridor.centerline_s, 100.0)  # Large positive = far from keepout
        
        for keepout in corridor.keepout_regions:
            keepout_s = keepout.get('s', 0)
            keepout_l = keepout.get('l', 0)
            keepout_width = keepout.get('width', 2)
            
            # Distance in s-l space
            dist = np.sqrt(
                (corridor.centerline_s - keepout_s) ** 2 + 
                (corridor.centerline_l - keepout_l) ** 2
            )
            
            # Subtract keepout size
            dist = dist - keepout_width / 2
            
            sdf = np.minimum(sdf, dist)
        
        return sdf


def create_corridor_manager(n_corridors: int = 4) -> CorridorManager:
    """Factory function"""
    config = CorridorConfig(n_corridors=n_corridors)
    return CorridorManager(config)


if __name__ == "__main__":
    # Test corridor manager
    manager = create_corridor_manager(n_corridors=4)
    
    # Update with ego state
    ego_state = {'x': 0, 'y': 0, 'heading': 0, 'speed': 15}
    corridors = manager.update(ego_state)
    
    print(f"Generated {len(corridors)} corridors:")
    for corr in corridors:
        print(f"  - {corr.corridor_type.value}: confidence={corr.confidence:.2f}")
    
    # Get SDF handles for first corridor
    if corridors:
        sdf = manager.get_sdf_handles(corridors[0])
        print(f"SDF drivable range: [{sdf['sdf_drivable'].min():.2f}, {sdf['sdf_drivable'].max():.2f}]")

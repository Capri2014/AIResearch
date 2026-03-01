"""
SDF-based Collision Checker

GPU-friendly collision checking using Signed Distance Fields.
Uses rasterized occupancy grids for fast lookups.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SDFConfig:
    """Configuration for SDF collision checker"""
    grid_resolution: float = 0.1  # meters per pixel
    horizon: float = 150.0  # meters
    width: float = 20.0  # lateral width
    n_time_slices: int = 60
    time_step: float = 0.1  # seconds


class SDFCollisionChecker:
    """
    Fast collision checker using Signed Distance Fields.
    
    Key optimizations:
    - Pre-computed SDF for static environment
    - Time-indexed occupancy for dynamic obstacles
    - Rasterized representation (GPU-friendly texture fetches)
    """
    
    def __init__(self, config: Optional[SDFConfig] = None):
        self.config = config or SDFConfig()
        
        # Grid dimensions
        self.n_s = int(self.config.horizon / self.config.grid_resolution)
        self.n_l = int(self.config.width / self.config.grid_resolution)
        
        # SDF grids
        self.sdf_drivable = np.zeros((self.n_s, self.n_l))
        self.sdf_keepout = np.full((self.n_s, self.n_l), 100.0)
        
        # Time-indexed occupancy
        self.occupancy = np.zeros((self.config.n_time_slices, self.n_s, self.n_l))
        
    def update_static_environment(self, 
                                 left_bound: np.ndarray, 
                                 right_bound: np.ndarray,
                                 keepout_regions: List[Dict]):
        """Update SDF based on static environment (road boundaries, keepouts)"""
        
        # Reset SDF
        self.sdf_drivable = np.full((self.n_s, self.n_l), -1.0)
        self.sdf_keepout = np.full((self.n_s, self.n_l), 100.0)
        
        # Compute drivable area SDF
        for i in range(self.n_s):
            s = i * self.config.grid_resolution
            
            # Find closest bounds at this s
            if i < len(left_bound):
                left = left_bound[i]
                right = right_bound[i]
                
                for j in range(self.n_l):
                    l = (j - self.n_l / 2) * self.config.grid_resolution
                    
                    # Distance to boundaries
                    dist_left = left - l
                    dist_right = l - right
                    
                    # SDF: positive = inside drivable
                    self.sdf_drivable[i, j] = min(dist_left, dist_right)
        
        # Compute keepout SDF
        for keepout in keepout_regions:
            keepout_s = keepout.get('s', 0)
            keepout_l = keepout.get('l', 0)
            keepout_width = keepout.get('width', 2)
            keepout_length = keepout.get('length', 4.5)
            
            # Mark keepout region
            s_center = int(keepout_s / self.config.grid_resolution)
            l_center = int(keepout_l / self.config.grid_resolution + self.n_l / 2)
            
            s_half = int(keepout_length / 2 / self.config.grid_resolution)
            l_half = int(keepout_width / 2 / self.config.grid_resolution)
            
            for di in range(-s_half, s_half + 1):
                for dj in range(-l_half, l_half + 1):
                    si, li = s_center + di, l_center + dj
                    if 0 <= si < self.n_s and 0 <= li < self.n_l:
                        dist = np.sqrt((di * self.config.grid_resolution)**2 + 
                                      (dj * self.config.grid_resolution)**2)
                        self.sdf_keepout[si, li] = min(self.sdf_keepout[si, li], 
                                                        keepout_width/2 - dist)
    
    def update_dynamic_obstacles(self, obstacles: List[Dict]):
        """
        Update time-indexed occupancy for dynamic obstacles.
        
        Args:
            obstacles: List of obstacle dicts with:
                - 's', 'l': position in Frenet coordinates
                - 'width', 'length': size
                - 'trajectory': list of (s, l, t) positions
        """
        # Reset occupancy
        self.occupancy = np.zeros((self.config.n_time_slices, self.n_s, self.n_l))
        
        for obs in obstacles:
            trajectory = obs.get('trajectory', [])
            width = obs.get('width', 2)
            length = obs.get('length', 4.5)
            
            for pos_s, pos_l, t in trajectory:
                # Find time slice
                t_idx = int(t / self.config.time_step)
                if t_idx >= self.config.n_time_slices:
                    continue
                
                # Mark occupancy
                s_center = int(pos_s / self.config.grid_resolution)
                l_center = int(pos_l / self.config.grid_resolution + self.n_l / 2)
                
                s_half = int(length / 2 / self.config.grid_resolution)
                l_half = int(width / 2 / self.config.grid_resolution)
                
                for di in range(-s_half, s_half + 1):
                    for dj in range(-l_half, l_half + 1):
                        si, li = s_center + di, l_center + dj
                        if 0 <= si < self.n_s and 0 <= li < self.n_l:
                            self.occupancy[t_idx, si, li] = 1.0
    
    def check_collision(self, 
                        trajectory: List[Tuple[float, float, float]]) -> Tuple[bool, float, int]:
        """
        Check if trajectory collides with obstacles.
        
        Args:
            trajectory: List of (s, l, t) positions
            
        Returns:
            (collision, min_distance, first_collision_time_step)
        """
        min_distance = float('inf')
        first_collision_step = -1
        
        for t_idx, (s, l, t) in enumerate(trajectory):
            # Get grid position
            s_idx = int(s / self.config.grid_resolution)
            l_idx = int(l / self.config.grid_resolution + self.n_l / 2)
            
            # Check bounds
            if s_idx < 0 or s_idx >= self.n_s or l_idx < 0 or l_idx >= self.n_l:
                continue
            
            # Check drivable SDF
            dist_drivable = self.sdf_drivable[s_idx, l_idx]
            if dist_drivable < 0:
                # Outside drivable area
                return True, dist_drivable, t_idx
            
            # Check keepout SDF
            dist_keepout = self.sdf_keepout[s_idx, l_idx]
            if dist_keepout < 0:
                # In keepout zone
                return True, dist_keepout, t_idx
            
            # Check time-indexed occupancy
            time_idx = int(t / self.config.time_step)
            if time_idx < self.config.n_time_slices:
                if self.occupancy[time_idx, s_idx, l_idx] > 0:
                    return True, 0.0, t_idx
            
            # Track minimum distance
            min_distance = min(min_distance, dist_drivable, dist_keepout)
        
        return False, min_distance, first_collision_step
    
    def compute_boundary_margin(self, s: float, l: float) -> float:
        """Compute distance to nearest boundary"""
        s_idx = int(s / self.config.grid_resolution)
        l_idx = int(l / self.config.grid_resolution + self.n_l / 2)
        
        if 0 <= s_idx < self.n_s and 0 <= l_idx < self.n_l:
            return self.sdf_drivable[s_idx, l_idx]
        return 0.0
    
    def compute_keepout_margin(self, s: float, l: float) -> float:
        """Compute distance to nearest keepout zone"""
        s_idx = int(s / self.config.grid_resolution)
        l_idx = int(l / self.config.grid_resolution + self.n_l / 2)
        
        if 0 <= s_idx < self.n_s and 0 <= l_idx < self.n_l:
            return self.sdf_keepout[s_idx, l_idx]
        return 100.0
    
    def batch_check(self, 
                    trajectories: List[List[Tuple[float, float, float]]]) -> List[Dict]:
        """
        Check multiple trajectories efficiently.
        
        Returns:
            List of collision results per trajectory
        """
        results = []
        for traj in trajectories:
            collision, min_dist, step = self.check_collision(traj)
            results.append({
                'collision': collision,
                'min_distance': min_dist,
                'first_collision_step': step,
                'is_feasible': not collision
            })
        return results


def create_collision_checker() -> SDFCollisionChecker:
    """Factory function"""
    config = SDFConfig(
        grid_resolution=0.1,
        horizon=150.0,
        width=20.0,
        n_time_slices=60,
        time_step=0.1
    )
    return SDFCollisionChecker(config)


if __name__ == "__main__":
    # Test collision checker
    checker = create_collision_checker()
    
    # Update with simple environment
    left_bound = np.array([5.0] * 50 + [3.0] * 50)
    right_bound = np.array([-5.0] * 50 + [-3.0] * 50)
    checker.update_static_environment(left_bound, right_bound, [])
    
    # Check trajectory
    trajectory = [(10, 0, 0), (20, 0, 1), (30, 0, 2)]
    collision, dist, step = checker.check_collision(trajectory)
    print(f"Collision: {collision}, Min dist: {dist:.2f}")
    
    # Batch check
    trajectories = [
        [(10, 0, 0), (20, 0, 1)],
        [(10, 6, 0), (20, 6, 1)],  # Outside bounds
    ]
    results = checker.batch_check(trajectories)
    print(f"Results: {results}")

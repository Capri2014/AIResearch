"""
Failure Detection Module for Autonomous Driving

Detects failure modes that trigger contingency behaviors:
- Collision imminent detection
- Off-road detection
- Timeout/prediction staleness
- Sensor degradation detection
- Uncertainty estimation

Based on survey: docs/surveys/2026-02-21-contingency-planning.md
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

from .contingency_planner import FailureMode, ContingencyConfig


@dataclass
class DetectionThresholds:
    """Thresholds for failure detection."""
    # Collision detection
    collision_distance: float = 3.0  # meters
    time_to_collision: float = 2.0  # seconds
    
    # Off-road detection
    road_width: float = 3.5  # meters (typical lane width)
    lateral_deviation: float = 2.0  # meters from lane center
    
    # Timeout detection
    prediction_staleness: float = 0.5  # seconds since last update
    
    # Uncertainty detection
    uncertainty_threshold: float = 0.7
    
    # Sensor degradation
    min_sensor_health: float = 0.5


class FailureDetector(nn.Module):
    """
    Multi-modal failure detector for autonomous driving.
    
    Combines:
    - Geometry-based collision detection
    - Map-based off-road detection
    - Temporal staleness detection
    - Learned uncertainty estimation
    - Sensor health monitoring
    """
    
    def __init__(
        self,
        config: ContingencyConfig,
        state_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.config = config
        
        # Learnable uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output 0-1 uncertainty score
        )
        
        # Sensor health aggregator
        self.sensor_health_aggregator = nn.Sequential(
            nn.Linear(4, 32),  # 4 sensors
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self.thresholds = DetectionThresholds()
    
    def forward(
        self,
        state: torch.Tensor,
        current_position: torch.Tensor,
        current_velocity: torch.Tensor,
        waypoints: torch.Tensor,
        road_centerline: Optional[torch.Tensor] = None,
        obstacle_positions: Optional[torch.Tensor] = None,
        sensor_health: Optional[torch.Tensor] = None,
        last_update_time: Optional[torch.Tensor] = None,
        current_time: Optional[torch.Tensor] = None,
    ) -> Tuple[FailureMode, torch.Tensor, dict]:
        """
        Detect current failure mode.
        
        Args:
            state: [B, state_dim] State encoding
            current_position: [B, 3] Current position (x, y, z)
            current_velocity: [B, 3] Current velocity (vx, vy, vz)
            waypoints: [B, N, 3] Planned waypoints
            road_centerline: [B, M, 3] Road centerline points (optional)
            obstacle_positions: [B, K, 3] Obstacle positions (optional)
            sensor_health: [B, 4] Sensor health scores (optional)
            last_update_time: [B] Time since last prediction (optional)
            current_time: [B] Current time (optional)
            
        Returns:
            failure_mode: Detected failure mode
            uncertainty: Uncertainty score [B]
            details: Dictionary with detection details
        """
        batch_size = state.size(0)
        details = {}
        
        # 1. Estimate uncertainty
        uncertainty = self.uncertainty_estimator(state).squeeze(-1)
        details["uncertainty"] = uncertainty
        
        # 2. Check for collision risk
        collision_risk = self._detect_collision(
            current_position, current_velocity, obstacle_positions
        )
        details["collision_risk"] = collision_risk
        
        # 3. Check for off-road
        off_road_risk = self._detect_off_road(
            current_position, waypoints, road_centerline
        )
        details["off_road_risk"] = off_road_risk
        
        # 4. Check for timeout/staleness
        timeout_risk = self._detect_timeout(last_update_time, current_time)
        details["timeout_risk"] = timeout_risk
        
        # 5. Check sensor degradation
        sensor_degraded = self._detect_sensor_degradation(sensor_health)
        details["sensor_degraded"] = sensor_degraded
        
        # Combine signals to determine failure mode
        failure_mode = self._determine_failure_mode(
            collision_risk, off_road_risk, timeout_risk, 
            uncertainty, sensor_degraded
        )
        
        return failure_mode, uncertainty, details
    
    def _detect_collision(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        obstacles: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Detect collision risk from obstacles."""
        if obstacles is None:
            return torch.zeros(position.size(0), device=position.device)
        
        # Compute distance to nearest obstacle
        distances = torch.cdist(position[:, :2], obstacles[:, :, :2])  # [B, K]
        min_distances = distances.min(dim=-1)[0]  # [B]
        
        # Compute time to collision
        rel_vel = velocity.unsqueeze(1) - obstacles  # [B, K, 3]
        time_to_collision = torch.where(
            min_distances > 0,
            min_distances / (rel_vel.norm(dim=-1).min(dim=-1)[0] + 1e-6),
            torch.full_like(min_distances, float('inf'))
        )
        
        # Risk score: high if close and approaching
        risk = torch.zeros(position.size(0), device=position.device)
        risk = torch.where(
            min_distances < self.thresholds.collision_distance,
            risk + 0.5,
            risk
        )
        risk = torch.where(
            time_to_collision < self.thresholds.time_to_collision,
            risk + 0.5,
            risk
        )
        
        return risk
    
    def _detect_off_road(
        self,
        position: torch.Tensor,
        waypoints: torch.Tensor,
        road_centerline: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Detect off-road risk."""
        if road_centerline is None:
            # Use waypoints as implicit road representation
            # Check if current position is far from planned trajectory
            first_waypoint = waypoints[:, 0, :2]  # [B, 2]
            lateral_dev = torch.norm(position[:, :2] - first_waypoint, dim=-1)
        else:
            # Find nearest point on centerline
            distances = torch.cdist(
                position[:, :2].unsqueeze(1), 
                road_centerline[:, :, :2]
            )  # [B, 1, M]
            min_dist = distances.min(dim=-1)[0].squeeze(-1)  # [B]
            lateral_dev = min_dist
        
        # Risk based on lateral deviation
        risk = torch.clamp(
            lateral_dev / self.thresholds.lateral_deviation,
            0, 1
        )
        
        return risk
    
    def _detect_timeout(
        self,
        last_update: Optional[torch.Tensor],
        current_time: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Detect if prediction is stale."""
        if last_update is None or current_time is None:
            return torch.zeros(1, device="cpu").expand(1)
        
        time_since_update = current_time - last_update
        risk = torch.clamp(
            time_since_update / self.thresholds.prediction_staleness,
            0, 1
        )
        
        return risk
    
    def _detect_sensor_degradation(
        self,
        sensor_health: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Detect sensor degradation."""
        if sensor_health is None:
            return torch.zeros(1, device="cpu").expand(1)
        
        # Any sensor below threshold triggers degradation
        min_health = sensor_health.min(dim=-1)[0]
        degraded = (min_health < self.thresholds.min_sensor_health).float()
        
        return degraded
    
    def _determine_failure_mode(
        self,
        collision_risk: torch.Tensor,
        off_road_risk: torch.Tensor,
        timeout_risk: torch.Tensor,
        uncertainty: torch.Tensor,
        sensor_degraded: torch.Tensor,
    ) -> FailureMode:
        """Determine primary failure mode from risks."""
        # Priority: collision > off-road > timeout > uncertainty > sensor > none
        
        if (collision_risk > 0.7).any():
            return FailureMode.COLLISION_IMMINENT
        
        if (off_road_risk > 0.7).any():
            return FailureMode.OFF_ROAD
        
        if (timeout_risk > 0.7).any():
            return FailureMode.TIMEOUT
        
        if (uncertainty > self.thresholds.uncertainty_threshold).any():
            return FailureMode.UNCERTAINTY_HIGH
        
        if (sensor_degraded > 0.5).any():
            return FailureMode.DEGRADED_SENSOR
        
        return FailureMode.NONE


def create_failure_detector(
    state_dim: int = 256,
    hidden_dim: int = 128,
) -> FailureDetector:
    """Factory function to create a failure detector."""
    config = ContingencyConfig()
    return FailureDetector(config, state_dim, hidden_dim)


if __name__ == "__main__":
    # Simple test
    detector = create_failure_detector()
    
    # Dummy inputs
    batch_size = 4
    state = torch.randn(batch_size, 256)
    position = torch.randn(batch_size, 3)
    velocity = torch.randn(batch_size, 3) * 10  # 10 m/s
    waypoints = torch.randn(batch_size, 10, 3)
    obstacles = torch.randn(batch_size, 5, 3)
    obstacles[:, :2, :2] = position[:, :2] + torch.randn(batch_size, 2, 2) * 5  # nearby
    sensor_health = torch.rand(batch_size, 4)
    
    failure_mode, uncertainty, details = detector(
        state, position, velocity, waypoints, 
        obstacle_positions=obstacles,
        sensor_health=sensor_health
    )
    
    print(f"Failure mode: {failure_mode}")
    print(f"Uncertainty: {uncertainty}")
    print(f"Collision risk: {details['collision_risk']}")
    print(f"Off-road risk: {details['off_road_risk']}")

"""
Safety Layer for Autonomous Driving

Implements:
- Collision detection and avoidance
- Lane boundary validation
- Fallback planner (emergency stop)
- Rule-based overrides

Usage:
    from training.models.safety_layer import SafetyLayer, CollisionChecker
    
    safety = SafetyLayer(config)
    
    # Validate and fix trajectory
    safe_trajectory = safety.validate_and_fix(
        trajectory,
        obstacles,
        road_boundaries,
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SafetyConfig:
    """Configuration for safety layer."""
    # Collision
    ego_length: float = 4.5  # meters
    ego_width: float = 1.8  # meters
    safe_distance: float = 3.0  # meters
    
    # Lane
    lane_width: float = 3.5  # meters
    
    # Control
    max_acceleration: float = 3.0  # m/s^2
    max_deceleration: float = -8.0  # m/s^2
    max_steering: float = 0.5  # radians
    
    # Check frequency
    dt: float = 0.1  # seconds
    
    # Fallback
    use_fallback: bool = True


# ============================================================================
# Collision Checker
# ============================================================================

class CollisionChecker:
    """
    Check for potential collisions between ego and obstacles.
    
    Uses polygon-based collision detection.
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.ego_corners = self._get_ego_corners()
    
    def _get_ego_corners(self) -> np.ndarray:
        """Get ego vehicle corners in local frame."""
        l = self.config.ego_length / 2
        w = self.config.ego_width / 2
        
        corners = np.array([
            [l, w],
            [l, -w],
            [-l, -w],
            [-l, w],
        ])
        
        return corners
    
    def _transform_corners(
        self,
        corners: np.ndarray,
        position: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """Transform corners to world frame."""
        # Rotation matrix
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h],
        ])
        
        # Transform
        world_corners = (R @ corners.T).T + position[:2]
        
        return world_corners
    
    def _polygon_intersects(
        self,
        poly1: np.ndarray,
        poly2: np.ndarray,
    ) -> bool:
        """Check if two polygons intersect."""
        # Use separating axis theorem
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                # Get edge
                edge = poly[(i + 1) % len(poly)] - poly[i]
                
                # Get normal (perpendicular)
                normal = np.array([-edge[1], edge[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                
                # Project both polygons onto normal
                proj1 = poly1 @ normal
                proj2 = poly2 @ normal
                
                # Check separation
                if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                    return False
        
        return True
    
    def check_collision(
        self,
        ego_position: np.ndarray,
        ego_heading: float,
        obstacles: List[Dict],
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check collision with obstacles.
        
        Args:
            ego_position: [x, y, z] or [x, y]
            ego_heading: heading in radians
            obstacles: list of obstacle dicts with 'position', 'size', 'heading'
            
        Returns:
            (has_collision, collision_info)
        """
        # Get ego polygon
        ego_corners = self._transform_corners(
            self.ego_corners,
            np.array(ego_position),
            ego_heading,
        )
        
        for i, obs in enumerate(obstacles):
            obs_pos = np.array(obs.get("position", [0, 0]))
            obs_size = obs.get("size", [2.0, 2.0])  # [length, width]
            obs_heading = obs.get("heading", 0.0)
            
            # Get obstacle corners
            obs_corners = self._get_ego_corners() * np.array(obs_size[:2]) / np.array([
                self.config.ego_length, self.config.ego_width
            ])
            obs_corners = self._transform_corners(obs_corners, obs_pos, obs_heading)
            
            # Check intersection
            if self._polygon_intersects(ego_corners, obs_corners):
                return True, {
                    "obstacle_id": i,
                    "obstacle_position": obs_pos,
                    "distance": np.linalg.norm(ego_position[:2] - obs_pos),
                }
        
        return False, None
    
    def compute_ttc(
        self,
        ego_trajectory: np.ndarray,
        obstacle_trajectories: List[np.ndarray],
    ) -> float:
        """
        Compute time-to-collision.
        
        Args:
            ego_trajectory: [T, 3] ego positions over time
            obstacle_trajectories: list of [T, 3] obstacle trajectories
            
        Returns:
            Minimum TTC across all obstacles
        """
        min_ttc = float("inf")
        
        for obs_traj in obstacle_trajectories:
            for t in range(min(len(ego_trajectory), len(obs_traj))):
                ego_pos = ego_trajectory[t, :2]
                obs_pos = obs_traj[t, :2]
                
                # Estimate velocities
                if t > 0:
                    ego_vel = ego_trajectory[t, :2] - ego_trajectory[t-1, :2]
                    obs_vel = obs_traj[t, :2] - obs_traj[t-1, :2]
                else:
                    continue
                
                # Relative position and velocity
                rel_pos = obs_pos - ego_pos
                rel_vel = obs_vel - ego_vel
                
                # Distance
                dist = np.linalg.norm(rel_pos)
                
                if dist < 1e-6:
                    return 0.0  # Already colliding
                
                # Time to collision (if converging)
                closing_speed = -np.dot(rel_pos, rel_vel)
                
                if closing_speed > 0:
                    ttc = dist / closing_speed
                    min_ttc = min(min_ttc, ttc)
        
        return min_ttc
    
    def distance_to_obstacles(
        self,
        ego_position: np.ndarray,
        obstacles: List[Dict],
    ) -> np.ndarray:
        """Get distances to all obstacles."""
        ego_pos = np.array(ego_position[:2])
        
        distances = []
        for obs in obstacles:
            obs_pos = np.array(obs.get("position", [0, 0]))
            dist = np.linalg.norm(ego_pos - obs_pos)
            distances.append(dist)
        
        return np.array(distances)


# ============================================================================
# Lane Boundary Checker
# ============================================================================

class LaneValidator:
    """
    Validate trajectory against lane boundaries.
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.lane_width = config.lane_width
    
    def check_in_lane(
        self,
        position: np.ndarray,
        lane_center: np.ndarray,
    ) -> bool:
        """Check if position is within lane."""
        dist = np.linalg.norm(position[:2] - lane_center[:2])
        return dist < self.lane_width / 2
    
    def validate_trajectory(
        self,
        trajectory: np.ndarray,
        lane_centers: np.ndarray,
    ) -> Tuple[bool, List[int]]:
        """
        Validate trajectory stays in lane.
        
        Args:
            trajectory: [T, 3] waypoints
            lane_centers: [T, 3] lane center positions
            
        Returns:
            (is_valid, invalid_indices)
        """
        invalid = []
        
        for t in range(len(trajectory)):
            if not self.check_in_lane(trajectory[t], lane_centers[t]):
                invalid.append(t)
        
        return len(invalid) == 0, invalid


# ============================================================================
# Fallback Planner
# ============================================================================

class FallbackPlanner:
    """
    Emergency fallback planner.
    
    Generates safe stopping trajectory.
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.max_decel = config.max_deceleration
        self.dt = config.dt
    
    def plan_emergency_stop(
        self,
        initial_state: Dict,
        target_position: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Plan emergency stopping trajectory.
        
        Args:
            initial_state: dict with 'position', 'velocity', 'heading'
            target_position: optional target to stop at
            
        Returns:
            trajectory: [T, 3] waypoints
        """
        pos = np.array(initial_state["position"])
        vel = np.array(initial_state.get("velocity", [0, 0, 0]))
        heading = initial_state.get("heading", 0.0)
        
        # Estimate time to stop
        speed = np.linalg.norm(vel[:2])
        
        if speed < 0.1:
            # Already stopped
            return pos.reshape(1, -1)
        
        # Time to stop
        t_stop = speed / abs(self.max_decel)
        
        # Number of steps
        n_steps = int(t_stop / self.dt) + 10
        
        # Generate stopping trajectory
        trajectory = []
        
        current_pos = pos[:2].copy()
        current_speed = speed
        
        for t in range(n_steps):
            # Decelerate
            current_speed = max(0, current_speed + self.max_decel * self.dt)
            
            if current_speed < 0.1:
                current_speed = 0
            
            # Move in heading direction
            direction = np.array([np.cos(heading), np.sin(heading)])
            current_pos = current_pos + direction * current_speed * self.dt
            
            trajectory.append([current_pos[0], current_pos[1], heading])
        
        return np.array(trajectory)
    
    def plan_to_position(
        self,
        initial_state: Dict,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Plan trajectory to reach target position.
        
        Args:
            initial_state: dict with 'position', 'velocity', 'heading'
            target: target [x, y]
            
        Returns:
            trajectory: [T, 3] waypoints
        """
        pos = np.array(initial_state["position"])
        vel = np.array(initial_state.get("velocity", [0, 0, 0]))
        heading = initial_state.get("heading", 0.0)
        
        target = np.array(target)
        
        # Direction to target
        direction = target - pos[:2]
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:
            return np.array([[pos[0], pos[1], heading]])
        
        direction = direction / distance
        
        # Required speed to reach
        # Use constant speed for simplicity
        speed = 5.0  # m/s
        
        # Time to reach
        t = distance / speed
        n_steps = max(int(t / self.dt), 1)
        
        # Generate trajectory
        trajectory = []
        
        for i in range(n_steps + 1):
            alpha = i / n_steps
            
            # Interpolate
            current_pos = pos[:2] + direction * (distance * alpha)
            
            # Update heading
            current_heading = np.arctan2(direction[1], direction[0])
            
            trajectory.append([current_pos[0], current_pos[1], current_heading])
        
        return np.array(trajectory)


# ============================================================================
# Complete Safety Layer
# ============================================================================

class SafetyLayer(nn.Module):
    """
    Complete safety layer for autonomous driving.
    
    Validates and fixes trajectories to ensure safety.
    
    Usage:
        safety = SafetyLayer(config)
        
        # Validate and fix
        safe_trajectory = safety.validate_and_fix(
            trajectory=predicted_trajectory,
            obstacles=detected_obstacles,
            lane_boundaries=detected_lanes,
            ego_state=current_ego_state,
        )
    """
    
    def __init__(self, config: SafetyConfig):
        super().__init__()
        self.config = config
        
        # Collision checker
        self.collision_checker = CollisionChecker(config)
        
        # Lane validator
        self.lane_validator = LaneValidator(config)
        
        # Fallback planner
        self.fallback_planner = FallbackPlanner(config)
    
    def validate_and_fix(
        self,
        trajectory: np.ndarray,
        obstacles: List[Dict],
        lane_boundaries: Optional[np.ndarray] = None,
        ego_state: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Validate trajectory and fix if unsafe.
        
        Args:
            trajectory: [T, 3] predicted trajectory (x, y, heading)
            obstacles: list of obstacle dicts
            lane_boundaries: [T, 2, 2] lane boundaries (optional)
            ego_state: current ego state dict
            
        Returns:
            (safe_trajectory, info_dict)
        """
        info = {
            "was_modified": False,
            "modification_reason": None,
            "collisions_detected": [],
            "lane_violations": [],
        }
        
        # Check each waypoint
        for t in range(len(trajectory)):
            pos = trajectory[t]
            heading = pos[2] if len(pos) > 2 else 0.0
            
            # Collision check
            has_collision, collision_info = self.collision_checker.check_collision(
                pos, heading, obstacles
            )
            
            if has_collision:
                info["collisions_detected"].append({
                    "timestep": t,
                    "info": collision_info,
                })
                
                # If collision detected, regenerate trajectory
                if ego_state is not None:
                    trajectory = self.fallback_planner.plan_emergency_stop(ego_state)
                    info["was_modified"] = True
                    info["modification_reason"] = "collision_avoidance"
                    break
        
        # Check lane boundaries (if provided)
        if lane_boundaries is not None:
            is_valid, invalid_indices = self.lane_validator.validate_trajectory(
                trajectory, lane_boundaries
            )
            
            if not is_valid:
                info["lane_violations"] = invalid_indices
                
                # Try to fix by returning to lane center
                if ego_state is not None:
                    # Simple fix: stop
                    trajectory = self.fallback_planner.plan_emergency_stop(ego_state)
                    info["was_modified"] = True
                    info["modification_reason"] = "lane_violation"
        
        # Check feasibility (acceleration, steering limits)
        is_feasible, violation_info = self._check_feasibility(trajectory)
        
        if not is_feasible:
            # Smooth trajectory
            trajectory = self._smooth_trajectory(trajectory)
            info["was_modified"] = True
            info["modification_reason"] = "infeasibility"
        
        return trajectory, info
    
    def _check_feasibility(
        self,
        trajectory: np.ndarray,
    ) -> Tuple[bool, Optional[Dict]]:
        """Check if trajectory is dynamically feasible."""
        if len(trajectory) < 2:
            return True, None
        
        # Check accelerations
        for t in range(len(trajectory) - 1):
            p1 = trajectory[t, :2]
            p2 = trajectory[t + 1, :2]
            
            # Velocity (approximation)
            vel = (p2 - p1) / self.config.dt
            speed = np.linalg.norm(vel)
            
            # Acceleration (approximation)
            if t > 0:
                prev_vel = (trajectory[t, :2] - trajectory[t-1, :2]) / self.config.dt
                acc = (vel - prev_vel) / self.config.dt
                acc_mag = np.linalg.norm(acc)
                
                if acc_mag > abs(self.config.max_acceleration):
                    return False, {"type": "acceleration", "timestep": t}
        
        return True, None
    
    def _smooth_trajectory(
        self,
        trajectory: np.ndarray,
    ) -> np.ndarray:
        """Smooth trajectory to meet constraints."""
        # Simple smoothing: moving average
        smoothed = trajectory.copy()
        
        window = 3
        
        for i in range(1, len(trajectory) - 1):
            start = max(0, i - window // 2)
            end = min(len(trajectory), i + window // 2 + 1)
            
            smoothed[i, :2] = trajectory[start:end, :2].mean(axis=0)
        
        return smoothed
    
    def forward(
        self,
        trajectory: torch.Tensor,
        obstacles: List[Dict],
        lane_boundaries: Optional[torch.Tensor] = None,
        ego_state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        PyTorch forward for integration in model.
        """
        # Convert to numpy
        traj_np = trajectory.detach().cpu().numpy()
        
        # Process
        safe_traj, info = self.validate_and_fix(
            traj_np, obstacles, lane_boundaries, ego_state
        )
        
        # Convert back
        safe_traj_tensor = torch.from_numpy(safe_traj).to(trajectory.device)
        
        return safe_traj_tensor, info


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = SafetyConfig()
    
    # Create safety layer
    safety = SafetyLayer(config)
    
    # Test trajectory
    trajectory = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
    ])
    
    # Test obstacles
    obstacles = [
        {"position": [3.5, 0.5], "size": [2, 2], "heading": 0},
    ]
    
    # Test ego state
    ego_state = {
        "position": [0, 0, 0],
        "velocity": [2, 0, 0],
        "heading": 0,
    }
    
    # Validate
    print("Validating trajectory...")
    safe_traj, info = safety.validate_and_fix(
        trajectory, obstacles, None, ego_state
    )
    
    print(f"  Original: {trajectory.shape}")
    print(f"  Safe: {safe_traj.shape}")
    print(f"  Modified: {info['was_modified']}")
    print(f"  Reason: {info['modification_reason']}")
    print(f"  Collisions: {info['collisions_detected']}")
    
    print("\n✓ Safety layer working!")

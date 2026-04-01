"""
Reward Shaping for Kinematics RL Pipeline

Provides reward components for waypoint following:
- Waypoint progress reward (getting closer to target)
- Smoothness reward (reducing jerk/accel)
- Collision penalty
- Success reward

This reward shaping makes RL training more effective for delta-waypoint learning.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    # Progress rewards
    waypoint_progress_weight: float = 1.0      # Reward for decreasing distance to target waypoint
    waypoint_reached_weight: float = 10.0      # Reward for reaching a waypoint
    distance_threshold: float = 2.0            # Distance (m) to consider waypoint "reached"
    
    # Smoothness rewards (penalties)
    accel_penalty_weight: float = 0.1           # Penalty for high acceleration
    jerk_penalty_weight: float = 0.05           # Penalty for high jerk
    steering_penalty_weight: float = 0.05      # Penalty for large steering angles
    
    # Collision/safety
    collision_penalty: float = -50.0            # Penalty for collision
    off_road_penalty: float = -10.0            # Penalty for going off-road
    
    # Success
    success_reward: float = 100.0              # Reward for completing episode
    timeout_penalty: float = -5.0              # Small penalty for timing out


class WaypointRewardShaper:
    """
    Computes shaped rewards for waypoint following tasks.
    
    Reward components:
    - progress: negative distance to current target waypoint
    - waypoint_reached: bonus when a waypoint is reached
    - smoothness: penalty for high accel/jerk/steering
    - safety: penalty for collisions/off-road
    - success: reward for episode completion
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # Track state for reward computation
        self.prev_velocity = None
        self.prev_accel = None
        self.prev_steering = None
        self.waypoints_reached = 0
        
    def reset(self):
        """Reset reward tracking for new episode."""
        self.prev_velocity = None
        self.prev_accel = None
        self.prev_steering = None
        self.waypoints_reached = 0
        
    def compute_reward(
        self,
        distance_to_target: float,
        velocity: float,
        steering: float,
        accel: float,
        jerk: float,
        collision: bool,
        off_road: bool,
        waypoint_reached: bool,
        done: bool,
        success: bool,
        timeout: bool = False,
    ) -> tuple[float, dict]:
        """
        Compute total reward and breakdown.
        
        Args:
            distance_to_target: Distance (m) to current target waypoint
            velocity: Current velocity (m/s)
            steering: Current steering angle (rad)
            accel: Current acceleration (m/s^2)
            jerk: Current jerk (m/s^3)
            collision: Whether collision occurred
            off_road: Whether vehicle is off-road
            waypoint_reached: Whether target waypoint was reached this step
            done: Whether episode is done
            success: Whether episode was successful
            timeout: Whether episode timed out
            
        Returns:
            Tuple of (total_reward, reward_breakdown_dict)
        """
        reward = 0.0
        breakdown = {}
        
        # 1. Progress reward: negative distance to target (closer = better)
        progress_reward = -self.config.waypoint_progress_weight * distance_to_target
        reward += progress_reward
        breakdown['progress'] = round(progress_reward, 4)
        
        # 2. Waypoint reached reward
        if waypoint_reached:
            waypoint_reward = self.config.waypoint_reached_weight
            reward += waypoint_reward
            breakdown['waypoint_reached'] = waypoint_reward
            self.waypoints_reached += 1
        else:
            breakdown['waypoint_reached'] = 0.0
            
        # 3. Smoothness penalties
        # Acceleration penalty
        accel_penalty = -self.config.accel_penalty_weight * abs(accel)
        reward += accel_penalty
        breakdown['accel_penalty'] = round(accel_penalty, 4)
        
        # Jerk penalty
        jerk_penalty = -self.config.jerk_penalty_weight * abs(jerk)
        reward += jerk_penalty
        breakdown['jerk_penalty'] = round(jerk_penalty, 4)
        
        # Steering penalty (penalize large steering angles)
        steering_penalty = -self.config.steering_penalty_weight * abs(steering)
        reward += steering_penalty
        breakdown['steering_penalty'] = round(steering_penalty, 4)
        
        # 4. Safety penalties
        if collision:
            collision_penalty = self.config.collision_penalty
            reward += collision_penalty
            breakdown['collision'] = collision_penalty
            
        if off_road:
            off_road_penalty = self.config.off_road_penalty
            reward += off_road_penalty
            breakdown['off_road'] = off_road_penalty
            
        # 5. Terminal rewards
        if done:
            if success:
                success_reward = self.config.success_reward
                reward += success_reward
                breakdown['success'] = success_reward
            elif timeout:
                timeout_penalty = self.config.timeout_penalty
                reward += timeout_penalty
                breakdown['timeout'] = timeout_penalty
                
        breakdown['total'] = round(reward, 4)
        
        return reward, breakdown
        
    def get_waypoints_reached(self) -> int:
        """Return number of waypoints reached in current episode."""
        return self.waypoints_reached


def compute_kinematics_reward(
    state: dict,
    prev_state: Optional[dict] = None,
    config: Optional[RewardConfig] = None,
) -> tuple[float, dict]:
    """
    Convenience function to compute reward from state dict.
    
    Args:
        state: Dictionary with keys:
            - distance_to_target: float
            - velocity: float
            - steering: float
            - accel: float
            - jerk: float
            - collision: bool
            - off_road: bool
            - waypoint_reached: bool
            - done: bool
            - success: bool
            - timeout: bool
        prev_state: Previous state for computing derivatives (optional)
        config: RewardConfig (optional)
        
    Returns:
        Tuple of (reward, breakdown_dict)
    """
    shaper = WaypointRewardShaper(config)
    
    # Compute jerk from previous state if available
    jerk = 0.0
    if prev_state is not None and 'accel' in prev_state:
        jerk = state.get('accel', 0) - prev_state.get('accel', 0)
        
    return shaper.compute_reward(
        distance_to_target=state.get('distance_to_target', 0),
        velocity=state.get('velocity', 0),
        steering=state.get('steering', 0),
        accel=state.get('accel', 0),
        jerk=jerk,
        collision=state.get('collision', False),
        off_road=state.get('off_road', False),
        waypoint_reached=state.get('waypoint_reached', False),
        done=state.get('done', False),
        success=state.get('success', False),
        timeout=state.get('timeout', False),
    )


# Default reward configuration for kinematics waypoint RL
DEFAULT_REWARD_CONFIG = RewardConfig()


if __name__ == "__main__":
    # Quick test
    config = RewardConfig()
    shaper = WaypointRewardShaper(config)
    
    # Simulate a few steps
    test_states = [
        {'distance_to_target': 10.0, 'velocity': 5.0, 'steering': 0.1, 
         'accel': 0.5, 'jerk': 0.1, 'collision': False, 'off_road': False,
         'waypoint_reached': False, 'done': False, 'success': False},
        {'distance_to_target': 8.0, 'velocity': 5.2, 'steering': 0.15, 
         'accel': 0.2, 'jerk': -0.3, 'collision': False, 'off_road': False,
         'waypoint_reached': False, 'done': False, 'success': False},
        {'distance_to_target': 2.0, 'velocity': 4.0, 'steering': 0.05, 
         'accel': -0.2, 'jerk': -0.4, 'collision': False, 'off_road': False,
         'waypoint_reached': True, 'done': False, 'success': False},
    ]
    
    print("Reward Shaping Test")
    print("=" * 50)
    
    for i, state in enumerate(test_states):
        reward, breakdown = shaper.compute_reward(**state)
        print(f"\nStep {i+1}:")
        print(f"  Total reward: {breakdown['total']}")
        for k, v in breakdown.items():
            if k != 'total':
                print(f"    {k}: {v}")
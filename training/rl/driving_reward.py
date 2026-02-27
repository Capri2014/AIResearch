"""
Reward Function Utilities for Autonomous Driving.

Provides modular reward components for driving scenarios:
- Progress rewards (distance to goal, route completion)
- Safety rewards (collision avoidance, off-road penalty)
- Efficiency rewards (speed, time)
- Comfort rewards (jerk, lateral acceleration)

Supports multi-objective weighting and scenario-specific configs.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """Reward component weights for multi-objective optimization."""
    goal_progress: float = 1.0       # Distance reduction toward goal
    route_completion: float = 0.5    # Fraction of route completed
    speed_efficiency: float = 0.3     # Optimal speed reward
    collision_penalty: float = -10.0   # Collision punishment
    off_road_penalty: float = -5.0    # Off-road/driving off road
    timeout_penalty: float = -2.0     # Timeout punishment
    comfort_jerk: float = -0.1        # Longitudinal jerk penalty
    comfort_lateral: float = -0.1     # Lateral acceleration penalty
    smoothness: float = 0.2           # Action smoothness reward
    goal_reached: float = 10.0        # Bonus for reaching goal


@dataclass
class RewardMetrics:
    """Computed reward metrics for logging/debugging."""
    goal_progress: float = 0.0
    route_completion: float = 0.0
    speed_efficiency: float = 0.0
    collision_penalty: float = 0.0
    off_road_penalty: float = 0.0
    timeout_penalty: float = 0.0
    comfort_jerk: float = 0.0
    comfort_lateral: float = 0.0
    smoothness: float = 0.0
    goal_reached: float = 0.0
    total: float = 0.0


class DrivingRewardFunction:
    """
    Modular reward function for autonomous driving.
    
    Supports configurable weights and computes individual components
    for analysis/debugging.
    """
    
    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        goal_threshold: float = 2.0,      # Distance to goal to consider "reached"
        max_speed: float = 15.0,           # m/s (~55 km/h)
        optimal_speed: float = 10.0,       # m/s (~36 km/h)
        jerk_penalty_threshold: float = 2.0,  # m/s^3
        lateral_acc_threshold: float = 2.0,  # m/s^2
    ):
        self.weights = weights or RewardWeights()
        self.goal_threshold = goal_threshold
        self.max_speed = max_speed
        self.optimal_speed = optimal_speed
        self.jerk_penalty_threshold = jerk_penalty_threshold
        self.lateral_acc_threshold = lateral_acc_threshold
        
        # State tracking for comfort metrics
        self.prev_speed = None
        self.prev_lateral_speed = None
        self.prev_heading_rate = None
        self.prev_actions: List[np.ndarray] = []
        
    def reset(self):
        """Reset internal state for new episode."""
        self.prev_speed = None
        self.prev_lateral_speed = None
        self.prev_heading_rate = None
        self.prev_actions = []
    
    def compute(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        info: Dict[str, Any],
    ) -> tuple[float, RewardMetrics]:
        """
        Compute total reward and individual components.
        
        Args:
            state: Current state [x, y, heading, speed, goal_x, goal_y, ...]
            action: Action taken [dx, dy] (delta waypoint)
            next_state: Next state after action
            done: Episode done flag
            info: Environment info dict
            
        Returns:
            Tuple of (total_reward, RewardMetrics)
        """
        metrics = RewardMetrics()
        
        # Extract relevant info
        distance_to_goal = info.get('distance_to_goal', np.linalg.norm(
            next_state[4:6] - next_state[:2]  # goal - position
        ))
        prev_distance_to_goal = info.get('prev_distance_to_goal', distance_to_goal)
        
        # 1. Goal progress reward (distance reduction)
        progress = prev_distance_to_goal - distance_to_goal
        metrics.goal_progress = progress * self.weights.goal_progress
        
        # 2. Route completion
        route_completion = info.get('route_completion', 0.0)
        metrics.route_completion = route_completion * self.weights.route_completion
        
        # 3. Speed efficiency
        speed = next_state[3] if len(next_state) > 3 else info.get('speed', 0.0)
        speed_ratio = min(speed / self.optimal_speed, 1.0)
        metrics.speed_efficiency = speed_ratio * self.weights.speed_efficiency
        
        # 4. Collision penalty
        if info.get('collision', False):
            metrics.collision_penalty = self.weights.collision_penalty
            
        # 5. Off-road penalty
        if info.get('off_road', False) or info.get('driving_off_road', False):
            metrics.off_road_penalty = self.weights.off_road_penalty
            
        # 6. Timeout penalty
        if done and not info.get('goal_reached', False):
            metrics.timeout_penalty = self.weights.timeout_penalty
            
        # 7. Comfort - jerk
        if self.prev_speed is not None:
            jerk = abs(speed - self.prev_speed)  # Simplified longitudinal jerk
            if jerk > self.jerk_penalty_threshold:
                metrics.comfort_jerk = -(jerk - self.jerk_penalty_threshold) * self.weights.comfort_jerk
                
        # 8. Comfort - lateral acceleration
        if self.prev_heading_rate is not None and self.prev_speed is not None:
            lateral_acc = abs(self.prev_heading_rate) * speed
            if lateral_acc > self.lateral_acc_threshold:
                metrics.comfort_lateral = -(lateral_acc - self.lateral_acc_threshold) * self.weights.comfort_lateral
                
        # 9. Smoothness (action magnitude penalty)
        action_magnitude = np.linalg.norm(action)
        metrics.smoothness = -0.1 * action_magnitude * self.weights.smoothness
        
        # 10. Goal reached bonus
        if distance_to_goal < self.goal_threshold:
            metrics.goal_reached = self.weights.goal_reached
        
        # Compute total
        metrics.total = (
            metrics.goal_progress +
            metrics.route_completion +
            metrics.speed_efficiency +
            metrics.collision_penalty +
            metrics.off_road_penalty +
            metrics.timeout_penalty +
            metrics.comfort_jerk +
            metrics.comfort_lateral +
            metrics.smoothness +
            metrics.goal_reached
        )
        
        # Update state for next step
        self.prev_speed = speed
        if len(action) >= 2:
            self.prev_lateral_speed = action[1] if len(action) > 1 else 0.0
        self.prev_actions.append(action.copy())
        if len(self.prev_actions) > 10:
            self.prev_actions.pop(0)
            
        return metrics.total, metrics
    
    def compute_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, List[RewardMetrics]]:
        """Compute rewards for a batch of transitions."""
        rewards = []
        metrics_list = []
        
        for i in range(len(states)):
            r, m = self.compute(states[i], actions[i], next_states[i], dones[i], infos[i])
            rewards.append(r)
            metrics_list.append(m)
            
        return np.array(rewards), metrics_list


class ScenarioRewardConfig:
    """Predefined reward configs for different driving scenarios."""
    
    @staticmethod
    def highway() -> RewardWeights:
        """High-speed highway driving - focus on efficiency and comfort."""
        return RewardWeights(
            goal_progress=1.0,
            route_completion=0.5,
            speed_efficiency=1.0,         # Higher - highway is about speed
            collision_penalty=-20.0,      # Severe - high speed collisions dangerous
            off_road_penalty=-10.0,
            timeout_penalty=-1.0,
            comfort_jerk=-0.2,             # Higher - comfort matters at speed
            comfort_lateral=-0.3,          # Higher - lane changes matter
            smoothness=0.5,               # Higher - smooth driving important
            goal_reached=20.0,
        )
    
    @staticmethod
    def urban() -> RewardWeights:
        """Urban driving - focus on safety and progress."""
        return RewardWeights(
            goal_progress=2.0,            # Higher - navigation is key
            route_completion=1.0,
            speed_efficiency=0.3,         # Lower - not about speed
            collision_penalty=-30.0,      # Very severe - pedestrians
            off_road_penalty=-10.0,
            timeout_penalty=-5.0,         # Higher - getting stuck is bad
            comfort_jerk=-0.1,
            comfort_lateral=-0.1,
            smoothness=0.2,
            goal_reached=15.0,
        )
    
    @staticmethod
    def parking() -> RewardWeights:
        """Parking scenario - precision over speed."""
        return RewardWeights(
            goal_progress=3.0,            # High - precision matters
            route_completion=0.5,
            speed_efficiency=-0.5,        # Negative - slow is good for parking
            collision_penalty=-50.0,      # Very severe - contact is failure
            off_road_penalty=-5.0,
            timeout_penalty=-1.0,
            comfort_jerk=-0.05,
            comfort_lateral=-0.05,
            smoothness=1.0,                # Very high - smooth parking
            goal_reached=30.0,
        )
    
    @staticmethod
    def defensive() -> RewardWeights:
        """Defensive driving - maximum safety margin."""
        return RewardWeights(
            goal_progress=0.8,
            route_completion=0.3,
            speed_efficiency=0.1,
            collision_penalty=-100.0,     # Extreme - no collisions
            off_road_penalty=-50.0,
            timeout_penalty=-0.5,
            comfort_jerk=-0.2,
            comfort_lateral=-0.2,
            smoothness=1.0,
            goal_reached=10.0,
        )


def create_reward_function(
    scenario: str = 'urban',
    custom_weights: Optional[RewardWeights] = None,
    **kwargs
) -> DrivingRewardFunction:
    """
    Factory function to create reward functions.
    
    Args:
        scenario: One of 'highway', 'urban', 'parking', 'defensive'
        custom_weights: Optional custom weights to override preset
        **kwargs: Additional arguments to DrivingRewardFunction
        
    Returns:
        Configured DrivingRewardFunction
    """
    if custom_weights is not None:
        weights = custom_weights
    elif scenario == 'highway':
        weights = ScenarioRewardConfig.highway()
    elif scenario == 'parking':
        weights = ScenarioRewardConfig.parking()
    elif scenario == 'defensive':
        weights = ScenarioRewardConfig.defensive()
    else:
        weights = ScenarioRewardConfig.urban()
        
    return DrivingRewardFunction(weights=weights, **kwargs)


# CLI for testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Driving Reward Function CLI')
    parser.add_argument('--scenario', type=str, default='urban',
                        choices=['highway', 'urban', 'parking', 'defensive'],
                        help='Driving scenario type')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of test episodes')
    args = parser.parse_args()
    
    # Create reward function
    reward_fn = create_reward_function(args.scenario)
    
    print(f"=== {args.scenario.title()} Scenario Reward Config ===")
    print(f"Weights:")
    for k, v in vars(reward_fn.weights).items():
        print(f"  {k}: {v}")
    print(f"\nParameters:")
    print(f"  goal_threshold: {reward_fn.goal_threshold}")
    print(f"  max_speed: {reward_fn.max_speed}")
    print(f"  optimal_speed: {reward_fn.optimal_speed}")
    
    # Simulate a simple episode
    reward_fn.reset()
    state = np.array([0, 0, 0, 5, 100, 0])  # x, y, heading, speed, goal_x, goal_y
    
    print(f"\n=== Sample Episode ===")
    total_reward = 0
    
    for step in range(50):
        action = np.array([1.0, 0.1])  # delta waypoint
        next_state = state + np.array([0.5, 0, 0, 0, -1, 0])  # Simplified
        distance_to_goal = np.linalg.norm(next_state[4:6] - next_state[:2])
        
        info = {
            'distance_to_goal': distance_to_goal,
            'prev_distance_to_goal': distance_to_goal + 0.5,
            'route_completion': min(step / 50, 1.0),
            'speed': 5.0,
            'collision': False,
            'off_road': False,
            'goal_reached': distance_to_goal < reward_fn.goal_threshold,
        }
        
        done = info['goal_reached'] or step == 49
        reward, metrics = reward_fn.compute(state, action, next_state, done, info)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.2f}, progress={metrics.goal_progress:.2f}, "
                  f"total={total_reward:.2f}")
        
        if done:
            break
        state = next_state
        
    print(f"\nFinal episode reward: {total_reward:.2f}")
    print(f"Final metrics: goal_progress={metrics.goal_progress:.2f}, "
          f"route_completion={metrics.route_completion:.2f}")

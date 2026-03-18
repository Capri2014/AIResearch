"""
Traffic-Aware Waypoint Environment.

Extends the kinematic waypoint environment with dynamic traffic obstacles.
This bridges the gap between simple kinematic training and real CARLA scenarios.

Features:
- Dynamic traffic vehicles that follow predefined paths
- Collision detection with traffic
- Traffic-aware reward shaping
- Multiple traffic density levels

Usage:
    python -m training.rl.traffic_aware_waypoint_env --episodes 10
    python -m training.rl.traffic_aware_waypoint_env --traffic-density high --episodes 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from training.rl.kinematic_waypoint_env import (
    KinematicWaypointConfig,
    KinematicVehicle,
    KinematicWaypointEnv,
)


# ============================================================================
# Configuration
# ============================================================================

class TrafficDensity(Enum):
    """Traffic density levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TrafficVehicle:
    """A traffic vehicle in the environment."""
    x: float
    y: float
    heading: float  # radians
    speed: float  # m/s
    path: List[Tuple[float, float]]  # Waypoints to follow
    path_index: int = 0
    width: float = 2.0  # meters
    length: float = 4.5  # meters
    
    def update(self, dt: float):
        """Update position along path."""
        if self.path_index >= len(self.path) - 1:
            # Reached end of path, stay in place
            return
        
        # Target waypoint
        target = self.path[self.path_index]
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist < 1.0:
            # Reached waypoint, move to next
            self.path_index += 1
            if self.path_index >= len(self.path) - 1:
                return
        
        # Move towards target
        if dist > 0.1:
            self.heading = math.atan2(dy, dx)
            self.x += math.cos(self.heading) * self.speed * dt
            self.y += math.sin(self.heading) * self.speed * dt


@dataclass
class TrafficAwareWaypointConfig(KinematicWaypointConfig):
    """Configuration for traffic-aware waypoint environment."""
    # Traffic
    traffic_density: TrafficDensity = TrafficDensity.LOW
    num_static_obstacles: int = 3
    num_dynamic_obstacles: int = 5
    obstacle_radius: float = 2.0  # meters
    
    # Traffic vehicle settings
    traffic_speed_range: Tuple[float, float] = (3.0, 8.0)  # m/s
    traffic_spawn_margin: float = 20.0  # meters from ego vehicle
    
    # Collision
    collision_margin: float = 1.5  # meters
    
    # Rewards
    collision_penalty: float = -20.0
    near_collision_penalty: float = -2.0
    traffic_clearance_reward: float = 0.5


# ============================================================================
# Traffic Scenario Generators
# ============================================================================

def generate_straight_traffic(
    num_vehicles: int,
    world_size: float,
    ego_start: Tuple[float, float],
    direction: str = "opposite",
) -> List[TrafficVehicle]:
    """Generate traffic vehicles moving in straight lines.
    
    Args:
        num_vehicles: Number of traffic vehicles
        world_size: Size of the world
        ego_start: Starting position of ego vehicle
        direction: "same" or "opposite" direction relative to ego
    
    Returns:
        List of traffic vehicles
    """
    vehicles = []
    margin = 20.0
    
    for i in range(num_vehicles):
        # Random lane position
        lane_offset = random.choice([-3.5, 3.5])  # Typical lane width
        y = ego_start[1] + lane_offset + random.uniform(-1, 1)
        
        # Ensure within world bounds
        if y < -world_size / 2 + margin or y > world_size / 2 - margin:
            continue
        
        # Position ahead or behind ego
        if direction == "opposite":
            x = ego_start[0] + random.uniform(-world_size / 2, -10)
            heading = 0.0  # Facing ego
        else:
            x = ego_start[0] + random.uniform(10, world_size / 2)
            heading = math.pi  # Same direction as ego
        
        speed = random.uniform(3.0, 8.0)
        
        # Simple straight-line path
        path = [(x, y), (x + 100 * math.cos(heading), y + 100 * math.sin(heading))]
        
        vehicles.append(TrafficVehicle(
            x=x, y=y, heading=heading, speed=speed, path=path
        ))
    
    return vehicles


def generate_cross_traffic(
    num_vehicles: int,
    world_size: float,
    ego_start: Tuple[float, float],
) -> List[TrafficVehicle]:
    """Generate traffic vehicles crossing the intersection.
    
    Args:
        num_vehicles: Number of traffic vehicles
        world_size: Size of the world
        ego_start: Starting position of ego vehicle
    
    Returns:
        List of traffic vehicles
    """
    vehicles = []
    
    for i in range(num_vehicles):
        # Choose crossing direction
        if random.random() < 0.5:
            # Crossing from left to right
            x = -world_size / 4 + random.uniform(-10, 10)
            y = ego_start[1] + random.uniform(-20, 20)
            heading = 0.0
        else:
            # Crossing from right to left
            x = world_size / 4 + random.uniform(-10, 10)
            y = ego_start[1] + random.uniform(-20, 20)
            heading = math.pi
        
        speed = random.uniform(4.0, 8.0)
        
        path = [
            (x, y),
            (x + 50 * math.cos(heading), y + 50 * math.sin(heading))
        ]
        
        vehicles.append(TrafficVehicle(
            x=x, y=y, heading=heading, speed=speed, path=path
        ))
    
    return vehicles


def generate_turn_traffic(
    num_vehicles: int,
    world_size: float,
    ego_start: Tuple[float, float],
) -> List[TrafficVehicle]:
    """Generate traffic vehicles turning at intersections.
    
    Args:
        num_vehicles: Number of traffic vehicles
        world_size: Size of the world
        ego_start: Starting position of ego vehicle
    
    Returns:
        List of traffic vehicles
    """
    vehicles = []
    
    for i in range(num_vehicles):
        # Start from one of 4 directions
        direction = random.randint(0, 3)
        
        if direction == 0:  # North to East
            x, y = -10, world_size / 4
            path = [(-10, y), (0, y), (10, y + 10)]
            heading = math.pi / 2
        elif direction == 1:  # East to South
            x, y = world_size / 4, 10
            path = [(x, 10), (x, 0), (x - 10, -10)]
            heading = -math.pi / 2
        elif direction == 2:  # South to West
            x, y = 10, -world_size / 4
            path = [(10, y), (0, y), (-10, y - 10)]
            heading = -math.pi / 2
        else:  # West to North
            x, y = -world_size / 4, -10
            path = [(x, -10), (x, 0), (x + 10, 10)]
            heading = math.pi / 2
        
        speed = random.uniform(3.0, 6.0)
        
        vehicles.append(TrafficVehicle(
            x=x, y=y, heading=heading, speed=speed, path=path
        ))
    
    return vehicles


def generate_static_obstacles(
    num_obstacles: int,
    world_size: float,
    ego_start: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Generate static obstacle positions.
    
    Args:
        num_obstacles: Number of static obstacles
        world_size: Size of the world
        ego_start: Starting position of ego vehicle
    
    Returns:
        List of (x, y) obstacle positions
    """
    obstacles = []
    margin = 10.0
    
    for _ in range(num_obstacles):
        # Random position not too close to ego
        for _ in range(10):  # Try 10 times
            x = random.uniform(-world_size / 2 + margin, world_size / 2 - margin)
            y = random.uniform(-world_size / 2 + margin, world_size / 2 - margin)
            
            # Check distance from ego
            dx = x - ego_start[0]
            dy = y - ego_start[1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist > 15.0:  # Not too close to ego
                obstacles.append((x, y))
                break
    
    return obstacles


# ============================================================================
# Collision Detection
# ============================================================================

def check_collision(
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    other_x: float = 0.0,
    other_y: float = 0.0,
    other_length: float = 4.5,
    other_width: float = 2.0,
) -> bool:
    """Check collision between two vehicles using simple circle approximation.
    
    Args:
        ego_x, ego_y: Ego vehicle position
        ego_heading: Ego vehicle heading
        ego_length, ego_width: Ego vehicle dimensions
        other_x, other_y: Other vehicle position
        other_length, other_width: Other vehicle dimensions
    
    Returns:
        True if collision detected
    """
    # Simple circle-based collision
    ego_radius = max(ego_length, ego_width) / 2
    other_radius = max(other_length, other_width) / 2
    
    dx = ego_x - other_x
    dy = ego_y - other_y
    dist = math.sqrt(dx * dx + dy * dy)
    
    return dist < (ego_radius + other_radius)


def check_near_collision(
    ego_x: float,
    ego_y: float,
    other_x: float,
    other_y: float,
    threshold: float = 5.0,
) -> bool:
    """Check if vehicle is near collision.
    
    Args:
        ego_x, ego_y: Ego vehicle position
        other_x, other_y: Other vehicle position
        threshold: Distance threshold for near-collision
    
    Returns:
        True if near collision
    """
    dx = ego_x - other_x
    dy = ego_y - other_y
    dist = math.sqrt(dx * dx + dy * dy)
    
    return dist < threshold


# ============================================================================
# Traffic-Aware Environment
# ============================================================================

class TrafficAwareWaypointEnv:
    """Waypoint follower environment with dynamic traffic.
    
    This environment extends the base kinematic waypoint environment
    with:
    - Static obstacles
    - Dynamic traffic vehicles
    - Collision detection
    - Traffic-aware rewards
    """
    
    def __init__(self, config: TrafficAwareWaypointConfig):
        self.config = config
        self.vehicle = KinematicVehicle(config)
        self._reset()
    
    def _reset(self):
        """Reset environment state."""
        # Reset ego vehicle
        self.vehicle.reset()
        
        # Generate goal
        self._generate_goal()
        
        # Generate traffic
        self._generate_traffic()
        
        # Generate static obstacles
        self.static_obstacles = generate_static_obstacles(
            self.config.num_static_obstacles,
            self.config.world_size,
            (self.vehicle.x, self.vehicle.y),
        )
        
        # Episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.collision_occurred = False
        self.near_collisions = 0
    
    def _generate_goal(self):
        """Generate a random goal position."""
        margin = 20.0
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(20, self.config.world_size / 2 - margin)
        
        self.goal_x = self.vehicle.x + math.cos(angle) * distance
        self.goal_y = self.vehicle.y + math.sin(angle) * distance
    
    def _generate_traffic(self):
        """Generate traffic vehicles based on density."""
        density = self.config.traffic_density
        
        if density == TrafficDensity.NONE:
            self.traffic = []
            return
        
        # Number of vehicles based on density
        if density == TrafficDensity.LOW:
            num_straight = 2
            num_cross = 1
            num_turn = 0
        elif density == TrafficDensity.MEDIUM:
            num_straight = 4
            num_cross = 2
            num_turn = 1
        else:  # HIGH
            num_straight = 6
            num_cross = 3
            num_turn = 2
        
        ego_start = (self.vehicle.x, self.vehicle.y)
        
        # Generate traffic
        self.traffic = []
        self.traffic.extend(generate_straight_traffic(
            num_straight, self.config.world_size, ego_start, "opposite"
        ))
        self.traffic.extend(generate_cross_traffic(
            num_cross, self.config.world_size, ego_start
        ))
        self.traffic.extend(generate_turn_traffic(
            num_turn, self.config.world_size, ego_start
        ))
    
    def _get_waypoints(self) -> np.ndarray:
        """Get SFT waypoint predictions (straight-line to goal)."""
        num_wp = self.config.num_waypoints
        spacing = self.config.waypoint_spacing
        
        waypoints = np.zeros((num_wp, 2))
        for i in range(num_wp):
            t = (i + 1) / num_wp
            waypoints[i, 0] = self.vehicle.x + (self.goal_x - self.vehicle.x) * t
            waypoints[i, 1] = self.vehicle.y + (self.goal_y - self.vehicle.y) * t
        
        return waypoints
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation.
        
        Returns:
            State vector: [ego_x, ego_y, ego_heading, ego_speed, 
                          goal_x, goal_y, num_traffic, traffic_0_x, traffic_0_y, ...]
        """
        state = [
            self.vehicle.x,
            self.vehicle.y,
            self.vehicle.heading,
            self.vehicle.speed,
            self.goal_x,
            self.goal_y,
            len(self.traffic),
        ]
        
        # Add traffic positions (up to max)
        max_traffic = 10
        for i in range(max_traffic):
            if i < len(self.traffic):
                t = self.traffic[i]
                state.extend([t.x, t.y, t.speed])
            else:
                state.extend([0.0, 0.0, 0.0])
        
        # Add static obstacles (up to max)
        max_obstacles = 5
        for i in range(max_obstacles):
            if i < len(self.static_obstacles):
                o = self.static_obstacles[i]
                state.extend([o[0], o[1]])
            else:
                state.extend([0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
    
    def _compute_reward(
        self,
        waypoints: np.ndarray,
        prev_distance_to_goal: float,
    ) -> float:
        """Compute reward for current step."""
        # Distance to goal
        dx = self.goal_x - self.vehicle.x
        dy = self.goal_y - self.vehicle.y
        distance_to_goal = math.sqrt(dx * dx + dy * dy)
        
        # Progress reward
        progress = prev_distance_to_goal - distance_to_goal
        reward = progress * self.config.progress_weight
        
        # Waypoint tracking reward
        if len(waypoints) > 0:
            closest_idx = np.argmin(
                np.sum((waypoints - [self.vehicle.x, self.vehicle.y]) ** 2, axis=1)
            )
            wp = waypoints[closest_idx]
            wp_dist = math.sqrt(
                (wp[0] - self.vehicle.x) ** 2 + (wp[1] - self.vehicle.y) ** 2
            )
            reward -= wp_dist * 0.1
        
        # Time penalty
        reward += self.config.time_penalty
        
        # Collision penalty
        if self.collision_occurred:
            reward += self.config.collision_penalty
        
        # Near collision penalty
        reward += self.near_collisions * self.config.near_collision_penalty
        
        # Success bonus
        if distance_to_goal < self.config.success_radius:
            reward += self.config.success_bonus
        
        return reward
    
    def step(
        self,
        steer: float,
        throttle: float,
        waypoints: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            steer: Steering angle
            throttle: Throttle value
            waypoints: Optional waypoint predictions from SFT model
        
        Returns:
            observation: State vector
            reward: Reward for this step
            done: Whether episode is done
            info: Additional info dict
        """
        # Get waypoints from SFT if not provided
        if waypoints is None:
            waypoints = self._get_waypoints()
        
        # Previous distance to goal
        prev_dx = self.goal_x - self.vehicle.x
        prev_dy = self.goal_y - self.vehicle.y
        prev_distance_to_goal = math.sqrt(prev_dx * prev_dx + prev_dy * prev_dy)
        
        # Update ego vehicle
        self.vehicle.step(steer, throttle, self.config.dt)
        
        # Update traffic
        for t in self.traffic:
            t.update(self.config.dt)
        
        # Check collisions
        self.collision_occurred = False
        self.near_collisions = 0
        
        # Check static obstacle collisions
        for obs in self.static_obstacles:
            dx = self.vehicle.x - obs[0]
            dy = self.vehicle.y - obs[1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < self.config.collision_margin:
                self.collision_occurred = True
            
            if dist < self.config.collision_margin * 2:
                self.near_collisions += 1
        
        # Check traffic collisions
        for t in self.traffic:
            if check_collision(
                self.vehicle.x, self.vehicle.y, self.vehicle.heading,
                other_x=t.x, other_y=t.y,
            ):
                self.collision_occurred = True
            
            if check_near_collision(
                self.vehicle.x, self.vehicle.y, t.x, t.y, threshold=5.0
            ):
                self.near_collisions += 1
        
        # Compute reward
        reward = self._compute_reward(waypoints, prev_distance_to_goal)
        
        # Check done conditions
        self.step_count += 1
        self.episode_reward += reward
        
        # Distance to goal
        dx = self.goal_x - self.vehicle.x
        dy = self.goal_y - self.vehicle.y
        distance_to_goal = math.sqrt(dx * dx + dy * dy)
        
        done = (
            self.step_count >= self.config.max_episode_steps or
            distance_to_goal < self.config.success_radius or
            self.collision_occurred
        )
        
        info = {
            "step": self.step_count,
            "distance_to_goal": distance_to_goal,
            "collision": self.collision_occurred,
            "near_collisions": self.near_collisions,
            "num_traffic": len(self.traffic),
        }
        
        return self._get_state(), reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset the environment.
        
        Returns:
            Initial observation
        """
        self._reset()
        return self._get_state()
    
    @property
    def state_dim(self) -> int:
        """Get state dimension."""
        # ego_x, ego_y, ego_heading, ego_speed, goal_x, goal_y, num_traffic
        # + 10 traffic * 3 (x, y, speed) + 5 obstacles * 2 (x, y)
        return 7 + 10 * 3 + 5 * 2


# ============================================================================
# RL Training Integration
# ============================================================================

def train_traffic_aware(
    episodes: int = 100,
    traffic_density: TrafficDensity = TrafficDensity.MEDIUM,
    output_dir: str = "out/traffic_aware_waypoint",
) -> Dict:
    """Train a simple policy on traffic-aware environment.
    
    Args:
        episodes: Number of training episodes
        traffic_density: Traffic density level
        output_dir: Output directory for checkpoints
    
    Returns:
        Training metrics dict
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Configuration
    config = TrafficAwareWaypointConfig(
        traffic_density=traffic_density,
        num_dynamic_obstacles=5,
    )
    
    env = TrafficAwareWaypointEnv(config)
    
    # Simple MLP policy
    class Policy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh(),
            )
        
        def forward(self, x):
            return self.net(x)
    
    policy = Policy(env.state_dim, 2)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    # Training metrics
    rewards = []
    collisions = 0
    successes = 0
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(config.max_episode_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            # Get action from policy
            with torch.no_grad():
                action = policy(obs_tensor).numpy()
            
            steer = action[0] * config.max_steer
            throttle = (action[1] + 1) / 2  # [0, 1]
            
            # Step environment
            next_obs, reward, done, info = env.step(steer, throttle)
            
            # Simple gradient update (policy gradient)
            optimizer.zero_grad()
            
            # Simple loss: maximize reward
            # Convert reward to tensor for gradient computation
            reward_tensor = torch.tensor(reward, dtype=torch.float32, requires_grad=True)
            loss = -reward_tensor * 0.01
            loss.backward()
            optimizer.step()
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        rewards.append(episode_reward)
        if info["collision"]:
            collisions += 1
        if info["distance_to_goal"] < config.success_radius:
            successes += 1
        
        if (episode + 1) % 10 == 0:
            mean_reward = np.mean(rewards[-10:])
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Mean reward: {mean_reward:.2f}, "
                  f"Success rate: {successes / (episode + 1):.2%}, "
                  f"Collision rate: {collisions / (episode + 1):.2%}")
    
    # Save checkpoint
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), f"{output_dir}/policy.pt")
    
    metrics = {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "final_reward": float(rewards[-1]),
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
    }
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Traffic-aware waypoint environment")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--traffic-density",
        type=str,
        default="low",
        choices=["none", "low", "medium", "high"],
        help="Traffic density level",
    )
    parser.add_argument("--output-dir", type=str, default="out/traffic_aware", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    
    density = TrafficDensity(args.traffic_density)
    
    if args.test:
        # Smoke test
        config = TrafficAwareWaypointConfig(traffic_density=density)
        env = TrafficAwareWaypointEnv(config)
        
        obs = env.reset()
        print(f"State shape: {obs.shape}")
        print(f"State dim: {env.state_dim}")
        
        # Run a few steps
        for i in range(5):
            steer = random.uniform(-0.5, 0.5)
            throttle = random.uniform(0.3, 0.7)
            obs, reward, done, info = env.step(steer, throttle)
            print(f"Step {i + 1}: reward={reward:.2f}, done={done}, info={info}")
        
        print("Smoke test: PASSED")
        return
    
    # Train
    print(f"Training with {args.traffic_density} traffic density for {args.episodes} episodes")
    metrics = train_traffic_aware(
        episodes=args.episodes,
        traffic_density=density,
        output_dir=args.output_dir,
    )
    
    print(f"\nTraining complete!")
    print(f"Mean reward: {metrics['mean_reward']:.2f}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Collision rate: {metrics['collision_rate']:.2%}")


if __name__ == "__main__":
    main()

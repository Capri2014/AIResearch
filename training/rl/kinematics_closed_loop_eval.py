#!/usr/bin/env python3
"""
Kinematics-based Closed-Loop Evaluation for RL Waypoint Policy

Uses the kinematics RL environment to evaluate the RL-refined waypoint policy
in a closed-loop simulation without needing CARLA.

This provides a sanity check before CARLA evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add training to path
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))


class KinematicsClosedLoopEnv:
    """
    Simplified closed-loop kinematics environment for waypoint evaluation.
    
    Features:
    - 2D vehicle kinematics (bicycle model)
    - Waypoint following
    - Collision detection with obstacles
    - Route completion tracking
    """
    
    def __init__(self, num_waypoints: int = 20, dt: float = 0.1):
        self.num_waypoints = num_waypoints
        self.dt = dt
        
        # Vehicle params (typical sedan)
        self.wheelbase = 2.5  # meters
        self.max_steer = np.radians(35)
        self.max_speed = 15.0  # m/s
        
        # State: [x, y, yaw, speed]
        self.state = None
        self.waypoints = None
        self.target_waypoint = None  # Current target
        self.episode_step = 0
        self.max_episode_steps = 500
        
        # Obstacles (static boxes)
        self.obstacles = []
        
    def reset(self, start_pos: Tuple[float, float, float] = (0, 0, 0),
              waypoints: np.ndarray = None) -> np.ndarray:
        """Reset environment to starting position."""
        self.state = np.array([start_pos[0], start_pos[1], start_pos[2], 0.0])
        self.episode_step = 0
        
        if waypoints is not None:
            self.waypoints = waypoints.copy()
            self.target_waypoint = waypoints[0].copy()  # Start with first waypoint
        else:
            # Default straight road
            t = np.linspace(0, 100, self.num_waypoints)
            self.waypoints = np.stack([
                t,
                np.zeros(self.num_waypoints),
                np.zeros(self.num_waypoints)
            ], axis=1)
            self.target_waypoint = self.waypoints[0].copy()
            
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation for policy (state + waypoints)."""
        # State: 4 values (x, y, yaw, speed)
        # Waypoints: 20 waypoints * 2 (x, y) = 40 values
        # Target: 2 values (target_x, target_y)
        obs = np.zeros(46)
        obs[:4] = self.state
        if self.waypoints is not None:
            obs[4:44] = self.waypoints[:, :2].flatten()
        if self.target_waypoint is not None:
            obs[44:] = self.target_waypoint[:2]
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step the environment.
        
        Action: [dx_0, dy_0, dx_1, dy_1, ...] = waypoint deltas (40 values)
        """
        # Apply action (waypoint deltas) to adjust target waypoint
        if len(action) >= 2:
            delta = action[:2]
            self.target_waypoint[:2] += delta * 0.5  # Scale down deltas
            
        # Get current target for steering
        target = self.target_waypoint[:2]
        
        # Bicycle model kinematics
        x, y, yaw, speed = self.state
        
        # Compute steering to reach target
        dx = target[0] - x
        dy = target[1] - y
        target_yaw = np.arctan2(dy, dx)
        yaw_error = target_yaw - yaw
        
        # Normalize to [-pi, pi]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
            
        steer = np.clip(yaw_error / np.radians(35), -1, 1) * np.radians(35)
        
        # Speed control (simple)
        dist_to_target = np.linalg.norm(target - self.state[:2])
        target_speed = min(self.max_speed, dist_to_target / 2)
        throttle = (target_speed - speed) / self.max_speed
        throttle = np.clip(throttle, -1, 1) * self.max_speed
        
        # Update state
        speed = np.clip(speed + throttle * self.dt, 0, self.max_speed)
        
        if speed > 0.1:
            dx = speed * np.cos(yaw) * self.dt
            dy = speed * np.sin(yaw) * self.dt
            dyaw = speed / self.wheelbase * np.tan(steer) * self.dt
        else:
            dx = 0
            dy = 0
            dyaw = 0
            
        x += dx
        y += dy
        yaw += dyaw
        
        self.state = np.array([x, y, yaw, speed])
        
        # Update target waypoint if close enough
        if self.waypoints is not None:
            for i in range(len(self.waypoints)):
                dist = np.linalg.norm(self.waypoints[i][:2] - self.state[:2])
                if dist < 3.0:
                    self.target_waypoint = self.waypoints[i].copy()
                    break
                
        self.episode_step += 1
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check done
        done = self._is_done()
        
        info = {
            "progress": self._compute_progress(),
            "speed": speed,
            "distance_to_goal": np.linalg.norm(
                self.waypoints[-1][:2] - self.state[:2]
            ) if self.waypoints is not None else 0
        }
        
        return self._get_obs(), reward, done, info
    
    def _compute_reward(self) -> float:
        """Compute reward based on progress and behavior."""
        if self.waypoints is None:
            return 0
            
        # Distance to target waypoint
        target = self.target_waypoint[:2] if self.target_waypoint is not None else self.waypoints[0][:2]
        dist = np.linalg.norm(target - self.state[:2])
        
        # Reward for being close to waypoint
        reward = -dist * 0.1
        
        # Progress reward
        reward += self._compute_progress() * 10
        
        # Speed reward
        reward += self.state[3] * 0.1
        
        return reward
    
    def _compute_progress(self) -> float:
        """Compute route completion progress."""
        if self.waypoints is None:
            return 0
            
        # Find closest waypoint
        min_dist = float('inf')
        closest_idx = 0
        for i, wp in enumerate(self.waypoints):
            dist = np.linalg.norm(wp[:2] - self.state[:2])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx / max(1, len(self.waypoints) - 1)
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Max steps
        if self.episode_step >= self.max_episode_steps:
            return True
            
        # Reached goal
        if self.waypoints is not None:
            dist_to_goal = np.linalg.norm(
                self.waypoints[-1][:2] - self.state[:2]
            )
            if dist_to_goal < 2.0:
                return True
                
        # Collision with obstacles
        for ox, oy, w, h in self.obstacles:
            if (abs(self.state[0] - ox) < w/2 + 1 and 
                abs(self.state[1] - oy) < h/2 + 1):
                return True
                
        return False
    
    def render(self):
        """Render (stub for visualization)."""
        pass


class PPOWaypointPolicy:
    """PPO policy that loads from rl_refine_from_bc.py checkpoint."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.policy_net = None
        self.exploration_std = 0.1
        
        if model_path and os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        """Load PPO policy from checkpoint."""
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            
            if isinstance(checkpoint, dict):
                # PPO checkpoint with policy and value networks
                if "policy_state_dict" in checkpoint:
                    self.policy_net = self._build_policy_network()
                    self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
                    self.policy_net.eval()
                    print(f"Loaded PPO policy from {self.model_path}")
                    
        except Exception as e:
            print(f"Could not load model: {e}")
            self.policy_net = None
    
    def _build_policy_network(self):
        """Build PPO policy network matching rl_refine_from_bc.py."""
        class PPOPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_std = nn.Parameter(torch.zeros(40))
                self.net = nn.Sequential(
                    nn.Linear(46, 128),  # 20 waypoints * 2 + 4 state + 2 target
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 40),  # waypoint deltas
                    nn.Tanh()
                )
                
            def forward(self, x):
                mean = self.net(x)
                return mean
                
        return PPOPolicy()
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action from observation.
        
        Returns: [dx_0, dy_0, ..., dx_19, dy_19] waypoint deltas
        """
        if self.policy_net is not None:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = self.policy_net(obs_tensor).numpy().squeeze()
            
            # Add exploration noise
            noise = np.random.randn(40) * self.exploration_std
            action = np.clip(action + noise, -1, 1)
        else:
            # No model - return zeros
            action = np.zeros(40)
        
        return action
    
    def reset(self):
        """Reset policy state."""
        pass


def run_kinematics_eval(
    model_path: str,
    num_episodes: int = 10,
    waypoint_noise: float = 0.0
) -> Dict:
    """
    Run kinematics closed-loop evaluation.
    
    Args:
        model_path: Path to RL model
        num_episodes: Number of evaluation episodes
        waypoint_noise: Std dev of noise to add to waypoints
        
    Returns:
        Evaluation metrics
    """
    print("=" * 60)
    print("Kinematics Closed-Loop Evaluation (PPO Policy)")
    print("=" * 60)
    
    # Create environment and policy
    env = KinematicsClosedLoopEnv()
    policy = PPOWaypointPolicy(model_path)
    
    results = []
    
    for episode in range(num_episodes):
        # Generate route (straight with slight curves)
        t = np.linspace(0, 1, env.num_waypoints)
        waypoints = np.zeros((env.num_waypoints, 3))
        waypoints[:, 0] = t * 100  # 100m route
        waypoints[:, 1] = np.sin(t * np.pi * 2) * 5  # slight curve
        waypoints[:, 2] = 0  # yaw (will be computed)
        
        # Add noise if requested
        if waypoint_noise > 0:
            waypoints[:, :2] += np.random.randn(*waypoints[:, :2].shape) * waypoint_noise
        
        # Compute waypoint yaws
        for i in range(len(waypoints) - 1):
            dx = waypoints[i+1, 0] - waypoints[i, 0]
            dy = waypoints[i+1, 1] - waypoints[i, 1]
            waypoints[i, 2] = np.arctan2(dy, dx)
        waypoints[-1, 2] = waypoints[-2, 2]
        
        # Reset environment
        obs = env.reset(start_pos=(0, 0, 0), waypoints=waypoints)
        policy.reset()
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Get action from policy
            action = policy.act(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
        # Record results
        results.append({
            "episode": episode,
            "total_reward": episode_reward,
            "steps": steps,
            "progress": info["progress"],
            "success": info["distance_to_goal"] < 5.0,
        })
        
        print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, "
              f"progress={info['progress']:.2f}, success={info['distance_to_goal'] < 5.0}")
        
    # Compute metrics
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_progress = np.mean([r["progress"] for r in results])
    
    metrics = {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_progress": avg_progress,
        "num_episodes": num_episodes,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Kinematics closed-loop evaluation for RL waypoint policy"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="out/rl_refine_from_bc/model_final.pt",
        help="Path to RL model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/kinematics_closed_loop_eval",
        help="Output directory",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--waypoint-noise",
        type=float,
        default=0.5,
        help="Standard deviation of waypoint noise (meters)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    metrics = run_kinematics_eval(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        waypoint_noise=args.waypoint_noise,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_file = os.path.join(args.output_dir, "metrics.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
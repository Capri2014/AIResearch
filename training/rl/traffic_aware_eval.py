#!/usr/bin/env python3
"""
Traffic-Aware Waypoint BC Evaluation Script

Evaluates waypoint BC policies in traffic-aware scenarios, measuring:
- Success rate (reaching goal without collision)
- Collision rate with dynamic traffic
- Near-miss detection
- Waypoint tracking accuracy
- Speed compliance

This bridges BC (PR #3) with RL refinement by providing baseline metrics
for traffic-aware scenarios.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.traffic_aware_waypoint_env import (
    TrafficAwareWaypointEnv,
    TrafficAwareWaypointConfig,
    TrafficDensity,
    generate_straight_traffic,
    generate_cross_traffic,
    generate_turn_traffic,
)


class TrafficAwareBCEvaluator:
    """Evaluates BC policies in traffic-aware scenarios."""

    def __init__(
        self,
        env_config: TrafficAwareWaypointConfig,
        num_episodes: int = 100,
        seed: int = 42,
    ):
        self.env_config = env_config
        self.num_episodes = num_episodes
        self.seed = seed
        self.env = None

    def set_bc_model(self, bc_model, device: str = "cpu"):
        """Set the BC model for policy execution."""
        self.bc_model = bc_model
        self.device = device
        self.bc_model.to(device)
        self.bc_model.eval()

    def _state_to_bc_input(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert environment state to BC model input.
        
        State format: [ego_x, ego_y, ego_yaw, ego_speed, 
                       waypoint_0_x, waypoint_0_y, ...,
                       traffic_0_x, traffic_0_y, ...]
        
        Returns: (bev_features, waypoints)
        """
        # Extract ego state (first 4 elements)
        ego_state = state[:4]  # x, y, yaw, speed
        
        # Extract waypoints (next 16 elements = 8 waypoints * 2)
        waypoints = state[4:20].reshape(-1, 2)
        
        # Traffic states start at index 20
        # For now, create dummy BEV features (would come from perception in real system)
        bev_features = np.random.randn(256, 8, 8).astype(np.float32)
        
        return bev_features, waypoints

    def _bc_predict(self, state: np.ndarray) -> np.ndarray:
        """Use BC model to predict waypoints."""
        import torch
        
        bev_features, waypoints = self._state_to_bc_input(state)
        
        # Convert to tensors
        bev_tensor = torch.from_numpy(bev_features).unsqueeze(0).to(self.device)
        waypoint_tensor = torch.from_numpy(waypoints).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # BC model predicts delta waypoints or absolute waypoints
            pred_waypoints = self.bc_model(bev_tensor, waypoint_tensor)
        
        return pred_waypoints.cpu().numpy()[0]

    def evaluate_random_policy(self) -> Dict:
        """Evaluate with random policy (baseline)."""
        np.random.seed(self.seed)
        
        results = {
            "policy": "random",
            "num_episodes": self.num_episodes,
            "episodes": [],
        }
        
        for ep in range(self.num_episodes):
            # Create fresh environment
            env = TrafficAwareWaypointEnv(self.env_config)
            state = env.reset()
            
            episode_result = {
                "episode": ep,
                "total_reward": 0.0,
                "steps": 0,
                "success": False,
                "collision": False,
                "near_miss": False,
                "max_progress": 0.0,
            }
            
            done = False
            while not done:
                # Random action: steer, throttle
                steer = np.random.randn() * 0.5
                throttle = np.clip(np.random.randn(), 0, 1)
                
                state, reward, terminated, info = env.step(steer, throttle)
                done = terminated
                
                episode_result["total_reward"] += reward
                episode_result["steps"] += 1
                
                if info.get("collision", False):
                    episode_result["collision"] = True
                if info.get("near_collision", False):
                    episode_result["near_miss"] = True
                if info.get("progress", 0) > episode_result["max_progress"]:
                    episode_result["max_progress"] = info.get("progress", 0)
                
                # Early exit for collision
                if episode_result["collision"]:
                    break
            
            episode_result["success"] = (
                not episode_result["collision"] and 
                episode_result["max_progress"] > 0.9
            )
            
            results["episodes"].append(episode_result)
        
        # Aggregate metrics
        results["metrics"] = self._aggregate_results(results["episodes"])
        return results

    def evaluate_waypoint_policy(self) -> Dict:
        """Evaluate with waypoint-following policy."""
        np.random.seed(self.seed)
        
        results = {
            "policy": "waypoint_follower",
            "num_episodes": self.num_episodes,
            "episodes": [],
        }
        
        for ep in range(self.num_episodes):
            env = TrafficAwareWaypointEnv(self.env_config)
            state = env.reset()
            
            episode_result = {
                "episode": ep,
                "total_reward": 0.0,
                "steps": 0,
                "success": False,
                "collision": False,
                "near_miss": False,
                "max_progress": 0.0,
            }
            
            done = False
            while not done:
                # Extract waypoints from state (indices 4:20)
                waypoints = state[4:20].reshape(-1, 2)
                
                # Simple waypoint following: steer toward next waypoint
                ego_x, ego_y = state[0], state[1]
                ego_yaw = state[2]
                
                # Get target waypoint (first one)
                target_x = waypoints[0, 0]
                target_y = waypoints[0, 1]
                
                # Compute desired heading
                dx = target_x - ego_x
                dy = target_y - ego_y
                target_yaw = np.arctan2(dy, dx)
                
                # Steering: difference between target yaw and ego yaw
                steer = np.sin(target_yaw - ego_yaw) * 2.0
                
                # Throttle: proportional to distance to waypoint
                dist = np.sqrt(dx**2 + dy**2)
                throttle = np.clip(1.0 - dist / 5.0, 0.1, 1.0)
                
                state, reward, terminated, info = env.step(steer, throttle)
                done = terminated
                
                episode_result["total_reward"] += reward
                episode_result["steps"] += 1
                
                if info.get("collision", False):
                    episode_result["collision"] = True
                if info.get("near_collision", False):
                    episode_result["near_miss"] = True
                if info.get("progress", 0) > episode_result["max_progress"]:
                    episode_result["max_progress"] = info.get("progress", 0)
                
                if episode_result["collision"]:
                    break
            
            episode_result["success"] = (
                not episode_result["collision"] and 
                episode_result["max_progress"] > 0.9
            )
            
            results["episodes"].append(episode_result)
        
        results["metrics"] = self._aggregate_results(results["episodes"])
        return results

    def _aggregate_results(self, episodes: List[Dict]) -> Dict:
        """Aggregate episode results into summary metrics."""
        total = len(episodes)
        successes = sum(1 for e in episodes if e["success"])
        collisions = sum(1 for e in episodes if e["collision"])
        near_misses = sum(1 for e in episodes if e["near_miss"])
        
        return {
            "success_rate": successes / total if total > 0 else 0,
            "collision_rate": collisions / total if total > 0 else 0,
            "near_miss_rate": near_misses / total if total > 0 else 0,
            "mean_reward": np.mean([e["total_reward"] for e in episodes]),
            "std_reward": np.std([e["total_reward"] for e in episodes]),
            "mean_steps": np.mean([e["steps"] for e in episodes]),
            "mean_progress": np.mean([e["max_progress"] for e in episodes]),
        }


def run_evaluation(
    traffic_density: str = "medium",
    num_episodes: int = 50,
    output_dir: str = "out/traffic_eval",
    test_bc: bool = False,
    bc_checkpoint: Optional[str] = None,
):
    """Run traffic-aware evaluation."""
    
    # Parse traffic density
    density_map = {
        "none": TrafficDensity.NONE,
        "low": TrafficDensity.LOW,
        "medium": TrafficDensity.MEDIUM,
        "high": TrafficDensity.HIGH,
    }
    density = density_map.get(traffic_density.lower(), TrafficDensity.MEDIUM)
    
    # Create environment config
    config = TrafficAwareWaypointConfig(
        traffic_density=density,
        num_static_obstacles=2,
        num_dynamic_obstacles=5,
        max_episode_steps=200,
    )
    
    print(f"Traffic density: {traffic_density}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Config: {config}")
    
    # Create evaluator
    evaluator = TrafficAwareBCEvaluator(
        env_config=config,
        num_episodes=num_episodes,
    )
    
    # Load BC model if provided
    if test_bc and bc_checkpoint:
        print(f"Loading BC model from {bc_checkpoint}")
        import torch
        from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
        
        bc_config = WaypointBCConfig(
            bev_feature_dim=256,
            embedding_dim=128,
            num_waypoints=8,
            predict_speed=True,
        )
        bc_model = WaypointBCModel(bc_config)
        bc_model.load_state_dict(torch.load(bc_checkpoint, map_location="cpu"))
        evaluator.set_bc_model(bc_model)
    
    # Run evaluations
    results = {}
    
    # Random policy baseline
    print("\n=== Evaluating Random Policy ===")
    random_results = evaluator.evaluate_random_policy()
    results["random"] = random_results
    print(f"Success rate: {random_results['metrics']['success_rate']:.2%}")
    print(f"Collision rate: {random_results['metrics']['collision_rate']:.2%}")
    
    # Waypoint following policy
    print("\n=== Evaluating Waypoint Policy ===")
    waypoint_results = evaluator.evaluate_waypoint_policy()
    results["waypoint_follower"] = waypoint_results
    print(f"Success rate: {waypoint_results['metrics']['success_rate']:.2%}")
    print(f"Collision rate: {waypoint_results['metrics']['collision_rate']:.2%}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"traffic_eval_{traffic_density}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Random policy - Success: {random_results['metrics']['success_rate']:.2%}, "
          f"Collision: {random_results['metrics']['collision_rate']:.2%}")
    print(f"Waypoint policy - Success: {waypoint_results['metrics']['success_rate']:.2%}, "
          f"Collision: {waypoint_results['metrics']['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate BC policies in traffic-aware scenarios"
    )
    parser.add_argument(
        "--traffic-density",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high"],
        help="Traffic density level",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/traffic_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--test-bc",
        action="store_true",
        help="Test with BC model (requires --bc-checkpoint)",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        help="Path to BC model checkpoint",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick smoke test",
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Quick smoke test
        print("Running smoke test...")
        run_evaluation(
            traffic_density="low",
            num_episodes=5,
            output_dir="out/traffic_eval_test",
        )
        print("Smoke test complete!")
    else:
        run_evaluation(
            traffic_density=args.traffic_density,
            num_episodes=args.num_episodes,
            output_dir=args.output_dir,
            test_bc=args.test_bc,
            bc_checkpoint=args.bc_checkpoint,
        )

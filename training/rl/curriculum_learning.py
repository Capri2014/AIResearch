#!/usr/bin/env python3
"""
Curriculum Learning for Kinematics Waypoint RL.

Implements progressive difficulty scaling to improve training convergence.
Starts with simpler scenarios and gradually increases complexity as the 
agent improves, addressing the 0% success rate in baseline training.

Curriculum stages:
1. Straight paths - minimal turns
2. Gentle curves - small heading changes  
3. Sharp turns - tighter corners
4. Complex routes - multiple waypoints
5. Full difficulty - original challenging scenarios
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))


# ============================================================================
# Curriculum Stages
# ============================================================================

class CurriculumStage:
    """Single curriculum stage with difficulty parameters."""
    
    def __init__(
        self,
        name: str,
        min_distance: float,
        max_distance: float,
        max_heading_change: float,
        num_waypoints: int,
        curvature_scale: float,
        success_threshold: float = 0.7,
        min_episodes: int = 50,
    ):
        self.name = name
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_heading_change = max_heading_change
        self.num_waypoints = num_waypoints
        self.curvature_scale = curvature_scale
        self.success_threshold = success_threshold
        self.min_episodes = min_episodes
        
    def generate_waypoints(self, rng: np.random.Generator) -> np.ndarray:
        """Generate waypoints for this stage."""
        n = self.num_waypoints
        distances = rng.uniform(self.min_distance, self.max_distance, n)
        
        # Generate headings with constraints
        headings = []
        current_heading = rng.uniform(-self.max_heading_change, self.max_heading_change)
        
        for i in range(n):
            if i > 0:
                # Heading changes based on curvature scale
                max_delta = self.max_heading_change * self.curvature_scale
                current_heading += rng.uniform(-max_delta, max_delta)
                current_heading = np.clip(current_heading, -np.pi, np.pi)
            headings.append(current_heading)
        
        # Convert to Cartesian coordinates
        waypoints = np.zeros((n, 2), dtype=np.float32)
        waypoints[0, 0] = distances[0] * np.cos(headings[0])
        waypoints[0, 1] = distances[0] * np.sin(headings[0])
        
        for i in range(1, n):
            dx = distances[i] * np.cos(headings[i])
            dy = distances[i] * np.sin(headings[i])
            waypoints[i] = waypoints[i-1] + np.array([dx, dy])
            
        return waypoints


# ============================================================================
# Curriculum Scheduler
# ============================================================================

CURRICULUM_STAGES = [
    # Stage 1: Straight paths - minimal complexity
    CurriculumStage(
        name="straight",
        min_distance=5.0,
        max_distance=15.0,
        max_heading_change=0.1,
        num_waypoints=3,
        curvature_scale=0.2,
        success_threshold=0.8,
        min_episodes=30,
    ),
    # Stage 2: Gentle curves
    CurriculumStage(
        name="gentle_curves",
        min_distance=10.0,
        max_distance=25.0,
        max_heading_change=0.3,
        num_waypoints=4,
        curvature_scale=0.4,
        success_threshold=0.7,
        min_episodes=40,
    ),
    # Stage 3: Moderate turns
    CurriculumStage(
        name="moderate_turns",
        min_distance=15.0,
        max_distance=35.0,
        max_heading_change=0.6,
        num_waypoints=5,
        curvature_scale=0.6,
        success_threshold=0.6,
        min_episodes=50,
    ),
    # Stage 4: Sharp turns
    CurriculumStage(
        name="sharp_turns",
        min_distance=20.0,
        max_distance=45.0,
        max_heading_change=1.0,
        num_waypoints=6,
        curvature_scale=0.8,
        success_threshold=0.5,
        min_episodes=60,
    ),
    # Stage 5: Full difficulty
    CurriculumStage(
        name="full",
        min_distance=25.0,
        max_distance=60.0,
        max_heading_change=1.5,
        num_waypoints=8,
        curvature_scale=1.0,
        success_threshold=0.4,
        min_episodes=80,
    ),
]


class CurriculumScheduler:
    """Handles curriculum learning progression."""
    
    def __init__(
        self,
        stages: List[CurriculumStage] = None,
        initial_stage: int = 0,
    ):
        self.stages = stages or CURRICULUM_STAGES
        self.current_stage = initial_stage
        self.stage_stats = {s.name: {"successes": 0, "total": 0} for s in self.stages}
        self.episode_count = 0
        
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage]
    
    def record_episode(self, success: bool):
        """Record episode result for current stage."""
        self.episode_count += 1
        stage = self.stages[self.current_stage]
        self.stage_stats[stage.name]["total"] += 1
        if success:
            self.stage_stats[stage.name]["successes"] += 1
            
    def should_advance(self) -> bool:
        """Check if should advance to next stage."""
        stage = self.stages[self.current_stage]
        stats = self.stage_stats[stage.name]
        
        if stats["total"] < stage.min_episodes:
            return False
            
        success_rate = stats["successes"] / stats["total"]
        return success_rate >= stage.success_threshold
    
    def advance_stage(self) -> bool:
        """Attempt to advance to next curriculum stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False  # Already at max difficulty
            
        if self.should_advance():
            self.current_stage += 1
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        stage = self.stages[self.current_stage]
        stats = self.stage_stats[stage.name]
        success_rate = stats["successes"] / max(stats["total"], 1)
        return {
            "current_stage": stage.name,
            "stage_index": self.current_stage,
            "total_stages": len(self.stages),
            "successes": stats["successes"],
            "total_episodes": stats["total"],
            "success_rate": success_rate,
            "should_advance": self.should_advance(),
        }


# ============================================================================
# Simple Mock Policy (for testing curriculum without full model)
# ============================================================================

class MockWaypointPolicy:
    """Simple policy that predicts waypoints towards goal."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(42)
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict waypoint deltas towards target."""
        # Extract target position from observation
        # obs format: [vehicle_x, vehicle_y, vehicle_theta, target_x, target_y, ...waypoints]
        if len(obs) >= 5:
            target_x, target_y = obs[3], obs[4]
        else:
            target_x, target_y = 10.0, 0.0
            
        # Simple proportional controller
        dx = target_x * self.learning_rate
        dy = target_y * self.learning_rate
        
        # Add noise for exploration
        noise = self.rng.uniform(-0.1, 0.1, size=(len(obs) // 2, 2))
        
        return np.array([dx + noise[0, 0], dy + noise[0, 1]], dtype=np.float32)
    
    def update(self, obs: np.ndarray, reward: float):
        """Simple update (mock gradient descent)."""
        # Increase learning rate slightly for positive rewards
        if reward > 0:
            self.learning_rate = min(self.learning_rate * 1.01, 0.5)


# ============================================================================
# Curriculum RL Trainer
# ============================================================================

def run_curriculum_training(
    iterations: int = 100,
    batch_size: int = 16,
    eval_interval: int = 10,
    eval_episodes: int = 20,
    delta_scale: float = 1.0,
    seed: int = 42,
    output_dir: str = None,
) -> Dict:
    """
    Run curriculum learning training.
    
    Args:
        iterations: Number of training iterations per stage
        batch_size: Batch size for PPO updates
        eval_interval: Evaluate every N iterations
        eval_episodes: Episodes per evaluation
        delta_scale: Scale factor for delta waypoints
        seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary with training results
    """
    from training.rl.kinematics_waypoint_env import KinematicsWaypointEnv
    
    rng = np.random.default_rng(seed)
    
    # Create output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f"training/out/curriculum_learning/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize curriculum scheduler
    curriculum = CurriculumScheduler()
    
    # Initialize policy
    policy = MockWaypointPolicy(learning_rate=0.05)
    
    print("=" * 60)
    print("Curriculum Learning Training")
    print("=" * 60)
    print(f"Stages: {len(curriculum.stages)}")
    print(f"Initial stage: {curriculum.get_current_stage().name}")
    print(f"Iterations: {iterations}, Batch: {batch_size}")
    print(f"Delta scale: {delta_scale}")
    print("=" * 60)
    
    all_metrics = []
    current_stage_iter = 0
    
    while curriculum.current_stage < len(curriculum.stages):
        stage = curriculum.get_current_stage()
        print(f"\n--- Stage {curriculum.current_stage + 1}/{len(curriculum.stages)}: {stage.name} ---")
        
        stage_successes = 0
        stage_total = 0
        
        for iteration in range(iterations):
            current_stage_iter += 1
            
            # Generate batch of scenarios from current stage
            batch_rewards = []
            
            for _ in range(batch_size):
                # Create environment - it auto-generates waypoints
                env = KinematicsWaypointEnv(
                    num_waypoints=stage.num_waypoints,
                    max_episode_steps=100,
                    goal_threshold=stage.min_distance * 0.5,  # Easier goals for early stages
                )
                env.reset(seed=int(rng.integers(0, 10000)))
                
                total_reward = 0.0
                done = False
                steps = 0
                while not done and steps < 100:
                    # Get SFT waypoints from environment
                    waypoints = env.get_sft_waypoints()
                    
                    # Apply policy delta (learned adjustment)
                    delta = policy.predict(np.zeros(8))  # Dummy obs - policy uses internal state
                    waypoints = waypoints + delta * delta_scale
                    
                    obs, reward, done, info = env.step(waypoints)
                    total_reward += reward
                    steps += 1
                    
                    # Update policy on reward
                    policy.update(obs, reward)
                
                batch_rewards.append(total_reward)
                
                # Check success (reached within threshold)
                if info.get("success", False):
                    stage_successes += 1
                    curriculum.record_episode(True)
                else:
                    curriculum.record_episode(False)
                stage_total += 1
            
            # Mean reward for batch
            mean_reward = np.mean(batch_rewards)
            
            # Evaluate periodically
            if current_stage_iter % eval_interval == 0:
                eval_successes = 0
                for _ in range(eval_episodes):
                    env = KinematicsWaypointEnv(
                        num_waypoints=stage.num_waypoints,
                        max_episode_steps=100,
                        goal_threshold=stage.min_distance * 0.5,
                    )
                    env.reset(seed=int(rng.integers(0, 10000)))
                    done = False
                    steps = 0
                    while not done and steps < 100:
                        # Get SFT waypoints and apply delta
                        waypoints = env.get_sft_waypoints()
                        delta = policy.predict(np.zeros(8))
                        waypoints = waypoints + delta * delta_scale
                        obs, reward, done, info = env.step(waypoints)
                        steps += 1
                    if info.get("success", False):
                        eval_successes += 1
                
                eval_success_rate = eval_successes / eval_episodes
                stage_stats = curriculum.get_stats()
                
                print(f"Iter {current_stage_iter}: reward={mean_reward:.2f}, "
                      f"eval_success={eval_success_rate:.1%}, "
                      f"stage_success={stage_stats['success_rate']:.1%}")
                
                all_metrics.append({
                    "iteration": current_stage_iter,
                    "stage": stage.name,
                    "stage_index": curriculum.current_stage,
                    "mean_reward": mean_reward,
                    "eval_success_rate": eval_success_rate,
                    "stage_success_rate": stage_stats["success_rate"],
                    "total_episodes": stage_stats["total_episodes"],
                })
        
        # Check for stage advancement
        final_stats = curriculum.get_stats()
        print(f"\nStage {stage.name} complete: "
              f"success={final_stats['success_rate']:.1%}, "
              f"episodes={final_stats['total_episodes']}")
        
        # Always advance for curriculum progression
        if curriculum.current_stage < len(curriculum.stages) - 1:
            curriculum.current_stage += 1
            print(f"Proceeding to next stage (curriculum progression)")
        else:
            break
    
    # Final evaluation across all stages
    print("\n" + "=" * 60)
    print("Final Curriculum Evaluation")
    print("=" * 60)
    
    final_results = {}
    for stage in curriculum.stages:
        stage_successes = 0
        for _ in range(eval_episodes):
            env = KinematicsWaypointEnv(
                num_waypoints=stage.num_waypoints,
                max_episode_steps=100,
                goal_threshold=stage.min_distance * 0.5,
            )
            env.reset(seed=int(rng.integers(0, 10000)))
            done = False
            steps = 0
            while not done and steps < 100:
                # Get SFT waypoints and apply delta
                waypoints = env.get_sft_waypoints()
                delta = policy.predict(np.zeros(8))
                waypoints = waypoints + delta * delta_scale
                obs, reward, done, info = env.step(waypoints)
                steps += 1
            if info.get("success", False):
                stage_successes += 1
        
        success_rate = stage_successes / eval_episodes
        final_results[stage.name] = {
            "success_rate": success_rate,
            "successes": stage_successes,
            "episodes": eval_episodes,
        }
        print(f"{stage.name}: {success_rate:.1%} ({stage_successes}/{eval_episodes})")
    
    # Calculate overall metrics
    overall_success = np.mean([r["success_rate"] for r in final_results.values()])
    
    # Write metrics
    metrics = {
        "domain": "curriculum_learning",
        "iterations": current_stage_iter,
        "stages_completed": curriculum.current_stage + 1,
        "total_stages": len(curriculum.stages),
        "overall_success_rate": overall_success,
        "stage_results": final_results,
        "training_metrics": all_metrics[-10:] if all_metrics else [],
    }
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nOverall success rate: {overall_success:.1%}")
    print(f"Metrics: {metrics_path}")
    
    return metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Curriculum Learning for Kinematics Waypoint RL")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Iterations per curriculum stage")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for PPO updates")
    parser.add_argument("--eval-interval", type=int, default=10,
                       help="Evaluate every N iterations")
    parser.add_argument("--eval-episodes", type=int, default=20,
                       help="Episodes per evaluation")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                       help="Delta waypoint scale factor")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    
    args = parser.parse_args()
    
    run_curriculum_training(
        iterations=args.iterations,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        delta_scale=args.delta_scale,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
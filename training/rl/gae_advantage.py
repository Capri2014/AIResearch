#!/usr/bin/env python3
"""
GAE (Generalized Advantage Estimation) Utility for Waypoint RL

This is a small slice of the RL-after-SFT stack for waypoint deltas.
Provides reusable GAE computation for PPO/GRPO training.

Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(z)

Output: out/<run_id>/train_metrics.json (schema-compliant)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GAEConfig:
    """Configuration for GAE advantage estimation."""
    gamma: float = 0.99      # Discount factor
    gae_lambda: float = 0.95 # GAE lambda (bias-variance tradeoff)
    normalize: bool = True   # Normalize advantages


# ============================================================================
# GAE Advantage Estimation
# ============================================================================

class GAEAdvantage:
    """
    Generalized Advantage Estimation (GAE) for RL training.
    
    GAE provides a bias-variance tradeoff in advantage estimation:
    - lambda=0: TD(0) advantage (low variance, high bias)
    - lambda=1: Monte Carlo advantage (high variance, low bias)
    
    Paper: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    """

    def __init__(self, config: Optional[GAEConfig] = None):
        self.config = config or GAEConfig()

    def compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and value targets.
        
        Args:
            rewards: [T] rewards at each timestep
            values: [T+1] value estimates (includes bootstrapping value)
            dones: [T] done flags at each timestep
            
        Returns:
            advantages: [T] GAE advantages
            value_targets: [T] value targets for value function training
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        
        gae = 0.0
        for t in reversed(range(T)):
            # TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.config.gae_lambda * self.config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            
            # GAE accumulator: A_t = delta_t + gamma * lambda * A_{t+1}
            gae = delta + self.config.gae_lambda * self.config.gamma * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute value targets: targets = advantages + values
        value_targets = advantages + values[:-1]
        
        # Optional normalization
        if self.config.normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, value_targets

    def compute_advantages_single_trajectory(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE for a single trajectory (list format)."""
        rewards_arr = np.array(rewards, dtype=np.float32)
        values_arr = np.array(values + [0.0], dtype=np.float32)  # bootstrapping from terminal
        dones_arr = np.array(dones, dtype=np.float32)
        
        advantages, targets = self.compute_advantages(rewards_arr, values_arr, dones_arr)
        
        return advantages.tolist(), targets.tolist()


# ============================================================================
#gae_advantage.py - GAE Advantage Estimator for PPO Waypoint RL
# ============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Functional interface for GAE computation.
    
    Args:
        rewards: [T] rewards
        values: [T+1] value estimates  
        gamma: discount factor
        gae_lambda: GAE lambda
        normalize: whether to normalize advantages
        
    Returns:
        advantages, value_targets
    """
    config = GAEConfig(gamma=gamma, gae_lambda=gae_lambda, normalize=normalize)
    estimator = GAEAdvantage(config)
    return estimator.compute_advantages(rewards, values, np.zeros_like(rewards))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GAE Advantage Estimation for Waypoint RL")
    parser.add_argument("--rewards", type=str, help="JSON array of rewards")
    parser.add_argument("--values", type=str, help="JSON array of values (T+1)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--no-normalize", action="store_true", help="Disable advantage normalization")
    parser.add_argument("--output", type=str, default="out/gae_test/metrics.json", help="Output path")
    args = parser.parse_args()
    
    # Default test values if not provided
    if args.rewards:
        rewards = np.array(json.loads(args.rewards), dtype=np.float32)
    else:
        rewards = np.array([0.0, -0.1, 0.5, 1.0, -0.2, 0.8, 1.0, 0.0], dtype=np.float32)
    
    if args.values:
        values = np.array(json.loads(args.values), dtype=np.float32)
    else:
        # Values for 9 timesteps (rewards + 1 bootstrap)
        values = np.array([1.0, 0.9, 0.8, 1.2, 1.0, 1.5, 1.2, 0.8, 0.0], dtype=np.float32)
    
    dones = np.zeros_like(rewards)
    dones[-1] = 1.0  # Terminal state
    
    # Compute GAE
    config = GAEConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        normalize=not args.no_normalize
    )
    estimator = GAEAdvantage(config)
    advantages, value_targets = estimator.compute_advantages(rewards, values, dones)
    
    # Output results
    print("=" * 60)
    print("GAE Advantage Estimation Test")
    print("=" * 60)
    print(f"Rewards:        {rewards.tolist()}")
    print(f"Values:         {values.tolist()}")
    print(f"GAE Lambda:     {args.gae_lambda}")
    print(f"Gamma:          {args.gamma}")
    print("-" * 60)
    print(f"Advantages:     {[f'{a:.4f}' for a in advantages]}")
    print(f"Value Targets:  {[f'{t:.4f}' for t in value_targets]}")
    print(f"Adv Mean:       {advantages.mean():.4f}")
    print(f"Adv Std:        {advantages.std():.4f}")
    print("=" * 60)
    
    # Save to output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        "rewards": rewards.tolist(),
        "values": values.tolist(),
        "advantages": advantages.tolist(),
        "value_targets": value_targets.tolist(),
        "config": {
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "normalize": not args.no_normalize
        }
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
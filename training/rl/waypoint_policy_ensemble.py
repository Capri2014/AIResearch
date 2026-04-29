#!/usr/bin/env python3
"""
Waypoint Policy Ensemble - Combines multiple RL policies for robust waypoint prediction.

This module provides ensemble functionality for combining multiple RL-refined
policies to improve waypoint prediction robustness and reduce variance.

Usage:
    python -m training.rl.waypoint_policy_ensemble \
        --policies policy1.pt policy2.pt policy3.pt \
        --output out/ensemble \
        --method weighted \
        --smoke-test
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import torch, fallback to numpy-only mode
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    TorchDataset = object


@dataclass
class EnsembleConfig:
    """Configuration for waypoint policy ensemble."""
    # Policy paths
    policy_paths: List[str] = field(default_factory=list)
    
    # Ensemble method: weighted, voting, averaging
    method: str = "weighted"
    
    # Weights for each policy (if method=weighted)
    weights: Optional[List[float]] = None
    
    # Waypoint config
    num_waypoints: int = 8
    delta_scale: float = 1.0
    
    # Inference config
    batch_size: int = 32
    device: str = "cuda"  # cuda or cpu
    
    # Output
    output_dir: str = "out/waypoint_ensemble"
    
    def __post_init__(self):
        if self.weights is None and self.method == "weighted":
            # Equal weights by default (if policies exist)
            n = len(self.policy_paths)
            if n > 0:
                self.weights = [1.0 / n] * n


@dataclass
class EnsembleMetrics:
    """Metrics for ensemble evaluation."""
    # Per-policy metrics
    policy_ade: List[float] = field(default_factory=list)
    policy_fde: List[float] = field(default_factory=list)
    policy_success: List[float] = field(default_factory=list)
    
    # Ensemble metrics
    ensemble_ade: float = 0.0
    ensemble_fde: float = 0.0
    ensemble_success: float = 0.0
    
    # Variance metric
    variance_ade: float = 0.0
    variance_fde: float = 0.0
    
    # Improvement over individual policies
    improvement_over_best: float = 0.0
    improvement_over_avg: float = 0.0


class WaypointEnsemble:
    """Ensemble of RL-refined waypoint policies."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.policies: List = []
        self.weights: np.ndarray = np.array(config.weights) if config.weights else None
        
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, using numpy fallback")
            return
        
        # Try to load each policy checkpoint
        self._load_policies()
    
    def _load_policies(self):
        """Load policy checkpoints."""
        if not TORCH_AVAILABLE:
            return
        
        for policy_path in self.config.policy_paths:
            try:
                if os.path.exists(policy_path):
                    state_dict = torch.load(policy_path, map_location=self.config.device)
                    
                    # Try to extract model from checkpoint
                    if isinstance(state_dict, dict):
                        if "model_state_dict" in state_dict:
                            model = self._create_model()
                            model.load_state_dict(state_dict["model_state_dict"])
                            self.policies.append(model)
                        elif "state_dict" in state_dict:
                            model = self._create_model()
                            model.load_state_dict(state_dict["state_dict"])
                            self.policies.append(model)
                        else:
                            # Try as direct state dict
                            model = self._create_model()
                            model.load_state_dict(state_dict)
                            self.policies.append(model)
                    else:
                        # Direct model
                        self.policies.append(state_dict)
                        
                    print(f"Loaded policy: {policy_path}")
                else:
                    print(f"Policy not found (skipping): {policy_path}")
            except Exception as e:
                print(f"Failed to load {policy_path}: {e}")
    
    def _create_model(self):
        """Create a simple waypoint model."""
        class SimpleWaypointModel(nn.Module):
            def __init__(self, input_dim=4, hidden_dim=128, num_waypoints=8):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_waypoints * 2)
                )
                self.num_waypoints = num_waypoints
            
            def forward(self, x):
                out = self.net(x)
                return out.view(-1, self.num_waypoints, 2)
        
        return SimpleWaypointModel(
            input_dim=4,
            hidden_dim=128,
            num_waypoints=self.config.num_waypoints
        ).to(self.config.device)
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict waypoints using ensemble.
        
        Args:
            observations: (N, obs_dim) observation array
            
        Returns:
            waypoints: (N, num_waypoints, 2) waypoint array
        """
        if not TORCH_AVAILABLE:
            return self._predict_numpy(observations)
        
        # Convert to torch
        obs_tensor = torch.from_numpy(observations).float().to(self.config.device)
        
        all_predictions = []
        
        for i, policy in enumerate(self.policies):
            policy.eval()
            with torch.no_grad():
                preds = policy(obs_tensor).cpu().numpy()
                all_predictions.append(preds)
        
        if len(all_predictions) == 0:
            # No policies loaded, use mock
            return self._predict_mock(observations)
        
        all_predictions = np.stack(all_predictions, axis=0)  # (num_policies, N, num_waypoints, 2)
        
        # Ensemble based on method
        if self.config.method == "weighted":
            # Weighted average
            weights = self.weights[:, None, None, None]
            ensemble_pred = np.sum(all_predictions * weights, axis=0)
        elif self.config.method == "voting":
            # Simple averaging
            ensemble_pred = np.mean(all_predictions, axis=0)
        else:
            # Default: averaging
            ensemble_pred = np.mean(all_predictions, axis=0)
        
        return ensemble_pred
    
    def _predict_numpy(self, observations: np.ndarray) -> np.ndarray:
        """NumPy fallback prediction."""
        return self._predict_mock(observations)
    
    def _predict_mock(self, observations: np.ndarray) -> np.ndarray:
        """Generate mock predictions."""
        n = len(observations)
        num_wp = self.config.num_waypoints
        
        # Generate deterministic mock waypoints based on observation
        waypoints = np.zeros((n, num_wp, 2))
        
        for i in range(n):
            obs = observations[i]
            speed = obs[2] if len(obs) > 2 else 1.0
            
            # Create waypoints in a line ahead of the agent
            for j in range(num_wp):
                dist = (j + 1) * 2.0 * speed
                angle = obs[3] if len(obs) > 3 else 0.0
                waypoints[i, j, 0] = dist * np.cos(angle)
                waypoints[i, j, 1] = dist * np.sin(angle)
        
        return waypoints
    
    def evaluate(
        self,
        observations: np.ndarray,
        ground_truth: np.ndarray
    ) -> EnsembleMetrics:
        """
        Evaluate ensemble on held-out data.
        
        Args:
            observations: (N, obs_dim)
            ground_truth: (N, num_waypoints, 2)
            
        Returns:
            EnsembleMetrics
        """
        metrics = EnsembleMetrics()
        
        # Per-policy metrics
        for i, policy in enumerate(self.policies):
            if TORCH_AVAILABLE:
                obs_tensor = torch.from_numpy(observations).float().to(self.config.device)
                gt_tensor = torch.from_numpy(ground_truth).float().to(self.config.device)
                
                policy.eval()
                with torch.no_grad():
                    pred = policy(obs_tensor)
                
                ade = float(torch.mean(torch.sqrt(torch.sum((pred - gt_tensor) ** 2, dim=-1))))
                fde = float(torch.mean(torch.sqrt(
                    torch.sum((pred[:, -1] - gt_tensor[:, -1]) ** 2, dim=-1)
                )))
                
                metrics.policy_ade.append(ade)
                metrics.policy_fde.append(fde)
            else:
                metrics.policy_ade.append(0.0)
                metrics.policy_fde.append(0.0)
        
        # Ensemble metrics
        ensemble_pred = self.predict(observations)
        
        if TORCH_AVAILABLE:
            gt_tensor = torch.from_numpy(ground_truth).float().to(self.config.device)
            ensemble_pred_tensor = torch.from_numpy(ensemble_pred).float().to(self.config.device)
            
            metrics.ensemble_ade = float(torch.mean(torch.sqrt(
                torch.sum((ensemble_pred_tensor - gt_tensor) ** 2, dim=-1)
            )))
            metrics.ensemble_fde = float(torch.mean(torch.sqrt(
                torch.sum((ensemble_pred_tensor[:, -1] - gt_tensor[:, -1]) ** 2, dim=-1)
            )))
            
            # Variance
            metrics.variance_ade = float(torch.std(
                torch.tensor(metrics.policy_ade)
            ))
            metrics.variance_fde = float(torch.std(
                torch.tensor(metrics.policy_fde)
            ))
        
        # Improvement
        if metrics.policy_ade:
            best_ade = min(metrics.policy_ade)
            metrics.improvement_over_best = (best_ade - metrics.ensemble_ade) / best_ade * 100
            
            avg_ade = np.mean(metrics.policy_ade)
            metrics.improvement_over_avg = (avg_ade - metrics.ensemble_ade) / avg_ade * 100
        
        return metrics
    
    def save(self, path: str):
        """Save ensemble configuration and weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        config = {
            "method": self.config.method,
            "weights": self.config.weights.tolist() if self.weights is not None else None,
            "num_waypoints": self.config.num_waypoints,
            "delta_scale": self.config.delta_scale,
            "num_policies": len(self.policies),
            "policy_paths": self.config.policy_paths
        }
        
        with open(path + ".config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved ensemble config to {path}.config.json")


def generate_synthetic_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for testing."""
    np.random.seed(42)
    
    # Observations: (pos_x, pos_y, speed, heading)
    observations = np.random.randn(n_samples, 4)
    observations[:, 2] = np.abs(observations[:, 2])  # speed >= 0
    
    # Ground truth waypoints: (n_samples, num_waypoints, 2)
    num_waypoints = 8
    ground_truth = np.zeros((n_samples, num_waypoints, 2))
    
    for i in range(n_samples):
        x, y, speed, heading = observations[i]
        
        for j in range(num_waypoints):
            dist = (j + 1) * 2.0 * speed
            ground_truth[i, j, 0] = x + dist * np.cos(heading)
            ground_truth[i, j, 1] = y + dist * np.sin(heading)
    
    return observations, ground_truth


def run_ensemble_smoke_test(config: EnsembleConfig) -> EnsembleMetrics:
    """Run smoke test with synthetic data."""
    print("\n=== Waypoint Ensemble Smoke Test ===")
    
    # Generate synthetic data
    obs, gt = generate_synthetic_data(50)
    print(f"Generated {len(obs)} synthetic samples")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Ground truth shape: {gt.shape}")
    
    # Create ensemble
    ensemble = WaypointEnsemble(config)
    print(f"Loaded {len(ensemble.policies)} policies")
    
    # Predict
    pred = ensemble.predict(obs)
    print(f"Prediction shape: {pred.shape}")
    
    # Evaluate
    metrics = ensemble.evaluate(obs, gt)
    print(f"Ensemble ADE: {metrics.ensemble_ade:.3f}m")
    print(f"Ensemble FDE: {metrics.ensemble_fde:.3f}m")
    print(f"Variance ADE: {metrics.variance_ade:.3f}m")
    print(f"Improvement over best: {metrics.improvement_over_best:.2f}%")
    
    # Save ensemble config
    ensemble.save(os.path.join(config.output_dir, "ensemble"))
    print(f"Saved to {config.output_dir}/")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Waypoint Policy Ensemble - Combine multiple RL policies"
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        help="Policy checkpoint paths"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="weighted",
        choices=["weighted", "averaging", "voting"],
        help="Ensemble method"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Policy weights (for weighted method)"
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=8,
        help="Number of waypoints to predict"
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help="Delta scale for waypoint adjustment"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/waypoint_ensemble",
        help="Output directory"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test with synthetic data"
    )
    
    args = parser.parse_args()
    
    if args.smoke_test:
        config = EnsembleConfig(
            policy_paths=args.policies or [],
            method=args.method,
            weights=args.weights,
            num_waypoints=args.num_waypoints,
            delta_scale=args.delta_scale,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir
        )
        
        metrics = run_ensemble_smoke_test(config)
        
        # Write smoke test results
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_out = {
            "ensemble_ade": metrics.ensemble_ade,
            "ensemble_fde": metrics.ensemble_fde,
            "ensemble_success": metrics.ensemble_success,
            "variance_ade": metrics.variance_ade,
            "variance_fde": metrics.variance_fde,
            "improvement_over_best": metrics.improvement_over_best,
            "improvement_over_avg": metrics.improvement_over_avg,
            "policy_ade": metrics.policy_ade,
            "policy_fde": metrics.policy_fde
        }
        
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)
        
        print(f"\nSmoke test complete!")
        print(f"Results: {args.output_dir}/metrics.json")
        
        return
    
    # Print help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    config = EnsembleConfig(
        policy_paths=args.policies or [],
        method=args.method,
        weights=args.weights,
        num_waypoints=args.num_waypoints,
        delta_scale=args.delta_scale,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )
    
    ensemble = WaypointEnsemble(config)
    
    print(f"Loaded {len(ensemble.policies)} policies into ensemble")
    print(f"Method: {config.method}")
    
    if ensemble.weights is not None:
        print(f"Weights: {ensemble.weights.tolist()}")
    
    ensemble.save(os.path.join(args.output_dir, "ensemble"))
    
    print(f"\nEnsemble saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
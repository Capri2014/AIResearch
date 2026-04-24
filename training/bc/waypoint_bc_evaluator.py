#!/usr/bin/env python3
"""
Waypoint BC Evaluator - Evaluate trained BC models on waypoint cache data.

This script:
- Loads a trained BC checkpoint (or uses inline model)
- Evaluates on the waypoint cache test set
- Computes ADE, FDE, and success metrics
- Outputs schema-compliant metrics.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import numpy as np


@dataclass
class BCEvaluationConfig:
    """Configuration for BC evaluation."""
    # Model
    checkpoint_path: Optional[str] = None
    hidden_dim: int = 256
    num_layers: int = 3
    num_waypoints: int = 8
    dropout: float = 0.1
    
    # Data
    cache_dir: str = "data/waymo/waypoint_cache"
    split: str = "val"
    
    # Evaluation
    batch_size: int = 64
    max_samples: Optional[int] = None
    
    # Output
    output_dir: str = "out/bc_eval"


class ResidualWaypointMLP(nn.Module):
    """MLP for waypoint prediction with progress conditioning."""
    
    def __init__(
        self,
        input_dim: int = 4,  # pos_x, pos_y, speed, heading
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_waypoints: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Progress embedding
        self.progress_embed = nn.Linear(1, hidden_dim // 4)
        
        # MLP layers
        dims = [input_dim + hidden_dim // 4] + [hidden_dim] * num_layers + [num_waypoints * 2]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, 4) - pos_x, pos_y, speed, heading
            progress: (B, 1) - episode progress [0, 1]
        Returns:
            waypoints: (B, num_waypoints, 2)
        """
        prog_emb = self.progress_embed(progress)  # (B, hidden_dim // 4)
        x = torch.cat([obs, prog_emb], dim=-1)  # (B, input_dim + hidden_dim // 4)
        out = self.mlp(x)  # (B, num_waypoints * 2)
        return out.view(-1, self.num_waypoints, 2)


def load_waypoint_cache(cache_dir: str, split: str = "val") -> List[Dict[str, Any]]:
    """Load waypoint cache data."""
    cache_path = Path(cache_dir)
    episodes = []
    
    for ep_file in sorted(cache_path.glob("episode_*.json")):
        with open(ep_file) as f:
            data = json.load(f)
        
        # Determine split based on episode ID (80/20 split)
        ep_id = int(ep_file.stem.split("_")[1])
        if (split == "train" and ep_id % 5 != 0) or (split == "val" and ep_id % 5 == 0):
            episodes.append(data)
    
    return episodes


def compute_ade(pred_waypoints: np.ndarray, gt_waypoints: np.ndarray) -> float:
    """Compute Average Displacement Error."""
    return np.mean(np.linalg.norm(pred_waypoints - gt_waypoints, axis=-1))


def compute_fde(pred_waypoints: np.ndarray, gt_waypoints: np.ndarray) -> float:
    """Compute Final Displacement Error."""
    return np.linalg.norm(pred_waypoints[:, -1] - gt_waypoints[:, -1])


def compute_success_rate(ade: float, threshold: float = 2.0) -> float:
    """Compute success rate based on ADE threshold."""
    return 1.0 if ade < threshold else 0.0


class BCEvaluator:
    """Evaluator for waypoint BC models."""
    
    def __init__(self, config: BCEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model
        self.model = ResidualWaypointMLP(
            input_dim=4,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_waypoints=config.num_waypoints,
            dropout=config.dropout,
        ).to(self.device)
        
        # Load checkpoint if provided
        if config.checkpoint_path and os.path.exists(config.checkpoint_path):
            print(f"Loading checkpoint: {config.checkpoint_path}")
            state_dict = torch.load(config.checkpoint_path, map_location=self.device)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict)
        else:
            print("Using untrained model (for testing)")
        
        self.model.eval()
    
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on waypoint cache."""
        print(f"Loading {self.config.split} data from {self.config.cache_dir}")
        episodes = load_waypoint_cache(self.config.cache_dir, self.config.split)
        
        if not episodes:
            print("No episodes found, using synthetic data")
            episodes = self._generate_synthetic_data()
        
        # Collect all samples
        samples = []
        for ep in episodes:
            for frame in ep.get("frames", []):
                samples.append({
                    "observation": frame.get("observation", [0, 0, 0, 0]),
                    "waypoints": frame.get("waypoints", [[0, 0]] * self.config.num_waypoints),
                    "progress": frame.get("progress", 0),
                })
        
        if self.config.max_samples:
            samples = samples[:self.config.max_samples]
        
        print(f"Evaluating {len(samples)} samples")
        
        # Run evaluation in batches
        all_ades = []
        all_fdes = []
        
        with torch.no_grad():
            for i in range(0, len(samples), self.config.batch_size):
                batch = samples[i:i + self.config.batch_size]
                
                obs = torch.tensor(
                    [s["observation"] for s in batch],
                    dtype=torch.float32,
                ).to(self.device)
                
                progress = torch.tensor(
                    [s["progress"] for s in batch],
                    dtype=torch.float32,
                ).unsqueeze(-1).to(self.device)
                
                gt_waypoints = np.array([s["waypoints"] for s in batch])
                
                # Forward pass
                pred = self.model(obs, progress).cpu().numpy()
                
                # Compute metrics
                for j in range(len(batch)):
                    ade = compute_ade(pred[j], gt_waypoints[j])
                    fde = compute_fde(pred[j], gt_waypoints[j])
                    all_ades.append(ade)
                    all_fdes.append(fde)
        
        # Aggregate metrics
        ade_mean = np.mean(all_ades)
        ade_std = np.std(all_ades)
        fde_mean = np.mean(all_fdes)
        fde_std = np.std(all_fdes)
        
        # Success rates at different thresholds
        success_2m = np.mean([1.0 if ade < 2.0 else 0.0 for ade in all_ades])
        success_5m = np.mean([1.0 if ade < 5.0 else 0.0 for ade in all_ades])
        
        metrics = {
            "ade": {
                "mean": float(ade_mean),
                "std": float(ade_std),
                "unit": "meters",
            },
            "fde": {
                "mean": float(fde_mean),
                "std": float(fde_std),
                "unit": "meters",
            },
            "success_rate": {
                "2m": float(success_2m),
                "5m": float(success_5m),
            },
            "num_samples": len(samples),
            "split": self.config.split,
        }
        
        return metrics
    
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic data for testing."""
        np.random.seed(42)
        samples = []
        
        for i in range(100):
            progress = i / 100.0
            obs = [np.random.randn(4) * 0.1]
            
            # Generate waypoints in a line
            waypoints = []
            for j in range(self.config.num_waypoints):
                waypoints.append([j * 2.0 + np.random.randn() * 0.1, np.random.randn() * 0.1])
            
            samples.append({
                "observation": obs[0].tolist(),
                "waypoints": waypoints,
                "progress": progress,
            })
        
        return [{"frames": samples}]


def save_metrics(metrics: Dict[str, Any], output_dir: str) -> None:
    """Save metrics to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate waypoint BC models")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to BC checkpoint")
    parser.add_argument("--cache-dir", type=str, default="data/waymo/waypoint_cache")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="out/bc_eval")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-waypoints", type=int, default=8)
    
    args = parser.parse_args()
    
    config = BCEvaluationConfig(
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_waypoints=args.num_waypoints,
    )
    
    evaluator = BCEvaluator(config)
    metrics = evaluator.evaluate()
    
    # Print summary
    print("\n" + "=" * 50)
    print("BC Evaluation Results")
    print("=" * 50)
    print(f"ADE: {metrics['ade']['mean']:.4f}m ± {metrics['ade']['std']:.4f}m")
    print(f"FDE: {metrics['fde']['mean']:.4f}m ± {metrics['fde']['std']:.4f}m")
    print(f"Success @ 2m: {metrics['success_rate']['2m']*100:.1f}%")
    print(f"Success @ 5m: {metrics['success_rate']['5m']*100:.1f}%")
    print("=" * 50)
    
    # Save metrics
    save_metrics(metrics, args.output_dir)


if __name__ == "__main__":
    main()
"""Waypoint BC trainer with evaluation metrics (ADE/FDE).

This module implements behavior cloning for waypoint prediction with proper
evaluation metrics integrated during training (evaluation-first design).

Architecture Pattern (from MEMORY.md):
- Residual Delta Learning: final_waypoints = sft_waypoints + delta_head(z)
- Keep SFT model fixed, train only a small delta head
- More sample-efficient, safer, modular

Evaluation Metrics:
- ADE (Average Displacement Error): mean Euclidean distance to all waypoints
- FDE (Final Displacement Error): distance to final waypoint only
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_ade(pred_waypoints: np.ndarray, gt_waypoints: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        pred_waypoints: (T, 2) predicted waypoints [x, y]
        gt_waypoints: (T, 2) ground truth waypoints

    Returns:
        Mean Euclidean distance across all timesteps
    """
    errors = np.linalg.norm(pred_waypoints - gt_waypoints, axis=1)
    return float(np.mean(errors))


def compute_fde(pred_waypoints: np.ndarray, gt_waypoints: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        pred_waypoints: (T, 2) predicted waypoints
        gt_waypoints: (T, 2) ground truth waypoints

    Returns:
        Euclidean distance to final waypoint
    """
    return float(np.linalg.norm(pred_waypoints[-1] - gt_waypoints[-1]))


def compute_metrics(
    pred_waypoints: np.ndarray,
    gt_waypoints: np.ndarray,
    prefix: str = "",
) -> dict:
    """Compute comprehensive waypoint metrics.

    Args:
        pred_waypoints: (..., T, 2) predicted waypoints (any leading dims)
        gt_waypoints: (..., T, 2) ground truth waypoints
        prefix: Metric name prefix for logging

    Returns:
        Dictionary of metrics
    ]
    """
    # Handle batch dimensions
    if pred_waypoints.ndim == 3:
        # Batch case: (B, T, 2)
        ade = np.mean([
            compute_ade(p, g) for p, g in zip(pred_waypoints, gt_waypoints)
        ])
        fde = np.mean([
            compute_fde(p, g) for p, g in zip(pred_waypoints, gt_waypoints)
        ])
    else:
        # Single case: (T, 2)
        ade = compute_ade(pred_waypoints, gt_waypoints)
        fde = compute_fde(pred_waypoints, gt_waypoints)

    metrics = {
        f"{prefix}ade": ade,
        f"{prefix}fde": fde,
    }
    return metrics


# ============================================================================
# Delta Head Model
# ============================================================================

class DeltaHead(nn.Module):
    """Small delta head for residual waypoint learning.

    Architecture: z (latent features) -> delta_waypoints
    Keeps SFT backbone frozen, only trains this head.
    """

    def __init__(self, latent_dim: int = 512, num_waypoints: int = 4):
        super().__init__()
        self.num_waypoints = num_waypoints
        # Residual delta: small adjustments to SFT waypoints
        self.delta_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_waypoints * 2),  # x, y per waypoint
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict delta waypoints.

        Args:
            z: (B, latent_dim) latent features from frozen backbone

        Returns:
            delta: (B, T, 2) delta waypoint adjustments
        """
        delta = self.delta_net(z)  # (B, T*2)
        delta = delta.view(-1, self.num_waypoints, 2)  # (B, T, 2)
        return delta


class WaypointBCModel(nn.Module):
    """Full waypoint BC model with frozen SFT backbone + trainable delta head.

    Architecture: final_waypoints = sft_waypoints + delta_head(z)
    """

    def __init__(
        self,
        sft_waypoints: np.ndarray | None = None,
        latent_dim: int = 512,
        num_waypoints: int = 4,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        # Frozen SFT waypoints (could be loaded from checkpoint)
        if sft_waypoints is not None:
            self.register_buffer(
                "sft_waypoints",
                torch.from_numpy(sft_waypoints.astype(np.float32)),
            )
        else:
            self.register_buffer(
                "sft_waypoints",
                torch.zeros(num_waypoints, 2),
            )
        self.sft_waypoints.requires_grad = False

        # Trainable delta head
        self.delta_head = DeltaHead(latent_dim, num_waypoints)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict final waypoints.

        Args:
            z: (B, latent_dim) latent features

        Returns:
            waypoints: (B, T, 2) final waypoints = SFT + delta
        """
        delta = self.delta_head(z)  # (B, T, 2)
        # Expand SFT waypoints to batch size
        sft_expanded = self.sft_waypoints.unsqueeze(0).expand(delta.size(0), -1, -1)
        final_waypoints = sft_expanded + delta
        return final_waypoints


# ============================================================================
# Dataset (Synthetic for demo)
# ============================================================================

class WaypointDataset(Dataset):
    """Synthetic waypoint dataset for demonstration.

    In production, this would load real trajectories from Waymo.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        num_waypoints: int = 4,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.num_waypoints = num_waypoints
        self.rng = np.random.default_rng(seed)

        # Generate synthetic latent features and waypoints
        # Latent: represents perception + prediction features
        self.latents = self.rng.normal(0, 1, (n_samples, 512)).astype(np.float32)

        # Waypoints: synthetic trajectory [x, y] over T steps
        self.waypoints = np.zeros((n_samples, num_waypoints, 2), dtype=np.float32)
        for i in range(n_samples):
            # Random curve parameters
            curvature = self.rng.uniform(-0.5, 0.5)
            speed = self.rng.uniform(1.0, 5.0)
            offset_x = self.rng.uniform(-2, 2)
            offset_y = self.rng.uniform(-1, 1)

            for t in range(num_waypoints):
                t_norm = t / (num_waypoints - 1)
                self.waypoints[i, t, 0] = offset_x + speed * t_norm + curvature * t_norm**2
                self.waypoints[i, t, 1] = offset_y + curvature * t_norm

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.latents[idx], self.waypoints[idx]


# ============================================================================
# Training Loop with Integrated Evaluation
# ============================================================================

@dataclass
class BCConfig:
    """Configuration for waypoint BC training."""
    out_dir: Path = Path("out/waypoint_bc")
    n_samples: int = 2000
    num_waypoints: int = 4
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 50
    seed: int = 0
    # Delta learning specific
    latent_dim: int = 512


def train_waypoint_bc(
    config: BCConfig | None = None,
    model: WaypointBCModel | None = None,
    train_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    eval_fn: Callable | None = None,
) -> tuple[WaypointBCModel, dict]:
    """Train waypoint BC model with integrated evaluation.

    Args:
        config: Training configuration
        model: Model to train (or None to create new)
        train_dataset: Training data
        eval_dataset: Evaluation data (optional)
        eval_fn: Custom evaluation function (optional)

    Returns:
        Trained model and training metrics dict
    """
    if config is None:
        config = BCConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets if not provided
    if train_dataset is None:
        train_dataset = WaypointDataset(config.n_samples, config.num_waypoints, config.seed)
    if eval_dataset is None:
        # Use last 20% for eval
        eval_size = max(1, len(train_dataset) // 5)
        train_size = len(train_dataset) - eval_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(config.seed),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Create model if not provided
    if model is None:
        model = WaypointBCModel(
            latent_dim=config.latent_dim,
            num_waypoints=config.num_waypoints,
        )
    model = model.to(device)

    # Freeze SFT backbone, only train delta head
    for name, param in model.named_parameters():
        if "delta_head" not in name:
            param.requires_grad = False

    # Count trainable parameters
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[waypoint_bc] Trainable parameters: {n_trainable:,}")
    print(f"[waypoint_bc] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer only for delta head
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
    )
    criterion = nn.MSELoss()

    # Training metrics tracking
    metrics = {
        "train_loss": [],
        "eval_ade": [],
        "eval_fde": [],
        "epoch": [],
    }

    best_fde = float("inf")

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for latents, waypoints in train_loader:
            latents = latents.to(device)
            waypoints = waypoints.to(device)

            optimizer.zero_grad()
            pred_waypoints = model(latents)
            loss = criterion(pred_waypoints, waypoints)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        metrics["train_loss"].append(avg_loss)
        metrics["epoch"].append(epoch)

        # Evaluation phase (every epoch for evaluation-first design)
        model.eval()
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for latents, waypoints in eval_loader:
                latents = latents.to(device)
                pred = model(latents)
                all_preds.append(pred.cpu().numpy())
                all_gts.append(waypoints.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)

        eval_metrics = compute_metrics(all_preds, all_gts, prefix="eval_")
        metrics["eval_ade"].append(eval_metrics["eval_ade"])
        metrics["eval_fde"].append(eval_metrics["eval_fde"])

        # Track best checkpoint by FDE
        if eval_metrics["eval_fde"] < best_fde:
            best_fde = eval_metrics["eval_fde"]
            # Save best checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": metrics.copy(),
            }
            torch.save(checkpoint, config.out_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[waypoint_bc] Epoch {epoch+1}/{config.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ADE: {eval_metrics['eval_ade']:.4f} | "
                  f"FDE: {eval_metrics['eval_fde']:.4f}")

    # Final metrics
    final_metrics = {
        "final_train_loss": metrics["train_loss"][-1],
        "final_eval_ade": metrics["eval_ade"][-1],
        "final_eval_fde": metrics["eval_fde"][-1],
        "best_eval_fde": best_fde,
        "epochs": config.epochs,
        "trainable_params": n_trainable,
        "architecture": "residual_delta",
        "formula": "final_waypoints = sft_waypoints + delta_head(z)",
    }

    # Save metrics
    with open(config.out_dir / "training_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save model config
    model_config = {
        "type": "WaypointBCModel",
        "num_waypoints": config.num_waypoints,
        "latent_dim": config.latent_dim,
        "trainable_head": "DeltaHead",
    }
    with open(config.out_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n[waypoint_bc] Training complete!")
    print(f"[waypoint_bc] Best FDE: {best_fde:.4f}")
    print(f"[waypoint_bc] Outputs: {config.out_dir}")

    return model, final_metrics


def main() -> None:
    """CLI entry point."""
    config = BCConfig()
    model, metrics = train_waypoint_bc(config)

    print("\n=== Training Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

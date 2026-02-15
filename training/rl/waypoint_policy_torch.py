"""Waypoint policy wrapper for PyTorch inference.

This module provides a clean interface for:
- Loading SFT waypoint checkpoints
- Running inference (encoder + head)
- Computing ADE/FDE metrics
- Integration with CARLA/ScenarioRunner rollouts

Usage
-----
# Load checkpoint and run inference
python -m training.rl.waypoint_policy_torch --help
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np


def _require_torch():
    try:
        import torch
        from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
        from training.sft.train_waypoint_bc_torch_v0 import WaypointHead
    except Exception as e:
        raise RuntimeError("This script requires PyTorch and the training modules.") from e
    return torch, TinyMultiCamEncoder, WaypointHead


@dataclass
class WaypointPolicyConfig:
    """Configuration for waypoint policy."""
    checkpoint: Path
    cam: str = "front"
    horizon_steps: int = 20
    device: str = "auto"


class WaypointPolicyTorch:
    """Combined encoder + head for waypoint prediction.
    
    This is the base SFT policy. For RL refinement, use:
    - WaypointPolicyWithDelta: wraps this + delta head
    """
    
    def __init__(self, cfg: WaypointPolicyConfig | None = None):
        cfg = cfg or WaypointPolicyConfig()
        self.cfg = cfg
        
        torch, Encoder, Head = _require_torch()
        self.torch = torch
        
        # Determine device
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        
        # Initialize model
        self.encoder = Encoder(out_dim=128).to(self.device)
        self.head = Head(torch=torch, in_dim=128, horizon_steps=cfg.horizon_steps).to(self.device)
        
        # Load checkpoint
        self._load_checkpoint(cfg.checkpoint)
        
        self.encoder.eval()
        self.head.eval()
        
        # Cache for efficiency
        self._cam = cfg.cam
    
    def _load_checkpoint(self, path: Path):
        """Load encoder + head from checkpoint."""
        ckpt = self.torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            enc_sd = ckpt.get("encoder")
            head_sd = ckpt.get("head")
            
            if enc_sd is None:
                # Try loading entire checkpoint as encoder
                self.encoder.load_state_dict(ckpt)
            else:
                if head_sd is None:
                    raise ValueError(f"Checkpoint missing 'head' state dict: {path}")
                self.encoder.load_state_dict(enc_sd)
                self.head.load_state_dict(head_sd)
        else:
            raise ValueError(f"Unexpected checkpoint format: {path}")
        
        print(f"[waypoint_policy] loaded checkpoint from {path}")
    
    @torch.no_grad()
    def __call__(
        self,
        images: Dict[str, torch.Tensor],
        image_valid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Predict waypoints from images.
        
        Args:
            images: Dict of camera_name -> (B, C, H, W) tensor
            image_valid: Optional dict of camera_name -> (B,) bool tensor
        
        Returns:
            waypoints: (B, horizon_steps, 2) tensor in world coordinates
        """
        if image_valid is None:
            image_valid = {k: torch.ones(v.shape[0], dtype=torch.bool, device=self.device) 
                          for k, v in images.items()}
        
        z = self.encoder(images, image_valid_by_cam=image_valid)
        waypoints = self.head(z)
        return waypoints
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: np.ndarray,
        image_valid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict waypoints from numpy images (B, H, W, C).
        
        Returns:
            waypoints: (B, horizon_steps, 2) numpy array
        """
        # Convert to tensor
        x = torch.from_numpy(images).float().to(self.device) / 255.0
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=self.device)
        if image_valid is not None:
            valid = torch.from_numpy(image_valid).bool().to(self.device)
        
        waypoints = self({self.cfg.cam: x}, image_valid={self.cfg.cam: valid})
        return waypoints.cpu().numpy()


class WaypointPolicyWithDelta(WaypointPolicyTorch):
    """SFT policy + delta head for residual learning.
    
    Usage:
        # Load SFT base
        base = WaypointPolicyTorch(cfg)
        
        # Wrap with delta head
        policy = WaypointPolicyWithDelta(
            base_policy=base,
            delta_head_state_dict=delta_ckpt["delta_head"],
        )
        
        # Combined prediction
        sft_waypoints = base(images)
        delta = delta_head(z)
        final_waypoints = sft_waypoints + delta
    """
    
    def __init__(
        self,
        base_policy: WaypointPolicyTorch,
        delta_head_state_dict: Optional[Dict] = None,
    ):
        super().__init__(base_policy.cfg)
        
        # Copy base weights
        self.encoder.load_state_dict(base_policy.encoder.state_dict())
        self.head.load_state_dict(base_policy.head.state_dict())
        
        # Add delta head
        self.torch, Encoder, Head = _require_torch()
        self.delta_head = Head(
            torch=self.torch,
            in_dim=128,
            horizon_steps=base_policy.cfg.horizon_steps,
        ).to(self.device)
        
        if delta_head_state_dict is not None:
            self.delta_head.load_state_dict(delta_head_state_dict)
        
        self.delta_head.eval()
    
    @torch.no_grad()
    def predict_with_delta(
        self,
        images: Dict[str, torch.Tensor],
        image_valid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict waypoints with delta correction.
        
        Returns:
            base_waypoints: (B, H, 2) from SFT
            delta_waypoints: (B, H, 2) correction
        """
        if image_valid is None:
            image_valid = {k: torch.ones(v.shape[0], dtype=torch.bool, device=self.device)
                          for k, v in images.items()}
        
        z = self.encoder(images, image_valid_by_cam=image_valid)
        base_waypoints = self.head(z)
        delta = self.delta_head(z)
        
        return base_waypoints, delta
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Full forward: base + delta."""
        base = self.head(z)
        delta = self.delta_head(z)
        return base + delta


def compute_ade_fde(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute ADE and FDE with per-example breakdown.
    
    Args:
        predictions: (N, H, 2) predicted waypoints
        targets: (N, H, 2) target waypoints
    
    Returns:
        ade: average displacement error
        fde: final displacement error
        ade_per_example: (N,) ADE for each example
        fde_per_example: (N,) FDE for each example
    """
    # Per-example ADE
    distances = np.linalg.norm(predictions - targets, axis=2)  # (N, H)
    ade_per_example = distances.mean(axis=1)  # (N,)
    ade = float(ade_per_example.mean())
    
    # Per-example FDE (final timestep only)
    fde_per_example = distances[:, -1]  # (N,)
    fde = float(fde_per_example.mean())
    
    return ade, fde, ade_per_example, fde_per_example


@dataclass
class EvalResult:
    """Result of evaluating a waypoint policy."""
    ade: float
    fde: float
    ade_std: float
    fde_std: float
    num_examples: int
    ade_per_example: np.ndarray
    fde_per_example: np.ndarray


def evaluate_policy(
    policy: WaypointPolicyTorch,
    dataset: "EpisodesWaypointBCDataset",
    batch_size: int = 32,
) -> EvalResult:
    """Evaluate a waypoint policy on a dataset."""
    from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset
    
    torch = _require_torch()[0]
    
    all_preds = []
    all_targets = []
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    
    for b in loader:
        images = b["image"]
        targets = b["waypoints"]
        
        # Skip invalid
        if b.get("image_valid") is None or b.get("waypoints_valid") is None:
            continue
        
        valid = b["image_valid"] & b["waypoints_valid"]
        if valid.sum() < 1:
            continue
        
        images = images.to(policy.device)[valid]
        targets = targets.to(policy.device)[valid]
        
        preds = policy(
            {policy.cfg.cam: images},
            image_valid={policy.cfg.cam: torch.ones(valid.sum(), dtype=torch.bool, device=policy.device)},
        )
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    if not all_preds:
        raise ValueError("No valid predictions found")
    
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    ade, fde, ade_per, fde_per = compute_ade_fde(predictions, targets)
    
    return EvalResult(
        ade=ade,
        fde=fde,
        ade_std=float(ade_per.std()),
        fde_std=float(fde_per.std()),
        num_examples=len(predictions),
        ade_per_example=ade_per,
        fde_per_example=fde_per,
    )


def main():
    """CLI for policy evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Waypoint Policy Evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes-glob", type=str, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset
    
    cfg = WaypointPolicyConfig(
        checkpoint=args.checkpoint,
    )
    policy = WaypointPolicyTorch(cfg)
    
    ds = EpisodesWaypointBCDataset(
        args.episodes_glob,
        cam=cfg.cam,
        horizon_steps=cfg.horizon_steps,
        decode_images=True,
    )
    
    result = evaluate_policy(policy, ds, batch_size=args.batch_size)
    
    print(f"\nEvaluation Results:")
    print(f"  ADE: {result.ade:.4f} ± {result.ade_std:.4f}")
    print(f"  FDE: {result.fde:.4f} ± {result.fde_std:.4f}")
    print(f"  N: {result.num_examples}")
    
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = {
            "ade": result.ade,
            "fde": result.fde,
            "ade_std": result.ade_std,
            "fde_std": result.fde_std,
            "num_examples": result.num_examples,
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        
        # Save per-example errors
        errors = {
            "ade_per_example": result.ade_per_example.tolist(),
            "fde_per_example": result.fde_per_example.tolist(),
        }
        (args.output_dir / "errors.json").write_text(json.dumps(errors, indent=2))
        
        print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()

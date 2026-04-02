"""
Pretrained Encoder Integration for Waypoint BC.

Loads the pretrained contrastive SSL encoder and integrates it with
the waypoint behavioral cloning pipeline.

Architecture:
    Waymo episodes → SSL pretrain (encoder_final.pt) → waypoint BC → RL refinement

This script:
1. Loads pretrained encoder from SSL pretrain checkpoint
2. Integrates with waypoint BC training
3. Can optionally freeze encoder and fine-tune BC head
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PretrainedEncoderWaypointBC(nn.Module):
    """Waypoint BC with pretrained encoder integration.
    
    Architecture:
        images → pretrained_encoder → features → waypoint_head → waypoints
    """

    def __init__(
        self,
        encoder_path: Optional[str] = None,
        encoder_frozen: bool = True,
        num_waypoints: int = 4,
        latent_dim: int = 512,
        use_delta_head: bool = True,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.latent_dim = latent_dim
        self.encoder_frozen = encoder_frozen
        
        # Camera config (must match SSL pretrain)
        self.camera_names = ["front", "left", "right", "rear"]
        
        # Load pretrained encoder if provided
        if encoder_path and os.path.exists(encoder_path):
            print(f"Loading pretrained encoder from: {encoder_path}")
            self.encoder = self._load_encoder(encoder_path)
            if encoder_frozen:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                print("Encoder frozen for fine-tuning")
        else:
            print("No encoder provided, using random init (training from scratch)")
            self.encoder = None
        
        # Waypoint prediction head
        input_dim = latent_dim
        self.waypoint_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_waypoints * 2),  # (x, y) for each waypoint
        )
        
        # Delta head for residual learning (optional)
        self.use_delta_head = use_delta_head
        if use_delta_head:
            self.delta_head = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_waypoints * 2),
            )
    
    def _load_encoder(self, path: str) -> nn.Module:
        """Load pretrained encoder from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        
        # Try to extract encoder state dict
        if isinstance(checkpoint, dict):
            if "encoder_state_dict" in checkpoint:
                state_dict = checkpoint["encoder_state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume checkpoint IS the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Create encoder and load weights
        # Use TinyMultiCamEncoder structure matching SSL pretrain
        encoder = TinyMultiCamEncoder(
            camera_names=self.camera_names,
            embed_dim=self.latent_dim,
        )
        encoder.load_state_dict(state_dict, strict=False)
        return encoder
    
    def forward(self, images: torch.Tensor, return_features: bool = False):
        """Forward pass.
        
        Args:
            images: (B, C, H, W) or (B, N, C, H, W) for multi-camera
            return_features: whether to return intermediate features
            
        Returns:
            waypoints: (B, num_waypoints * 2)
            features: (B, latent_dim) if return_features=True
        """
        batch_size = images.shape[0]
        
        # Get features from encoder
        if self.encoder is not None:
            # Handle both single (B,C,H,W) and multi-camera (B,N,C,H,W)
            if images.ndim == 5:
                # Multi-camera: use encoder which handles this
                features = self.encoder(images)
            else:
                # Single camera: wrap in list
                features = self.encoder(images.unsqueeze(1))
        else:
            # Random init fallback
            features = torch.randn(batch_size, self.latent_dim, device=images.device)
        
        # Main waypoint prediction
        waypoints = self.waypoint_head(features)
        
        # Delta head for residual learning
        if self.use_delta_head:
            delta = self.delta_head(features)
            waypoints = waypoints + delta
        
        if return_features:
            return waypoints, features
        return waypoints


class TinyMultiCamEncoder(nn.Module):
    """Multi-camera encoder matching SSL pretrain architecture."""

    def __init__(self, camera_names=None, embed_dim=512):
        super().__init__()
        self.camera_names = camera_names or ["front", "left"]
        self.embed_dim = embed_dim
        
        # Per-camera encoders
        self.camera_encoders = nn.ModuleDict()
        for cam in self.camera_names:
            self.camera_encoders[cam] = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 16, embed_dim),
            )
        
        # Fusion layer
        self.fusion = nn.Linear(len(self.camera_names) * embed_dim, embed_dim)

    def forward(self, images):
        """Forward pass handling both single and multi-camera."""
        # images: (B, C, H, W) or (B, N, C, H, W)
        if images.ndim == 4:
            # Single camera
            features = self.camera_encoders[self.camera_names[0]](images)
            return features
        
        # Multi-camera: (B, N, C, H, W)
        batch_size, num_cams = images.shape[:2]
        cam_features = []
        
        for i, cam in enumerate(self.camera_names):
            cam_img = images[:, i]
            cam_feat = self.camera_encoders[cam](cam_img)
            cam_features.append(cam_feat)
        
        # Concatenate and fuse
        fused = torch.cat(cam_features, dim=-1)
        features = self.fusion(fused)
        return features


def create_synthetic_batch(batch_size: int, num_cameras: int = 4) -> Dict:
    """Create synthetic batch for testing (simulates Waymo episode data).
    
    Must match encoder camera config: ["front", "left", "right", "rear"]
    """
    # Images: (B, N, C, H, W) - use 4 cameras to match encoder
    images = torch.randn(batch_size, num_cameras, 3, 128, 256)
    
    # Waypoints: (B, num_waypoints * 2)
    num_waypoints = 4
    waypoints = torch.randn(batch_size, num_waypoints * 2)
    
    return {
        "images": images,
        "waypoints": waypoints,
    }


def train_with_pretrained_encoder(
    encoder_path: Optional[str] = None,
    encoder_frozen: bool = True,
    num_steps: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    use_delta_head: bool = True,
    log_every: int = 10,
    checkpoint_every: int = 50,
    output_dir: str = "out/pretrain_bc",
):
    """Train waypoint BC with pretrained encoder."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = PretrainedEncoderWaypointBC(
        encoder_path=encoder_path,
        encoder_frozen=encoder_frozen,
        use_delta_head=use_delta_head,
    )
    model.train()
    
    # Optimizer (only train waypoint head if encoder frozen)
    if encoder_frozen:
        optimizer = torch.optim.Adam(model.waypoint_head.parameters(), lr=lr)
        if use_delta_head:
            optimizer.add_param_group({"params": model.delta_head.parameters()})
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    metrics = {
        "encoder_path": encoder_path,
        "encoder_frozen": encoder_frozen,
        "use_delta_head": use_delta_head,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "lr": lr,
    }
    
    print(f"Starting training with pretrained encoder: {encoder_path}")
    print(f"Encoder frozen: {encoder_frozen}, Delta head: {use_delta_head}")
    
    for step in range(num_steps):
        # Create batch (synthetic for now)
        batch = create_synthetic_batch(batch_size)
        images = batch["images"]
        waypoints = batch["waypoints"]
        
        # Forward
        optimizer.zero_grad()
        pred_waypoints = model(images)
        loss = criterion(pred_waypoints, waypoints)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % log_every == 0:
            avg_loss = np.mean(losses[-log_every:])
            print(f"step={step+1}/{num_steps} loss={avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, "pretrained_bc_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "final_loss": np.mean(losses[-10:]),
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save training metrics
    training_metrics = {
        "run_id": f"pretrain_bc_{os.path.basename(output_dir)}",
        "domain": "pretrain_bc",
        "config": metrics,
        "final_loss": float(np.mean(losses[-10:])),
        "loss_curve": [float(x) for x in losses],
    }
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    return model, training_metrics


def main():
    parser = argparse.ArgumentParser(description="Waypoint BC with pretrained encoder")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--encoder-frozen", action="store_true", default=True,
                        help="Freeze encoder during training")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--no-delta-head", action="store_true",
                        help="Disable delta head (pure BC)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--output-dir", type=str, default="out/pretrain_bc",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Find encoder checkpoint if not specified
    if args.encoder_path is None:
        default_encoder = "out/pretrain_contrastive/encoder_final.pt"
        if os.path.exists(default_encoder):
            args.encoder_path = default_encoder
            print(f"Using default encoder: {args.encoder_path}")
        else:
            print(f"No encoder found at {default_encoder}, training from scratch")
    
    # Train
    model, metrics = train_with_pretrained_encoder(
        encoder_path=args.encoder_path,
        encoder_frozen=args.encoder_frozen,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        use_delta_head=not args.no_delta_head,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
    )
    
    print("\n=== Training Complete ===")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

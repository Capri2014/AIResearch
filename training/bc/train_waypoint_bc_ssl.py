"""Waypoint BC training with SSL encoder transfer learning.

This script trains a waypoint prediction model using:
1. SSL-pretrained encoder (from PR #2) 
2. Waypoint prediction head
3. Behavioral cloning loss on Waymo episodes

The encoder can be:
- Loaded from a pretrained SSL checkpoint
- Created as a stub for testing

Usage
-----
python -m training.bc.train_waypoint_bc_ssl --help

Examples:
    # Train with SSL encoder
    python -m training.bc.train_waypoint_bc_ssl \
        --episode-dir data/waymo_episodes \
        --ssl-checkpoint out/pretrain/ssl_model.pt \
        --output-dir out/bc_ssl \
        --num-steps 50000

    # Test without pretrained weights
    python -m training.bc.train_waypoint_bc_ssl --stub --test

    # Smoke test
    python -m training.bc.train_waypoint_bc_ssl --test
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class BCConfig:
    """Configuration for waypoint BC training."""
    # Data
    episode_dir: Path = Path("data/waymo_episodes")
    batch_size: int = 32
    num_workers: int = 4
    
    # Model
    ssl_checkpoint: Optional[Path] = None
    encoder_out_dim: int = 128
    horizon_steps: int = 20
    
    # Training
    num_steps: int = 10000
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Output
    output_dir: Path = Path("out/bc")
    save_every: int = 1000
    log_every: int = 100
    
    # Device
    device: str = "auto"
    
    # Debug
    stub_encoder: bool = False
    test_mode: bool = False


class WaypointBCWithSSLDataset:
    """Dataset that combines Waymo episodes with SSL encoder features.
    
    This dataset:
    1. Loads camera images from Waymo episodes
    2. Passes images through SSL encoder to get BEV features
    3. Returns (encoded_features, waypoints) for BC training
    
    Args:
        episode_dir: Directory containing Waymo episode data
        encoder: SSL encoder to use for feature extraction
        horizon_steps: Number of future waypoints to predict
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        episode_dir: Path,
        encoder,
        horizon_steps: int = 20,
        transform=None,
    ):
        self.episode_dir = Path(episode_dir)
        self.encoder = encoder
        self.horizon_steps = horizon_steps
        self.transform = transform
        
        # Find all episode files
        self.episode_files = sorted(self.episode_dir.glob("*.pt"))
        
        if len(self.episode_files) == 0:
            # Create stub data for testing
            print(f"[WaypointBCDataset] No episodes found in {episode_dir}, using stub data")
            self.use_stub = True
            self.num_samples = 1000
        else:
            print(f"[WaypointBCDataset] Found {len(self.episode_files)} episode files")
            self.use_stub = False
            # Load episode metadata
            self._load_metadata()
    
    def _load_metadata(self):
        """Load episode metadata."""
        # For now, just count total frames
        self.num_samples = 0
        for ep_file in self.episode_files:
            try:
                data = torch.load(ep_file, map_location="cpu")
                if isinstance(data, dict):
                    self.num_samples += len(data.get("frames", []))
                else:
                    self.num_samples += len(data)
            except Exception as e:
                print(f"[WaypointBCDataset] Failed to load {ep_file}: {e}")
        
        print(f"[WaypointBCDataset] Total samples: {self.num_samples}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single sample.
        
        Returns:
            features: (encoder_out_dim,) SSL encoder features
            waypoints: (horizon_steps, 2) future waypoints in ego frame
        """
        if self.use_stub:
            # Return stub data for testing
            features = np.random.randn(self.encoder.out_dim).astype(np.float32)
            waypoints = np.random.randn(self.horizon_steps, 2).astype(np.float32) * 0.5
            return features, waypoints
        
        # Load from episode
        ep_idx = idx % len(self.episode_files)
        frame_idx = idx // len(self.episode_files)
        
        try:
            data = torch.load(self.episode_files[ep_idx], map_location="cpu")
            
            if isinstance(data, dict):
                frames = data.get("frames", [])
            else:
                frames = data
            
            frame = frames[frame_idx % len(frames)]
            
            # Extract image and waypoints
            if isinstance(frame, dict):
                image = frame.get("image", np.zeros((256, 256, 3), dtype=np.uint8))
                waypoints = frame.get("waypoints", np.zeros((self.horizon_steps, 2)))
            else:
                image = frame[0] if len(frame) > 0 else np.zeros((256, 256, 3))
                waypoints = frame[1] if len(frame) > 1 else np.zeros((self.horizon_steps, 2))
            
            # Encode image through SSL encoder
            with torch.no_grad():
                if isinstance(image, np.ndarray):
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                else:
                    image_tensor = image.unsqueeze(0)
                
                features = self.encoder(image_tensor)
                features = features.squeeze(0).cpu().numpy()
            
            return features, waypoints
            
        except Exception as e:
            print(f"[WaypointBCDataset] Error loading sample {idx}: {e}")
            # Return stub on error
            features = np.random.randn(self.encoder.out_dim).astype(np.float32)
            waypoints = np.random.randn(self.horizon_steps, 2).astype(np.float32) * 0.5
            return features, waypoints


def create_stub_ssl_encoder(out_dim: int = 128):
    """Create a stub SSL encoder for testing without pretrained weights.
    
    This is useful for:
    - Testing the training pipeline
    - Debugging without requiring SSL pretraining to complete
    
    Args:
        out_dim: Output feature dimension
    
    Returns:
        Stub encoder (torch.nn.Module) with forward() method
    """
    import torch.nn as nn
    
    class StubEncoder(nn.Module):
        """Stub encoder that returns random features."""
        
        def __init__(self, out_dim: int):
            super().__init__()
            self.out_dim = out_dim
            # Dummy layer to make it a valid Module
            self.dummy = nn.Linear(1, 1)
        
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Return random features."""
            import torch
            batch_size = x.shape[0]
            return torch.randn(batch_size, self.out_dim, device=x.device)
    
    print(f"[create_stub_ssl_encoder] Created stub encoder with out_dim={out_dim}")
    return StubEncoder(out_dim)


class WaypointMLPHead(nn.Module):
    """MLP head for waypoint prediction.
    
    Maps encoder features to waypoint coordinates.
    """
    
    def __init__(self, in_dim: int, horizon_steps: int, hidden_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.horizon_steps = horizon_steps
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon_steps * 2),  # (x, y) for each step
        )
    
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (B, in_dim) encoder features
        
        Returns:
            waypoints: (B, horizon_steps, 2) predicted waypoints
        """
        out = self.net(x)  # (B, horizon_steps * 2)
        return out.view(-1, self.horizon_steps, 2)


def load_ssl_encoder(checkpoint_path: Path, device: str = "auto") -> Tuple:
    """Load pretrained SSL encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to SSL checkpoint file
        device: Device to load encoder on ("auto", "cuda", "cpu")
    
    Returns:
        Tuple of (config, encoder)
    """
    import torch
    from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Load checkpoint
    print(f"[load_ssl_encoder] Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract config and encoder
    if isinstance(ckpt, dict):
        config = ckpt.get("config", {})
        encoder = TinyMultiCamEncoder(out_dim=config.get("encoder_out_dim", 128))
        
        # Try to load encoder weights
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        elif "model" in ckpt:
            encoder.load_state_dict(ckpt["model"])
        else:
            print(f"[load_ssl_encoder] Warning: No encoder weights found in checkpoint")
    else:
        # Try loading as direct encoder state
        config = {}
        encoder = TinyMultiCamEncoder(out_dim=128)
        try:
            encoder.load_state_dict(ckpt)
        except Exception as e:
            print(f"[load_ssl_encoder] Failed to load encoder: {e}")
    
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f"[load_ssl_encoder] Loaded encoder to {device}")
    return config, encoder


class WaypointBCTraining:
    """Training loop for waypoint BC with SSL encoder.
    
    This class handles:
    - Model setup (encoder + head)
    - Data loading
    - Training loop with mixed precision
    - Checkpoint saving
    """
    
    def __init__(self, config: BCConfig):
        self.config = config
        self._setup_device()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
    
    def _setup_device(self):
        """Setup computation device."""
        import torch
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        print(f"[WaypointBCTraining] Using device: {self.device}")
    
    def _setup_model(self):
        """Setup encoder and waypoint prediction head."""
        self.torch = torch
        
        # Setup encoder
        if self.config.stub_encoder:
            self.encoder = create_stub_ssl_encoder(self.config.encoder_out_dim)
        elif self.config.ssl_checkpoint:
            _, self.encoder = load_ssl_encoder(self.config.ssl_checkpoint, self.config.device)
        else:
            print("[WaypointBCTraining] No SSL checkpoint, using stub encoder")
            self.encoder = create_stub_ssl_encoder(self.config.encoder_out_dim)
        
        # Setup waypoint head (our own nn.Module version)
        self.head = WaypointMLPHead(
            in_dim=self.config.encoder_out_dim,
            horizon_steps=self.config.horizon_steps,
        ).to(self.device)
        
        # Combined model - only include nn.Modules
        self.model = nn.ModuleDict({
            "head": self.head,
        })
        
        # Encoder is kept separately (may not be nn.Module)
        self._encoder_trainable = isinstance(self.encoder, nn.Module)
        
        print(f"[WaypointBCTraining] Model setup complete")
    
    def _setup_data(self):
        """Setup data loader."""
        self.dataset = WaypointBCWithSSLDataset(
            episode_dir=self.config.episode_dir,
            encoder=self.encoder,
            horizon_steps=self.config.horizon_steps,
        )
        
        # Simple iterable dataset
        self.dataloader = iter([
            self.dataset[i] for i in range(min(len(self.dataset), self.config.batch_size))
        ])
        
        print(f"[WaypointBCTraining] Dataset size: {len(self.dataset)}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        # Only optimize the head (not encoder, which may be frozen or not an nn.Module)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_steps,
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
    
    def train_step(self, batch: Tuple) -> Dict[str, float]:
        """Single training step."""
        import torch
        
        features, waypoints = batch
        features = torch.from_numpy(features).to(self.device)
        waypoints = torch.from_numpy(waypoints).to(self.device)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            pred_waypoints = self.head(features)
            
            # L2 loss
            loss = torch.nn.functional.mse_loss(pred_waypoints, waypoints)
        
        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        import torch
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "step": step,
            "encoder": self.encoder.state_dict() if hasattr(self.encoder, 'state_dict') else {},
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": {
                "encoder_out_dim": self.config.encoder_out_dim,
                "horizon_steps": self.config.horizon_steps,
            },
        }
        
        path = self.config.output_dir / f"checkpoint_{step:06d}.pt"
        torch.save(checkpoint, path)
        print(f"[WaypointBCTraining] Saved checkpoint: {path}")
    
    def train(self):
        """Run training loop."""
        import torch
        
        print(f"[WaypointBCTraining] Starting training for {self.config.num_steps} steps")
        
        self.model.train()
        
        for step in range(1, self.config.num_steps + 1):
            try:
                batch = next(self.dataloader)
            except StopIteration:
                # Reset dataloader
                self.dataloader = iter([
                    self.dataset[i] for i in range(min(len(self.dataset), self.config.batch_size))
                ])
                batch = next(self.dataloader)
            
            metrics = self.train_step(batch)
            
            if step % self.config.log_every == 0:
                print(f"Step {step}/{self.config.num_steps} | Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e}")
            
            if step % self.config.save_every == 0:
                self.save_checkpoint(step)
        
        # Save final checkpoint
        self.save_checkpoint(self.config.num_steps)
        print("[WaypointBCTraining] Training complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Waypoint BC with SSL Encoder")
    
    # Data
    parser.add_argument("--episode-dir", type=Path, default=Path("data/waymo_episodes"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model
    parser.add_argument("--ssl-checkpoint", type=Path, default=None)
    parser.add_argument("--encoder-out-dim", type=int, default=128)
    parser.add_argument("--horizon-steps", type=int, default=20)
    
    # Training
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("out/bc"))
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    
    # Debug
    parser.add_argument("--stub", action="store_true", help="Use stub encoder")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    
    args = parser.parse_args()
    
    # Convert to config
    config = BCConfig(
        episode_dir=args.episode_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ssl_checkpoint=args.ssl_checkpoint,
        encoder_out_dim=args.encoder_out_dim,
        horizon_steps=args.horizon_steps,
        num_steps=args.num_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        stub_encoder=args.stub,
        test_mode=args.test,
    )
    
    if args.test:
        print("=" * 50)
        print("Running smoke tests...")
        print("=" * 50)
        
        # Test stub encoder creation
        print("\n[TEST 1] Creating stub encoder...")
        encoder = create_stub_ssl_encoder(128)
        print(f"  Encoder out_dim: {encoder.out_dim}")
        
        # Test with torch
        import torch
        print("\n[TEST 2] Running encoder forward...")
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        
        # Test dataset
        print("\n[TEST 3] Creating dataset...")
        dataset = WaypointBCWithSSLDataset(
            episode_dir=Path("data/waymo_episodes"),
            encoder=encoder,
            horizon_steps=20,
        )
        print(f"  Dataset size: {len(dataset)}")
        
        # Test sample
        print("\n[TEST 4] Loading sample...")
        features, waypoints = dataset[0]
        print(f"  Features shape: {features.shape}")
        print(f"  Waypoints shape: {waypoints.shape}")
        
        # Test training class initialization
        print("\n[TEST 5] Initializing training...")
        config.stub_encoder = True
        trainer = WaypointBCTraining(config)
        print("  Training class initialized OK")
        
        # Test a training step
        print("\n[TEST 6] Running training step...")
        batch = (features, waypoints)
        metrics = trainer.train_step(batch)
        print(f"  Loss: {metrics['loss']:.4f}")
        
        print("\n" + "=" * 50)
        print("✓ All smoke tests passed!")
        print("=" * 50)
        return
    
    # Run training
    trainer = WaypointBCTraining(config)
    trainer.train()


if __name__ == "__main__":
    main()

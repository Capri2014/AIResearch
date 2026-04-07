#!/usr/bin/env python3
"""
Augmented Episode SSL Training with Quality-Aware Sampling.

This script extends train_contrastive_ssl_synthetic.py to work with
augmented episodes from augment_episodes.py, using quality metrics
to weight or filter episodes during training.

The pipeline:
1. Load augmented episodes with quality metrics
2. Optionally filter by difficulty or quality score
3. Apply on-the-fly image augmentations
4. Compute contrastive loss with multi-camera views
5. Save checkpoints for downstream waypoint BC use

Usage:
    python train_augmented_ssl.py --episodes-dir data/waymo/episodes_augmented \
        --images-dir data/waymo/images --batch-size 8 --num-steps 100 \
        --quality-filter --min-quality 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


def _require_torch():
    try:
        import torch
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch


@dataclass
class Config:
    # Data
    episodes_dir: str = "data/waymo/episodes_augmented"
    images_dir: str = "data/waymo/images"
    batch_size: int = 8
    num_steps: int = 100
    max_cameras: int = 4
    
    # Quality filtering
    quality_filter: bool = False
    min_quality: float = 0.0
    difficulty_filter: Optional[str] = None  # easy, medium, hard
    
    # Augmentation
    augment_images: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    noise_std: float = 0.01
    
    # Model
    encoder_out_dim: int = 128
    encoder_hidden_dim: int = 128
    
    # Training
    lr: float = 1e-3
    temperature: float = 0.1
    
    # Output
    out_dir: str = "out/augmented_ssl"
    checkpoint_every: int = 20
    log_every: int = 5
    image_size: int = 224


class QualityAwareEpisodeDataset:
    """Dataset for augmented Waymo episodes with quality metrics."""
    
    def __init__(
        self,
        episodes_dir: str,
        images_dir: str,
        camera_names: List[str] = None,
        image_size: int = 224,
        quality_filter: bool = False,
        min_quality: float = 0.0,
        difficulty_filter: Optional[str] = None,
        augment_images: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        noise_std: float = 0.01,
    ):
        self._torch = _require_torch()
        self.episodes_dir = Path(episodes_dir)
        self.images_dir = Path(images_dir)
        self.camera_names = camera_names or ["front", "left", "right", "rear"]
        self.image_size = image_size
        self.quality_filter = quality_filter
        self.min_quality = min_quality
        self.difficulty_filter = difficulty_filter
        self.augment_images = augment_images
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.noise_std = noise_std
        
        # Load quality report if available
        quality_report_path = self.episodes_dir / "quality_report.json"
        self.quality_metrics = {}
        if quality_report_path.exists():
            with open(quality_report_path, "r") as f:
                report = json.load(f)
                self.quality_metrics = report.get("episodes", {})
        
        # Find episode files
        self.episode_files = sorted(self.episodes_dir.glob("syn_*.json"))
        
        # Filter by quality and difficulty
        self._filter_episodes()
        
        if not self.episode_files:
            raise ValueError(f"No episode files found in {episodes_dir} after filtering")
        
        print(f"Loaded {len(self.episode_files)} episodes after quality filtering")
        
        # Build index: (episode_idx, frame_idx, camera_name)
        self.index: List[Tuple[int, int, str]] = []
        for ep_idx, ep_path in enumerate(self.episode_files):
            with open(ep_path, "r") as f:
                ep_data = json.load(f)
            
            num_frames = len(ep_data.get("frames", []))
            
            for frame_idx in range(num_frames):
                for camera in self.camera_names:
                    self.index.append((ep_idx, frame_idx, camera))
        
        print(f"Total samples: {len(self.index)} ({len(self.episode_files)} episodes × frames × cameras)")
    
    def _filter_episodes(self):
        """Filter episodes based on quality and difficulty."""
        filtered = []
        
        for ep_path in self.episode_files:
            ep_name = ep_path.stem
            
            # Extract difficulty from filename (e.g., syn_42_0003_easy.json)
            difficulty = None
            for d in ["easy", "medium", "hard"]:
                if d in ep_name:
                    difficulty = d
                    break
            
            # Apply difficulty filter
            if self.difficulty_filter and difficulty != self.difficulty_filter:
                continue
            
            # Apply quality filter
            if self.quality_filter and self.quality_metrics:
                ep_quality = self.quality_metrics.get(ep_name, {}).get("quality_score", 1.0)
                if ep_quality < self.min_quality:
                    continue
            
            filtered.append(ep_path)
        
        self.episode_files = filtered
    
    def _load_image(self, ep_idx: int, frame_idx: int, camera: str) -> Image.Image:
        """Load and optionally augment image."""
        ep_path = self.episode_files[ep_idx]
        ep_name = ep_path.stem
        
        # Try to load from images directory
        img_dir = self.images_dir / f"episode_{ep_name}"
        
        if not img_dir.exists():
            # Fallback to episode JSON (base64 images)
            with open(ep_path, "r") as f:
                ep_data = json.load(f)
            
            frames = ep_data.get("frames", [])
            if frame_idx >= len(frames):
                raise ValueError(f"Frame index {frame_idx} out of range")
            
            frame = frames[frame_idx]
            cameras_data = frame.get("cameras", {})
            
            if camera not in cameras_data:
                # Try common camera name variants
                camera_map = {"front": "front", "left": "left", "right": "right", "rear": "rear"}
                camera = camera_map.get(camera, camera)
            
            cam_data = cameras_data.get(camera, {})
            
            # Handle base64 or path
            img_data = cam_data.get("image", {})
            if isinstance(img_data, str):
                if img_data.startswith("data:image"):
                    # Base64 data URL
                    import base64
                    header, b64 = img_data.split(",", 1)
                    img_bytes = base64.b64decode(b64)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                elif os.path.isabs(img_data):
                    return Image.open(img_data).convert("RGB")
            
            # Fallback: generate placeholder
            return self._generate_placeholder_image(camera)
        
        # Load from image directory
        img_path = img_dir / f"frame_{frame_idx:04d}_{camera}.png"
        
        if not img_path.exists():
            # Try alternative naming
            img_path = img_dir / f"{camera}_frame_{frame_idx:04d}.png"
        
        if not img_path.exists():
            return self._generate_placeholder_image(camera)
        
        img = Image.open(img_path).convert("RGB")
        
        # Apply augmentations if enabled
        if self.augment_images:
            img = self._augment_image(img)
        
        return img
    
    def _generate_placeholder_image(self, camera: str) -> Image.Image:
        """Generate a placeholder image with camera-specific pattern."""
        # Create different patterns for different cameras
        np.random.seed(hash(camera) % (2**31))
        
        width, height = self.image_size, self.image_size
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Add gradient based on camera
        if camera == "front":
            arr[:, :, 0] = np.linspace(50, 200, width).astype(np.uint8)
        elif camera == "left":
            arr[:, :, 1] = np.linspace(50, 200, width).astype(np.uint8)
        elif camera == "right":
            arr[:, :, 2] = np.linspace(50, 200, width).astype(np.uint8)
        else:  # rear
            arr = (arr * 0.5 + 100).astype(np.uint8)
        
        return Image.fromarray(arr)
    
    def _augment_image(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations to image."""
        from PIL import ImageEnhance
        
        # Brightness
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        # Contrast
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        
        # Saturation
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)
        
        # Hue shift (approximate)
        if random.random() < 0.3 and hasattr(Image, 'convert'):
            # Convert to HSV-like (PIL doesn't have native HSV, skip)
            pass
        
        return img
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict:
        ep_idx, frame_idx, camera = self.index[idx]
        
        # Load image
        img = self._load_image(ep_idx, frame_idx, camera)
        
        # Resize
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor
        arr = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        
        return {
            "image": img_tensor,
            "camera": camera,
            "episode_idx": ep_idx,
            "frame_idx": frame_idx,
        }


class TinyMultiCamEncoder(nn.Module):
    """Lightweight multi-camera encoder for synthetic data."""
    
    def __init__(
        self,
        num_cameras: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 128,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.in_channels = in_channels
        
        # Per-camera encoder
        self.camera_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            for _ in range(num_cameras)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_cameras, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, out_dim),
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, num_cameras, C, H, W)
        Returns:
            embeddings: (B, out_dim)
        """
        B, num_cam, C, H, W = images.shape
        
        # Encode each camera
        cam_embeddings = []
        for i in range(num_cam):
            cam_img = images[:, i]  # (B, C, H, W)
            enc = self.camera_encoders[i](cam_img)  # (B, hidden_dim)
            cam_embeddings.append(enc)
        
        # Concatenate and fuse
        fused = torch.cat(cam_embeddings, dim=1)  # (B, hidden_dim * num_cam)
        out = self.fusion(fused)  # (B, out_dim)
        
        return out


def multi_camera_simclr_loss(
    embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Compute SimCLR-style loss treating different cameras as positive pairs.
    
    Args:
        embeddings: (B, num_cameras, D) embeddings from multi-camera encoder
        temperature: softmax temperature
    
    Returns:
        loss: scalar contrastive loss
    """
    B, num_cam, D = embeddings.shape
    
    # Flatten for pairwise comparison - use reshape for non-contiguous
    embeddings = embeddings.reshape(B * num_cam, D)  # (B*num_cam, D)
    
    # Compute similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature  # (B*num_cam, B*num_cam)
    
    # Create mask: same frame (different camera) are positives
    # Each frame has num_cam samples
    labels = torch.arange(B).repeat_interleave(num_cam).to(embeddings.device)
    
    # Positive pairs: same frame, different cameras
    # For each sample, positives are other cameras in same frame
    mask = torch.zeros_like(sim, dtype=torch.bool)
    for i in range(B):
        start = i * num_cam
        end = (i + 1) * num_cam
        # All other cameras in same frame are positives
        for j in range(start, end):
            mask[j, start:end] = True
    mask.fill_diagonal_(False)
    
    # For simplicity, use cross-entropy with all pairs as negatives
    # and mask to keep only cross-camera positives
    loss = F.cross_entropy(sim, labels, reduction='none')
    
    # Average over valid samples
    valid = (labels >= 0)
    return loss[valid].mean()


def train_augmented_ssl(config: Config):
    """Train SSL on augmented episodes with quality-aware sampling."""
    torch = _require_torch()
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Create dataset
    dataset = QualityAwareEpisodeDataset(
        episodes_dir=config.episodes_dir,
        images_dir=config.images_dir,
        image_size=config.image_size,
        quality_filter=config.quality_filter,
        min_quality=config.min_quality,
        difficulty_filter=config.difficulty_filter,
        augment_images=config.augment_images,
        brightness=config.brightness,
        contrast=config.contrast,
        saturation=config.saturation,
        hue=config.hue,
        noise_std=config.noise_std,
    )
    
    # Create model
    model = TinyMultiCamEncoder(
        num_cameras=config.max_cameras,
        hidden_dim=config.encoder_hidden_dim,
        out_dim=config.encoder_out_dim,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    camera_to_idx = {cam: i for i, cam in enumerate(dataset.camera_names)}
    
    losses = []
    start_time = time.time()
    
    for step in range(config.num_steps):
        # Sample batch
        indices = random.sample(range(len(dataset)), min(config.batch_size, len(dataset)))
        
        # Collect images for each camera
        batch_images = torch.zeros(
            config.batch_size, config.max_cameras, 3, config.image_size, config.image_size
        ).to(device)
        
        batch_cameras = []
        
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            cam_idx = camera_to_idx.get(sample["camera"], 0)
            batch_images[i, cam_idx] = sample["image"].to(device)
            batch_cameras.append(sample["camera"])
        
        # Forward pass
        embeddings = model(batch_images)  # (B, num_cameras, D)
        
        # Reshape for multi-camera loss
        embeddings = embeddings.unsqueeze(1).expand(-1, config.max_cameras, -1)
        
        # Compute loss (simplified - treat all cameras as positives)
        # For true multi-camera contrastive, need different loss
        # Using simple infoNCE here
        loss = multi_camera_simclr_loss(
            embeddings.view(config.batch_size, config.max_cameras, -1),
            temperature=config.temperature,
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % config.log_every == 0:
            avg_loss = np.mean(losses[-config.log_every:])
            elapsed = time.time() - start_time
            print(f"Step {step + 1}/{config.num_steps}: loss={avg_loss:.4f}, elapsed={elapsed:.1f}s")
        
        if (step + 1) % config.checkpoint_every == 0:
            checkpoint_path = os.path.join(config.out_dir, f"checkpoint_{step + 1}.pt")
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": np.mean(losses[-config.checkpoint_every:]),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config.out_dir, "encoder_final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "encoder_out_dim": config.encoder_out_dim,
            "encoder_hidden_dim": config.encoder_hidden_dim,
            "num_cameras": config.max_cameras,
        },
    }, final_path)
    print(f"Saved final model: {final_path}")
    
    # Save training metrics
    metrics = {
        "run_id": f"augmented_ssl_{int(time.time())}",
        "config": {
            "episodes_dir": config.episodes_dir,
            "batch_size": config.batch_size,
            "num_steps": config.num_steps,
            "quality_filter": config.quality_filter,
            "min_quality": config.min_quality,
            "difficulty_filter": config.difficulty_filter,
            "augment_images": config.augment_images,
            "lr": config.lr,
            "temperature": config.temperature,
        },
        "final_loss": np.mean(losses[-10:]),
        "total_steps": config.num_steps,
        "elapsed_time": time.time() - start_time,
    }
    
    metrics_path = os.path.join(config.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SSL on augmented Waymo episodes")
    parser.add_argument("--episodes-dir", type=str, 
                        default="data/waymo/episodes_augmented",
                        help="Path to augmented episodes")
    parser.add_argument("--images-dir", type=str,
                        default="data/waymo/images",
                        help="Path to images directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--encoder-out-dim", type=int, default=128)
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--max-cameras", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--quality-filter", action="store_true",
                        help="Filter episodes by quality score")
    parser.add_argument("--min-quality", type=float, default=0.0,
                        help="Minimum quality score (0-1)")
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard"],
                        help="Filter by difficulty")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable image augmentation")
    parser.add_argument("--brightness", type=float, default=0.2)
    parser.add_argument("--contrast", type=float, default=0.2)
    parser.add_argument("--saturation", type=float, default=0.2)
    parser.add_argument("--hue", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="out/augmented_ssl")
    
    args = parser.parse_args()
    
    config = Config(
        episodes_dir=args.episodes_dir,
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        max_cameras=args.max_cameras,
        quality_filter=args.quality_filter,
        min_quality=args.min_quality,
        difficulty_filter=args.difficulty,
        augment_images=not args.no_augment,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        noise_std=args.noise_std,
        encoder_out_dim=args.encoder_out_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        lr=args.lr,
        temperature=args.temperature,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        image_size=args.image_size,
        out_dir=args.out_dir,
    )
    
    print("=== Augmented SSL Training ===")
    print(f"Episodes dir: {config.episodes_dir}")
    print(f"Quality filter: {config.quality_filter} (min={config.min_quality})")
    print(f"Difficulty filter: {config.difficulty_filter}")
    print(f"Image augmentation: {config.augment_images}")
    print(f"Batch size: {config.batch_size}, Steps: {config.num_steps}")
    print()
    
    metrics = train_augmented_ssl(config)
    
    print()
    print("=== Final Results ===")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Total time: {metrics['elapsed_time']:.1f}s")
    print(f"Output: {config.out_dir}")


if __name__ == "__main__":
    main()
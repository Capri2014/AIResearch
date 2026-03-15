"""SSL pretrain using Waymo episodes with temporal contrastive learning.

This script performs self-supervised pretraining on Waymo driving data using
a temporal contrastive objective: align embeddings between frame t and t+Δt.

Usage:
  python -m training.pretrain.train_waymo_ssl \
    --episode-dir /path/to/waymo/episodes \
    --split train \
    --batch-size 32 \
    --num-steps 1000

The script:
1. Loads Waymo episodes via WaymoTemporalPairDataset
2. Encodes anchor and positive frames using a CNN encoder
3. Applies temporal InfoNCE loss to align embeddings
4. Saves encoder checkpoints for downstream BC fine-tuning
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.pretrain.waymo_ssl_dataset import (
    WaymoTemporalPairDataset,
    collate_temporal_pairs,
    create_waymo_ssl_dataloader,
)
from training.episodes.waymo_episode_dataset import WaymoEpisodeDatasetConfig


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("PyTorch is required for Waymo SSL training.") from e
    return torch


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WaymoSSLConfig:
    """Configuration for Waymo SSL pretraining."""

    # Data
    episode_dir: str = "/tmp/waymo_episodes"
    split: str = "train"
    cameras: List[str] = field(default_factory=lambda: ["front"])
    future_waypoints: int = 8

    # Temporal pairs
    delta_t_range: Tuple[float, float] = (0.5, 2.0)

    # Training
    batch_size: int = 32
    num_steps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1

    # Model
    encoder_type: str = "resnet34"  # resnet34, resnet50, efficientnet_b0
    embedding_dim: int = 128

    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    decode_images: bool = True
    image_cache_size: int = 2048

    # DataLoader
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True

    # Output
    out_dir: Path = Path("out/waymo_ssl_pretrain")
    save_every: int = 500
    log_every: int = 20

    # Device
    device: str = "auto"
    seed: int = 42


# =============================================================================
# Encoder Model
# =============================================================================

class SimpleEncoder(nn.Module):
    """Simple CNN encoder for SSL pretraining.

    Uses a ResNet backbone (pretrained on ImageNet) and projects to
    a lower-dimensional embedding space.
    """

    def __init__(
        self,
        encoder_type: str = "resnet34",
        embedding_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim

        # Import torchvision for pretrained models
        try:
            import torchvision.models as models
            torchvision_available = True
        except ImportError:
            torchvision_available = False

        if torchvision_available:
            if encoder_type == "resnet34":
                backbone = models.resnet34(pretrained=pretrained)
                feature_dim = 512
            elif encoder_type == "resnet50":
                backbone = models.resnet50(pretrained=pretrained)
                feature_dim = 2048
            elif encoder_type == "efficientnet_b0":
                backbone = models.efficientnet_b0(pretrained=pretrained)
                feature_dim = 1280
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

            # Remove final FC layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = feature_dim
        else:
            # Fallback: simple CNN
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.feature_dim = 128

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to embedding space.

        Args:
            x: Images of shape (B, C, H, W)

        Returns:
            Embeddings of shape (B, embedding_dim)
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        # L2 normalize for contrastive learning
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return embeddings


# =============================================================================
# Temporal InfoNCE Loss
# =============================================================================

def temporal_info_nce_loss(
    anchor_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute temporal InfoNCE loss.

    Given anchor and positive embeddings, computes the contrastive loss
    where anchors and positives are treated as matching pairs, and all
    other combinations (including within-batch negatives) are treated
    as non-matches.

    Args:
        anchor_embeds: Anchor embeddings (B, D)
        positive_embeds: Positive embeddings (B, D)
        temperature: Temperature scaling for logits

    Returns:
        Scalar loss
    """
    B = anchor_embeds.shape[0]

    # Normalize embeddings
    anchor_embeds = nn.functional.normalize(anchor_embeds, dim=1)
    positive_embeds = nn.functional.normalize(positive_embeds, dim=1)

    # Compute similarity matrix (B, B)
    # Diagonal elements are anchor-positive pairs
    # Off-diagonal are negative pairs
    logits = torch.matmul(anchor_embeds, positive_embeds.T) / temperature

    # Labels are on the diagonal (each anchor matches its positive)
    labels = torch.arange(B, device=anchor_embeds.device)

    loss = nn.functional.cross_entropy(logits, labels)
    return loss


# =============================================================================
# Training
# =============================================================================

def train_step(
    encoder: nn.Module,
    batch: Dict[str, Any],
    temperature: float,
    device: torch.device,
) -> Dict[str, float]:
    """Single training step."""
    # Extract anchor and positive images
    anchor_images = batch["anchor"]["images"]
    positive_images = batch["positive"]["images"]

    if anchor_images is None or positive_images is None:
        return {"loss": 0.0, "n_samples": 0}

    # Get front camera (or first available)
    cam = list(anchor_images.keys())[0]
    anchor_img = anchor_images[cam]["images"]  # (B, C, H, W)
    positive_img = positive_images[cam]["images"]

    if anchor_img is None or positive_img is None:
        return {"loss": 0.0, "n_samples": 0}

    # Get valid mask
    anchor_valid = anchor_images[cam]["valid"]
    positive_valid = positive_images[cam]["valid"]
    valid = anchor_valid & positive_valid

    # Filter to valid samples
    anchor_img = anchor_img[valid].to(device)
    positive_img = positive_img[valid].to(device)

    if valid.sum() < 2:
        return {"loss": 0.0, "n_samples": 0}

    # Encode
    anchor_embeds = encoder(anchor_img)
    positive_embeds = encoder(positive_img)

    # Compute loss
    loss = temporal_info_nce_loss(anchor_embeds, positive_embeds, temperature)

    return {
        "loss": loss.item(),
        "n_samples": valid.sum().item(),
    }


def run_training(cfg: WaymoSSLConfig) -> Dict[str, Any]:
    """Run the SSL training loop."""
    torch = _require_torch()

    # Set seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    print(f"[waymo-ssl] Using device: {device}")

    # Create dataloader
    print(f"[waymo-ssl] Loading episodes from: {cfg.episode_dir}")
    loader = create_waymo_ssl_dataloader(
        episode_dir=cfg.episode_dir,
        split=cfg.split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        delta_t_range=cfg.delta_t_range,
        cameras=cfg.cameras,
        future_waypoints=cfg.future_waypoints,
        image_size=cfg.image_size,
        decode_images=cfg.decode_images,
        image_cache_size=cfg.image_cache_size,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
    )

    print(f"[waymo-ssl] Dataset size: {len(loader.dataset)} temporal pairs")

    # Create model
    encoder = SimpleEncoder(
        encoder_type=cfg.encoder_type,
        embedding_dim=cfg.embedding_dim,
        pretrained=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Training loop
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = cfg.out_dir / "config.json"
    with open(config_path, "w") as f:
        # Convert to dict for JSON serialization
        cfg_dict = {
            **cfg.__dict__,
            "out_dir": str(cfg.out_dir),
            "delta_t_range": list(cfg.delta_t_range),
            "cameras": cfg.cameras,
            "image_size": list(cfg.image_size),
        }
        json.dump(cfg_dict, f, indent=2)
    print(f"[waymo-ssl] Saved config to {config_path}")

    step = 0
    metrics_history: List[Dict[str, float]] = []
    it = iter(loader)

    while step < cfg.num_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # Training step
        metrics = train_step(encoder, batch, cfg.temperature, device)

        if metrics["n_samples"] == 0:
            if step % 50 == 0:
                print(f"[waymo-ssl] step={step} skipping - no valid samples")
            step += 1
            continue

        loss = torch.tensor(metrics["loss"], requires_grad=True).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % cfg.log_every == 0:
            delta_t = batch.get("delta_t")
            delta_t_str = f" dt~{delta_t[0].item():.2f}s" if delta_t is not None else ""
            print(f"[waymo-ssl] step={step} loss={metrics['loss']:.4f}{delta_t_str}")

        # Save checkpoint
        if (step + 1) % cfg.save_every == 0:
            checkpoint_path = cfg.out_dir / f"checkpoint_{step + 1}.pt"
            torch.save({
                "step": step + 1,
                "encoder_type": cfg.encoder_type,
                "embedding_dim": cfg.embedding_dim,
                "encoder_state": encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"[waymo-ssl] Saved checkpoint to {checkpoint_path}")

        # Record metrics
        metrics_history.append({
            "step": step,
            "loss": metrics["loss"],
            "n_samples": metrics["n_samples"],
        })

        step += 1

    # Save final model
    final_path = cfg.out_dir / "encoder_final.pt"
    torch.save({
        "encoder_type": cfg.encoder_type,
        "embedding_dim": cfg.embedding_dim,
        "encoder_state": encoder.state_dict(),
        "config": {
            "encoder_type": cfg.encoder_type,
            "embedding_dim": cfg.embedding_dim,
            "temperature": cfg.temperature,
            "delta_t_range": list(cfg.delta_t_range),
            "cameras": cfg.cameras,
        },
    }, final_path)
    print(f"[waymo-ssl] Saved final encoder to {final_path}")

    # Save metrics
    metrics_path = cfg.out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"[waymo-ssl] Saved metrics to {metrics_path}")

    # Training summary
    avg_loss = sum(m["loss"] for m in metrics_history) / len(metrics_history) if metrics_history else 0.0
    summary = {
        "total_steps": step,
        "final_loss": metrics_history[-1]["loss"] if metrics_history else 0.0,
        "avg_loss": avg_loss,
        "encoder_type": cfg.encoder_type,
        "embedding_dim": cfg.embedding_dim,
        "output_dir": str(cfg.out_dir),
    }

    summary_path = cfg.out_dir / "train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[waymo-ssl] Saved summary to {summary_path}")

    return summary


def parse_args() -> WaymoSSLConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Waymo SSL Pretraining")

    # Data
    parser.add_argument("--episode-dir", type=str, default="/tmp/waymo_episodes")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--cameras", type=str, nargs="+", default=["front"])
    parser.add_argument("--future-waypoints", type=int, default=8)

    # Temporal pairs
    parser.add_argument("--delta-t-min", type=float, default=0.5)
    parser.add_argument("--delta-t-max", type=float, default=2.0)

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)

    # Model
    parser.add_argument("--encoder-type", type=str, default="resnet34",
                        choices=["resnet34", "resnet50", "efficientnet_b0"])
    parser.add_argument("--embedding-dim", type=int, default=128)

    # Image processing
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--decode-images", action="store_true", default=True)
    parser.add_argument("--no-decode-images", action="store_false", dest="decode_images")
    parser.add_argument("--image-cache-size", type=int, default=2048)

    # DataLoader
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--no-drop-last", action="store_false", dest="drop_last")

    # Output
    parser.add_argument("--out-dir", type=Path, default=Path("out/waymo_ssl_pretrain"))
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)

    # Device
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return WaymoSSLConfig(
        episode_dir=args.episode_dir,
        split=args.split,
        cameras=args.cameras,
        future_waypoints=args.future_waypoints,
        delta_t_range=(args.delta_t_min, args.delta_t_max),
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        encoder_type=args.encoder_type,
        embedding_dim=args.embedding_dim,
        image_size=tuple(args.image_size),
        decode_images=args.decode_images,
        image_cache_size=args.image_cache_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        out_dir=args.out_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        seed=args.seed,
    )


def main():
    cfg = parse_args()
    summary = run_training(cfg)
    print(f"[waymo-ssl] Training complete: {summary}")


if __name__ == "__main__":
    main()

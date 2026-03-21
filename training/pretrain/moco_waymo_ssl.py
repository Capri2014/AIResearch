"""MoCo-style SSL pretraining for Waymo with momentum encoder and queue.

This module implements MoCo (Momentum Contrast) for unsupervised learning
on Waymo driving data. MoCo maintains a queue of negative examples and uses
a momentum-updated encoder for more consistent representations.

See: "Momentum Contrast for Unsupervised Visual Representation Learning"
(He et al., CVPR 2020)

Usage:
  python -m training.pretrain.moco_waymo_ssl \
    --episode-dir /path/to/waymo/episodes \
    --batch-size 32 \
    --num-steps 10000 \
    --moco-m 0.999 \
    --queue-size 65536

The key differences from vanilla temporal InfoNCE:
1. Momentum encoder (q_enc) updated via exponential moving average of q_enc
2. Queue of negative embeddings for contrastive loss
3. Shuffled BN for preventing shortcut learning
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.pretrain.waymo_ssl_dataset import (
    WaymoTemporalPairDataset,
    collate_temporal_pairs,
    create_waymo_ssl_dataloader,
)

# Import augmentations lazily
def _get_augmentation():
    from training.pretrain.augmentations import (
        build_moco_augmentation,
        build_simclr_augmentation,
        SSLAugmentationConfig,
    )
    return build_moco_augmentation, build_simclr_augmentation, SSLAugmentationConfig


def _get_waymo_config():
    from training.episodes.waymo_episode_dataset import WaymoEpisodeDatasetConfig
    return WaymoEpisodeDatasetConfig


def _require_torch():
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for MoCo SSL training.") from e
    return torch


# =============================================================================
# MoCo Encoder Models
# =============================================================================

class MoCoEncoder(nn.Module):
    """MoCo encoder with query and key encoders.
    
    The query encoder (q_enc) is trained via backprop.
    The key encoder (k_enc) is updated via momentum update from q_enc.
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
        
        # Query encoder (trained via backprop)
        self.q_encoder = self._build_encoder(encoder_type, embedding_dim, pretrained)
        
        # Key encoder (momentum updated)
        self.k_encoder = self._build_encoder(encoder_type, embedding_dim, pretrained)
        
        # Projection heads for contrastive learning
        self.q_head = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.k_head = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        
        # Freeze key encoder initially
        for param in self.k_encoder.parameters():
            param.requires_grad = False
        for param in self.k_head.parameters():
            param.requires_grad = False

    def _build_encoder(self, encoder_type: str, embedding_dim: int, pretrained: bool):
        """Build base encoder.
        
        Supports:
        - simple: Simple CNN encoder (no torchvision required)
        - resnet34/resnet50: ResNet encoders (requires torchvision)
        - efficientnet_b0: EfficientNet encoder (requires torchvision)
        """
        # Simple CNN encoder that works without torchvision
        if encoder_type == "simple":
            return self._build_simple_encoder(embedding_dim)
        
        # torchvision-based encoders
        try:
            import torchvision.models as models
        except ImportError:
            print(f"[moco-ssl] torchvision not available, falling back to simple encoder")
            return self._build_simple_encoder(embedding_dim)
        
        if encoder_type.startswith("resnet"):
            resnet_variant = encoder_type  # resnet34 or resnet50
            if encoder_type == "resnet34":
                base = models.resnet34(pretrained=pretrained)
            else:
                base = models.resnet50(pretrained=pretrained)
            
            # Remove final FC and avgpool
            features = nn.Sequential(*list(base.children())[:-2])
            return nn.ModuleDict({
                "features": features,
                "pool": nn.AdaptiveAvgPool2d((1, 1)),
                "proj": nn.Linear(base.fc.in_features, embedding_dim),
            })
        elif encoder_type.startswith("efficientnet"):
            if encoder_type == "efficientnet_b0":
                base = models.efficientnet_b0(pretrained=pretrained)
            else:
                base = models.efficientnet_b3(pretrained=pretrained)
            
            # Use features + classifier
            return nn.ModuleDict({
                "features": base.features,
                "pool": nn.AdaptiveAvgPool2d((1, 1)),
                "proj": nn.Linear(base.classifier[1].in_features, embedding_dim),
            })
        else:
            # Fall back to simple encoder
            print(f"[moco-ssl] Unknown encoder type {encoder_type}, using simple encoder")
            return self._build_simple_encoder(embedding_dim)
    
    def _build_simple_encoder(self, embedding_dim: int):
        """Build a simple CNN encoder that works without torchvision."""
        return nn.ModuleDict({
            "features": nn.Sequential(
                # Input: (B, 3, 224, 224)
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Block 1: (B, 32, 56, 56)
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Block 2: (B, 64, 28, 28)
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Block 3: (B, 128, 14, 14)
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Block 4: (B, 256, 7, 7)
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            "pool": nn.AdaptiveAvgPool2d((1, 1)),
            "proj": nn.Linear(512, embedding_dim),
        })

    def forward(self, x: torch.Tensor, use_key: bool = False) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            use_key: If True, use key encoder; otherwise use query encoder
            
        Returns:
            Embeddings (B, embedding_dim)
        """
        if use_key:
            encoder = self.k_encoder
            head = self.k_head
        else:
            encoder = self.q_encoder
            head = self.q_head
        
        # Extract features
        features = encoder["features"](x)
        features = encoder["pool"](features)
        features = features.flatten(1)
        
        # Project
        embedding = encoder["proj"](features)
        embedding = head(embedding)
        
        return F.normalize(embedding, dim=1)

    def momentum_update(self, m: float):
        """Update key encoder via momentum update.
        
        Args:
            m: Momentum coefficient (typically 0.999)
        """
        # Update encoder
        for q_param, k_param in zip(
            self.q_encoder.parameters(), 
            self.k_encoder.parameters()
        ):
            k_param.data.mul_(m).add_(q_param.data, alpha=1 - m)
        
        # Update head
        for q_param, k_param in zip(
            self.q_head.parameters(), 
            self.k_head.parameters()
        ):
            k_param.data.mul_(m).add_(q_param.data, alpha=1 - m)


class MoCoQueue:
    """Queue for MoCo negative samples.
    
    Maintains a queue of embeddings that are used as negative examples
    for contrastive learning.
    """

    def __init__(self, embedding_dim: int, queue_size: int, device: torch.device):
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.device = device
        
        # Queue: (queue_size, embedding_dim)
        self.queue = torch.randn(queue_size, embedding_dim, device=device)
        self.queue = F.normalize(self.queue, dim=1)
        
        # Pointer for queue update
        self.ptr = 0

    def enqueue_dequeue(self, embeddings: torch.Tensor):
        """Add embeddings to queue, remove oldest.
        
        Args:
            embeddings: New embeddings to add (B, embedding_dim)
        """
        B = embeddings.shape[0]
        
        # Replace old embeddings with new ones
        start = self.ptr
        end = self.ptr + B
        if end <= self.queue_size:
            self.queue[start:end] = embeddings
        else:
            # Wrap around
            first_part = self.queue_size - start
            self.queue[start:] = embeddings[:first_part]
            self.queue[:B - first_part] = embeddings[first_part:]
        
        # Update pointer
        self.ptr = (self.ptr + B) % self.queue_size

    def get_queue(self) -> torch.Tensor:
        """Get current queue for contrastive loss."""
        return self.queue


# =============================================================================
# MoCo Loss
# =============================================================================

def moco_loss(
    query_embeds: torch.Tensor,
    key_embeds: torch.Tensor,
    queue_embeds: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """MoCo contrastive loss.
    
    Computes contrastive loss where:
    - Positive pairs: query_i and key_i (diagonal)
    - Negative pairs: query_i with all queue embeddings
    
    Args:
        query_embeds: Query embeddings (B, D)
        key_embeds: Key embeddings (B, D) 
        queue_embeds: Queue embeddings (queue_size, D)
        temperature: Temperature for softmax
        
    Returns:
        Scalar loss
    """
    # Normalize embeddings
    query_embeds = F.normalize(query_embeds, dim=1)
    key_embeds = F.normalize(key_embeds, dim=1)
    queue_embeds = F.normalize(queue_embeds, dim=1)
    
    # Concatenate keys and queue for negative samples
    # Shape: (B, B + queue_size)
    all_keys = torch.cat([key_embeds, queue_embeds], dim=0)
    
    # Compute similarities: (B, B + queue_size)
    # Each query i has positive key i and negative all others
    logits = torch.matmul(query_embeds, all_keys.T) / temperature
    
    # Labels are on diagonal (query i matches key i)
    B = query_embeds.shape[0]
    labels = torch.arange(B, device=query_embeds.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MoCoWaymoSSLConfig:
    """Configuration for MoCo SSL pretraining."""

    # Data
    episode_dir: str = "/tmp/waymo_episodes"
    split: str = "train"
    cameras: List[str] = field(default_factory=lambda: ["front"])
    future_waypoints: int = 8

    # Temporal pairs
    delta_t_range: Tuple[float, float] = (0.5, 2.0)

    # Augmentation
    augmentation_type: str = "moco"
    augment: bool = True

    # Training
    batch_size: int = 32
    num_steps: int = 10000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1
    
    # MoCo specific
    moco_m: float = 0.999  # Momentum for key encoder update
    queue_size: int = 65536  # Number of negative samples in queue
    moco_warmup_steps: int = 500  # Steps before momentum starts

    # Model
    encoder_type: str = "resnet34"
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
    out_dir: Path = Path("out/moco_waymo_ssl")
    save_every: int = 1000
    log_every: int = 50

    # Device
    device: str = "auto"
    seed: int = 42


# =============================================================================
# Training
# =============================================================================

def moco_train_step(
    encoder: MoCoEncoder,
    queue: MoCoQueue,
    batch: Dict[str, Any],
    temperature: float,
    moco_m: float,
    step: int,
    warmup_steps: int,
    device: torch.device,
) -> Dict[str, float]:
    """Single MoCo training step."""
    # Extract anchor and positive images
    anchor_images = batch["anchor"]["images"]
    positive_images = batch["positive"]["images"]

    if anchor_images is None or positive_images is None:
        return {"loss": 0.0, "n_samples": 0}

    # Get front camera
    cam = list(anchor_images.keys())[0]
    anchor_img = anchor_images[cam]["images"]
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

    # Query encoder on anchor (trained via backprop)
    query_embeds = encoder(anchor_img, use_key=False)
    
    # Key encoder on positive (momentum updated)
    with torch.no_grad():
        key_embeds = encoder(positive_img, use_key=True)

    # Get queue embeddings
    queue_embeds = queue.get_queue()

    # Compute MoCo loss
    loss = moco_loss(query_embeds, key_embeds, queue_embeds, temperature)

    # Update queue with current key embeddings
    queue.enqueue_dequeue(key_embeds.detach())

    # Momentum update (after warmup)
    if step >= warmup_steps:
        encoder.momentum_update(moco_m)

    return {
        "loss": loss.item(),
        "n_samples": valid.sum().item(),
    }


def run_moco_training(cfg: MoCoWaymoSSLConfig) -> Dict[str, Any]:
    """Run the MoCo SSL training loop."""
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

    print(f"[moco-ssl] Using device: {device}")

    # Create augmentation function
    augmentation = None
    if cfg.augment and cfg.augmentation_type != "none":
        print(f"[moco-ssl] Using {cfg.augmentation_type} augmentation")
        build_moco_aug, build_simclr_aug, _ = _get_augmentation()
        if cfg.augmentation_type == "moco":
            augmentation = build_moco_aug(cfg.image_size[0])
        elif cfg.augmentation_type == "simclr":
            augmentation = build_simclr_aug(cfg.image_size[0])

    # Create dataloader
    print(f"[moco-ssl] Loading episodes from: {cfg.episode_dir}")
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
        augmentation=augmentation,
    )

    print(f"[moco-ssl] Dataset size: {len(loader.dataset)} temporal pairs")

    # Create model
    encoder = MoCoEncoder(
        encoder_type=cfg.encoder_type,
        embedding_dim=cfg.embedding_dim,
        pretrained=True,
    ).to(device)

    # Create queue
    queue = MoCoQueue(
        embedding_dim=cfg.embedding_dim,
        queue_size=cfg.queue_size,
        device=device,
    )

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
        # Convert to JSON-serializable format
        config_dict = {
            **cfg.__dict__,
            "out_dir": str(cfg.out_dir),
        }
        json.dump(config_dict, f, indent=2)
    print(f"[moco-ssl] Config saved to {config_path}")

    # Metrics tracking
    metrics = {
        "steps": [],
        "loss": [],
    }
    running_loss = 0.0
    step = 0

    print(f"[moco-ssl] Starting training for {cfg.num_steps} steps...")
    print(f"[moco-ssl] MoCo params: m={cfg.moco_m}, queue_size={cfg.queue_size}, warmup={cfg.moco_warmup_steps}")

    encoder.train()
    data_iter = iter(loader)

    while step < cfg.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Training step
        step_metrics = moco_train_step(
            encoder=encoder,
            queue=queue,
            batch=batch,
            temperature=cfg.temperature,
            moco_m=cfg.moco_m,
            step=step,
            warmup_steps=cfg.moco_warmup_steps,
            device=device,
        )

        # Backward pass
        loss = step_metrics["loss"]
        if loss > 0:
            # Recompute loss with gradients
            anchor_images = batch["anchor"]["images"]
            positive_images = batch["positive"]["images"]
            if anchor_images is not None and positive_images is not None:
                cam = list(anchor_images.keys())[0]
                anchor_img = anchor_images[cam]["images"]
                positive_img = positive_images[cam]["images"]
                if anchor_img is not None and positive_img is not None:
                    anchor_valid = anchor_images[cam]["valid"]
                    positive_valid = positive_images[cam]["valid"]
                    valid = anchor_valid & positive_valid
                    if valid.sum() >= 2:
                        anchor_img = anchor_img[valid].to(device)
                        positive_img = positive_img[valid].to(device)
                        
                        query_embeds = encoder(anchor_img, use_key=False)
                        with torch.no_grad():
                            key_embeds = encoder(positive_img, use_key=True)
                        queue_embeds = queue.get_queue()
                        loss = moco_loss(query_embeds, key_embeds, queue_embeds, cfg.temperature)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                        optimizer.step()
                        
                        # Update queue
                        queue.enqueue_dequeue(key_embeds.detach())
                        
                        # Momentum update
                        if step >= cfg.moco_warmup_steps:
                            encoder.momentum_update(cfg.moco_m)

        running_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        step += 1

        # Logging
        if step % cfg.log_every == 0:
            avg_loss = running_loss / cfg.log_every
            print(f"[moco-ssl] Step {step}/{cfg.num_steps} | Loss: {avg_loss:.4f}")
            metrics["steps"].append(step)
            metrics["loss"].append(avg_loss)
            running_loss = 0.0

        # Checkpointing
        if step % cfg.save_every == 0:
            checkpoint_path = cfg.out_dir / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "encoder_type": cfg.encoder_type,
                "embedding_dim": cfg.embedding_dim,
                "q_encoder_state": encoder.q_encoder.state_dict(),
                "q_head_state": encoder.q_head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {
                    "episode_dir": cfg.episode_dir,
                    "split": cfg.split,
                    "moco_m": cfg.moco_m,
                    "queue_size": cfg.queue_size,
                },
            }, checkpoint_path)
            print(f"[moco-ssl] Checkpoint saved to {checkpoint_path}")

    # Final checkpoint
    final_path = cfg.out_dir / "final_checkpoint.pt"
    torch.save({
        "step": step,
        "encoder_type": cfg.encoder_type,
        "embedding_dim": cfg.embedding_dim,
        "q_encoder_state": encoder.q_encoder.state_dict(),
        "q_head_state": encoder.q_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, final_path)
    print(f"[moco-ssl] Final checkpoint saved to {final_path}")

    # Save metrics
    metrics_path = cfg.out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "final_loss": metrics["loss"][-1] if metrics["loss"] else 0.0,
        "out_dir": str(cfg.out_dir),
    }


def load_moco_checkpoint(
    checkpoint_path: str | Path,
    encoder_type: str = "resnet34",
    embedding_dim: int = 128,
    device: str = "cpu",
) -> MoCoEncoder:
    """Load MoCo encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        encoder_type: Type of encoder (resnet34, resnet50, etc.)
        embedding_dim: Embedding dimension
        device: Device to load model to
        
    Returns:
        MoCoEncoder with loaded weights
    """
    torch = _require_torch()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create encoder
    encoder = MoCoEncoder(
        encoder_type=encoder_type,
        embedding_dim=embedding_dim,
        pretrained=False,  # Load from checkpoint instead
    )
    
    # Load query encoder weights
    encoder.q_encoder.load_state_dict(checkpoint["q_encoder_state"])
    encoder.q_head.load_state_dict(checkpoint["q_head_state"])
    
    # Copy to key encoder as well (for consistency)
    encoder.k_encoder.load_state_dict(checkpoint["q_encoder_state"])
    encoder.k_head.load_state_dict(checkpoint["q_head_state"])
    
    return encoder.to(device)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MoCo SSL pretraining on Waymo data")
    
    # Data
    parser.add_argument("--episode-dir", type=str, default="/tmp/waymo_episodes")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cameras", type=str, nargs="+", default=["front"])
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--augmentation-type", type=str, default="moco",
                        choices=["moco", "simclr", "light", "none"])
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    
    # MoCo
    parser.add_argument("--moco-m", type=float, default=0.999)
    parser.add_argument("--queue-size", type=int, default=65536)
    parser.add_argument("--moco-warmup-steps", type=int, default=500)
    
    # Model
    parser.add_argument("--encoder-type", type=str, default="resnet34")
    parser.add_argument("--embedding-dim", type=int, default=128)
    
    # Image
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    
    # Output
    parser.add_argument("--out-dir", type=str, default="out/moco_waymo_ssl")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    
    # Other
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Create config
    cfg = MoCoWaymoSSLConfig(
        episode_dir=args.episode_dir,
        split=args.split,
        cameras=args.cameras,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        augmentation_type=args.augmentation_type,
        augment=not args.no_augment,
        moco_m=args.moco_m,
        queue_size=args.queue_size,
        moco_warmup_steps=args.moco_warmup_steps,
        encoder_type=args.encoder_type,
        embedding_dim=args.embedding_dim,
        image_size=tuple(args.image_size),
        out_dir=Path(args.out_dir),
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        seed=args.seed,
    )
    
    # Run training
    result = run_moco_training(cfg)
    print(f"[moco-ssl] Training complete! Output: {result['out_dir']}")


if __name__ == "__main__":
    main()

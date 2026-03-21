"""
SimCLR Self-Supervised Learning for Waymo Data

SimCLR (Simple Contrastive Learning of Representations) implementation
for Waymo driving data. Unlike MoCo, SimCLR uses large batch sizes
and symmetric projector architecture.

Reference: "A Simple Framework for Contrastive Learning of Visual Representations"
(Chen et al., 2020)

Architecture:
    image → encoder → projector → contrastive loss
           (frozen)  (trainable)
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from training.episodes.waymo_episode_dataset import (
    WaymoEpisodeDataset,
    WaymoEpisodeDatasetConfig,
    create_waymo_dataloader,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SimCLRConfig:
    """Configuration for SimCLR SSL training on Waymo data."""

    # Data
    episode_dir: str = "data/waymo_episodes"
    batch_size: int = 64
    num_workers: int = 4
    image_size: int = 256

    # Model
    encoder_type: str = "resnet34"  # resnet34, resnet50, efficientnet_b0
    embedding_dim: int = 128
    projection_hidden_dim: int = 256

    # SimCLR-specific
    temperature: float = 0.07  # NT-Xent temperature
    use_broadcast: bool = True  # Use Gather:0 for distributed training

    # Training
    num_steps: int = 10000
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    warmup_steps: int = 500

    # Augmentations
    color_jitter: float = 0.4
    gaussian_blur: bool = True
    random_resized_crop: bool = True

    # Output
    out_dir: str = "out/simclr_waymo"
    log_interval: int = 10
    checkpoint_interval: int = 1000


# ============================================================================
# SimCLR Augmentations
# ============================================================================


class SimCLRAugmentation:
    """SimCLR-style augmentations for Waymo camera images."""

    def __init__(self, image_size: int = 256, color_jitter: float = 0.4,
                 gaussian_blur: bool = True):
        self.image_size = image_size
        self.color_jitter = color_jitter
        self.gaussian_blur = gaussian_blur

        # Try to import torchvision, use simple fallback if unavailable
        try:
            from torchvision import transforms
            self.use_torchvision = True
        except ImportError:
            self.use_torchvision = False

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply SimCLR augmentations to a single image."""
        if self.use_torchvision:
            from torchvision import transforms

            # Random resized crop
            if self.random_resized_crop:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.2, 1.0), ratio=(0.75, 1.33)
                )
                image = transforms.functional.resized_crop(
                    image, i, j, h, w, (self.image_size, self.image_size)
                )
            else:
                image = transforms.functional.resize(
                    image, (self.image_size, self.image_size)
                )

            # Random horizontal flip
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)

            # Color jitter (simplified)
            if self.color_jitter > 0:
                # Brightness/contrast adjustment
                image = image * (1 + random.uniform(-self.color_jitter, self.color_jitter))
                image = torch.clamp(image, 0, 1)

            # Normalize
            image = transforms.functional.normalize(
                image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            # Simple fallback augmentations without torchvision
            # Resize
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Random horizontal flip
            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])

            # Simple color adjustment
            if self.color_jitter > 0:
                # Brightness
                brightness_factor = 1 + random.uniform(-self.color_jitter, self.color_jitter)
                image = image * brightness_factor

                # Contrast
                mean = image.mean(dim=(1, 2), keepdim=True)
                contrast_factor = 1 + random.uniform(-self.color_jitter, self.color_jitter)
                image = (image - mean) * contrast_factor + mean

            image = torch.clamp(image, 0, 1)

            # Normalize
            image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                    torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        return image


class SimCLRPairTransform:
    """Creates two augmented views for contrastive learning."""

    def __init__(self, image_size: int = 256, color_jitter: float = 0.4,
                 gaussian_blur: bool = True):
        self.aug1 = SimCLRAugmentation(image_size, color_jitter, gaussian_blur)
        self.aug2 = SimCLRAugmentation(image_size, color_jitter, gaussian_blur)

    def __call__(self, image: torch.Tensor):
        """Return two augmented views of the same image."""
        return self.aug1(image), self.aug2(image)


# ============================================================================
# SimCLR Model Components
# ============================================================================


class SimCLREncoder(nn.Module):
    """Encoder network for SimCLR (ResNet backbone or simple CNN)."""

    def __init__(self, encoder_type: str = "resnet34", embedding_dim: int = 128):
        super().__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim

        # Try to import torchvision, fall back to simple CNN if unavailable
        try:
            import torchvision.models as models
            TORCHVISION_AVAILABLE = True
        except ImportError:
            TORCHVISION_AVAILABLE = False

        # Build encoder based on type
        if TORCHVISION_AVAILABLE and encoder_type.startswith("resnet"):
            if encoder_type == "resnet34":
                backbone = models.resnet34(pretrained=False)
                feature_dim = 512
            elif encoder_type == "resnet50":
                backbone = models.resnet50(pretrained=False)
                feature_dim = 2048
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

            # Remove final FC layer
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = feature_dim

        elif TORCHVISION_AVAILABLE and encoder_type == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=False)
            feature_dim = 1280
            self.encoder = nn.Sequential(
                backbone.features,
                backbone.avgpool
            )
            self.feature_dim = feature_dim

        else:
            # Simple CNN encoder (fallback)
            print(f"Using simple CNN encoder (torchvision not available)")
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_dim = 128

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, projection_hidden_dim(encoder_type)),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim(encoder_type), embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        features = self.encoder(x)  # [B, feature_dim, 1, 1]
        features = features.flatten(1)  # [B, feature_dim]
        embeddings = self.projector(features)  # [B, embedding_dim]
        return embeddings


def projection_hidden_dim(encoder_type: str = "resnet34") -> int:
    """Get projection hidden dimension based on encoder."""
    if encoder_type == "resnet50":
        return 1024
    elif encoder_type == "efficientnet_b0":
        return 512
    return 256


class SimCLRModel(nn.Module):
    """Complete SimCLR model with encoder and projector."""

    def __init__(self, encoder_type: str = "resnet34", embedding_dim: int = 128):
        super().__init__()
        self.encoder = SimCLREncoder(encoder_type, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings (includes projection)."""
        return self.encoder(x)

    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get pre-projection features (for downstream tasks)."""
        features = self.encoder.encoder(x)
        return features.flatten(1)


# ============================================================================
# SimCLR Loss (NT-Xent)
# ============================================================================


def simclr_loss(z_i: torch.Tensor, z_j: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    SimCLR NT-Xent loss (normalized temperature-scaled cross entropy).

    Args:
        z_i: Embeddings from view i [B, D]
        z_j: Embeddings from view j [B, D]

    Returns:
        Contrastive loss scalar
    """
    batch_size = z_i.shape[0]

    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Concatenate representations
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    # Create masks for positive pairs
    # Positive pairs are (i, i+B) and (i+B, i)
    mask = torch.eye(2 * batch_size, device=z.device)
    mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device)
    mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device)

    # Remove diagonal (self-similarity)
    sim_matrix = sim_matrix - mask * 1e9

    # Compute loss for each positive pair
    pos_sim_i = sim_matrix[:batch_size, batch_size:]  # [B, B]
    pos_sim_j = sim_matrix[batch_size:, :batch_size]  # [B, B]

    # For row i, the positive is at column i+B
    pos_sim = torch.cat([pos_sim_i.diag(), pos_sim_j.diag()], dim=0)  # [2B]

    # All other similarities are negatives
    neg_sim = sim_matrix.masked_fill(mask.bool(), float('-inf'))

    # Numerically stable softmax
    loss_i = -pos_sim_i.diag() + torch.logsumexp(pos_sim_i, dim=1)
    loss_j = -pos_sim_j.diag() + torch.logsumexp(pos_sim_j, dim=1)

    loss = (loss_i.mean() + loss_j.mean()) / 2

    return loss


def simclr_loss_v2(z_i: torch.Tensor, z_j: torch.Tensor,
                   temperature: float = 0.07) -> torch.Tensor:
    """
    Alternative SimCLR loss using the simplified formula.
    """
    batch_size = z_i.shape[0]

    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Concatenate all representations
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

    # Compute similarity matrix
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    # Create labels for positive pairs
    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels, labels], dim=0)  # [2B]

    # Use cross-entropy loss
    loss = F.cross_entropy(sim, labels)

    return loss


# ============================================================================
# Training Loop
# ============================================================================


def load_simclr_encoder(config: SimCLRConfig):
    """Load SimCLR encoder from checkpoint."""
    import training.pretrain

    checkpoint_path = os.path.join(config.out_dir, "encoder_final.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading SimCLR encoder from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = SimCLRModel(config.encoder_type, config.embedding_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint.get("config", None)

    return None, None


def simclr_training_loop(config: SimCLRConfig):
    """Main training loop for SimCLR on Waymo data."""
    print(f"Starting SimCLR training with config:")
    print(f"  Encoder: {config.encoder_type}")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Steps: {config.num_steps}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    os.makedirs(os.path.join(config.out_dir, "checkpoints"), exist_ok=True)

    # Create model
    model = SimCLRModel(config.encoder_type, config.embedding_dim)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - config.warmup_steps) /
                                       (config.num_steps - config.warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create data loader
    dataset_config = WaymoEpisodeDatasetConfig(
        episode_dir=config.episode_dir,
        image_size=config.image_size,
        cameras=["front"],
        waypoint_horizon=8,
    )

    try:
        dataset = WaymoEpisodeDataset(dataset_config)
        print(f"Loaded {len(dataset)} frames from {config.episode_dir}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Using stub data for testing...")
        # Return early for testing
        return model, {"test": True}

    # Note: For SimCLR, we need to apply augmentations per batch
    # For now, we'll use the raw dataset and apply augmentations in training
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda x: x  # Custom collate below
    )

    # Training loop
    model.train()
    step = 0
    epoch = 0

    # TensorBoard writer (optional)
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=config.out_dir)

    # Augmentation for contrastive views
    transform = SimCLRPairTransform(
        image_size=config.image_size,
        color_jitter=config.color_jitter,
        gaussian_blur=config.gaussian_blur
    )

    print("Starting training...")

    while step < config.num_steps:
        epoch += 1

        for batch in dataloader:
            if step >= config.num_steps:
                break

            # Apply augmentations to create two views
            # Each batch item has 'image' or 'images' key
            views_i = []
            views_j = []

            for item in batch:
                # Get image (handle different key names)
                if 'image' in item:
                    img = item['image']
                elif 'images' in item:
                    img = item['images'].get('front', item['images'].get(list(item['images'].keys())[0]))
                else:
                    # Skip if no image
                    continue

                # Apply augmentations
                if isinstance(img, str):
                    # Load image from path
                    from PIL import Image
                    import torchvision.transforms.functional as TF
                    img = Image.open(img).convert('RGB')
                    img = TF.to_tensor(img)

                aug_i, aug_j = transform(img)
                views_i.append(aug_i)
                views_j.append(aug_j)

            if len(views_i) == 0:
                continue

            # Stack views
            views_i = torch.stack(views_i).to(device)
            views_j = torch.stack(views_j).to(device)

            # Forward pass
            z_i = model(views_i)
            z_j = model(views_j)

            # Compute loss
            loss = simclr_loss(z_i, z_j, config.temperature)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            if step % config.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), step)
                    writer.add_scalar("train/lr", lr, step)

                print(f"Step {step}/{config.num_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

            # Checkpointing
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    config.out_dir, "checkpoints", f"checkpoint_{step}.pt"
                )
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "encoder_type": config.encoder_type,
                        "embedding_dim": config.embedding_dim,
                        "temperature": config.temperature,
                    }
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            step += 1

    # Save final model
    final_path = os.path.join(config.out_dir, "encoder_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": {
            "encoder_type": config.encoder_type,
            "embedding_dim": config.embedding_dim,
            "temperature": config.temperature,
        }
    }, final_path)
    print(f"Saved final model to {final_path}")

    if writer is not None:
        writer.close()

    return model, {"step": step, "loss": loss.item()}


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="SimCLR SSL Training for Waymo")
    parser.add_argument("--episode-dir", type=str, default="data/waymo_episodes",
                        help="Path to Waymo episode directory")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--encoder-type", type=str, default="resnet34",
                        choices=["resnet34", "resnet50", "efficientnet_b0"],
                        help="Encoder backbone type")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="NT-Xent temperature")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--out-dir", type=str, default="out/simclr_waymo",
                        help="Output directory")
    parser.add_argument("--test", action="store_true",
                        help="Run smoke test with stub data")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    config = SimCLRConfig(
        episode_dir=args.episode_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        encoder_type=args.encoder_type,
        embedding_dim=args.embedding_dim,
        temperature=args.temperature,
        learning_rate=args.lr,
        out_dir=args.out_dir,
    )

    # Run test mode
    if args.test:
        print("Running SimCLR smoke test...")

        # Test 1: Model creation
        model = SimCLRModel("resnet34", 128)
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} params")

        # Test 2: Forward pass
        x = torch.randn(4, 3, 256, 256)
        z = model(x)
        print(f"✓ Forward pass: {z.shape}")

        # Test 3: Augmentation
        aug = SimCLRPairTransform(256)
        x_img = torch.randn(3, 256, 256)
        aug_i, aug_j = aug(x_img)
        print(f"✓ Augmentation: {aug_i.shape}, {aug_j.shape}")

        # Test 4: Loss computation
        loss = simclr_loss(z, z)
        print(f"✓ Loss computation: {loss.item():.4f}")

        # Test 5: Save/load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": model.state_dict()}, f.name)
            checkpoint = torch.load(f.name)
            model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Save/load checkpoint")

        print("\nAll smoke tests passed!")
        return

    # Run training
    simclr_training_loop(config)


if __name__ == "__main__":
    main()

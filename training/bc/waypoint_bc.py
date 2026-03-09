"""
Waypoint Behavior Cloning (BC) Module

Supervised learning for waypoint prediction using pre-trained SSL encoder features.
Bridges SSL pretrain → waypoint BC → RL refinement pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json
from pathlib import Path


@dataclass
class BCConfig:
    """Configuration for Waypoint BC training."""
    # Model architecture
    encoder_dim: int = 256
    hidden_dim: int = 512
    num_waypoints: int = 8
    future_len: int = 4.0  # seconds of future waypoints
    
    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 50
    grad_clip: float = 1.0
    
    # Loss weights
    loss_waypoint_weight: float = 1.0
    loss_speed_weight: float = 0.1
    
    # Checkpoint
    ssl_encoder_path: Optional[str] = None  # Path to pre-trained SSL encoder
    checkpoint_dir: str = "out/waypoint_bc"


class SSLEncoder(nn.Module):
    """
    Mock SSL Encoder that extracts features from perception input.
    In production, loads pre-trained SSL encoder (JEPA, contrastive, etc.).
    """
    def __init__(self, input_dim: int = 3, encoder_dim: int = 256):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        # Mock encoder backbone (CNN + MLP)
        # In production: load real SSL encoder checkpoint
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] perception input (e.g., bird's eye view)
        Returns:
            features: [B, encoder_dim] latent features
        """
        B = x.shape[0]
        x = self.conv(x)
        x = x.flatten(1)
        features = self.fc(x)
        return features


class WaypointHead(nn.Module):
    """
    Predicts future waypoints from encoder features.
    Outputs (x, y) coordinates for each future timestep.
    """
    def __init__(self, encoder_dim: int = 256, hidden_dim: int = 512, num_waypoints: int = 8):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Waypoint prediction head
        self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * 2)  # (x, y) for each waypoint
        
        # Speed prediction head
        self.speed_head = nn.Linear(hidden_dim, 1)  # scalar speed
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, encoder_dim] encoder features
        Returns:
            waypoints: [B, num_waypoints, 2] (x, y) coordinates in meters
            speed: [B, 1] predicted speed in m/s
        """
        hidden = self.mlp(features)
        
        waypoints = self.waypoint_head(hidden)  # [B, num_waypoints * 2]
        waypoints = waypoints.view(-1, self.num_waypoints, 2)  # [B, num_waypoints, 2]
        
        speed = self.speed_head(hidden)  # [B, 1]
        
        return waypoints, speed


class WaypointBCModel(nn.Module):
    """
    Full Waypoint Behavior Cloning model.
    Combines SSL encoder + waypoint prediction head.
    """
    def __init__(self, config: BCConfig, load_encoder: bool = True):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = SSLEncoder(input_dim=3, encoder_dim=config.encoder_dim)
        
        # Waypoint head
        self.waypoint_head = WaypointHead(
            encoder_dim=config.encoder_dim,
            hidden_dim=config.hidden_dim,
            num_waypoints=config.num_waypoints
        )
        
        # Load pre-trained encoder if path provided
        if load_encoder and config.ssl_encoder_path:
            self.load_encoder(config.ssl_encoder_path)
            
    def load_encoder(self, path: str):
        """Load pre-trained SSL encoder weights."""
        # In production: load real SSL encoder
        # For now: mock implementation
        pass
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] perception input
        Returns:
            waypoints: [B, num_waypoints, 2] predicted waypoints
            speed: [B, 1] predicted speed
        """
        features = self.encoder(x)
        waypoints, speed = self.waypoint_head(features)
        return waypoints, speed
    
    def predict(self, x: torch.Tensor) -> dict:
        """
        Prediction mode - returns dict for downstream use.
        """
        waypoints, speed = self.forward(x)
        return {
            "waypoints": waypoints,  # [B, num_waypoints, 2]
            "speed": speed,          # [B, 1]
        }


def compute_bc_loss(
    pred_waypoints: torch.Tensor,
    pred_speed: torch.Tensor,
    target_waypoints: torch.Tensor,
    target_speed: torch.Tensor,
    config: BCConfig
) -> dict:
    """
    Compute Behavior Cloning loss.
    
    Args:
        pred_waypoints: [B, num_waypoints, 2] predicted waypoints
        pred_speed: [B, 1] predicted speed
        target_waypoints: [B, num_waypoints, 2] ground truth waypoints
        target_speed: [B, 1] ground truth speed
        config: BC configuration
    Returns:
        dict of losses
    """
    # Waypoint L2 loss (per waypoint, then average)
    waypoint_loss = F.mse_loss(pred_waypoints, target_waypoints)
    
    # Speed L2 loss
    speed_loss = F.mse_loss(pred_speed, target_speed)
    
    # Total loss
    total_loss = (
        config.loss_waypoint_weight * waypoint_loss +
        config.loss_speed_weight * speed_loss
    )
    
    return {
        "loss": total_loss,
        "loss_waypoint": waypoint_loss,
        "loss_speed": speed_loss,
    }


class WaypointBCDataset(torch.utils.data.Dataset):
    """
    Dataset for Waypoint BC training.
    Loads (image, waypoints, speed) tuples.
    In production: loads from Waymo episodes with waypoint extraction.
    """
    def __init__(
        self,
        data_path: Optional[str] = None,
        num_samples: int = 1000,
        num_waypoints: int = 8,
        image_size: Tuple[int, int] = (128, 128)
    ):
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
        self.image_size = image_size
        
        # In production: load real data from data_path
        # For now: generate synthetic data
        self.images = torch.randn(num_samples, 3, *image_size)
        self.waypoints = torch.randn(num_samples, num_waypoints, 2) * 10  # meters
        self.speeds = torch.randn(num_samples, 1).abs() * 5  # m/s
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            "waypoints": self.waypoints[idx],
            "speed": self.speeds[idx],
        }


def train_epoch(
    model: WaypointBCModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: BCConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """Train one epoch."""
    model.train()
    total_metrics = {"loss": 0.0, "loss_waypoint": 0.0, "loss_speed": 0.0}
    
    for batch in dataloader:
        images = batch["image"].to(device)
        target_waypoints = batch["waypoints"].to(device)
        target_speed = batch["speed"].to(device)
        
        # Forward pass
        pred_waypoints, pred_speed = model(images)
        
        # Compute loss
        losses = compute_bc_loss(pred_waypoints, pred_speed, target_waypoints, target_speed, config)
        
        # Backward pass
        optimizer.zero_grad()
        losses["loss"].backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
        optimizer.step()
        
        # Accumulate metrics
        for k, v in losses.items():
            total_metrics[k] += v.item() * images.shape[0]
    
    # Average metrics
    n = len(dataloader.dataset)
    for k in total_metrics:
        total_metrics[k] /= n
        
    return total_metrics


@torch.no_grad()
def evaluate(
    model: WaypointBCModel,
    dataloader: torch.utils.data.DataLoader,
    config: BCConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_metrics = {"loss": 0.0, "loss_waypoint": 0.0, "loss_speed": 0.0}
    
    for batch in dataloader:
        images = batch["image"].to(device)
        target_waypoints = batch["waypoints"].to(device)
        target_speed = batch["speed"].to(device)
        
        # Forward pass
        pred_waypoints, pred_speed = model(images)
        
        # Compute loss
        losses = compute_bc_loss(pred_waypoints, pred_speed, target_waypoints, target_speed, config)
        
        # Accumulate metrics
        for k, v in losses.items():
            total_metrics[k] += v.item() * images.shape[0]
    
    # Average metrics
    n = len(dataloader.dataset)
    for k in total_metrics:
        total_metrics[k] /= n
        
    return total_metrics


def save_checkpoint(
    model: WaypointBCModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: BCConfig,
    checkpoint_path: Path
):
    """Save training checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "encoder_dim": config.encoder_dim,
            "hidden_dim": config.hidden_dim,
            "num_waypoints": config.num_waypoints,
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    config: BCConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[WaypointBCModel, torch.optim.Optimizer, dict]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = WaypointBCModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model, optimizer, checkpoint.get("metrics", {})


if __name__ == "__main__":
    # Smoke test
    config = BCConfig(num_waypoints=8, batch_size=4, epochs=2)
    model = WaypointBCModel(config)
    
    # Create dummy batch
    images = torch.randn(4, 3, 128, 128)
    target_waypoints = torch.randn(4, 8, 2)
    target_speed = torch.randn(4, 1)
    
    # Forward pass
    pred_waypoints, pred_speed = model(images)
    print(f"Pred waypoints shape: {pred_waypoints.shape}")
    print(f"Pred speed shape: {pred_speed.shape}")
    
    # Compute loss
    losses = compute_bc_loss(pred_waypoints, pred_speed, target_waypoints, target_speed, config)
    print(f"Loss: {losses['loss'].item():.4f}")
    
    print("✓ WaypointBC smoke test passed")

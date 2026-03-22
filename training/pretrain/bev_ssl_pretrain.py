"""
BEV SSL Pretraining for Waymo Data.

Self-supervised pretraining of BEV encoder using temporal contrastive learning
and cross-modal alignment between camera and LiDAR features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np

from training.pretrain.bev_encoder import BEVEncoder, BEVConfig, create_bev_encoder


@dataclass
class BEVSSLConfig:
    """Configuration for BEV SSL pretraining."""
    # Model
    encoder_dim: int = 128
    bev_channels: int = 64
    fusion_method: str = "concat"
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.1
    
    # Contrastive
    queue_size: int = 16384
    momentum: float = 0.999
    num_negatives: int = 4096
    
    # Augmentation
    temporal_stride: int = 3  # frames between positive pairs
    noise_scale: float = 0.01
    
    # Data
    episode_dir: str = "data/waymo_episodes"
    num_workers: int = 4
    
    # Output
    output_dir: str = "out/bev_ssl"


class bevQueue:
    """Queue for BEV SSL contrastive learning."""
    
    def __init__(self, queue_size: int, feature_dim: int, device: torch.device):
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.device = device
        
        # Initialize queue
        self.queue = torch.zeros(queue_size, feature_dim, device=device)
        self.queue_ptr = 0
        self.queue_full = False
    
    def enqueue_dequeue(self, keys: torch.Tensor):
        """Add new keys to queue, remove oldest."""
        batch_size = keys.shape[0]
        
        # Update pointers
        ptr = self.queue_ptr
        
        # Handle wrap-around
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Split across boundary
            first_size = self.queue_size - ptr
            self.queue[ptr:] = keys[:first_size]
            self.queue[:batch_size - first_size] = keys[first_size:]
        
        # Update pointer
        self.queue_ptr = (ptr + batch_size) % self.queue_size
        
        # Mark as full
        if ptr + batch_size >= self.queue_size:
            self.queue_full = True
    
    def get_queue(self) -> torch.Tensor:
        """Get current queue for contrastive loss."""
        if self.queue_full:
            return self.queue
        else:
            return self.queue[:self.queue_ptr]


class TemporalContrastiveLoss(nn.Module):
    """Temporal contrastive loss for BEV features."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        pos_key: torch.Tensor,
        neg_keys: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            query: (B, D) - query features
            pos_key: (B, D) - positive key features (temporal neighbor)
            neg_keys: (B, K, D) - negative keys from queue
        Returns:
            loss: scalar loss
            metrics: dict of metrics
        """
        # Normalize features
        query = F.normalize(query, dim=-1)
        pos_key = F.normalize(pos_key, dim=-1)
        neg_keys = F.normalize(neg_keys, dim=-1)
        
        # Compute positive similarity
        pos_sim = (query * pos_key).sum(dim=-1) / self.temperature  # (B,)
        
        # Compute negative similarity
        # query: (B, D), neg_keys: (B, K, D) -> (B, K)
        neg_sim = torch.bmm(query.unsqueeze(1), neg_keys.transpose(1, 2)).squeeze(1) / self.temperature
        
        # Combine
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1+K)
        
        # Labels are 0 (positive is first)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # Metrics
        with torch.no_grad():
            pos_sim_mean = pos_sim.exp().mean()
            neg_sim_mean = neg_sim.exp().mean()
        
        metrics = {
            "loss": loss.item(),
            "pos_sim": pos_sim_mean.item(),
            "neg_sim": neg_sim_mean.item(),
        }
        
        return loss, metrics


class CrossModalAlignmentLoss(nn.Module):
    """Align camera and LiDAR features in BEV space."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        camera_features: torch.Tensor,
        lidar_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            camera_features: (B, D) camera-derived features
            lidar_features: (B, D) LiDAR-derived features
        Returns:
            loss: alignment loss
            metrics: dict of metrics
        """
        # Normalize
        camera_features = F.normalize(camera_features, dim=-1)
        lidar_features = F.normalize(lidar_features, dim=-1)
        
        # Similarity matrix
        sim = torch.mm(camera_features, lidar_features.t()) / self.temperature
        
        # Labels are on diagonal
        labels = torch.arange(camera_features.size(0), device=camera_features.device)
        
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        
        with torch.no_grad():
            alignment = torch.diagonal(sim).exp().mean()
        
        metrics = {
            "alignment_loss": loss.item(),
            "alignment": alignment.item()
        }
        
        return loss, metrics


class WaymoBEVDataset(Dataset):
    """Dataset for BEV SSL pretraining from Waymo episodes."""
    
    def __init__(
        self,
        episode_dir: str,
        temporal_stride: int = 3,
        transform: Optional[callable] = None
    ):
        self.episode_dir = Path(episode_dir)
        self.temporal_stride = temporal_stride
        self.transform = transform
        
        # Load episode metadata
        self.episodes = []
        self._load_episodes()
    
    def _load_episodes(self):
        """Load available episodes."""
        if not self.episode_dir.exists():
            print(f"Warning: Episode directory {self.episode_dir} not found")
            return
        
        for episode_path in sorted(self.episode_dir.iterdir()):
            if episode_path.is_dir():
                # Check for frames
                frames_dir = episode_path / "frames"
                if frames_dir.exists():
                    num_frames = len(list(frames_dir.glob("*.pt")))
                    if num_frames > self.temporal_stride:
                        self.episodes.append({
                            "path": episode_path,
                            "num_frames": num_frames
                        })
    
    def __len__(self) -> int:
        return sum(ep["num_frames"] - self.temporal_stride for ep in self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a temporal pair of frames."""
        # Find episode and frame index
        for ep in self.episodes:
            if idx < ep["num_frames"] - self.temporal_stride:
                break
            idx -= ep["num_frames"] - self.temporal_stride
        
        frame_idx = idx
        pos_frame_idx = frame_idx + self.temporal_stride
        
        # Load frames
        try:
            # Try to load from frames directory
            frames_dir = ep["path"] / "frames"
            
            # For now, return dummy data if not found
            # In practice, would load actual camera + LiDAR data
            if list(frames_dir.glob("*.pt")):
                frame_curr = torch.load(frames_dir / f"frame_{frame_idx:05d}.pt")
                frame_pos = torch.load(frames_dir / f"frame_{pos_frame_idx:05d}.pt")
            else:
                # Stub data for testing
                frame_curr = self._dummy_frame()
                frame_pos = self._dummy_frame()
        except Exception as e:
            # Fallback to dummy data
            frame_curr = self._dummy_frame()
            frame_pos = self._dummy_frame()
        
        return {
            "images": frame_curr["images"],
            "lidar": frame_curr["lidar"],
            "images_pos": frame_pos["images"],
            "lidar_pos": frame_pos["lidar"],
        }
    
    def _dummy_frame(self) -> Dict[str, torch.Tensor]:
        """Generate dummy frame for testing."""
        return {
            "images": torch.randn(6, 3, 224, 224),  # 6 cameras
            "lidar": torch.randn(1000, 2) * 50,  # LiDAR points
        }


class BEVSSLTrainer:
    """Trainer for BEV SSL pretraining."""
    
    def __init__(self, config: BEVSSLConfig):
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model - query encoder (trained) and key encoder (momentum)
        self.bev_config = BEVConfig(
            encoder_dim=config.encoder_dim,
            bev_channels=config.bev_channels,
            fusion_method=config.fusion_method
        )
        
        self.query_encoder = create_bev_encoder(self.bev_config).to(self.device)
        self.key_encoder = create_bev_encoder(self.bev_config).to(self.device)
        
        # Freeze key encoder
        for param in self.key_encoder.parameters():
            param.requires_grad = False
        
        # Queue
        self.queue = bevQueue(
            queue_size=config.queue_size,
            feature_dim=config.encoder_dim,
            device=self.device
        )
        
        # Losses
        self.temporal_loss = TemporalContrastiveLoss(temperature=config.temperature)
        self.alignment_loss = CrossModalAlignmentLoss(temperature=config.temperature)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.query_encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Dataset
        self.dataset = WaymoBEVDataset(
            episode_dir=config.episode_dir,
            temporal_stride=config.temporal_stride
        )
        
        # Output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.epoch = 0
    
    @torch.no_grad()
    def _update_key_encoder(self):
        """Update key encoder via momentum update."""
        m = self.config.momentum
        
        for (name_q, param_q), (name_k, param_k) in zip(
            self.query_encoder.named_parameters(),
            self.key_encoder.named_parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1 - m)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Move to device
        images = batch["images"].to(self.device)
        lidar = batch["lidar"].to(self.device)
        images_pos = batch["images_pos"].to(self.device)
        lidar_pos = batch["lidar_pos"].to(self.device)
        
        # Query encoder forward
        q_outputs = self.query_encoder(images=images, lidar_points=lidar)
        q_bev = q_outputs["bev_features"]
        
        # Key encoder forward (no grad)
        with torch.no_grad():
            k_outputs = self.key_encoder(images=images_pos, lidar_points=lidar_pos)
            k_bev = k_outputs["bev_features"]
        
        # Get negatives from queue
        neg_keys = self.queue.get_queue()  # (K, D)
        
        # Expand negatives to match batch
        B = q_bev.size(0)
        neg_indices = torch.randint(0, neg_keys.size(0), (B, self.config.num_negatives), device=self.device)
        neg_keys_batch = neg_keys[neg_indices]  # (B, K, D)
        
        # Temporal contrastive loss
        temporal_loss, temporal_metrics = self.temporal_loss(q_bev, k_bev, neg_keys_batch)
        
        # Cross-modal alignment loss
        if "camera_features" in q_outputs and "lidar_features" in q_outputs:
            align_loss, align_metrics = self.alignment_loss(
                q_outputs["camera_features"].mean(dim=1),
                q_outputs["lidar_features"]
            )
        else:
            align_loss = torch.tensor(0.0, device=self.device)
            align_metrics = {}
        
        # Total loss
        loss = temporal_loss + 0.5 * align_loss
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update key encoder
        self._update_key_encoder()
        
        # Update queue
        with torch.no_grad():
            self.queue.enqueue_dequeue(k_bev)
        
        # Metrics
        metrics = {
            "step": self.step,
            "epoch": self.epoch,
            **temporal_metrics,
            **align_metrics,
            "total_loss": loss.item()
        }
        
        self.step += 1
        
        return metrics
    
    def train(self):
        """Full training loop."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Starting BEV SSL training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Episodes: {len(self.dataset)}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            for batch in dataloader:
                metrics = self.train_step(batch)
                
                if self.step % 100 == 0:
                    print(f"Step {self.step}: loss={metrics['total_loss']:.4f}, "
                          f"pos_sim={metrics['pos_sim']:.4f}, align={metrics.get('alignment', 0):.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        # Save final
        self.save_checkpoint("final.pt")
        print(f"Training complete. Saved to {self.output_dir}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "config": self.config.__dict__,
            "query_encoder": self.query_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")


def bev_ssl_training_loop(
    episode_dir: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    output_dir: str = "out/bev_ssl",
    **kwargs
):
    """Main entry point for BEV SSL pretraining."""
    
    config = BEVSSLConfig(
        episode_dir=episode_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        output_dir=output_dir,
        **kwargs
    )
    
    trainer = BEVSSLTrainer(config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BEV SSL Pretraining")
    parser.add_argument("--episode-dir", type=str, default="data/waymo_episodes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="out/bev_ssl")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--queue-size", type=int, default=16384)
    parser.add_argument("--momentum", type=float, default=0.999)
    
    args = parser.parse_args()
    
    trainer = bev_ssl_training_loop(
        episode_dir=args.episode_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        temperature=args.temperature,
        queue_size=args.queue_size,
        momentum=args.momentum
    )

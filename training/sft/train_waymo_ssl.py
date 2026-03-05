"""
SSL Pretraining on Waymo Episodes

Self-supervised pretraining on Waymo motion data using:
- Masked trajectory prediction
- Map-agent contrastive learning
- Future prediction auxiliary task

This provides pretrained encoders for the downstream waypoint BC pipeline.

Usage:
    python -m training.sft.train_waymo_ssl \
        --tfrecords /path/to/waymo/*.tfrecord \
        --output_dir out/ssl_waymo \
        --epochs 50 \
        --batch_size 32 \
        --lr 1e-4 \
        --mask_ratio 0.15
"""

import os
import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from training.sft.dataloader_waymo import WaymoMotionDataset, WaymoToWaypointBCAdapter
from training.sft.scene_encoder import SceneEncoderConfig, SceneTransformerEncoder
from training.sft.proposal_waypoint_head import ProposalWaypointHead


@dataclass
class SSLConfig:
    """SSL pretraining configuration."""
    # Model
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # SSL tasks
    mask_ratio: float = 0.15
    future_steps: int = 30
    contrastive_temp: float = 0.1
    
    # Data
    historical_steps: int = 20
    num_agents: int = 64
    num_map_points: int = 10000
    
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "out/ssl_waymo"


class MaskedTrajectoryPredictor(nn.Module):
    """
    SSL head for masked trajectory reconstruction.
    
    Masks random agent timesteps and predicts the masked values.
    """
    
    def __init__(self, hidden_dim: int, num_features: int = 7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_features),
        )
    
    def forward(
        self, 
        encoded: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoded: (B, A, T, hidden_dim) - encoded trajectories
            mask: (B, A, T) - boolean mask (True = masked)
        
        Returns:
            predictions: (B, A, T, num_features) - predicted values
        """
        # Encode each timestep
        predictions = self.encoder(encoded)  # (B, A, T, num_features)
        return predictions


class ContrastiveMapAgentHead(nn.Module):
    """
    Contrastive learning head for map-agent alignment.
    
    Creates positive pairs between agents and their relevant map regions.
    """
    
    def __init__(self, hidden_dim: int, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
        self.agent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.map_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self, 
        agent_features: torch.Tensor, 
        map_features: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            agent_features: (B, A, hidden_dim) - agent embeddings
            map_features: (B, P, hidden_dim) - map embeddings
            positive_pairs: (B, K, 2) - indices of positive pairs
        
        Returns:
            contrastive_loss: scalar
        """
        # Project to common space
        agent_emb = F.normalize(self.agent_proj(agent_features), dim=-1)
        map_emb = F.normalize(self.map_proj(map_features), dim=-1)
        
        # Compute similarity matrix
        # (B, A, 1) @ (B, 1, P) -> (B, A, P)
        sim_matrix = torch.matmul(
            agent_emb.unsqueeze(2), 
            map_emb.unsqueeze(2).transpose(1, 2)
        ).squeeze(2) / self.temperature
        
        # If no positive pairs specified, use diagonal (agent i vs map i)
        if positive_pairs is None:
            batch_size = agent_emb.shape[0]
            num_agents = agent_emb.shape[1]
            positive_pairs = torch.arange(
                num_agents, device=agent_emb.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Gather positive similarities
        pos_sim = sim_matrix.gather(
            dim=2, 
            index=positive_pairs.unsqueeze(2).expand(-1, -1, 1)
        ).squeeze(2)
        
        # InfoNCE loss
        logits = sim_matrix
        labels = torch.arange(agent_emb.shape[1], device=agent_emb.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class WaymoSSLModel(nn.Module):
    """
    SSL-pretrained Scene Transformer model.
    
    Combines:
    - SceneEncoder for agent + map encoding
    - MaskedTrajectoryPredictor for masked reconstruction
    - ContrastiveMapAgentHead for map-agent alignment
    - FuturePredictionHead for future trajectory prediction
    """
    
    def __init__(self, config: SSLConfig):
        super().__init__()
        self.config = config
        
        # Scene encoder
        encoder_config = SceneEncoderConfig(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.scene_encoder = SceneTransformerEncoder(encoder_config)
        
        # SSL heads
        self.masked_predictor = MaskedTrajectoryPredictor(config.hidden_dim)
        self.contrastive_head = ContrastiveMapAgentHead(
            config.hidden_dim, 
            config.contrastive_temp
        )
        self.future_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.future_steps * 2),  # x, y
        )
        
        # Projection for temporal features
        self.temporal_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(
        self,
        agent_history: torch.Tensor,  # (B, A, T, 7)
        map_polylines: torch.Tensor,  # (B, P, M, 3)
        agent_valid: torch.Tensor = None,  # (B, A)
        polyline_valid: torch.Tensor = None,  # (B, P)
        mask_ratio: float = 0.15,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SSL objectives.
        
        Args:
            agent_history: Historical agent states
            map_polylines: Map polylines
            agent_valid: Valid agent mask
            polyline_valid: Valid polyline mask
            mask_ratio: Ratio of timesteps to mask
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary with losses and predictions
        """
        batch_size = agent_history.shape[0]
        num_agents = agent_history.shape[1]
        num_timesteps = agent_history.shape[2]
        
        # Create mask for SSL (randomly mask timesteps)
        if self.training and mask_ratio > 0:
            mask = torch.rand(
                batch_size, num_agents, num_timesteps, 
                device=agent_history.device
            ) < mask_ratio
        else:
            mask = torch.zeros(
                batch_size, num_agents, num_timesteps,
                dtype=torch.bool, device=agent_history.device
            )
        
        # Apply mask to input (replace with zeros or learned mask token)
        agent_history_masked = agent_history.clone()
        agent_history_masked[mask] = 0
        
        # Encode scene
        encoded = self.scene_encoder(
            agent_history=agent_history_masked,
            map_polylines=map_polylines,
            agent_valid=agent_valid,
            polyline_valid=polyline_valid,
        )  # (B, A, T, hidden_dim)
        
        # Temporal projection
        encoded_proj = self.temporal_proj(encoded)
        
        # Masked trajectory prediction
        if self.training:
            masked_pred = self.masked_predictor(encoded_proj, mask)
            # Loss only on masked positions
            masked_loss = F.mse_loss(
                masked_pred[mask], 
                agent_history[mask]
            )
        else:
            masked_loss = torch.tensor(0.0, device=agent_history.device)
        
        # Agent-level features (pool over time)
        if agent_valid is not None:
            mask_2d = ~agent_valid.unsqueeze(2)
        else:
            mask_2d = None
        
        agent_features = encoded.mean(dim=2)  # (B, A, hidden_dim)
        map_features = encoded.mean(dim=2)  # Pool over time for map
        
        # Contrastive loss
        if self.training:
            contrastive_loss = self.contrastive_head(
                agent_features, map_features
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=agent_history.device)
        
        # Future prediction (auxiliary task)
        future_pred = self.future_head(agent_features)  # (B, A, H*2)
        future_pred = future_pred.reshape(batch_size, num_agents, -1, 2)
        
        outputs = {
            "masked_loss": masked_loss,
            "contrastive_loss": contrastive_loss,
            "future_pred": future_pred,
            "encoded": encoded if return_features else None,
        }
        
        return outputs
    
    def get_encoder(self) -> SceneTransformerEncoder:
        """Return the encoder for downstream tasks."""
        return self.scene_encoder


def compute_total_loss(outputs: Dict[str, torch.Tensor], config: SSLConfig) -> torch.Tensor:
    """Compute weighted sum of SSL losses."""
    loss = 0.0
    
    # Masked reconstruction loss
    if "masked_loss" in outputs:
        loss += config.mask_ratio * outputs["masked_loss"]
    
    # Contrastive loss
    if "contrastive_loss" in outputs:
        loss += 0.1 * outputs["contrastive_loss"]
    
    # Future prediction (if targets available)
    # This would be added in training loop with actual future targets
    
    return loss


def train_epoch(
    model: WaymoSSLModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: SSLConfig,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_masked_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        agent_history = batch["agent_history"].to(device)
        map_polylines = batch["map_polylines"].to(device)
        agent_valid = batch.get("agent_valid", None)
        if agent_valid is not None:
            agent_valid = agent_valid.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            agent_history=agent_history,
            map_polylines=map_polylines,
            agent_valid=agent_valid,
            mask_ratio=config.mask_ratio,
        )
        
        # Compute loss
        loss = compute_total_loss(outputs, config)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        if "masked_loss" in outputs:
            total_masked_loss += outputs["masked_loss"].item()
        if "contrastive_loss" in outputs:
            total_contrastive_loss += outputs["contrastive_loss"].item()
        num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "masked_loss": total_masked_loss / num_batches,
        "contrastive_loss": total_contrastive_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: WaymoSSLModel,
    dataloader: DataLoader,
    config: SSLConfig,
    device: str,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        agent_history = batch["agent_history"].to(device)
        map_polylines = batch["map_polylines"].to(device)
        agent_valid = batch.get("agent_valid", None)
        if agent_valid is not None:
            agent_valid = agent_valid.to(device)
        
        outputs = model(
            agent_history=agent_history,
            map_polylines=map_polylines,
            agent_valid=agent_valid,
            mask_ratio=0.0,  # No masking during eval
        )
        
        loss = compute_total_loss(outputs, config)
        total_loss += loss.item()
        num_batches += 1
    
    return {"eval_loss": total_loss / num_batches}


def main():
    parser = argparse.ArgumentParser(description="SSL Pretraining on Waymo")
    parser.add_argument("--tfrecords", nargs="+", required=True,
                        help="Path to Waymo TFRecord files")
    parser.add_argument("--output_dir", type=str, default="out/ssl_waymo",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create config
    config = SSLConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Create dataset
    print(f"Loading Waymo dataset from {args.tfrecords}")
    dataset = WaymoMotionDataset(
        tfrecord_paths=args.tfrecords,
        historical_steps=config.historical_steps,
        future_steps=config.future_steps,
    )
    adapter = WaymoToWaypointBCAdapter(dataset)
    
    # For demo, create dummy data loader
    # In production, use real Waymo TFRecords
    from torch.utils.data import TensorDataset
    
    print("Creating dummy data for demo...")
    dummy_agent_history = torch.randn(32, 64, 20, 7)
    dummy_map_polylines = torch.randn(32, 100, 20, 3)
    dummy_agent_valid = torch.rand(32, 64) > 0.3
    
    dummy_dataset = TensorDataset(
        dummy_agent_history,
        dummy_map_polylines,
        dummy_agent_valid,
    )
    
    dataloader = DataLoader(
        dummy_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # Create model
    model = WaymoSSLModel(config).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    
    # Training loop
    print("Starting SSL pretraining...")
    best_loss = float("inf")
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        
        train_metrics = train_epoch(
            model, dataloader, optimizer, config, config.device
        )
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{config.epochs} ({epoch_time:.1f}s) - "
              f"loss: {train_metrics['loss']:.4f} - "
              f"masked: {train_metrics['masked_loss']:.4f} - "
              f"contrastive: {train_metrics['contrastive_loss']:.4f}")
        
        # Save checkpoint
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "metrics": train_metrics,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
    
    print("SSL pretraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Encoder saved to {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()

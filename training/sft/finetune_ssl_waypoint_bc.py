"""
SSL to Waypoint BC Fine-tuning

Loads SSL-pretrained encoder and fine-tunes on waypoint BC dataset.
This bridges the gap between SSL pretraining and downstream waypoint prediction.

Usage:
    # Fine-tune SSL encoder on waypoint BC
    python -m training.sft.finetune_ssl_waypoint_bc \
        --ssl_checkpoint out/ssl_waymo/best_model.pt \
        --output_dir out/finetune_ssl_bc \
        --epochs 30 \
        --batch_size 32 \
        --lr 1e-5 \
        --freeze_encoder

    # Continue training with unfrozen encoder
    python -m training.sft.finetune_ssl_waypoint_bc \
        --ssl_checkpoint out/finetune_ssl_bc/frozen_model.pt \
        --output_dir out/finetune_ssl_bc_unfrozen \
        --epochs 20 \
        --batch_size 32 \
        --lr 1e-4
"""

import os
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np

from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset
from training.sft.scene_encoder import SceneEncoderConfig, SceneTransformerEncoder
from training.sft.proposal_waypoint_head import ProposalWaypointHead
from training.sft.training_metrics import compute_batch_ade_fde


@dataclass
class FineTuneConfig:
    """Fine-tuning configuration."""
    # Model
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Waypoint prediction
    num_proposals: int = 5
    horizon_steps: int = 20
    
    # Training
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    
    # Freeze encoder (transfer learning)
    freeze_encoder: bool = True
    freeze_epochs: int = 10
    
    # Data
    data_dir: str = "data/waypoint_bc"
    num_workers: int = 4
    
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "out/finetune_ssl_bc"


class SSLToWaypointBCModel(nn.Module):
    """
    SSL-pretrained encoder + waypoint prediction head.
    
    Loads SSL-pretrained weights and adds waypoint prediction head.
    """
    
    def __init__(
        self,
        encoder_config: SceneEncoderConfig,
        num_proposals: int = 5,
        horizon_steps: int = 20,
    ):
        super().__init__()
        self.encoder = SceneTransformerEncoder(encoder_config)
        self.num_proposals = num_proposals
        self.horizon_steps = horizon_steps
        
        # Waypoint prediction head
        self.waypoint_head = ProposalWaypointHead(
            in_dim=encoder_config.hidden_dim,
            horizon_steps=horizon_steps,
            num_proposals=num_proposals,
            hidden_dim=encoder_config.hidden_dim,
        )
        
        # Store config for reference
        self.config = encoder_config
    
    def forward(
        self,
        agent_history: torch.Tensor,  # (B, A, T, 7)
        map_polylines: torch.Tensor,   # (B, P, M, 3)
        agent_mask: Optional[torch.Tensor] = None,  # (B, A) or (B, A, T)
        map_mask: Optional[torch.Tensor] = None,     # (B, P) or (B, P, M)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            agent_history: (B, A, T, 7) [x, y, z, vx, vy, yaw, type]
            map_polylines: (B, P, M, 3) [x, y, is_endpoint]
            agent_mask: (B, A) - valid agent mask (will be expanded to 3D)
            map_mask: (B, P) - valid polyline mask (will be expanded to 3D)
            
        Returns:
            proposals: (B, K, H, 2) - K trajectory proposals
            scores: (B, K) - proposal confidence scores
        """
        # Expand masks to 3D if needed
        B, A, T, _ = agent_history.shape
        _, P, M, _ = map_polylines.shape
        
        if agent_mask is not None:
            if agent_mask.dim() == 2:
                # (B, A) -> (B, A, T)
                agent_mask = agent_mask.unsqueeze(-1).expand(-1, -1, T)
        else:
            agent_mask = torch.ones(B, A, T, device=agent_history.device, dtype=torch.bool)
            
        if map_mask is not None:
            if map_mask.dim() == 2:
                # (B, P) -> (B, P, M)
                map_mask = map_mask.unsqueeze(-1).expand(-1, -1, M)
        else:
            map_mask = torch.ones(B, P, M, device=map_polylines.device, dtype=torch.bool)
        
        # Encode scene
        encoded_dict = self.encoder(
            agent_history=agent_history,
            map_polylines=map_polylines,
            agent_masks=agent_mask,
            polyline_masks=map_mask,
        )
        
        # Extract encoded features - use the agent embeddings from dict
        if isinstance(encoded_dict, dict):
            agent_embeddings = encoded_dict.get("agent_embeddings")  # (B, A, D)
        else:
            agent_embeddings = encoded_dict
        
        # Select first agent (ego vehicle) for waypoint prediction
        # TODO: Make agent selection configurable
        encoded = agent_embeddings[:, 0, :]  # (B, D)
        
        # Predict waypoints
        proposals, scores = self.waypoint_head(encoded)
        
        return proposals, scores
    
    def freeze_encoder(self):
        """Freeze the encoder weights for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze the encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True


def load_ssl_checkpoint(
    model: SSLToWaypointBCModel,
    checkpoint_path: str,
    device: str = "cpu",
) -> SSLToWaypointBCModel:
    """
    Load SSL pretrained weights into the model.
    
    Handles both full SSL model checkpoints and encoder-only checkpoints.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "encoder_state_dict" in checkpoint:
        # SSL checkpoint with separate encoder weights
        encoder_state = checkpoint["encoder_state_dict"]
        model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded SSL encoder from {checkpoint_path}")
    elif "model_state_dict" in checkpoint:
        # Full SSL model checkpoint
        state_dict = checkpoint["model_state_dict"]
        
        # Filter to only encoder weights
        encoder_state = {
            k.replace("encoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("encoder.")
        }
        
        if encoder_state:
            model.encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded encoder from full SSL checkpoint {checkpoint_path}")
        else:
            print(f"Warning: No encoder weights found in {checkpoint_path}")
    elif "state_dict" in checkpoint:
        # Generic checkpoint format
        state_dict = checkpoint["state_dict"]
        encoder_state = {
            k.replace("encoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("encoder.")
        }
        if encoder_state:
            model.encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded encoder from checkpoint {checkpoint_path}")
    else:
        # Assume it's a state dict directly
        model.encoder.load_state_dict(checkpoint, strict=False)
        print(f"Loaded encoder weights from {checkpoint_path}")
    
    return model


def compute_loss(
    proposals: torch.Tensor,      # (B, K, H, 2)
    scores: torch.Tensor,         # (B, K)
    targets: torch.Tensor,        # (B, H, 2)
    agent_mask: Optional[torch.Tensor] = None,  # (B,)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute waypoint prediction loss.
    
    Uses minimum-over-proposals loss: compute loss for each proposal,
    then select the best one for backprop.
    """
    batch_size = proposals.shape[0]
    num_proposals = proposals.shape[1]
    horizon_steps = proposals.shape[2]
    
    # Compute displacement for each proposal
    # proposals: (B, K, H, 2) -> (B, K, H-1, 2)
    displacements = proposals[:, :, 1:, :] - proposals[:, :, :-1, :]
    
    # Expand target for each proposal: (B, H, 2) -> (B, K, H, 2)
    targets_expanded = targets.unsqueeze(1).expand(-1, num_proposals, -1, -1)
    
    # Compute ADE for each proposal: (B, K)
    ade_per_proposal = torch.norm(proposals - targets_expanded, dim=-1).mean(dim=-1)
    
    # Find best proposal (minimum ADE)
    best_proposal_idx = ade_per_proposal.argmin(dim=1)  # (B,)
    
    # Select best proposal and target
    best_proposals = proposals[torch.arange(batch_size), best_proposal_idx]  # (B, H, 2)
    best_targets = targets  # (B, H, 2)
    
    # Waypoint L2 loss
    waypoint_loss = F.mse_loss(best_proposals, best_targets)
    
    # Scoring loss: encourage higher scores for better proposals
    # Use negative ADE as target score
    target_scores = F.one_hot(best_proposal_idx, num_proposals).float()  # (B, K)
    score_loss = F.binary_cross_entropy_with_logits(scores, target_scores)
    
    # Total loss
    loss = waypoint_loss + 0.1 * score_loss
    
    # Metrics
    with torch.no_grad():
        # ADE: average displacement error
        ade = torch.norm(best_proposals - best_targets, dim=-1).mean()
        # FDE: final displacement error
        fde = torch.norm(best_proposals[:, -1, :] - best_targets[:, -1, :], dim=-1).mean()
    
    metrics = {
        "loss": loss.item(),
        "waypoint_loss": waypoint_loss.item(),
        "score_loss": score_loss.item(),
        "ade": ade.item(),
        "fde": fde.item(),
    }
    
    return loss, metrics


def train_epoch(
    model: SSLToWaypointBCModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
    freeze_encoder: bool = True,
    epoch: int = 0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_metrics = {
        "loss": 0.0,
        "waypoint_loss": 0.0,
        "score_loss": 0.0,
        "ade": 0.0,
        "fde": 0.0,
    }
    
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        agent_history = batch["agent_history"].to(device)
        map_polylines = batch["map_polylines"].to(device)
        future_waypoints = batch["future_waypoints"].to(device)
        agent_mask = batch.get("agent_mask", None)
        map_mask = batch.get("map_mask", None)
        
        if agent_mask is not None:
            agent_mask = agent_mask.to(device)
        if map_mask is not None:
            map_mask = map_mask.to(device)
        
        # Forward pass
        proposals, scores = model(
            agent_history=agent_history,
            map_polylines=map_polylines,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )
        
        # Compute loss
        loss, metrics = compute_loss(proposals, scores, future_waypoints, agent_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        for k, v in metrics.items():
            total_metrics[k] += v
        
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"loss={metrics['loss']:.4f}, ade={metrics['ade']:.4f}")
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    return total_metrics


@torch.no_grad()
def evaluate(
    model: SSLToWaypointBCModel,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    
    total_metrics = {
        "loss": 0.0,
        "ade": 0.0,
        "fde": 0.0,
    }
    
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        agent_history = batch["agent_history"].to(device)
        map_polylines = batch["map_polylines"].to(device)
        future_waypoints = batch["future_waypoints"].to(device)
        agent_mask = batch.get("agent_mask", None)
        map_mask = batch.get("map_mask", None)
        
        if agent_mask is not None:
            agent_mask = agent_mask.to(device)
        if map_mask is not None:
            map_mask = map_mask.to(device)
        
        # Forward pass
        proposals, scores = model(
            agent_history=agent_history,
            map_polylines=map_polylines,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )
        
        # Compute loss
        _, metrics = compute_loss(proposals, scores, future_waypoints, agent_mask)
        
        # Accumulate metrics
        total_metrics["loss"] += metrics["loss"]
        total_metrics["ade"] += metrics["ade"]
        total_metrics["fde"] += metrics["fde"]
        
        num_batches += 1
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    return total_metrics


def save_checkpoint(
    model: SSLToWaypointBCModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: str,
    prefix: str = "checkpoint",
):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    
    # Save latest
    path = os.path.join(output_dir, f"{prefix}_latest.pt")
    torch.save(checkpoint, path)
    
    # Save best ADE
    best_ade_path = os.path.join(output_dir, f"{prefix}_best_ade.pt")
    if not os.path.exists(best_ade_path) or metrics.get("ade", float("inf")) < getattr(
        torch.load(best_ade_path, weights_only=False)["metrics"], "ade", float("inf")
    ):
        torch.save(checkpoint, best_ade_path)
    
    print(f"Saved checkpoint to {output_dir}")


def create_model(
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    num_proposals: int = 5,
    horizon_steps: int = 20,
) -> SSLToWaypointBCModel:
    """Create the model."""
    encoder_config = SceneEncoderConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    model = SSLToWaypointBCModel(
        encoder_config=encoder_config,
        num_proposals=num_proposals,
        horizon_steps=horizon_steps,
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="SSL to Waypoint BC Fine-tuning")
    parser.add_argument("--ssl_checkpoint", type=str, default=None,
                        help="Path to SSL pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="out/finetune_ssl_bc",
                        help="Output directory")
    parser.add_argument("--data_dir", type=str, default="data/waypoint_bc",
                        help="Waypoint BC data directory")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights initially")
    parser.add_argument("--freeze_epochs", type=int, default=10,
                        help="Number of epochs to keep encoder frozen")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of encoder layers")
    parser.add_argument("--num_proposals", type=int, default=5,
                        help="Number of trajectory proposals")
    parser.add_argument("--horizon_steps", type=int, default=20,
                        help="Planning horizon steps")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"=== SSL to Waypoint BC Fine-tuning ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_proposals=args.num_proposals,
        horizon_steps=args.horizon_steps,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load SSL checkpoint if provided
    if args.ssl_checkpoint is not None:
        print(f"\nLoading SSL checkpoint: {args.ssl_checkpoint}")
        model = load_ssl_checkpoint(model, args.ssl_checkpoint, device)
    
    # Freeze encoder if requested
    if args.freeze_encoder:
        print("Freezing encoder weights...")
        model.freeze_encoder()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters after freezing: {trainable_params:,}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    
    train_dataset = EpisodesWaypointBCDataset(
        data_dir=args.data_dir,
        split="train",
    )
    val_dataset = EpisodesWaypointBCDataset(
        data_dir=args.data_dir,
        split="val",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=EpisodesWaypointBCDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=EpisodesWaypointBCDataset.collate_fn,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.1,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )
    
    # Resume if requested
    start_epoch = 0
    best_ade = float("inf")
    
    if args.resume is not None:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_ade = checkpoint.get("metrics", {}).get("ade", float("inf"))
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n=== Starting Training ===")
    
    for epoch in range(start_epoch, args.epochs):
        # Unfreeze encoder after freeze_epochs
        if args.freeze_encoder and epoch == args.freeze_epochs:
            print(f"\nUnfreezing encoder at epoch {epoch}")
            model.unfreeze_encoder()
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters after unfreeze: {trainable_params:,}")
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            freeze_encoder=args.freeze_encoder and epoch < args.freeze_epochs,
            epoch=epoch,
        )
        
        print(f"Train - loss: {train_metrics['loss']:.4f}, "
              f"ade: {train_metrics['ade']:.4f}, fde: {train_metrics['fde']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Val   - loss: {val_metrics['loss']:.4f}, "
              f"ade: {val_metrics['ade']:.4f}, fde: {val_metrics['fde']:.4f}")
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_metrics, args.output_dir,
            prefix=f"epoch_{epoch + 1}",
        )
        
        # Track best
        if val_metrics["ade"] < best_ade:
            best_ade = val_metrics["ade"]
            print(f"New best ADE: {best_ade:.4f}")
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }) + "\n")
    
    print(f"\n=== Training Complete ===")
    print(f"Best ADE: {best_ade:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

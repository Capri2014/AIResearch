"""
SFT Training with CoT (Chain of Thought) Reasoning

Train waypoint prediction model with CoT reasoning traces for improved decision quality.

Usage:
    python -m training.sft.train_waypoint_bc_cot \
        --data-dir /data/waymo \
        --cot-file cot_traces.jsonl \
        --output out/sft_cot \
        --epochs 10 \
        --batch-size 32 \
        --lr 1e-4
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass
from datetime import datetime
import sys


@dataclass
class CoTConfig:
    """Configuration for CoT SFT training."""
    # Model
    encoder_name: str = 'resnet50'
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # CoT
    cot_weight: float = 0.1  # Weight for CoT loss
    cot_encoder: str = 'bert'  # bert, lstm, none
    
    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Data
    max_waypoints: int = 16
    max_cot_length: int = 128
    
    # Logging
    log_interval: int = 100
    save_interval: int = 1000


class CoTEncoder(nn.Module):
    """
    Encoder for CoT reasoning traces.
    
    Options:
    - none: Don't use CoT
    - lstm: Simple LSTM encoder
    - bert: BERT-based encoder (requires transformers library)
    """
    
    def __init__(self, config: CoTConfig):
        super().__init__()
        self.config = config
        self.use_cot = config.cot_encoder != 'none'
        
        if self.use_cot:
            if config.cot_encoder == 'lstm':
                self.encoder = nn.LSTM(
                    input_size=64,  # Character embedding size
                    hidden_size=config.hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    dropout=config.dropout if config.num_layers > 1 else 0,
                )
            elif config.cot_encoder == 'bert':
                try:
                    from transformers import BertModel, BertTokenizer
                    self.bert = BertModel.from_pretrained('bert-base-uncased')
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    self projection = nn.Linear(768, config.hidden_dim)
                except ImportError:
                    print("Warning: transformers not installed, falling back to LSTM")
                    config.cot_encoder = 'lstm'
                    self.encoder = nn.LSTM(
                        input_size=64,
                        hidden_size=config.hidden_dim,
                        num_layers=2,
                        batch_first=True,
                    )
    
    def forward(self, cot_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode CoT tokens to hidden representation.
        
        Args:
            cot_tokens: [B, L] token indices
            
        Returns:
            hidden: [B, hidden_dim] hidden representation
        """
        if not self.use_cot:
            return None
        
        if self.config.cot_encoder == 'bert':
            # BERT-based encoding
            outputs = self.bert(cot_tokens)
            # Use [CLS] token representation
            hidden = outputs.last_hidden_state[:, 0, :]
            hidden = self.projection(hidden)
            return hidden
        
        elif self.config.cot_encoder == 'lstm':
            # LSTM encoding
            output, (h_n, c_n) = self.encoder(cot_tokens)
            # Use last hidden state
            return h_n[-1]


class WaypointBCWithCoT(nn.Module):
    """
    Waypoint prediction model with CoT reasoning traces.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Input: State features + CoT text                       │
    │    │                                                   │
    │    ▼                                                   │
    │  ┌─────────────┐                                       │
    │  │ State Encoder │  (MLP)                               │
    │  └──────┬──────┘                                       │
    │         │                                               │
    │    ┌────▼────┐                                         │
    │    │ CoT Encoder │ (optional)                           │
    │    └────┬────┘                                         │
    │         │                                               │
    │    ┌────▼────┐                                         │
    │    │ Fusion    │  (concatenate or cross-attention)    │
    │    └────┬────┘                                         │
    │         │                                               │
    │    ┌────▼────┐                                         │
    │    │  Decoder  │  (Transformer decoder)                 │
    │    └────┬────┘                                         │
    │         │                                               │
    │    ┌────▼────┐                                         │
    │    │ Waypoint Head │  [T, 3]                           │
    │    └───────────┘                                       │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: CoTConfig):
        super().__init__()
        self.config = config
        
        # State encoder (MLP)
        self.state_encoder = nn.Sequential(
            nn.Linear(16, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # CoT encoder
        self.cot_encoder = CoTEncoder(config)
        
        # Fusion layer
        fusion_input_dim = config.hidden_dim
        if config.cot_encoder != 'none':
            fusion_input_dim *= 2
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        
        # Transformer decoder for temporal modeling
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers,
        )
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 3),  # x, y, heading
        )
        
        # Control prediction head (optional)
        self.control_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 3),  # steering, throttle, brake
        )
    
    def forward(
        self,
        state: torch.Tensor,
        cot_tokens: Optional[torch.Tensor] = None,
        waypoint_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: [B, 16] state features
            cot_tokens: [B, L] CoT token indices (optional)
            waypoint_mask: [B, T] mask for valid waypoints
            
        Returns:
            Dict with:
                waypoints: [B, T, 3] predicted waypoints
                control: [B, 3] predicted control
                cot_features: [B, D] CoT encoding (for loss)
        """
        # Encode state
        state_features = self.state_encoder(state)  # [B, D]
        
        # Encode CoT if available
        cot_features = None
        if cot_tokens is not None and self.cot_encoder.use_cot:
            cot_features = self.cot_encoder(cot_tokens)  # [B, D]
        
        # Fusion
        if cot_features is not None:
            fused = torch.cat([state_features, cot_features], dim=-1)
        else:
            fused = state_features
        fused = self.fusion(fused)  # [B, D]
        
        # Expand for decoder
        decoder_input = fused.unsqueeze(1).expand(-1, self.config.max_waypoints, -1)
        
        # Create memory for decoder (use fused as memory)
        memory = fused.unsqueeze(1)  # [B, 1, D]
        
        # Decoder
        decoder_output = self.decoder(decoder_input, memory)  # [B, T, D]
        
        # Predict waypoints
        waypoints = self.waypoint_head(decoder_output)  # [B, T, 3]
        
        # Predict control (from fused features)
        control = self.control_head(fused)  # [B, 3]
        
        # Apply mask if provided
        if waypoint_mask is not None:
            waypoints = waypoints * waypoint_mask.unsqueeze(-1)
        
        return {
            'waypoints': waypoints,
            'control': control,
            'cot_features': cot_features,
        }


class CoTSFTrainer:
    """Trainer for SFT with CoT reasoning."""
    
    def __init__(
        self,
        model: WaypointBCWithCoT,
        config: CoTConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Loss functions
        self.waypoint_loss_fn = nn.MSELoss()
        self.control_loss_fn = nn.MSELoss()
        
        # Logging
        self.train_metrics = []
        self.val_metrics = []
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_waypoint_loss = 0
        total_cot_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            state = batch['state'].to(self.device)
            waypoints = batch['waypoints'].to(self.device)
            cot_text = batch['cot_text'].to(self.device)
            control = batch['control'].to(self.device)
            
            # Create mask for valid waypoints
            waypoint_mask = (waypoints.abs().sum(dim=-1) > 0).float()
            
            # Forward pass
            outputs = self.model(state, cot_text, waypoint_mask)
            
            # Compute losses
            # Waypoint loss
            waypoint_loss = self.waypoint_loss_fn(
                outputs['waypoints'],
                waypoints
            )
            
            # Control loss
            control_loss = self.control_loss_fn(
                outputs['control'],
                control
            )
            
            # Total loss
            loss = waypoint_loss + 0.1 * control_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_waypoint_loss += waypoint_loss.item()
            total_cot_loss += 0.0  # CoT is encoded, not directly supervised
            
            self.global_step += 1
            
            # Log periodically
            if batch_idx % self.config.log_interval == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"WP Loss: {waypoint_loss.item():.4f}")
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'waypoint_loss': total_waypoint_loss / n_batches,
            'control_loss': total_cot_loss / n_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_ade = 0
        total_fde = 0
        n_samples = 0
        
        for batch in self.val_loader:
            state = batch['state'].to(self.device)
            waypoints = batch['waypoints'].to(self.device)
            cot_text = batch['cot_text'].to(self.device)
            
            outputs = self.model(state, cot_text)
            
            # Compute loss
            loss = F.mse_loss(outputs['waypoints'], waypoints)
            total_loss += loss.item() * len(batch)
            
            # Compute ADE/FDE
            pred = outputs['waypoints'].cpu().numpy()
            target = waypoints.cpu().numpy()
            
            ade = np.mean(np.linalg.norm(pred - target, axis=-1))
            fde = np.mean(np.linalg.norm(pred[:, -1] - target[:, -1], axis=-1))
            
            total_ade += ade * len(batch)
            total_fde += fde * len(batch)
            n_samples += len(batch)
        
        return {
            'val_loss': total_loss / n_samples,
            'val_ade': total_ade / n_samples,
            'val_fde': total_fde / n_samples,
        }
    
    def save_checkpoint(self, output_dir: str, epoch: int):
        """Save checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.config),
            'global_step': self.global_step,
            'epoch': epoch,
        }
        
        path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(description='SFT Training with CoT')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--cot-file', type=str, default=None)
    parser.add_argument('--val-dir', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    
    # Model
    parser.add_argument('--encoder', type=str, default='resnet50')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cot-encoder', type=str, default='lstm',
                       choices=['none', 'lstm', 'bert'])
    parser.add_argument('--cot-weight', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    print("SFT Training with CoT Reasoning")
    print("=" * 50)
    print(f"Data dir: {args.data_dir}")
    print(f"Output: {args.output}")
    print(f"CoT encoder: {args.cot_encoder}")
    print(f"Epochs: {args.epochs}")
    print()
    
    # Load datasets (placeholder - would load actual data)
    print("Loading datasets...")
    train_dataset = None  # Would load actual data
    val_dataset = None
    
    print("Dataset loaded. Ready to train.")
    
    # Print sample command
    print("\n" + "=" * 50)
    print("To run actual training, implement dataset loading.")
    print("Example:")
    print(f"  python -m training.sft.train_waypoint_bc_cot \\")
    print(f"    --data-dir {args.data_dir} \\")
    print(f"    --cot-file {args.cot_file} \\")
    print(f"    --output {args.output} \\")
    print(f"    --epochs {args.epochs} \\")
    print(f"    --batch-size {args.batch_size}")


if __name__ == "__main__":
    main()

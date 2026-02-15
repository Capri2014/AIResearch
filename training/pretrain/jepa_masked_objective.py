"""JEPA-style masked latent prediction objective for encoder pretraining.

This module adds an alternative/auxiliary pretraining objective beyond contrastive learning:
- Masked latent prediction: predict masked encoder embeddings from visible ones
- Can be combined with contrastive loss for multi-task training

Usage
-----
python -m training.pretrain.train_jepa_masked \
  --episodes-glob "out/episodes/**/*.json" \
  --mask-ratio 0.3 \
  --pred-depth 4

Combines with contrastive:
python -m training.pretrain.train_ssl_temporal_contrastive_v0 \
  --aux-jepa-loss 0.5 \
  --jepa-mask-ratio 0.3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JEPAConfig:
    """Configuration for masked latent prediction objective."""
    # Masking
    mask_ratio: float = 0.3  # Fraction of timesteps to mask
    mask_type: str = "random"  # "random", "temporal", "spatial"
    
    # Prediction head
    pred_depth: int = 4  # Number of transformer layers
    pred_dim: int = 128  # Hidden dimension
    num_heads: int = 4  # Attention heads
    
    # Loss weighting
    jepa_weight: float = 1.0  # Weight relative to contrastive loss
    
    # Output
    out_dir: Path = Path("out/pretrain_jepa_v0")


class MaskedPredictor(nn.Module):
    """Predicts masked latent representations from visible ones.
    
    Architecture:
    - Visible tokens attend to each other via self-attention
    - Masked tokens are predicted from visible context
    - Lightweight transformer decoder (no cross-attention to encoder)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, depth: int = 4, num_heads: int = 4):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Project input to hidden dimension
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        
        # Positional embedding (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_dim))
        
        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(depth)
        ])
        
        # Project back to latent space
        self.proj_out = nn.Linear(hidden_dim, in_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(
        self,
        visible_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        visible_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masked embeddings.
        
        Args:
            visible_embeds: (B, N_vis, D) - visible (unmasked) embeddings
            target_embeds: (B, N_total, D) - all embeddings (for computing loss)
            visible_mask: (B, N_total) - bool mask indicating visible tokens
        
        Returns:
            pred_embeds: (B, N_mask, D) - predicted embeddings for masked positions
            loss_embeds: (B, N_mask, D) - target embeddings for masked positions
        """
        B, N_total, D = target_embeds.shape
        
        # Determine visible vs masked positions
        if visible_mask is None:
            # All visible
            visible_mask = torch.ones(B, N_total, dtype=torch.bool, device=target_embeds.device)
        
        N_vis = visible_embeds.shape[1]
        
        # Create full sequence with masked placeholders
        full_seq = torch.zeros(B, N_total, D, device=target_embeds.device, dtype=target_embeds.dtype)
        full_seq[visible_mask] = visible_embeds.view(-1, D)
        
        # Add positional embeddings
        seq_len = full_seq.shape[1]
        pos = self.pos_embed[:, :seq_len, :]
        full_seq = full_seq + pos
        
        # Transformer forward
        for block in self.blocks:
            full_seq = block(full_seq)
        
        # Get predictions for masked positions
        pred_embeds = full_seq[~visible_mask]  # (B * N_mask, D)
        
        # Get target embeddings for masked positions
        loss_embeds = target_embeds[~visible_mask]  # (B * N_mask, D)
        
        return pred_embeds, loss_embeds


def compute_jepa_loss(
    pred_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute MSE loss between predicted and target embeddings."""
    loss = F.mse_loss(pred_embeds, target_embeds, reduction=reduction)
    return loss


@dataclass
class JEPAObjectiveConfig:
    """Configuration for JEPA objective when used as auxiliary loss."""
    enabled: bool = False
    mask_ratio: float = 0.3
    mask_type: str = "random"
    weight: float = 1.0
    pred_depth: int = 4
    pred_dim: int = 128


class JEPALoss:
    """JEPA loss module for integration with existing training loops."""
    
    def __init__(self, encoder_out_dim: int, config: JEPAObjectiveConfig | None = None):
        config = config or JEPAObjectiveConfig()
        self.config = config
        self.encoder_out_dim = encoder_out_dim
        
        if config.enabled:
            self.predictor = MaskedPredictor(
                in_dim=encoder_out_dim,
                hidden_dim=config.pred_dim,
                depth=config.pred_depth,
            )
        else:
            self.predictor = None
    
    def compute_loss(
        self,
        encoder_embeds: torch.Tensor,
        temporal_indices: Optional[torch.Tensor] = None,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute JEPA loss for a batch of embeddings.
        
        Args:
            encoder_embeds: (B, T, D) encoder output embeddings
            temporal_indices: Optional indices for temporal masking
            batch_indices: Optional batch indices for masking
        
        Returns:
            loss: JEPA loss tensor
            info: Dict with loss components for logging
        """
        if self.predictor is None or not self.config.enabled:
            return torch.tensor(0.0), {"jepa_loss": 0.0}
        
        B, T, D = encoder_embeds.shape
        
        # Generate mask
        mask = self._generate_mask(B, T)
        
        # Separate visible and masked
        visible_mask = ~mask
        visible_embeds = encoder_embeds[visible_mask].view(-1, D)
        
        # Predict
        pred_embeds, target_embeds = self.predictor(
            visible_embeds=visible_embeds,
            target_embeds=encoder_embeds,
            visible_mask=visible_mask,
        )
        
        # Compute loss
        loss = compute_jepa_loss(pred_embeds, target_embeds)
        loss = loss * self.config.weight
        
        info = {
            "jepa_loss": loss.item(),
            "jepa_mask_ratio": float(mask.sum() / mask.numel()),
        }
        
        return loss, info
    
    def _generate_mask(self, B: int, T: int) -> torch.Tensor:
        """Generate mask for JEPA prediction."""
        if self.config.mask_type == "random":
            # Random masking
            mask = torch.rand(B, T) < self.config.mask_ratio
        elif self.config.mask_type == "temporal":
            # Mask contiguous blocks (temporal structure)
            mask = torch.zeros(B, T, dtype=torch.bool)
            block_size = max(1, int(T * self.config.mask_ratio))
            for b in range(B):
                start = torch.randint(0, T - block_size + 1, (1,)).item()
                mask[b, start:start + block_size] = True
        else:
            # Default to random
            mask = torch.rand(B, T) < self.config.mask_ratio
        
        return mask


def main():
    """Quick test of JEPA objective."""
    import argparse
    
    parser = argparse.ArgumentParser(description="JEPA masked prediction test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    args = parser.parse_args()
    
    # Create dummy embeddings
    embeds = torch.randn(args.batch_size, args.timesteps, args.dim)
    
    # Compute JEPA loss
    config = JEPAObjectiveConfig(
        enabled=True,
        mask_ratio=args.mask_ratio,
        weight=1.0,
    )
    jepa = JEPALoss(encoder_out_dim=args.dim, config=config)
    loss, info = jepa.compute_loss(embeds)
    
    print(f"JEPA loss: {loss.item():.4f}")
    print(f"Info: {info}")


if __name__ == "__main__":
    main()

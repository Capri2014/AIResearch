"""Temporal Masked Prediction (TMP) objective for SSL pretraining.

This module adds a temporal prediction objective that complements
contrastive learning:
- Predicts future encoder embeddings from past context
- Uses temporal masking (forward prediction)
- Composes with contrastive loss for multi-task training

Pipeline stage 2: Waymo → SSL pretrain → waypoint BC

Usage:
    python3 -m training.pretrain.temporal_masked_objective --smoke
    python3 -m training.pretrain.run_ssl_pretrain --epochs 10 --enable-tmp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TMPConfig:
    """Configuration for Temporal Masked Prediction objective."""
    # Masking
    predict_horizon: int = 5  # Number of future timesteps to predict
    context_length: int = 10  # Number of past timesteps for context
    mask_ratio: float = 0.3  # Fraction of future timesteps to mask
    
    # Architecture
    pred_hidden_dim: int = 256
    num_heads: int = 4
    pred_depth: int = 3
    
    # Loss weighting
    weight: float = 0.5  # Weight relative to other SSL losses
    
    # Output
    out_dir: str = "out/pretrain_tmp"


# ============================================================================
# Temporal Prediction Model
# ============================================================================

class TemporalPredictor(nn.Module):
    """Predicts future encoder embeddings from past context.
    
    Architecture:
    - LSTM-based predictor for temporal sequences
    - Takes context embeddings and predicts future embeddings
    - Lightweight enough to run alongside main encoder
    """
    
    def __init__(
        self,
        encoder_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        predict_horizon: int = 5,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_horizon = predict_horizon
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        
        # Project to output dimension
        self.pred_head = nn.Linear(hidden_dim, encoder_dim)
        
        # Initialize LSTM hidden state
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.fill_(0.0)
    
    def forward(
        self,
        context_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict future embeddings from context.
        
        Args:
            context_embeds: (B, T_context, D) - past context embeddings
        
        Returns:
            pred_embeds: (B, predict_horizon, D) - predicted future embeddings
            hidden: Optional hidden state for debugging
        """
        B, T_ctx, D = context_embeds.shape
        
        # Encode context
        lstm_out, hidden = self.lstm(context_embeds)
        
        # Use last hidden state for prediction
        h_n = hidden[0][-1]  # (B, hidden_dim)
        
        # Decode future predictions autoregressively
        pred_embeds = []
        current_h = h_n
        
        for t in range(self.predict_horizon):
            # Project hidden state to prediction
            pred = self.pred_head(current_h)  # (B, D)
            pred_embeds.append(pred)
            
            # Update hidden state (simplified: reuse same hidden)
            # In full version: would use another LSTM cell
            current_h = F.gelu(self.pred_head(current_h))
        
        pred_embeds = torch.stack(pred_embeds, dim=1)  # (B, horizon, D)
        
        return pred_embeds, hidden[0]  # Return hidden for logging


class TemporalMaskedPredictor(nn.Module):
    """Transformer-based predictor for temporal masked prediction.
    
    More expressive than LSTM-based predictor:
    - Uses self-attention over context
    - Predicts masked timesteps directly
    """
    
    def __init__(
        self,
        encoder_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        predict_horizon: int = 5,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.predict_horizon = predict_horizon
        
        # Project input
        self.proj_in = nn.Linear(encoder_dim, hidden_dim)
        
        # Learnable sequence position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head (predicts multiple future timesteps)
        self.pred_head = nn.Linear(hidden_dim, encoder_dim * predict_horizon)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(
        self,
        context_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict future embeddings from context.
        
        Args:
            context_embeds: (B, T_context, D) - past context embeddings
        
        Returns:
            pred_embeds: (B, predict_horizon, D) - predicted future embeddings
        """
        B, T_ctx, D = context_embeds.shape
        
        # Project to hidden dimension
        h = self.proj_in(context_embeds)
        
        # Add positional embeddings
        h = h + self.pos_embed[:, :T_ctx, :]
        
        # Transformer processing
        h = self.transformer(h)
        
        # Use final timestep features
        final_h = h[:, -1, :]  # (B, hidden_dim)
        
        # Predict future
        pred = self.pred_head(final_h)  # (B, encoder_dim * predict_horizon)
        pred = pred.view(B, self.predict_horizon, self.encoder_dim)
        
        return pred


# ============================================================================
# Loss Computation
# ============================================================================

def compute_tmp_loss(
    pred_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute TMP loss between predicted and target future embeddings.
    
    Args:
        pred_embeds: (B, horizon, D) predicted future embeddings
        target_embeds: (B, horizon, D) target future embeddings
        reduction: loss reduction mode
    
    Returns:
        loss: scalar loss tensor
        info: dict with loss components
    """
    # MSE loss
    mse_loss = F.mse_loss(pred_embeds, target_embeds, reduction=reduction)
    
    # Alignment loss (encourages predictions to be close to targets in cosine space)
    pred_norm = F.normalize(pred_embeds, dim=-1)
    target_norm = F.normalize(target_embeds, dim=-1)
    cos_sim = (pred_norm * target_norm).sum(dim=-1).mean()
    align_loss = 1.0 - cos_sim
    
    # Combined loss
    loss = mse_loss + 0.1 * align_loss
    
    info = {
        "tmp_mse": mse_loss.item(),
        "tmp_align": align_loss.item(),
        "tmp_cos_sim": cos_sim.item(),
    }
    
    return loss, info


# ============================================================================
# TMP Objective Module
# ============================================================================

class TMPObjective(nn.Module):
    """Temporal Masked Prediction objective for integration with SSL training.
    
    This module:
    - Takes encoder embeddings over time
    - Masks future timesteps
    - Predicts them from context
    - Computes TMP loss
    """
    
    def __init__(self, encoder_dim: int, config: TMPConfig | None = None):
        super().__init__()
        config = config or TMPConfig()
        self.config = config
        self.encoder_dim = encoder_dim
        
        # Choose predictor architecture
        if config.pred_depth > 2:
            self.predictor = TemporalMaskedPredictor(
                encoder_dim=encoder_dim,
                hidden_dim=config.pred_hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.pred_depth,
                predict_horizon=config.predict_horizon,
            )
        else:
            self.predictor = TemporalPredictor(
                encoder_dim=encoder_dim,
                hidden_dim=config.pred_hidden_dim,
                num_layers=config.pred_depth + 1,
                predict_horizon=config.predict_horizon,
            )
    
    def forward(
        self,
        encoder_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute TMP loss for a sequence.
        
        Args:
            encoder_embeds: (B, T, D) encoder embeddings over time
        
        Returns:
            loss: TMP loss tensor
            info: dict with loss components
        """
        B, T, D = encoder_embeds.shape
        horizon = self.config.predict_horizon
        
        if T < self.config.context_length + horizon:
            # Not enough timesteps
            return torch.tensor(0.0, device=encoder_embeds.device), {"tmp_loss": 0.0}
        
        # Extract context (past) and targets (future)
        context_start = max(0, T - self.config.context_length - horizon)
        context_end = T - horizon
        target_start = context_end
        
        context = encoder_embeds[:, context_start:context_end, :]  # (B, context_len, D)
        targets = encoder_embeds[:, target_start:, :]  # (B, horizon, D)
        
        # Predict
        pred = self.predictor(context)
        
        # Compute loss
        loss, info = compute_tmp_loss(pred, targets)
        loss = loss * self.config.weight
        
        info["tmp_loss"] = loss.item()
        
        return loss, info


# ============================================================================
# Combined SSL Loss
# ============================================================================

class CombinedSSLLoss(nn.Module):
    """Combined SSL loss: contrastive + waypoint regression + TMP."""
    
    def __init__(
        self,
        encoder_dim: int,
        lambda_contrastive: float = 1.0,
        lambda_waypoint: float = 1.0,
        lambda_tmp: float = 0.5,
        tmp_config: TMPConfig | None = None,
    ):
        super().__init__()
        
        self.lambda_contrastive = lambda_contrastive
        self.lambda_waypoint = lambda_waypoint
        self.lambda_tmp = lambda_tmp
        
        # Existing losses
        self.contrastive_fn = ContrastiveLossFn()
        self.waypoint_fn = nn.MSELoss()
        
        # TMP objective
        self.tmp = TMPObjective(encoder_dim, tmp_config)
    
    def forward(
        self,
        encoder_embeds: torch.Tensor,
        view_i: torch.Tensor,
        view_j: torch.Tensor,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined SSL loss.
        
        Args:
            encoder_embeds: (B, T, D) sequence embeddings for TMP
            view_i: (B, D) view i embeddings
            view_j: (B, D) view j embeddings  
            pred_waypoints: (B, num_waypoints * 2)
            target_waypoints: (B, num_waypoints * 2)
        
        Returns:
            loss: combined loss tensor
            info: dict with loss components
        """
        loss = torch.tensor(0.0, device=encoder_embeds.device)
        info = {}
        
        # Contrastive loss
        if self.lambda_contrastive > 0:
            c_loss = self.contrastive_fn(view_i, view_j)
            loss = loss + self.lambda_contrastive * c_loss
            info["contrastive_loss"] = c_loss.item()
        
        # Waypoint regression loss
        if self.lambda_waypoint > 0:
            w_loss = self.waypoint_fn(pred_waypoints, target_waypoints)
            loss = loss + self.lambda_waypoint * w_loss
            info["waypoint_loss"] = w_loss.item()
        
        # TMP loss
        if self.lambda_tmp > 0:
            t_loss, t_info = self.tmp(encoder_embeds)
            loss = loss + t_loss
            info.update(t_info)
        
        info["total_loss"] = loss.item()
        
        return loss, info


class ContrastiveLossFn(nn.Module):
    """Simple contrastive loss for multi-view SSL."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute symmetric InfoNCE loss."""
        B = z_i.size(0)
        
        # Normalize
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Similarity matrix
        sim = torch.matmul(z_i, z_j.T) / self.temperature
        
        # Labels (diagonal = positive pairs)
        labels = torch.arange(B, device=z_i.device)
        
        # Loss
        loss_i = F.cross_entropy(sim, labels)
        loss_j = F.cross_entropy(sim.T, labels)
        
        return 0.5 * (loss_i + loss_j)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Smoke test for TMP objective."""
    parser = argparse.ArgumentParser(description="TMP smoke test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--predict-horizon", type=int, default=5)
    parser.add_argument("--context-length", type=int, default=10)
    args = parser.parse_args()
    
    # Create config
    config = TMPConfig(
        predict_horizon=args.predict_horizon,
        context_length=args.context_length,
    )
    
    # Create model
    model = TMPObjective(encoder_dim=args.encoder_dim, config=config)
    
    # Dummy embeddings
    embeds = torch.randn(args.batch_size, args.timesteps, args.encoder_dim)
    
    # Forward
    loss, info = model(embeds)
    
    print(f"TMP Loss: {loss.item():.4f}")
    print(f"Info: {info}")
    print("✓ TMP objective working")


if __name__ == "__main__":
    main()
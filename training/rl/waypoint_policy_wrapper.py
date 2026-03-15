"""
Waypoint Policy Wrapper for RL Refinement.

Wraps BC waypoint model for RL training (PPO/GRPO):
- Loads BC checkpoint as frozen base policy
- Adds residual delta head for RL fine-tuning
- Provides clean interface for RL training scripts

Design Pattern:
    final_waypoints = bc_base_waypoints + delta_head(state)

This enables:
1. Keep BC model frozen (behavioral cloning knowledge)
2. Train delta head via PPO/GRPO (learns corrections)
3. Combine at inference time

Usage:
    from training.rl.waypoint_policy_wrapper import (
        WaypointPolicyWithDelta,
        create_rl_refinement_model,
    )
    
    # Create model for RL training
    model = create_rl_refinement_model(
        bc_checkpoint="out/bc/model.pt",
        ssl_encoder=encoder,
        delta_hidden_dims=[128, 64],
        device="cuda",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLRefinementConfig:
    """Configuration for RL refinement model."""
    # BC base model
    bc_checkpoint: Optional[Path] = None
    freeze_bc: bool = True
    
    # SSL encoder (optional)
    ssl_encoder_checkpoint: Optional[Path] = None
    
    # Delta head architecture
    delta_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    delta_activation: str = "relu"
    
    # Output
    num_waypoints: int = 8
    waypoint_dim: int = 2
    predict_speed: bool = True
    
    # State encoding
    state_dim: int = 4  # [x, y, theta, speed]
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Delta Head
# ============================================================================

class DeltaWaypointHead(nn.Module):
    """
    Residual delta waypoint head.
    
    Learns to predict corrections to BC waypoints based on:
    - Current state (position, heading, speed)
    - BC waypoints (frozen base prediction)
    """
    
    def __init__(
        self,
        state_dim: int,
        num_waypoints: int,
        waypoint_dim: int,
        bc_waypoint_dim: int,  # BC output dim (with or without speed)
        hidden_dims: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Input: state + BC waypoints
        input_dim = state_dim + bc_waypoint_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # Output: delta waypoints + optional speed delta
        output_dim = num_waypoints * waypoint_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "gelu":
            return nn.GELU()
        elif name == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU(inplace=True)
    
    def forward(
        self,
        state: torch.Tensor,      # [B, state_dim]
        bc_waypoints: torch.Tensor,  # [B, num_waypoints * waypoint_dim] or [B, ...]
    ) -> torch.Tensor:
        """
        Predict delta waypoints.
        
        Args:
            state: Current state [B, state_dim]
            bc_waypoints: BC base waypoints [B, num_waypoints * waypoint_dim]
            
        Returns:
            Delta waypoints [B, num_waypoints * waypoint_dim]
        """
        # Flatten BC waypoints if needed
        if bc_waypoints.dim() == 3:
            # [B, num_waypoints, waypoint_dim] -> [B, num_waypoints * waypoint_dim]
            bc_waypoints = bc_waypoints.flatten(1)
        
        # Concatenate state and BC waypoints
        x = torch.cat([state, bc_waypoints], dim=-1)
        
        # Predict delta
        delta = self.mlp(x)
        
        # Reshape to waypoints
        delta = delta.view(-1, self.num_waypoints, self.waypoint_dim)
        
        return delta


# ============================================================================
# Combined Policy
# ============================================================================

class WaypointPolicyWithDelta(nn.Module):
    """
    Waypoint policy with residual delta learning.
    
    Combines:
    1. Frozen BC model (base waypoint predictions)
    2. Trainable delta head (learns corrections)
    
    Forward pass:
        final_waypoints = bc_waypoints + delta_head(state, bc_waypoints)
    """
    
    def __init__(
        self,
        bc_model: nn.Module,
        delta_head: nn.Module,
        freeze_bc: bool = True,
    ):
        super().__init__()
        
        self.bc_model = bc_model
        self.delta_head = delta_head
        self.freeze_bc = freeze_bc
        
        if freeze_bc:
            for param in bc_model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        bev_features: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through combined policy.
        
        Args:
            bev_features: BEV features from encoder [B, C, H, W]
            state: Current state [B, state_dim]
            
        Returns:
            Tuple of:
                - final_waypoints: [B, num_waypoints, waypoint_dim]
                - delta_waypoints: [B, num_waypoints, waypoint_dim]
        """
        # Get BC predictions (frozen)
        with torch.no_grad() if self.freeze_bc else torch.enable_grad():
            bc_output = self.bc_model(bev_features)
        
        if isinstance(bc_output, tuple):
            bc_waypoints = bc_output[0]  # [B, num_waypoints, waypoint_dim]
        else:
            bc_waypoints = bc_output  # [B, num_waypoints, waypoint_dim]
        
        # Get delta predictions
        delta = self.delta_head(state, bc_waypoints)
        
        # Combine
        final_waypoints = bc_waypoints + delta
        
        return final_waypoints, delta
    
    def get_bc_waypoints(self, bev_features: torch.Tensor) -> torch.Tensor:
        """Get only BC waypoints (no delta)."""
        with torch.no_grad():
            bc_output = self.bc_model(bev_features)
        if isinstance(bc_output, tuple):
            return bc_output[0]
        return bc_output
    
    def get_delta(self, state: torch.Tensor, bc_waypoints: torch.Tensor) -> torch.Tensor:
        """Get only delta waypoints."""
        return self.delta_head(state, bc_waypoints)


# ============================================================================
# Factory Functions
# ============================================================================

def create_rl_refinement_model(
    bc_checkpoint: Optional[Path | str] = None,
    ssl_encoder: Optional[nn.Module] = None,
    delta_hidden_dims: List[int] = [256, 128, 64],
    num_waypoints: int = 8,
    waypoint_dim: int = 2,
    predict_speed: bool = True,
    freeze_bc: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[WaypointPolicyWithDelta, Dict]:
    """
    Create a RL refinement model from BC checkpoint.
    
    Args:
        bc_checkpoint: Path to BC checkpoint
        ssl_encoder: Optional SSL encoder (if BC was trained with SSL)
        delta_hidden_dims: Hidden dimensions for delta head MLP
        num_waypoints: Number of waypoints to predict
        waypoint_dim: Dimension per waypoint (2 for x,y)
        predict_speed: Whether BC predicts speed
        freeze_bc: Whether to freeze BC parameters
        device: Device to place model on
        
    Returns:
        Tuple of (model, info_dict)
    """
    from training.rl.bc_checkpoint_loader import load_bc_waypoint_model
    
    # Load BC model
    if bc_checkpoint is not None:
        bc_model, bc_config = load_bc_waypoint_model(
            checkpoint=bc_checkpoint,
            ssl_encoder=ssl_encoder,
            device=device,
            eval_mode=True,
            strict=True,
        )
    else:
        # Create stub BC model for testing
        from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
        
        bc_config = WaypointBCConfig(
            num_waypoints=num_waypoints,
            waypoint_dim=waypoint_dim,
            predict_speed=predict_speed,
        )
        bc_model = WaypointBCModel(bc_config)
        bc_config_dict = bc_config.__dict__ if hasattr(bc_config, '__dict__') else {}
    
    # Determine BC output dimension
    bc_waypoint_dim = num_waypoints * waypoint_dim
    if predict_speed:
        bc_waypoint_dim += 1  # speed prediction
    
    # Create delta head
    delta_head = DeltaWaypointHead(
        state_dim=4,  # x, y, theta, speed
        num_waypoints=num_waypoints,
        waypoint_dim=waypoint_dim,
        bc_waypoint_dim=bc_waypoint_dim,
        hidden_dims=delta_hidden_dims,
    )
    
    # Combine
    model = WaypointPolicyWithDelta(
        bc_model=bc_model,
        delta_head=delta_head,
        freeze_bc=freeze_bc,
    )
    
    model = model.to(device)
    
    # Info dict
    info = {
        "bc_checkpoint": str(bc_checkpoint) if bc_checkpoint else None,
        "freeze_bc": freeze_bc,
        "num_waypoints": num_waypoints,
        "waypoint_dim": waypoint_dim,
        "predict_speed": predict_speed,
        "delta_params": sum(p.numel() for p in delta_head.parameters()),
        "bc_params": sum(p.numel() for p in bc_model.parameters()),
        "trainable_params": sum(p.numel() for p in delta_head.parameters()),
    }
    
    return model, info


def save_rl_model(
    model: WaypointPolicyWithDelta,
    path: Path | str,
    info: Optional[Dict] = None,
):
    """Save RL refinement model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "delta_head_state_dict": model.delta_head.state_dict(),
    }
    
    if info is not None:
        checkpoint["info"] = info
    
    torch.save(checkpoint, path)
    print(f"Saved RL model to {path}")


def load_rl_model(
    path: Path | str,
    bc_model: nn.Module,
    device: str = "cuda",
) -> WaypointPolicyWithDelta:
    """Load RL refinement model checkpoint."""
    path = Path(path)
    
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    # Build delta head (architecture must match)
    # This is a simplified loader - in practice you'd want to save config
    delta_head = DeltaWaypointHead(
        state_dim=4,
        num_waypoints=8,
        waypoint_dim=2,
        bc_waypoint_dim=17,  # 8*2 + 1
        hidden_dims=[256, 128, 64],
    )
    
    model = WaypointPolicyWithDelta(
        bc_model=bc_model,
        delta_head=delta_head,
        freeze_bc=True,
    )
    
    if "delta_head_state_dict" in ckpt:
        model.delta_head.load_state_dict(ckpt["delta_head_state_dict"])
    
    return model


# ============================================================================
# Inference Interface
# ============================================================================

class WaypointPolicyInference:
    """
    Simple inference interface for waypoint policy.
    
    Usage:
        policy = WaypointPolicyInference(checkpoint="out/rl/model.pt")
        waypoints = policy.predict(bev_features, state)
    """
    
    def __init__(
        self,
        bc_checkpoint: Optional[Path] = None,
        rl_checkpoint: Optional[Path] = None,
        ssl_encoder: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        self.device = device
        
        # Load BC model
        from training.rl.bc_checkpoint_loader import load_bc_waypoint_model
        
        if bc_checkpoint is not None:
            self.bc_model, _ = load_bc_waypoint_model(
                checkpoint=bc_checkpoint,
                ssl_encoder=ssl_encoder,
                device=device,
            )
        else:
            raise ValueError("bc_checkpoint is required")
        
        # Load RL delta if provided
        self.rl_model = None
        if rl_checkpoint is not None:
            self.rl_model = load_rl_model(rl_checkpoint, self.bc_model, device)
            self.use_rl = True
        else:
            self.use_rl = False
    
    @torch.no_grad()
    def predict(
        self,
        bev_features: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict waypoints.
        
        Args:
            bev_features: [B, C, H, W] BEV features
            state: [B, 4] state [x, y, theta, speed]
            
        Returns:
            waypoints: [B, num_waypoints, 2] predicted waypoints
        """
        bev_features = bev_features.to(self.device)
        state = state.to(self.device)
        
        if self.rl_model is not None:
            waypoints, _ = self.rl_model(bev_features, state)
        else:
            waypoints = self.bc_model(bev_features)
            if isinstance(waypoints, tuple):
                waypoints = waypoints[0]
        
        return waypoints


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create RL refinement model from BC checkpoint")
    parser.add_argument("--bc-checkpoint", type=Path, required=True, help="Path to BC checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output path for RL model")
    parser.add_argument("--delta-hidden", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    model, info = create_rl_refinement_model(
        bc_checkpoint=args.bc_checkpoint,
        delta_hidden_dims=args.delta_hidden,
        device=args.device,
    )
    
    print(f"Created RL refinement model:")
    print(f"  - BC params (frozen): {info['bc_params']:,}")
    print(f"  - Delta params (trainable): {info['delta_params']:,}")
    
    save_rl_model(model, args.output, info)
    

if __name__ == "__main__":
    main()

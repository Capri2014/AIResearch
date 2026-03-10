"""
SFT Checkpoint Loader for BC Model Integration.

Loads trained WaypointBCModel checkpoints and integrates with RL residual training.
"""

from __future__ import annotations

import json
import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.bc.waypoint_bc import WaypointBCModel, BCConfig, SSLEncoder, WaypointHead


class SFTWaypointLoader:
    """
    Loads and wraps trained BC checkpoint for use in RL residual training.
    
    Provides:
    - Frozen SFT base model (encoder + waypoint head)
    - Interface for getting waypoint predictions from perception input
    - Delta head integration for residual learning
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        freeze_encoder: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.freeze_encoder = freeze_encoder
        
        # Load config and model
        self.config = self._load_config(checkpoint_path)
        self.model = self._load_model(checkpoint_path)
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.waypoint_head.parameters():
                param.requires_grad = False
        
        self.model.to(device)
        self.model.eval()
        
    def _load_config(self, checkpoint_path: str) -> BCConfig:
        """Load config from checkpoint or use defaults."""
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Filter to only BCConfig fields
            bc_config_fields = {
                'encoder_dim', 'hidden_dim', 'num_waypoints', 'future_len',
                'lr', 'weight_decay', 'batch_size', 'epochs', 'grad_clip',
                'loss_waypoint_weight', 'loss_speed_weight', 'ssl_encoder_path',
                'checkpoint_dir'
            }
            filtered_dict = {k: v for k, v in config_dict.items() if k in bc_config_fields}
            return BCConfig(**filtered_dict)
        
        # Default config
        return BCConfig()
    
    def _load_model(self, checkpoint_path: str) -> WaypointBCModel:
        """Load trained BC model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Create model with config (using BCConfig object)
        config = self.config
        model = WaypointBCModel(config=config, load_encoder=False)
        
        # Load state dict - handle various checkpoint formats
        state_dict = None
        for key in ["model_state_dict", "model_state", "state_dict", "bc_model_state"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        else:
            # Try loading directly (skip metadata keys)
            filtered_state = {k: v for k, v in checkpoint.items() 
                            if not k.startswith(('epoch', 'optimizer', 'metrics', 'config'))}
            model.load_state_dict(filtered_state, strict=False)
        
        return model
    
    @torch.no_grad()
    def predict_waypoints(
        self,
        perception_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict waypoints from perception input.
        
        Args:
            perception_input: [B, C, H, W] or [B, encoder_dim] tensor
            
        Returns:
            waypoints: [B, num_waypoints, 2] (x, y) waypoints in world coords
            speed: [B, 1] predicted speed
        """
        self.model.eval()
        
        # If already encoded
        if perception_input.dim() == 2:
            features = perception_input
        else:
            # Encode perception input
            features = self.model.encoder(perception_input)
        
        # Predict waypoints
        waypoints, speed = self.model.waypoint_head(features)
        return waypoints, speed
    
    def get_encoder_features(
        self,
        perception_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get encoder features for delta head input.
        
        Args:
            perception_input: [B, C, H, W] or [B, encoder_dim]
            
        Returns:
            features: [B, encoder_dim] latent features
        """
        self.model.eval()
        
        if perception_input.dim() == 2:
            return perception_input
        
        return self.model.encoder(perception_input)
    
    def get_config(self) -> BCConfig:
        """Get model config."""
        return self.config


class DeltaHead(nn.Module):
    """
    Learnable delta head for residual waypoint learning.
    
    Takes SFT encoder features and predicts corrections to SFT waypoints.
    """
    
    def __init__(
        self,
        encoder_dim: int = 256,
        num_waypoints: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_waypoints = num_waypoints
        
        # Delta prediction head
        self.delta_net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),  # (dx, dy) for each waypoint
        )
        
    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoint deltas.
        
        Args:
            encoder_features: [B, encoder_dim]
            
        Returns:
            delta_waypoints: [B, num_waypoints, 2] (dx, dy) corrections
        """
        delta = self.delta_net(encoder_features)
        delta = delta.view(-1, self.num_waypoints, 2)
        return delta


class SFTResidualWrapper(nn.Module):
    """
    Wrapper combining frozen SFT model + learnable delta head.
    
    final_waypoints = sft_waypoints + delta_head(encoder_features)
    """
    
    def __init__(
        self,
        sft_loader: SFTWaypointLoader,
        delta_head: Optional[DeltaHead] = None,
        freeze_sft: bool = True,
    ):
        super().__init__()
        self.sft_loader = sft_loader
        self.freeze_sft = freeze_sft
        
        # Delta head (learnable)
        if delta_head is None:
            config = sft_loader.get_config()
            delta_head = DeltaHead(
                encoder_dim=config.encoder_dim,
                num_waypoints=config.num_waypoints,
                hidden_dim=128,
            )
        self.delta_head = delta_head
        
        # Freeze SFT if requested
        if self.freeze_sft:
            for param in sft_loader.model.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        perception_input: torch.Tensor,
        apply_delta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass combining SFT predictions + delta corrections.
        
        Args:
            perception_input: [B, C, H, W] perception input
            apply_delta: Whether to apply delta corrections
            
        Returns:
            sft_waypoints: [B, num_waypoints, 2] SFT predictions
            delta_waypoints: [B, num_waypoints, 2] delta corrections
            final_waypoints: [B, num_waypoints, 2] final predictions
        """
        # Get SFT predictions
        with torch.no_grad() if self.freeze_sft else torch.enable_grad():
            sft_waypoints, speed = self.sft_loader.predict_waypoints(perception_input)
        
        # Get encoder features for delta
        encoder_features = self.sft_loader.get_encoder_features(perception_input)
        
        # Predict delta
        delta_waypoints = self.delta_head(encoder_features)
        
        # Combine
        if apply_delta:
            final_waypoints = sft_waypoints + delta_waypoints
        else:
            final_waypoints = sft_waypoints
            
        return sft_waypoints, delta_waypoints, final_waypoints
    
    def get_delta_parameters(self):
        """Get delta head parameters for optimizer."""
        return self.delta_head.parameters()


def load_sft_for_rl(
    checkpoint_path: str,
    device: str = "cpu",
    freeze_encoder: bool = True,
    add_delta_head: bool = True,
) -> Tuple[SFTWaypointLoader, Optional[SFTResidualWrapper]]:
    """
    Load SFT checkpoint for RL training.
    
    Args:
        checkpoint_path: Path to BC checkpoint (best.pt)
        device: Device to load model on
        freeze_encoder: Whether to freeze SFT encoder
        add_delta_head: Whether to add delta head wrapper
        
    Returns:
        sft_loader: SFT waypoint loader
        residual_wrapper: Optional wrapper with delta head
    """
    # Load SFT checkpoint
    sft_loader = SFTWaypointLoader(
        checkpoint_path=checkpoint_path,
        device=device,
        freeze_encoder=freeze_encoder,
    )
    
    # Optionally wrap with delta head
    if add_delta_head:
        wrapper = SFTResidualWrapper(
            sft_loader=sft_loader,
            freeze_sft=freeze_encoder,
        )
        return sft_loader, wrapper
    
    return sft_loader, None


# CLI for testing
if __name__ == "__main__":
    import json
    
    # Test loading
    checkpoint = "out/waypoint_bc/run_20260309_163356/best.pt"
    
    if os.path.exists(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        
        sft_loader, wrapper = load_sft_for_rl(
            checkpoint_path=checkpoint,
            device="cpu",
            freeze_encoder=True,
            add_delta_head=True,
        )
        
        print(f"Config: encoder_dim={sft_loader.config.encoder_dim}, num_waypoints={sft_loader.config.num_waypoints}")
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(2, 3, 128, 128)
        sft_wp, delta_wp, final_wp = wrapper(dummy_input)
        
        print(f"SFT waypoints shape: {sft_wp.shape}")
        print(f"Delta waypoints shape: {delta_wp.shape}")
        print(f"Final waypoints shape: {final_wp.shape}")
        
        print("\n✓ SFT checkpoint loader working!")
    else:
        print(f"Checkpoint not found: {checkpoint}")

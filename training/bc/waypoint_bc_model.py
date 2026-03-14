"""
Waypoint BC Model - Core behavior cloning model for waypoint prediction.

This module provides the main WaypointBCModel that predicts future waypoints
from BEV features, optionally with speed prediction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class WaypointBCConfig:
    """Configuration for Waypoint BC Model."""
    # Input
    bev_feature_dim: int = 256
    bev_height: int = 200
    bev_width: int = 200
    
    # Waypoint prediction
    num_waypoints: int = 8
    waypoint_dim: int = 2  # x, y relative positions
    
    # Speed prediction (optional)
    predict_speed: bool = True
    speed_min: float = 0.0
    speed_max: float = 15.0
    speed_dim: int = 1
    
    # Architecture
    use_temporal: bool = True
    temporal_history: int = 3
    
    # MLP head
    mlp_hidden_dims: List[int] = None
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [512, 256, 128]


class MLP(nn.Module):
    """Multi-layer perceptron with residual connections."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class WaypointBCModel(nn.Module):
    """
    Waypoint Behavior Cloning Model.
    
    Predicts future waypoints from BEV features.
    Optionally predicts speed at each waypoint.
    """
    
    def __init__(
        self,
        config: WaypointBCConfig,
        ssl_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.ssl_encoder = ssl_encoder
        
        # Compute input dimension
        bev_dim = config.bev_feature_dim
        if config.use_temporal:
            bev_dim *= config.temporal_history
        
        # Waypoint prediction head
        self.waypoint_mlp = MLP(
            input_dim=bev_dim,
            hidden_dims=config.mlp_hidden_dims,
            output_dim=config.num_waypoints * config.waypoint_dim,
        )
        
        # Speed prediction head (optional)
        if config.predict_speed:
            # Input: bev features + waypoint positions (for conditioning)
            self.speed_mlp = MLP(
                input_dim=bev_dim + config.num_waypoints * config.waypoint_dim,
                hidden_dims=config.mlp_hidden_dims[:-1],
                output_dim=config.num_waypoints * config.speed_dim,
            )
        
        # Temporal encoding if enabled
        if config.use_temporal:
            self.temporal_encoder = nn.LSTM(
                input_size=config.bev_feature_dim,
                hidden_size=config.bev_feature_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
            )
    
    def forward(
        self,
        bev_features: torch.Tensor,
        return_speed: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            bev_features: [B, C, H, W] or [B, T, C, H, W] if temporal
            
        Returns:
            waypoints: [B, num_waypoints, waypoint_dim]
            speeds: [B, num_waypoints] or None if predict_speed=False
        """
        B = bev_features.shape[0]
        
        # Handle temporal dimension
        if self.config.use_temporal and bev_features.dim() == 5:
            # [B, T, C, H, W] -> [B, T, C, H*W]
            T = bev_features.shape[1]
            bev_flat = bev_features.flatten(3)  # [B, T, C, H*W]
            bev_flat = bev_flat.permute(0, 1, 3, 2)  # [B, T, H*W, C]
            bev_flat = bev_flat.reshape(B, T, -1)  # [B, T, H*W*C]
            
            # Temporal encoding
            _, (h_n, _) = self.temporal_encoder(bev_flat)
            bev_encoding = h_n[-1]  # [B, C]
        else:
            # Global average pooling over spatial dimensions
            bev_encoding = bev_features.flatten(2).mean(dim=2)  # [B, C]
        
        # Predict waypoints
        waypoint_flat = self.waypoint_mlp(bev_encoding)  # [B, num_waypoints * waypoint_dim]
        waypoints = waypoint_flat.reshape(B, self.config.num_waypoints, self.config.waypoint_dim)
        
        # Predict speeds (optional)
        speeds = None
        if return_speed and self.config.predict_speed:
            # Condition on waypoint positions
            waypoint_cond = waypoint_flat  # [B, num_waypoints * waypoint_dim]
            speed_input = torch.cat([bev_encoding, waypoint_cond], dim=1)
            speed_flat = self.speed_mlp(speed_input)
            speeds = speed_flat.reshape(B, self.config.num_waypoints)
            
            # Clip to valid range
            speeds = torch.clamp(
                speeds,
                self.config.speed_min,
                self.config.speed_max
            )
        
        return waypoints, speeds
    
    def predict(
        self,
        bev_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for prediction without gradient tracking.
        """
        with torch.no_grad():
            waypoints, speeds = self.forward(bev_features, return_speed=True)
        return waypoints, speeds


def create_waypoint_bc_model(
    bev_feature_dim: int = 256,
    num_waypoints: int = 8,
    predict_speed: bool = True,
    ssl_encoder_type: Optional[str] = None,
) -> WaypointBCModel:
    """
    Factory function to create a Waypoint BC model.
    
    Args:
        bev_feature_dim: Dimension of BEV features
        num_waypoints: Number of future waypoints to predict
        predict_speed: Whether to predict speed
        ssl_encoder_type: Type of SSL encoder ('resnet34', 'resnet50', 'jepa', etc.)
    
    Returns:
        WaypointBCModel instance
    """
    config = WaypointBCConfig(
        bev_feature_dim=bev_feature_dim,
        num_waypoints=num_waypoints,
        predict_speed=predict_speed,
    )
    
    # Load SSL encoder if specified
    ssl_encoder = None
    if ssl_encoder_type is not None:
        try:
            from training.sft.ssl_pretrained_loader import load_ssl_pretrained
            ssl_encoder = load_ssl_pretrained(ssl_encoder_type)
        except ImportError:
            print(f"Warning: Could not load SSL encoder {ssl_encoder_type}")
    
    return WaypointBCModel(config, ssl_encoder=ssl_encoder)


class WaypointBCWithSpeed(nn.Module):
    """
    Wrapper that combines waypoint prediction with speed prediction.
    Provides easier integration with CARLA policy wrapper.
    """
    
    def __init__(self, model: WaypointBCModel):
        super().__init__()
        self.model = model
    
    def forward(self, bev_features: torch.Tensor):
        return self.model(bev_features, return_speed=True)
    
    def predict(self, bev_features: torch.Tensor):
        return self.model.predict(bev_features)


# Loss functions
def waypoint_l1_loss(pred_waypoints: torch.Tensor, target_waypoints: torch.Tensor) -> torch.Tensor:
    """L1 loss for waypoint prediction."""
    return torch.abs(pred_waypoints - target_waypoints).mean()


def waypoint_mse_loss(pred_waypoints: torch.Tensor, target_waypoints: torch.Tensor) -> torch.Tensor:
    """MSE loss for waypoint prediction."""
    return nn.functional.mse_loss(pred_waypoints, target_waypoints)


def speed_l1_loss(pred_speeds: torch.Tensor, target_speeds: torch.Tensor) -> torch.Tensor:
    """L1 loss for speed prediction."""
    return torch.abs(pred_speeds - target_speeds).mean()


def speed_mse_loss(pred_speeds: torch.Tensor, target_speeds: torch.Tensor) -> torch.Tensor:
    """MSE loss for speed prediction."""
    return nn.functional.mse_loss(pred_speeds, target_speeds)


def compute_bc_loss(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    pred_speeds: Optional[torch.Tensor] = None,
    target_speeds: Optional[torch.Tensor] = None,
    speed_weight: float = 0.3,
) -> dict:
    """
    Compute combined BC loss for waypoint and speed prediction.
    
    Args:
        pred_waypoints: [B, N, 2] predicted waypoints
        target_waypoints: [B, N, 2] target waypoints
        pred_speeds: [B, N] predicted speeds
        target_speeds: [B, N] target speeds
        speed_weight: Weight for speed loss
    
    Returns:
        Dictionary with loss components and total loss
    """
    # Waypoint loss
    waypoint_loss = waypoint_l1_loss(pred_waypoints, target_waypoints)
    
    # Speed loss (optional)
    if pred_speeds is not None and target_speeds is not None:
        speed_loss = speed_l1_loss(pred_speeds, target_speeds)
        total_loss = (1 - speed_weight) * waypoint_loss + speed_weight * speed_loss
        return {
            'total_loss': total_loss,
            'waypoint_loss': waypoint_loss,
            'speed_loss': speed_loss,
        }
    else:
        return {
            'total_loss': waypoint_loss,
            'waypoint_loss': waypoint_loss,
        }


# Demo usage
if __name__ == "__main__":
    # Create model
    model = create_waypoint_bc_model(
        bev_feature_dim=256,
        num_waypoints=8,
        predict_speed=True,
    )
    
    # Dummy input
    B, C, H, W = 4, 256, 200, 200
    bev = torch.randn(B, C, H, W)
    
    # Forward pass
    waypoints, speeds = model(bev)
    print(f"Waypoints shape: {waypoints.shape}")  # [4, 8, 2]
    print(f"Speeds shape: {speeds.shape}")  # [4, 8]
    
    # Compute loss
    target_waypoints = torch.randn_like(waypoints)
    target_speeds = torch.rand(B, 8) * 10
    
    losses = compute_bc_loss(
        waypoints, target_waypoints,
        speeds, target_speeds,
    )
    print(f"Total loss: {losses['total_loss']:.4f}")
    print(f"Waypoint loss: {losses['waypoint_loss']:.4f}")
    print(f"Speed loss: {losses['speed_loss']:.4f}")

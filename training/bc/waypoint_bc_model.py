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
        freeze_ssl_encoder: bool = True,
    ):
        super().__init__()
        self.config = config
        self.ssl_encoder = ssl_encoder
        self.freeze_ssl_encoder = freeze_ssl_encoder
        
        # SSL encoder projection to match BEV feature dimension
        if ssl_encoder is not None:
            # Get SSL encoder output dimension
            ssl_out_dim = self._get_ssl_encoder_dim(ssl_encoder)
            self.ssl_projection = nn.Linear(ssl_out_dim, config.bev_feature_dim)
        
        # Input dimension for MLP is bev_feature_dim
        # (temporal encoding outputs single hidden state, not concatenated frames)
        bev_dim = config.bev_feature_dim
        
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
    
    def _get_ssl_encoder_dim(self, ssl_encoder) -> int:
        """Get output dimension of SSL encoder."""
        # Try to get the embedding dimension from the encoder
        if hasattr(ssl_encoder, 'embedding_dim'):
            return ssl_encoder.embedding_dim
        elif hasattr(ssl_encoder, 'out_features'):
            return ssl_encoder.out_features
        elif hasattr(ssl_encoder, 'fc') and hasattr(ssl_encoder.fc, 'out_features'):
            return ssl_encoder.fc.out_features
        else:
            # Default to common SSL embedding dimensions
            return 128
    
    def _encode_with_ssl(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images through SSL encoder and project to BEV feature space.
        
        Args:
            images: [B, C, H, W] input images
            
        Returns:
            bev_features: [B, bev_feature_dim] BEV feature encoding
        """
        if self.ssl_encoder is None:
            raise ValueError("SSL encoder not set. Pass ssl_encoder to constructor.")
        
        with torch.set_grad_enabled(not self.freeze_ssl_encoder):
            # Get SSL embeddings
            if hasattr(self.ssl_encoder, 'encode_image'):
                ssl_embeddings = self.ssl_encoder.encode_image(images)
            elif hasattr(self.ssl_encoder, 'forward'):
                ssl_embeddings = self.ssl_encoder(images)
            else:
                raise ValueError("SSL encoder must have encode_image or forward method")
            
            # Project to BEV feature space
            bev_features = self.ssl_projection(ssl_embeddings)
        
        return bev_features
    
    def forward(
        self,
        bev_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        return_speed: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Supports two modes:
        1. Direct BEV features: pass bev_features=[B, C, H, W] or [B, T, C, H, W]
        2. SSL encoder path: pass images=[B, C, H, W] to encode through SSL first
        
        Args:
            bev_features: [B, C, H, W] or [B, T, C, H, W] if temporal
            images: [B, C, H, W] images to encode through SSL encoder
            return_speed: Whether to return speed predictions
            
        Returns:
            waypoints: [B, num_waypoints, waypoint_dim]
            speeds: [B, num_waypoints] or None if predict_speed=False
        """
        # Handle SSL encoder path
        if images is not None:
            if self.ssl_encoder is None:
                raise ValueError("images provided but ssl_encoder is not set")
            bev_encoding = self._encode_with_ssl(images)
            # bev_encoding is [B, bev_feature_dim] - directly use for MLP
            B = bev_encoding.shape[0]
            
            # Predict waypoints directly from SSL-encoded features
            waypoint_flat = self.waypoint_mlp(bev_encoding)  # [B, num_waypoints * waypoint_dim]
            waypoints = waypoint_flat.reshape(B, self.config.num_waypoints, self.config.waypoint_dim)
            
            # Predict speeds (optional)
            speeds = None
            if return_speed and self.config.predict_speed:
                waypoint_cond = waypoint_flat
                speed_input = torch.cat([bev_encoding, waypoint_cond], dim=1)
                speed_flat = self.speed_mlp(speed_input)
                speeds = speed_flat.reshape(B, self.config.num_waypoints)
                speeds = torch.clamp(speeds, self.config.speed_min, self.config.speed_max)
            
            return waypoints, speeds
        
        if bev_features is None:
            raise ValueError("Must provide either bev_features or images")
        
        B = bev_features.shape[0]
        
        # Handle temporal dimension
        if self.config.use_temporal and bev_features.dim() == 5:
            # [B, T, C, H, W] - temporal sequence of features
            # Flatten spatial dimensions per frame
            bev_per_frame = bev_features.flatten(3)  # [B, T, C, H*W]
            bev_per_frame = bev_per_frame.mean(dim=-1)  # [B, T, C] - global pool spatial
            bev_per_frame = bev_per_frame.permute(0, 1, 2)  # [B, T, C]
            
            # Temporal encoding
            _, (h_n, _) = self.temporal_encoder(bev_per_frame)
            bev_encoding = h_n[-1]  # [B, C]
        else:
            # Global average pooling over spatial dimensions
            if bev_features.dim() == 4:
                bev_encoding = bev_features.flatten(2).mean(dim=2)  # [B, C]
            else:
                bev_encoding = bev_features  # Already [B, C] from SSL path
        
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
        bev_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for prediction without gradient tracking.
        
        Args:
            bev_features: Pre-computed BEV features [B, C, H, W]
            images: Images to encode through SSL [B, C, H, W]
            
        Returns:
            waypoints: [B, num_waypoints, 2]
            speeds: [B, num_waypoints]
        """
        with torch.no_grad():
            waypoints, speeds = self.forward(
                bev_features=bev_features,
                images=images,
                return_speed=True,
            )
        return waypoints, speeds


def create_waypoint_bc_model(
    bev_feature_dim: int = 256,
    num_waypoints: int = 8,
    predict_speed: bool = True,
    ssl_encoder_type: Optional[str] = None,
    freeze_ssl_encoder: bool = True,
    ssl_encoder_checkpoint: Optional[str] = None,
    use_temporal: bool = False,
    temporal_history: int = 3,
) -> WaypointBCModel:
    """
    Factory function to create a Waypoint BC model.
    
    Args:
        bev_feature_dim: Dimension of BEV features
        num_waypoints: Number of future waypoints to predict
        predict_speed: Whether to predict speed
        ssl_encoder_type: Type of SSL encoder ('resnet34', 'resnet50', 'jepa', etc.)
        freeze_ssl_encoder: Whether to freeze SSL encoder weights
        ssl_encoder_checkpoint: Path to SSL encoder checkpoint
        use_temporal: Whether to use temporal encoding
        temporal_history: Number of historical frames
        
    Returns:
        WaypointBCModel instance
    """
    config = WaypointBCConfig(
        bev_feature_dim=bev_feature_dim,
        num_waypoints=num_waypoints,
        predict_speed=predict_speed,
        use_temporal=use_temporal,
        temporal_history=temporal_history,
    )
    
    # Load SSL encoder if specified
    ssl_encoder = None
    if ssl_encoder_type is not None:
        try:
            from training.sft.ssl_pretrained_loader import load_ssl_pretrained
            ssl_encoder = load_ssl_pretrained(
                ssl_encoder_type,
                checkpoint_path=ssl_encoder_checkpoint,
            )
        except ImportError:
            print(f"Warning: Could not load SSL encoder {ssl_encoder_type}")
    
    return WaypointBCModel(
        config,
        ssl_encoder=ssl_encoder,
        freeze_ssl_encoder=freeze_ssl_encoder,
    )


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

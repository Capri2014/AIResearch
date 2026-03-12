"""
BEV Encoder Integration for Waypoint BC

Bridges the BEV encoder module (sim/driving/carla_srunner/bev_encoder.py)
with Waypoint BC training pipeline.

This enables using camera + LiDAR based BEV features for waypoint prediction,
extending the SSL encoder approach with multi-sensor fusion.

Usage:
    from training.bc.bev_encoder_integration import BEVBCConfig, create_bev_bc_model
    
    # Create BC model with BEV encoder
    config = BEVBCConfig(
        use_bev_encoder=True,
        input_types=["camera", "lidar"],
        feature_dim=256
    )
    model = create_bev_bc_model(config)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import sys

# Import BEV encoder components
# Note: Requires sim/driving/carla_srunner to be in path
try:
    from sim.driving.carla_srunner.bev_encoder import (
        BEVEncoder, LidarToBEV, CameraToBEV, BEVEncoderConfig, create_bev_encoder
    )
    BEV_ENCODER_AVAILABLE = True
except ImportError:
    BEV_ENCODER_AVAILABLE = False
    BEVEncoder = None
    BEVEncoderConfig = None


@dataclass
class BEVBCConfig:
    """
    Configuration for Waypoint BC with BEV encoder integration.
    
    Extends BCConfig with BEV-specific options for multi-sensor input.
    """
    # BEV Encoder options
    use_bev_encoder: bool = False  # Use BEV encoder instead of simple SSL encoder
    input_types: List[str] = field(default_factory=lambda: ["camera"])  # ["camera", "lidar"]
    bev_resolution: float = 0.5  # meters per pixel
    bev_size: Tuple[int, int] = (200, 200)  # (height, width) in pixels
    feature_dim: int = 256
    
    # SSL encoder fallback options (when use_bev_encoder=False)
    ssl_encoder_path: Optional[str] = None
    freeze_encoder: bool = True
    
    # Standard BC options
    encoder_dim: int = 256
    hidden_dim: int = 512
    num_waypoints: int = 8
    future_len: float = 4.0
    
    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 50
    grad_clip: float = 1.0
    
    # Loss weights
    loss_waypoint_weight: float = 1.0
    loss_speed_weight: float = 0.1
    
    # Checkpoint
    checkpoint_dir: str = "out/waypoint_bc"


class BEVWaypointBCModel(nn.Module):
    """
    Waypoint BC model with optional BEV encoder integration.
    
    Supports two modes:
    1. BEV encoder mode: Uses camera + LiDAR BEV features
    2. SSL encoder mode: Uses simple CNN encoder (fallback)
    
    Architecture (BEV mode):
        Camera Images → CameraToBEV → Camera BEV Features
        LiDAR Points → LidarToBEV → LiDAR BEV Features
                                      ↓
                              Feature Fusion
                                      ↓
                      Features [B, feature_dim, H, W]
                                      ↓
                              BEV Encoder
                                      ↓
                      Features [B, encoder_dim]
                                      ↓
                              WaypointHead
                                      ↓
                      waypoints [B, num_waypoints, 2], speed [B, 1]
    """
    def __init__(self, config: BEVBCConfig):
        super().__init__()
        self.config = config
        self.use_bev_encoder = config.use_bev_encoder and BEV_ENCODER_AVAILABLE
        
        if self.use_bev_encoder:
            # Initialize BEV encoder
            self._init_bev_encoder()
        else:
            # Use simple SSL encoder from waypoint_bc.py
            from training.bc.waypoint_bc import SSLEncoder
            self.encoder = SSLEncoder(input_dim=3, encoder_dim=config.encoder_dim)
            
            # Load pre-trained SSL encoder if path provided
            if config.ssl_encoder_path:
                self._load_ssl_encoder(config.ssl_encoder_path)
        
        # Waypoint prediction head
        self.waypoint_head = self._create_waypoint_head()
        
    def _init_bev_encoder(self):
        """Initialize BEV encoder with camera + LiDAR fusion."""
        # Use factory function instead of direct config
        self.bev_encoder = create_bev_encoder(
            input_types=self.config.input_types,
            bev_resolution=self.config.bev_resolution,
            bev_size=(self.config.bev_size[1], self.config.bev_size[0]),  # (width, height)
            feature_dim=self.config.feature_dim
        )
        
        # Determine actual output channels from BEV encoder (depends on internal config)
        # BEV encoder outputs [B, channels, H, W] where channels is determined internally
        # We'll use a dummy forward pass to get the channel count
        with torch.no_grad():
            dummy_cameras = [torch.randn(1, 3, 128, 128)]
            if 'lidar' in self.config.input_types:
                dummy_lidar = torch.randn(1, 1000, 3)
            else:
                dummy_lidar = None
            dummy_bev = self.bev_encoder.encode(cameras=dummy_cameras, lidar_points=dummy_lidar)
            bev_channels = dummy_bev.shape[1]  # Get actual channel count
            print(f'[BEV-BC] BEV encoder output channels: {bev_channels}')
        
        # BEV feature projection to encoder_dim (using actual bev_channels)
        self.bev_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(bev_channels, self.config.encoder_dim),
            nn.ReLU(),
            nn.Linear(self.config.encoder_dim, self.config.encoder_dim),
        )
        
    def _load_ssl_encoder(self, path: str):
        """Load pre-trained SSL encoder weights."""
        from training.bc.waypoint_bc import WaypointBCModel
        
        path = Path(path)
        if not path.exists():
            print(f"[BEV-BC] Warning: SSL encoder not found: {path}")
            return
            
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Handle different checkpoint formats
            state_dict = None
            if 'encoder' in checkpoint:
                state_dict = checkpoint['encoder']
            elif 'conv.0.weight' in checkpoint:
                state_dict = checkpoint
                
            if state_dict:
                loaded = self.encoder.load_state_dict(state_dict, strict=False)
                print(f"[BEV-BC] Loaded SSL encoder from: {path}")
                
                if self.config.freeze_encoder:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    print("[BEV-BC] SSL encoder is frozen")
                    
        except Exception as e:
            print(f"[BEV-BC] Warning: Failed to load encoder: {e}")
            
    def _create_waypoint_head(self):
        """Create waypoint prediction head."""
        from training.bc.waypoint_bc import WaypointHead
        return WaypointHead(
            encoder_dim=self.config.encoder_dim,
            hidden_dim=self.config.hidden_dim,
            num_waypoints=self.config.num_waypoints
        )
        
    def encode_bev(
        self,
        cameras: Optional[List[torch.Tensor]] = None,
        lidar_points: Optional[torch.Tensor] = None,
        camera_images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode multi-sensor input to BEV features.
        
        Args:
            cameras: List of camera images [B, C, H, W] (one per view)
            lidar_points: LiDAR point cloud [B, N, 3] or [B, N, 4]
            camera_images: Alternative to cameras - stacked [B, num_cameras, C, H, W]
            
        Returns:
            features: [B, encoder_dim] BEV features
        """
        if not self.use_bev_encoder:
            raise RuntimeError("BEV encoder not enabled. Set use_bev_encoder=True in config.")
            
        # Handle different input formats
        if camera_images is not None:
            # [B, num_cameras, C, H, W] -> list of [B, C, H, W]
            B, num_cams = camera_images.shape[:2]
            cameras = [camera_images[:, i] for i in range(num_cams)]
            
        # Encode with BEV encoder
        bev_features = self.bev_encoder.encode(
            cameras=cameras,
            lidar_points=lidar_points
        )
        
        # Project to encoder_dim
        features = self.bev_projection(bev_features)
        return features
        
    def forward(
        self,
        x: torch.Tensor,
        cameras: Optional[List[torch.Tensor]] = None,
        lidar_points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, C, H, W] BEV image input (used when use_bev_encoder=False)
            cameras: List of camera images [B, C, H, W] (used when use_bev_encoder=True)
            lidar_points: LiDAR point cloud [B, N, 3] or [B, N, 4]
            
        Returns:
            waypoints: [B, num_waypoints, 2] predicted waypoints
            speed: [B, 1] predicted speed
        """
        if self.use_bev_encoder:
            # Use BEV encoder path
            if cameras is not None or lidar_points is not None:
                features = self.encode_bev(cameras=cameras, lidar_points=lidar_points)
            else:
                raise ValueError("BEV encoder requires cameras and/or lidar_points input")
        else:
            # Use simple SSL encoder
            features = self.encoder(x)
            
        waypoints, speed = self.waypoint_head(features)
        return waypoints, speed
        
    def predict(
        self,
        x: torch.Tensor = None,
        cameras: Optional[List[torch.Tensor]] = None,
        lidar_points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prediction mode - returns dict for downstream use.
        """
        waypoints, speed = self.forward(x, cameras=cameras, lidar_points=lidar_points)
        return {
            "waypoints": waypoints,
            "speed": speed,
        }


def create_bev_bc_model(config: BEVBCConfig) -> BEVWaypointBCModel:
    """
    Factory function to create BEV-integrated BC model.
    
    Args:
        config: BEVBCConfig with model configuration
        
    Returns:
        BEVWaypointBCModel instance
    """
    if config.use_bev_encoder and not BEV_ENCODER_AVAILABLE:
        print("[BEV-BC] Warning: BEV encoder not available, falling back to SSL encoder")
        config.use_bev_encoder = False
        
    return BEVWaypointBCModel(config)


def find_latest_bev_encoder_checkpoint(checkpoint_dir: str = "out/bev_encoder") -> Optional[str]:
    """
    Find latest BEV encoder checkpoint.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
        
    # Find all .pt files
    checkpoints = list(checkpoint_path.glob("*.pt"))
    if not checkpoints:
        return None
        
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


# Re-export for convenience
__all__ = [
    "BEVBCConfig",
    "BEVWaypointBCModel", 
    "create_bev_bc_model",
    "find_latest_bev_encoder_checkpoint",
    "BEV_ENCODER_AVAILABLE",
]

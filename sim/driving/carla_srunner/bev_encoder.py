"""
BEV (Bird's Eye View) Encoder Module

Transforms multi-view camera and LiDAR data into unified BEV features.
Bridges perception (camera + LiDAR) → BEV features → waypoint BC model.

Usage:
    from sim.driving.carla_srunner.bev_encoder import BEVEncoder, create_bev_encoder

    # Create encoder
    encoder = create_bev_encoder(
        input_types=["camera", "lidar"],
        bev_resolution=0.5,  # meters per pixel
        bev_size=(200, 200),  # width x height in pixels
    )

    # Encode multi-view sensors to BEV
    bev_features = encoder.encode(
        cameras=camera_images,    # List of [C, H, W] tensors
        lidar_points=lidar_pc,    # [N, 3] or [N, 4] (x, y, z, intensity)
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
    )  # Returns [B, feature_dim, H, W] BEV features

    # Get BEV image for visualization
    bev_image = encoder.get_bev_image(bev_features)  # [B, 3, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from pathlib import Path
import json


@dataclass
class BEVEncoderConfig:
    """Configuration for BEV encoder."""
    # Input modalities
    input_types: List[str] = field(default_factory=lambda: ["camera"])  # ["camera", "lidar"]
    
    # BEV grid
    bev_resolution: float = 0.5  # meters per pixel
    bev_width: int = 200  # pixels
    bev_height: int = 200  # pixels
    bev_range_x: Tuple[float, float] = (-50.0, 50.0)  # meters (forward is +x)
    bev_range_y: Tuple[float, float] = (-50.0, 50.0)  # meters (left is +y)
    
    # Camera encoding
    camera_embed_dim: int = 128
    camera_num_layers: int = 4
    camera_use_latest: bool = True  # Use only latest frame
    
    # LiDAR encoding
    lidar_use_intensity: bool = True
    lidar_num_bins: int = 256
    lidar_max_points: int = 20000
    
    # Feature fusion
    fusion_type: str = "concat"  # "concat", "attention", "sum"
    feature_dim: int = 256
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LidarToBEV(nn.Module):
    """Convert LiDAR point cloud to BEV representation."""
    
    def __init__(self, config: BEVEncoderConfig):
        super().__init__()
        self.config = config
        
        # BEV grid parameters
        self.bev_w = config.bev_width
        self.bev_h = config.bev_height
        self.resolution = config.bev_resolution
        self.x_range = config.bev_range_x
        self.y_range = config.bev_range_y
        
        # Discretization
        self.x_bins = int((self.x_range[1] - self.x_range[0]) / self.resolution)
        self.y_bins = int((self.y_range[1] - self.y_range[0]) / self.resolution)
        
        # Height bins for elevation information
        self.height_bins = nn.Parameter(
            torch.linspace(-2.0, 4.0, 4), requires_grad=False
        )
        
        # LiDAR feature encoding (height bins + intensity)
        in_channels = 5 if config.lidar_use_intensity else 4
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, config.feature_dim // 2, kernel_size=1),
        )
    
    def points_to_bev(
        self,
        points: torch.Tensor,
        image_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project LiDAR points to BEV grid.
        
        Args:
            points: [N, 3] or [N, 4] (x, y, z, intensity)
            image_coords: Precomputed BEV grid indices [N, 2]
        
        Returns:
            bev: [B, C, bev_h, bev_w]
        """
        N = points.shape[0]
        
        # Extract coordinates and features
        x = points[:, 0]  # forward
        y = points[:, 1]  # left
        z = points[:, 2]  # up
        
        # Discretize to BEV grid
        x_idx = ((x - self.x_range[0]) / self.resolution).long()
        y_idx = ((y - self.y_range[0]) / self.resolution).long()
        
        # Clamp to valid range
        x_idx = x_idx.clamp(0, self.bev_w - 1)
        y_idx = y_idx.clamp(0, self.bev_h - 1)
        
        # Build BEV feature tensor (4 height bins + intensity)
        batch_size = 1
        bev = torch.zeros(
            batch_size, 5, self.bev_h, self.bev_w,
            device=points.device, dtype=torch.float32
        )
        
        # Height encoding (4 height bins)
        height_bin_idx = torch.searchsorted(self.height_bins, z)
        height_bin_idx = height_bin_idx.clamp(0, 3)  # 0-3 for 4 bins
        
        # Fill BEV - take max height for each cell
        for i in range(N):
            h = height_bin_idx[i].item()
            if h < 4:  # Safety check
                bev[0, h, y_idx[i], x_idx[i]] = max(
                    bev[0, h, y_idx[i], x_idx[i]], z[i] + 2.0
                )
        
        # Add intensity channel if available
        if points.shape[1] >= 4:
            intensity = points[:, 3:4]  # [N, 1]
            # Simple max pooling for intensity
            bev[0, -1, y_idx, x_idx] = torch.max(
                bev[0, -1, y_idx, x_idx], intensity.squeeze()
            )
        
        return bev
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode LiDAR points to BEV features.
        
        Args:
            points: [N, 3] or [N, 4] (x, y, z, intensity)
        
        Returns:
            features: [1, feature_dim//2, bev_h, bev_w]
        """
        # Handle empty point cloud
        if points.shape[0] == 0:
            return torch.zeros(
                1, self.config.feature_dim // 2, self.bev_h, self.bev_w,
                device=self.config.device
            )
        
        # Subsample if too many points
        if points.shape[0] > self.config.lidar_max_points:
            idx = torch.randperm(points.shape[0])[:self.config.lidar_max_points]
            points = points[idx]
        
        # Project to BEV
        bev = self.points_to_bev(points)
        
        # Encode
        features = self.lidar_encoder(bev)
        return features


class CameraToBEV(nn.Module):
    """Transform camera features to BEV via perspective transformation."""
    
    def __init__(self, config: BEVEncoderConfig):
        super().__init__()
        self.config = config
        
        # Camera feature backbone
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # BEV projection
        self.bev_projection = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, config.feature_dim // 2, kernel_size=1),
        )
    
    def forward(
        self,
        images: List[torch.Tensor],
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode camera images to BEV features.
        
        Args:
            images: List of [B, C, H, W] camera images
            intrinsics: Camera intrinsic matrices [B, 3, 3]
            extrinsics: Camera extrinsic matrices [B, 4, 4]
        
        Returns:
            features: [B, feature_dim//2, bev_h, bev_w]
        """
        # Encode each camera view
        encoded = []
        for img in images:
            features = self.camera_encoder(img)
            encoded.append(features)
        
        # Simple average pooling across views
        if len(encoded) > 1:
            encoded = torch.stack(encoded, dim=0)
            features = encoded.mean(dim=0)
        else:
            features = encoded[0]
        
        # Project to BEV (simplified - real implementation would use IPM)
        bev_features = self.bev_projection(features)
        
        # Resize to target BEV size
        bev_features = F.interpolate(
            bev_features,
            size=(self.config.bev_height, self.config.bev_width),
            mode="bilinear",
            align_corners=False
        )
        
        return bev_features


class BEVEncoder(nn.Module):
    """
    Unified BEV encoder that combines camera and LiDAR into BEV features.
    
    Architecture:
        Camera Images → CameraToBEV → Camera BEV Features
        LiDAR Points → LidarToBEV → LiDAR BEV Features
                                      ↓
                              Feature Fusion
                                      ↓
                              BEV Features [B, feature_dim, H, W]
    """
    
    def __init__(self, config: BEVEncoderConfig):
        super().__init__()
        self.config = config
        
        # Initialize encoders
        self.camera_encoder = None
        self.lidar_encoder = None
        
        if "camera" in config.input_types:
            self.camera_encoder = CameraToBEV(config)
        
        if "lidar" in config.input_types:
            self.lidar_encoder = LidarToBEV(config)
        
        # Feature fusion
        if config.fusion_type == "concat":
            self.fusion = nn.Identity()
            self.output_dim = config.feature_dim
        elif config.fusion_type == "attention":
            self.fusion = nn.Sequential(
                nn.Conv2d(config.feature_dim, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Attention2d(128, 128),
                nn.Conv2d(128, config.feature_dim, kernel_size=1),
            )
        else:  # sum
            self.fusion = nn.Identity()
            self.output_dim = config.feature_dim // 2
        
        self.to(config.device)
    
    def encode(
        self,
        cameras: Optional[List[torch.Tensor]] = None,
        lidar_points: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sensors to BEV features.
        
        Args:
            cameras: List of [B, C, H, W] camera images
            lidar_points: [N, 3] or [N, 4] LiDAR points (x, y, z, intensity)
            intrinsics: Camera intrinsics
            extrinsics: Camera extrinsics
        
        Returns:
            bev_features: [B, feature_dim, bev_h, bev_w]
        """
        features_list = []
        
        # Encode cameras
        if cameras is not None and self.camera_encoder is not None:
            camera_features = self.camera_encoder(
                cameras, intrinsics, extrinsics
            )
            features_list.append(camera_features)
        
        # Encode LiDAR
        if lidar_points is not None and self.lidar_encoder is not None:
            lidar_features = self.lidar_encoder(lidar_points)
            features_list.append(lidar_features)
        
        # Fuse features
        if len(features_list) == 0:
            raise ValueError("No valid inputs provided")
        elif len(features_list) == 1:
            bev_features = features_list[0]
        else:
            if self.config.fusion_type == "concat":
                bev_features = torch.cat(features_list, dim=1)
            elif self.config.fusion_type == "sum":
                bev_features = torch.stack(features_list, dim=0).sum(dim=0)
            else:
                bev_features = torch.cat(features_list, dim=1)
        
        # Apply fusion
        bev_features = self.fusion(bev_features)
        
        return bev_features
    
    def get_bev_image(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Convert BEV features to RGB image for visualization.
        
        Args:
            bev_features: [B, C, H, W] BEV features
        
        Returns:
            image: [B, 3, H, W] RGB image
        """
        # Simple visualization: take first 3 channels
        if bev_features.shape[1] >= 3:
            img = bev_features[:, :3, :, :]
            # Normalize
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:
            # Replicate channels
            img = bev_features[:, :1, :, :].repeat(1, 3, 1, 1)
        
        return img
    
    def forward(
        self,
        cameras: Optional[List[torch.Tensor]] = None,
        lidar_points: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass - same as encode."""
        return self.encode(cameras, lidar_points, intrinsics, extrinsics)


def create_bev_encoder(
    input_types: Optional[List[str]] = None,
    bev_resolution: float = 0.5,
    bev_size: Tuple[int, int] = (200, 200),
    feature_dim: int = 256,
    device: Optional[str] = None
) -> BEVEncoder:
    """
    Factory function to create BEV encoder.
    
    Args:
        input_types: List of input modalities ["camera", "lidar"]
        bev_resolution: Meters per pixel
        bev_size: (width, height) in pixels
        feature_dim: Output feature dimension
        device: Device to run on
    
    Returns:
        BEVEncoder instance
    """
    if input_types is None:
        input_types = ["camera"]
    
    config = BEVEncoderConfig(
        input_types=input_types,
        bev_resolution=bev_resolution,
        bev_width=bev_size[0],
        bev_height=bev_size[1],
        feature_dim=feature_dim,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    return BEVEncoder(config)


def demo():
    """Demo usage of BEV encoder."""
    print("BEV Encoder Demo")
    print("=" * 50)
    
    # Create encoder
    encoder = create_bev_encoder(
        input_types=["camera", "lidar"],
        bev_resolution=0.5,
        bev_size=(200, 200),
        feature_dim=256
    )
    print(f"Created encoder: {encoder.config.input_types}")
    print(f"BEV size: {encoder.config.bev_width}x{encoder.config.bev_height}")
    print(f"Resolution: {encoder.config.bev_resolution}m/pixel")
    
    # Dummy inputs
    batch_size = 1
    cameras = [torch.randn(batch_size, 3, 224, 224) for _ in range(4)]  # 4 cameras
    lidar = torch.randn(10000, 4)  # 10k points with intensity
    
    # Encode
    with torch.no_grad():
        bev_features = encoder.encode(cameras=cameras, lidar_points=lidar)
    
    print(f"\nInput:")
    print(f"  Cameras: {len(cameras)} views, shape {cameras[0].shape}")
    print(f"  LiDAR: {lidar.shape[0]} points")
    
    print(f"\nOutput:")
    print(f"  BEV features: {bev_features.shape}")
    
    # Get visualization
    bev_img = encoder.get_bev_image(bev_features)
    print(f"  BEV image: {bev_img.shape}")
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo()

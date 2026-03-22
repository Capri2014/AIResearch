"""
BEV (Bird's Eye View) Encoder for Waymo SSL Pretraining.

This module provides a BEV encoder that transforms multi-camera + LiDAR data
into a unified Bird's Eye View representation for downstream driving tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BEVConfig:
    """Configuration for BEV encoder."""
    # Input dimensions
    image_size: Tuple[int, int] = (224, 224)
    num_cameras: int = 6
    lidar_channels: int = 2  # x, y positions
    
    # BEV grid
    bev_resolution: float = 0.5  # meters per pixel
    bev_range: Tuple[float, float, float, float] = (-50, -50, 50, 50)  # x_min, y_min, x_max, y_max
    
    # Encoder
    camera_backbone: str = "resnet18"
    lidar_backbone: str = "point_pillars"
    encoder_dim: int = 256
    bev_channels: int = 64
    
    # Fusion
    fusion_method: str = "attention"  # 'attention', 'concat', 'add'


class ConvBlock(nn.Module):
    """Basic convolutional block with BN and ReLU."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CameraEncoder(nn.Module):
    """Encode multi-camera images to features."""
    
    def __init__(self, in_channels: int = 3, out_dim: int = 128):
        super().__init__()
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ConvBlock(128, 128, kernel_size=3, stride=2),
        )
        
        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(128, out_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, N, C, H, W) where N is num cameras
        Returns:
            features: (B, N, out_dim)
        """
        B, N, C, H, W = images.shape
        
        # Process each camera
        images = images.view(B * N, C, H, W)
        features = self.backbone(images)
        features = self.pool(features).squeeze(-1).squeeze(-1)
        features = self.proj(features)
        
        return features.view(B, N, -1)


class LidarEncoder(nn.Module):
    """Encode LiDAR points to BEV features."""
    
    def __init__(self, in_channels: int = 2, out_dim: int = 128, bev_res: float = 0.5, bev_range: Tuple = (-50, -50, 50, 50)):
        super().__init__()
        self.bev_res = bev_res
        self.bev_range = bev_range
        
        # BEV projection backbone (input is single-channel density map)
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, kernel_size=3, stride=2),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(128, out_dim)
    
    def points_to_bev(self, points: torch.Tensor, bev_h: int, bev_w: int) -> torch.Tensor:
        """
        Project LiDAR points to BEV grid.
        
        Args:
            points: (B, N, 3) - x, y, z or (B, N, 2) - x, y
        Returns:
            bev: (B, C, bev_h, bev_w)
        """
        B, N, _ = points.shape
        x_min, y_min, x_max, y_max = self.bev_range
        
        # Normalize to [0, 1]
        x_norm = (points[..., 0] - x_min) / (x_max - x_min)
        y_norm = (points[..., 1] - y_min) / (y_max - y_min)
        
        # To grid coordinates
        x_grid = (x_norm * (bev_w - 1)).long()
        y_grid = (y_norm * (bev_h - 1)).long()
        
        # Clip to valid range
        x_grid = torch.clamp(x_grid, 0, bev_w - 1)
        y_grid = torch.clamp(y_grid, 0, bev_h - 1)
        
        # Create BEV density map
        bev = torch.zeros(B, bev_h * bev_w, device=points.device)
        indices = y_grid * bev_w + x_grid
        
        # Use scatter to accumulate (density)
        for b in range(B):
            for idx, coord in zip(indices[b], torch.arange(N, device=points.device)):
                bev[b, idx] += 1
        
        bev = bev.view(B, 1, bev_h, bev_w)
        
        # Apply log to compress dynamic range
        bev = torch.log1p(bev)
        
        return bev
    
    def forward(self, lidar_points: torch.Tensor, bev_h: int = 200, bev_w: int = 200) -> torch.Tensor:
        """
        Args:
            lidar_points: (B, N, 3) or (B, N, 2)
        Returns:
            features: (B, out_dim)
        """
        # Convert to BEV
        bev = self.points_to_bev(lidar_points, bev_h, bev_w)
        
        # Encode
        features = self.backbone(bev)
        features = self.pool(features).squeeze(-1).squeeze(-1)
        features = self.proj(features)
        
        return features


class CrossViewAttention(nn.Module):
    """Cross-view attention to fuse camera and LiDAR features."""
    
    def __init__(self, camera_dim: int, lidar_dim: int, fused_dim: int):
        super().__init__()
        
        self.camera_proj = nn.Linear(camera_dim, fused_dim)
        self.lidar_proj = nn.Linear(lidar_dim, fused_dim)
        
        self.attention = nn.MultiheadAttention(fused_dim, num_heads=4, batch_first=True)
        
        self.output_norm = nn.LayerNorm(fused_dim)
    
    def forward(self, camera_features: torch.Tensor, lidar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_features: (B, N, camera_dim)
            lidar_features: (B, lidar_dim)
        Returns:
            fused: (B, fused_dim)
        """
        # Project to common space
        camera_proj = self.camera_proj(camera_features)  # (B, N, fused_dim)
        lidar_proj = self.lidar_proj(lidar_features)  # (B, fused_dim)
        lidar_expanded = lidar_proj.unsqueeze(1).expand(-1, camera_features.size(1), -1)  # (B, N, fused_dim)
        
        # Attention: query=lidar, keys/values=cameras
        fused, _ = self.attention(lidar_expanded, camera_proj, camera_proj)
        
        # Pool across cameras
        fused = fused.mean(dim=1)  # (B, fused_dim)
        
        return self.output_norm(fused)


class BEVEncoder(nn.Module):
    """
    Complete BEV encoder that fuses camera and LiDAR into unified representation.
    """
    
    def __init__(self, config: Optional[BEVConfig] = None):
        super().__init__()
        self.config = config or BEVConfig()
        
        # Camera encoder
        self.camera_encoder = CameraEncoder(
            in_channels=3,
            out_dim=self.config.encoder_dim
        )
        
        # LiDAR encoder
        self.lidar_encoder = LidarEncoder(
            in_channels=self.config.lidar_channels,
            out_dim=self.config.encoder_dim,
            bev_res=self.config.bev_resolution,
            bev_range=self.config.bev_range
        )
        
        # Fusion module
        if self.config.fusion_method == "attention":
            self.fusion = CrossViewAttention(
                self.config.encoder_dim,
                self.config.encoder_dim,
                self.config.bev_channels
            )
        elif self.config.fusion_method == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(self.config.encoder_dim * 2, self.config.bev_channels),
                nn.ReLU(),
                nn.Linear(self.config.bev_channels, self.config.bev_channels)
            )
        else:  # add
            self.fusion = nn.Linear(self.config.encoder_dim, self.config.bev_channels)
        
        # Output projection
        self.output_proj = nn.Linear(self.config.bev_channels, self.config.encoder_dim)
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        lidar_points: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, N, C, H, W) multi-camera images
            lidar_points: (B, N, 3) or (B, N, 2) LiDAR points
        Returns:
            Dictionary with:
                - 'bev_features': (B, encoder_dim) fused BEV representation
                - 'camera_features': (B, N, encoder_dim) if images provided
                - 'lidar_features': (B, encoder_dim) if lidar_points provided
        """
        outputs = {}
        
        # Encode cameras
        if images is not None:
            camera_features = self.camera_encoder(images)
            outputs['camera_features'] = camera_features
        
        # Encode LiDAR
        if lidar_points is not None:
            lidar_features = self.lidar_encoder(lidar_points)
            outputs['lidar_features'] = lidar_features
        
        # Fuse
        if images is not None and lidar_points is not None:
            if self.config.fusion_method == "attention":
                bev_features = self.fusion(camera_features, lidar_features)
            elif self.config.fusion_method == "concat":
                camera_pooled = camera_features.mean(dim=1)  # (B, encoder_dim)
                fused = torch.cat([camera_pooled, lidar_features], dim=-1)
                bev_features = self.fusion(fused)
            else:  # add
                camera_pooled = camera_features.mean(dim=1)
                bev_features = self.fusion(camera_pooled + lidar_features)
            
            outputs['bev_features'] = self.output_proj(bev_features)
        
        return outputs


def create_bev_encoder(config: Optional[BEVConfig] = None) -> BEVEncoder:
    """Factory function to create BEV encoder."""
    return BEVEncoder(config)


# Testing
if __name__ == "__main__":
    # Test BEV encoder
    config = BEVConfig(
        encoder_dim=128,
        bev_channels=64,
        fusion_method="concat"
    )
    
    encoder = create_bev_encoder(config)
    
    # Dummy inputs
    B, N, C, H, W = 2, 6, 3, 224, 224
    images = torch.randn(B, N, C, H, W)
    
    # LiDAR: (B, num_points, 2)
    num_points = 1000
    lidar = torch.randn(B, num_points, 2) * 50
    
    with torch.no_grad():
        outputs = encoder(images=images, lidar_points=lidar)
    
    print(f"Camera features shape: {outputs['camera_features'].shape}")
    print(f"LiDAR features shape: {outputs['lidar_features'].shape}")
    print(f"BEV features shape: {outputs['bev_features'].shape}")
    print("✓ BEV encoder test passed")

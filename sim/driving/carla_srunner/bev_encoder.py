"""
BEV (Bird's Eye View) Encoder Module

Unified encoder that combines camera and LiDAR inputs into a BEV representation
for the waypoint BC model.

Architecture:
- Camera Images → CameraToBEV → Camera BEV Features
- LiDAR Points → LidarToBEV → LiDAR BEV Features
- Feature Fusion → BEV Features [B, feature_dim, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum


class FusionType(str, Enum):
    """BEV feature fusion strategies."""
    CONCAT = "concat"
    ATTENTION = "attention"
    SUM = "sum"


@dataclass
class BEVEncoderConfig:
    """Configuration for BEV encoder."""
    input_types: List[str]  # ["camera", "lidar"]
    bev_resolution: float = 0.5  # meters per pixel
    bev_size: Tuple[int, int] = (200, 200)  # (height, width) in pixels
    feature_dim: int = 256  # output feature dimension
    camera_fov: float = 90.0  # camera field of view in degrees
    lidar_range: float = 50.0  # LiDAR max range in meters
    num_height_bins: int = 4  # number of height bins for LiDAR
    fusion_type: FusionType = FusionType.CONCAT
    use_intensity: bool = True


class LidarToBEV(nn.Module):
    """
    Convert LiDAR point clouds to BEV grid representation.
    
    Processes 3D point clouds into a 2D BEV image with:
    - Height bin encoding (multiple elevation slices)
    - Intensity channel (reflectance)
    - Density/occupancy channel
    """
    
    def __init__(
        self,
        bev_resolution: float = 0.5,
        bev_size: Tuple[int, int] = (200, 200),
        num_height_bins: int = 4,
        lidar_range: float = 50.0,
        use_intensity: bool = True,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.bev_resolution = bev_resolution
        self.bev_height, self.bev_width = bev_size
        self.num_height_bins = num_height_bins
        self.lidar_range = lidar_range
        self.use_intensity = use_intensity
        
        # BEV grid limits
        self.x_min = -lidar_range
        self.x_max = lidar_range
        self.y_min = -lidar_range
        self.y_max = lidar_range
        
        # Height bins (in meters, relative to ground)
        self.height_bins = torch.linspace(0, 5.0, num_height_bins + 1)
        
        # Projection to target feature dimension
        num_raw_channels = 2 if use_intensity else 1
        self.lidar_projection = nn.Sequential(
            nn.Conv2d(num_height_bins * num_raw_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, 1),
        )
    
    def points_to_voxel_indices(
        self, 
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert 3D points to voxel grid indices.
        
        Args:
            points: [N, 3] or [N, 4] (x, y, z, intensity)
            
        Returns:
            indices: [N, 3] voxel indices (batch, height_bin, y, x)
            features: [N, num_features] density + optional intensity
            mask: [N] valid points mask
        """
        xyz = points[..., :3]
        intensity = points[..., 3:] if points.shape[-1] == 4 else None
        
        # Filter points within range
        mask = (
            (xyz[:, 0] >= self.x_min) & (xyz[:, 0] <= self.x_max) &
            (xyz[:, 1] >= self.y_min) & (xyz[:, 1] <= self.y_max) &
            (xyz[:, 2] >= 0) & (xyz[:, 2] <= 5.0)
        )
        
        # Compute BEV pixel indices
        x_idx = ((xyz[:, 0] - self.x_min) / self.bev_resolution).long()
        y_idx = ((xyz[:, 1] - self.y_min) / self.bev_resolution).long()
        
        # Compute height bin indices
        z = xyz[:, 2]
        height_bin_idx = torch.bucketize(z, self.height_bins[1:-1]).long()
        height_bin_idx = torch.clamp(height_bin_idx, 0, self.num_height_bins - 1)
        
        # Build density feature
        density = torch.ones(xyz.shape[0], 1, device=xyz.device)
        
        # Build intensity feature if available
        if intensity is not None and self.use_intensity:
            features = torch.cat([density, intensity], dim=-1)
        else:
            features = density
            
        return height_bin_idx, x_idx, y_idx, features, mask
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert LiDAR points to BEV representation.
        
        Args:
            points: [B, N, 3] or [B, N, 4] point clouds
            
        Returns:
            bev: [B, feature_dim, H, W] BEV feature grid
        """
        B = points.shape[0]
        num_raw_channels = 2 if self.use_intensity else 1
        
        # Initialize BEV grid with zeros
        bev = torch.zeros(
            B, 
            self.num_height_bins * num_raw_channels, 
            self.bev_height, 
            self.bev_width,
            device=points.device
        )
        
        for b in range(B):
            pts = points[b]  # [N, 3] or [N, 4]
            if pts.shape[0] == 0:
                continue
                
            height_idx, x_idx, y_idx, features, mask = self.points_to_voxel_indices(pts)
            
            # Filter valid points
            height_idx = height_idx[mask]
            x_idx = x_idx[mask]
            y_idx = y_idx[mask]
            features = features[mask]
            
            if features.shape[0] == 0:
                continue
            
            # Clamp indices
            x_idx = torch.clamp(x_idx, 0, self.bev_width - 1)
            y_idx = torch.clamp(y_idx, 0, self.bev_height - 1)
            
            # Accumulate features using max pooling across height bins
            feat_dim = features.shape[1] if len(features.shape) > 1 else 1
            for ch in range(min(num_raw_channels, feat_dim)):
                if len(features.shape) > 1:
                    feat_values = features[:, ch]  # [N]
                else:
                    feat_values = features  # [N]
                
                # For each height bin, take max of points in that bin
                for hb in range(self.num_height_bins):
                    hb_mask = height_idx == hb
                    if hb_mask.any():
                        hb_x = x_idx[hb_mask]
                        hb_y = y_idx[hb_mask]
                        hb_vals = feat_values[hb_mask]
                        
                        # Max pooling within each pixel
                        for i in range(len(hb_vals)):
                            ch_idx = hb * num_raw_channels + ch
                            bev[b, ch_idx, hb_y[i], hb_x[i]] = max(
                                bev[b, ch_idx, hb_y[i], hb_x[i]], 
                                hb_vals[i]
                            )
        
        # Apply log transform to density for better visualization
        bev = torch.log1p(bev)
        
        # Project to target feature dimension
        bev = self.lidar_projection(bev)
        
        return bev


class CameraToBEV(nn.Module):
    """
    Transform camera features to BEV via perspective projection.
    
    Uses learned depth estimation or geometric projection to map
    image features to the BEV plane.
    """
    
    def __init__(
        self,
        camera_channels: int = 3,
        feature_dim: int = 256,
        bev_size: Tuple[int, int] = (200, 200),
        bev_resolution: float = 0.5,
        camera_fov: float = 90.0,
    ):
        super().__init__()
        self.camera_channels = camera_channels
        self.feature_dim = feature_dim
        self.bev_height, self.bev_width = bev_size
        self.bev_resolution = bev_resolution
        self.camera_fov = camera_fov
        
        # Camera intrinsic parameters (can be overridden)
        self.focal_length = bev_size[1] / (2 * torch.tan(torch.tensor(camera_fov * torch.pi / 360)))
        
        # Feature projection head
        self.depth_head = nn.Sequential(
            nn.Conv2d(camera_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),  # Depth prediction
        )
        
        # Feature transformation head
        self.feature_head = nn.Sequential(
            nn.Conv2d(camera_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, 3, padding=1),
        )
        
        # Also add a projection layer for arbitrary input channels
        self.input_proj = nn.Conv2d(camera_channels, camera_channels, 1) if camera_channels != 3 else nn.Identity()
    
    def perspective_transform(
        self, 
        features: torch.Tensor, 
        depth: torch.Tensor
    ) -> torch.Tensor:
        """
        Project image features to BEV via perspective transform.
        
        Args:
            features: [B, C, H, W] image features
            depth: [B, 1, H, W] depth prediction
            
        Returns:
            bev: [B, C, bev_h, bev_w] BEV features
        """
        B, C, H, W = features.shape
        
        # Create mesh grid for image
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=features.device),
            torch.arange(W, device=features.device),
            indexing='ij'
        )
        
        # Normalize coordinates
        x_norm = (2 * x_grid / (W - 1) - 1)
        y_norm = (2 * y_grid / (H - 1) - 1)
        
        # Compute rays (direction from camera)
        # Assuming camera is at height 1.5m looking forward
        camera_height = 1.5
        
        # Simple projection: map image to ground plane
        # This is a simplified version - real implementation would use
        # camera intrinsics and extrinsics
        
        # Create output BEV grid
        bev = torch.zeros(
            B, C, self.bev_height, self.bev_width,
            device=features.device
        )
        
        # Sample from features based on depth
        # For each BEV pixel, find corresponding image pixel
        bev_h_range = torch.linspace(-50, 50, self.bev_height, device=features.device)
        bev_w_range = torch.linspace(-50, 50, self.bev_width, device=features.device)
        
        for bi in range(B):
            for yi, y_dist in enumerate(bev_h_range):
                for xi, x_dist in enumerate(bev_w_range):
                    # Simple geometric projection
                    if y_dist > 0.1:  # Forward of camera
                        depth_val = y_dist / self.focal_length * 100
                        u = int(W / 2 + x_dist / depth_val * 50)
                        v = int(H / 2 - camera_height / depth_val * 50)
                        
                        if 0 <= u < W and 0 <= v < H:
                            bev[bi, :, yi, xi] = features[bi, :, v, u]
        
        return bev
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert camera images to BEV features.
        
        Args:
            images: [B, C, H, W] camera images or features
            
        Returns:
            bev: [B, feature_dim, bev_h, bev_w] BEV features
        """
        # Project input to standard channels
        images = self.input_proj(images)
        
        # Generate depth prediction
        depth = torch.sigmoid(self.depth_head(images))
        
        # Get image features
        features = self.feature_head(images)
        
        # Project to BEV
        bev = self.perspective_transform(features, depth)
        
        return bev


class AttentionFusion(nn.Module):
    """Attention-based fusion for multi-modal BEV features."""
    
    def __init__(self, feature_dim: int, num_modalities: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Cross-modal attention
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.scale = feature_dim ** 0.5
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple BEV features using cross-attention.
        
        Args:
            features_list: List of [B, C, H, W] tensors
            
        Returns:
            fused: [B, C, H, W] fused features
        """
        B, C, H, W = features_list[0].shape
        
        # Flatten spatial dimensions
        flat_features = []
        for feat in features_list:
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            flat_features.append(feat_flat)
        
        # Stack modalities: [B, H*W*num_mod, C]
        stacked = torch.cat(flat_features, dim=1)
        
        # Apply attention
        Q = self.query_proj(stacked)
        K = self.key_proj(stacked)
        V = self.value_proj(stacked)
        
        attn = torch.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        out = attn @ V
        
        # Project and reshape
        out = self.out_proj(out)
        
        # Average across modalities
        out = out.reshape(B, self.num_modalities, H * W, C)
        out = out.mean(dim=1)
        
        # Reshape to spatial format
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        
        return out


class BEVEncoder(nn.Module):
    """
    Unified BEV encoder combining camera + LiDAR inputs.
    
    This encoder bridges perception (camera + LiDAR) to the waypoint BC model
    by producing a unified BEV feature representation.
    """
    
    def __init__(self, config: BEVEncoderConfig):
        super().__init__()
        self.config = config
        
        # Initialize encoders based on input types
        self.encoders = nn.ModuleDict()
        
        if "camera" in config.input_types:
            # Determine camera channels - default to 3 for RGB, can be overridden
            camera_channels = getattr(config, 'camera_channels', 3)
            self.encoders["camera"] = CameraToBEV(
                camera_channels=camera_channels,
                feature_dim=config.feature_dim,
                bev_size=config.bev_size,
                bev_resolution=config.bev_resolution,
                camera_fov=config.camera_fov,
            )
        
        if "lidar" in config.input_types:
            self.encoders["lidar"] = LidarToBEV(
                bev_resolution=config.bev_resolution,
                bev_size=config.bev_size,
                num_height_bins=config.num_height_bins,
                lidar_range=config.lidar_range,
                use_intensity=config.use_intensity,
                feature_dim=config.feature_dim,
            )
        
        # Fusion module
        num_modalities = len(config.input_types)
        if config.fusion_type == FusionType.ATTENTION:
            self.fusion = AttentionFusion(config.feature_dim, num_modalities)
        elif config.fusion_type == FusionType.SUM:
            self.fusion = None  # Handle in forward
        else:  # CONCAT
            self.fusion = nn.Conv2d(
                config.feature_dim * num_modalities,
                config.feature_dim,
                1
            )
    
    def encode(
        self,
        cameras: Optional[List[torch.Tensor]] = None,
        lidar_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode multi-view sensors to BEV.
        
        Args:
            cameras: List of [B, C, H, W] camera images/features
            lidar_points: [B, N, 3] or [B, N, 4] LiDAR point clouds
            
        Returns:
            bev_features: [B, feature_dim, H, W] BEV features
        """
        features_list = []
        
        # Encode camera
        if cameras is not None and "camera" in self.config.input_types:
            if isinstance(cameras, list):
                # Multi-view: encode each and combine
                cam_features = [self.encoders["camera"](cam) for cam in cameras]
                cam_bev = torch.cat(cam_features, dim=1)
            else:
                cam_bev = self.encoders["camera"](cameras)
            features_list.append(cam_bev)
        
        # Encode LiDAR
        if lidar_points is not None and "lidar" in self.config.input_types:
            lidar_bev = self.encoders["lidar"](lidar_points)
            features_list.append(lidar_bev)
        
        # Fuse features
        if len(features_list) == 1:
            return features_list[0]
        
        if self.config.fusion_type == FusionType.CONCAT:
            concatenated = torch.cat(features_list, dim=1)
            return self.fusion(concatenated)
        elif self.config.fusion_type == FusionType.SUM:
            return torch.stack(features_list, dim=0).sum(dim=0)
        else:  # ATTENTION
            return self.fusion(features_list)
    
    def forward(self, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self.encode(**kwargs)
    
    def get_bev_image(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Get visualization-friendly BEV image.
        
        Args:
            bev_features: [B, C, H, W] BEV features
            
        Returns:
            image: [B, 3, H, W] RGB visualization
        """
        # Normalize features to [0, 1]
        bev_norm = (bev_features - bev_features.min()) / (bev_features.max() - bev_features.min() + 1e-8)
        
        # Take first 3 channels or repeat
        if bev_norm.shape[1] >= 3:
            return bev_norm[:, :3, :, :]
        else:
            # Repeat channels to get RGB
            return bev_norm.repeat(1, 3, 1, 1)[:, :3, :, :]


def create_bev_encoder(
    input_types: List[str] = ["camera", "lidar"],
    bev_resolution: float = 0.5,
    bev_size: Tuple[int, int] = (200, 200),
    feature_dim: int = 256,
    fusion_type: str = "concat",
    **kwargs
) -> BEVEncoder:
    """
    Factory function to create a BEV encoder.
    
    Args:
        input_types: List of input modalities ["camera", "lidar"]
        bev_resolution: Resolution in meters per pixel
        bev_size: (height, width) in pixels
        feature_dim: Output feature dimension
        fusion_type: "concat", "attention", or "sum"
        **kwargs: Additional config options
        
    Returns:
        BEVEncoder instance
    """
    config = BEVEncoderConfig(
        input_types=input_types,
        bev_resolution=bev_resolution,
        bev_size=bev_size,
        feature_dim=feature_dim,
        fusion_type=FusionType(fusion_type),
        **kwargs
    )
    return BEVEncoder(config)


# Example usage
if __name__ == "__main__":
    # Create encoder
    encoder = create_bev_encoder(
        input_types=["camera", "lidar"],
        bev_resolution=0.5,
        bev_size=(200, 200),
        feature_dim=256
    )
    
    # Example inputs
    B, C, H, W = 2, 3, 224, 224
    camera_images = torch.randn(B, C, H, W)
    lidar_points = torch.randn(B, 10000, 3)
    
    # Encode
    bev_features = encoder.encode(cameras=camera_images, lidar_points=lidar_points)
    print(f"BEV features shape: {bev_features.shape}")
    
    # Get visualization
    bev_image = encoder.get_bev_image(bev_features)
    print(f"BEV image shape: {bev_image.shape}")

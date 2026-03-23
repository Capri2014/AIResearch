"""BEV-specific augmentations for Waymo SSL pretraining.

This module provides augmentations specifically designed for Bird's Eye View (BEV)
representations in autonomous driving. These augmentations complement image-based
augmentations and help learn robust spatial representations.

Key augmentations:
- Spatial: random cropping, flipping, rotation, scaling
- Occlusion: random masking, dropout
- Temporal: frame dropping, speed perturbation
- Multi-modal: camera-LiDAR consistency augmentation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class BEVAugmentationConfig:
    """Configuration for BEV augmentations."""
    # Spatial transforms
    random_crop: bool = True
    crop_scale: Tuple[float, float] = (0.7, 1.0)
    horizontal_flip: bool = True
    random_rotation: bool = True
    max_rotation_deg: float = 15.0
    random_scale: bool = True
    scale_range: Tuple[float, float] = (0.9, 1.1)
    
    # Occlusion/dropout
    random_mask: bool = True
    mask_prob: float = 0.3
    mask_ratio: float = 0.15
    random_dropout: bool = True
    dropout_prob: float = 0.1
    dropout_ratio: float = 0.2
    
    # Temporal augmentations
    temporal_drop: bool = True
    temporal_drop_prob: float = 0.1
    
    # Noise
    add_gaussian_noise: bool = True
    noise_std: float = 0.01
    add_uniform_noise: bool = False
    uniform_noise_range: float = 0.02
    
    # BEV grid specifc
    grid_size: Tuple[int, int] = (200, 200)  # default BEV grid
    resolution: float = 0.1  # meters per cell


class BEVSpatialAugmentation:
    """Spatial augmentations for BEV features."""
    
    def __init__(self, config: BEVAugmentationConfig):
        self.config = config
    
    def __call__(self, bev: torch.Tensor) -> torch.Tensor:
        """Apply spatial augmentations to BEV tensor.
        
        Args:
            bev: BEV tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            Augmented BEV tensor
        """
        if bev.dim() == 3:
            bev = bev.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, channels, height, width = bev.shape
        device = bev.device
        
        # Random horizontal flip
        if self.config.horizontal_flip and random.random() < 0.5:
            bev = torch.flip(bev, dims=[3])  # flip along W axis
        
        # Random rotation
        if self.config.random_rotation and random.random() < 0.5:
            angle_deg = random.uniform(-self.config.max_rotation_deg, 
                                       self.config.max_rotation_deg)
            angle_rad = angle_deg * np.pi / 180.0
            
            # Create rotation matrix
            cos_a = torch.cos(torch.tensor(angle_rad, device=device))
            sin_a = torch.sin(torch.tensor(angle_rad, device=device))
            
            # Apply rotation using grid_sample
            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ], device=device, dtype=bev.dtype)
            theta = theta.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Create grid
            grid = F.affine_grid(theta, bev.shape, align_corners=False)
            bev = F.grid_sample(bev, grid, mode='bilinear', 
                              padding_mode='zeros', align_corners=False)
        
        # Random scale
        if self.config.random_scale and random.random() < 0.5:
            scale = random.uniform(self.config.scale_range[0],
                                  self.config.scale_range[1])
            # Scale around center
            theta = torch.tensor([
                [scale, 0, 0],
                [0, scale, 0]
            ], device=device, dtype=bev.dtype)
            theta = theta.unsqueeze(0).expand(batch_size, -1, -1)
            
            grid = F.affine_grid(theta, bev.shape, align_corners=False)
            bev = F.grid_sample(bev, grid, mode='bilinear',
                              padding_mode='zeros', align_corners=False)
        
        # Random crop (using adaptive pooling to simulate)
        if self.config.random_crop and random.random() < 0.5:
            crop_scale = random.uniform(self.config.crop_scale[0],
                                       self.config.crop_scale[1])
            new_h = int(height * crop_scale)
            new_w = int(width * crop_scale)
            
            # Random crop position
            top = random.randint(0, height - new_h)
            left = random.randint(0, width - new_w)
            
            # Crop and resize back
            bev = bev[:, :, top:top+new_h, left:left+new_w]
            bev = F.interpolate(bev, size=(height, width), 
                               mode='bilinear', align_corners=False)
        
        if squeeze_output:
            bev = bev.squeeze(0)
        
        return bev


class BEVOcclusionAugmentation:
    """Occlusion and dropout augmentations for BEV features."""
    
    def __init__(self, config: BEVAugmentationConfig):
        self.config = config
    
    def __call__(self, bev: torch.Tensor) -> torch.Tensor:
        """Apply occlusion augmentations to BEV tensor.
        
        Args:
            bev: BEV tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            Augmented BEV tensor
        """
        bev = bev.clone()
        
        if bev.dim() == 3:
            squeeze_output = True
            bev = bev.unsqueeze(0)
        else:
            squeeze_output = False
        
        batch_size, channels, height, width = bev.shape
        
        # Random spatial masking
        if self.config.random_mask and random.random() < self.config.mask_prob:
            mask_h = int(height * self.config.mask_ratio)
            mask_w = int(width * self.config.mask_ratio)
            
            for _ in range(batch_size):
                top = random.randint(0, height - mask_h)
                left = random.randint(0, width - mask_w)
                bev[_, :, top:top+mask_h, left:left+mask_w] = 0
        
        # Random channel dropout
        if self.config.random_dropout and random.random() < self.config.dropout_prob:
            num_drop = int(channels * self.config.dropout_ratio)
            drop_channels = random.sample(range(channels), 
                                          min(num_drop, channels))
            bev[:, drop_channels, :, :] = 0
        
        if squeeze_output:
            bev = bev.squeeze(0)
        
        return bev


class BEVNoiseAugmentation:
    """Noise augmentations for BEV features."""
    
    def __init__(self, config: BEVAugmentationConfig):
        self.config = config
    
    def __call__(self, bev: torch.Tensor) -> torch.Tensor:
        """Apply noise augmentations to BEV tensor.
        
        Args:
            bev: BEV tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            Augmented BEV tensor
        """
        device = bev.device
        
        # Gaussian noise
        if self.config.add_gaussian_noise and random.random() < 0.5:
            noise = torch.randn_like(bev) * self.config.noise_std
            bev = bev + noise
        
        # Uniform noise
        if self.config.add_uniform_noise and random.random() < 0.3:
            noise = torch.rand_like(bev) * self.config.uniform_noise_range * 2 \
                    - self.config.uniform_noise_range
            bev = bev + noise
        
        return bev


class BEVTemporalAugmentation:
    """Temporal augmentations for BEV sequences."""
    
    def __init__(self, config: BEVAugmentationConfig):
        self.config = config
    
    def __call__(self, bev_sequence: torch.Tensor) -> torch.Tensor:
        """Apply temporal augmentations to BEV sequence.
        
        Args:
            bev_sequence: BEV tensor of shape (T, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Augmented BEV sequence
        """
        if bev_sequence.dim() == 4:
            squeeze_output = True
            bev_sequence = bev_sequence.unsqueeze(0)
        else:
            squeeze_output = False
        
        batch_size, time_steps, channels, height, width = bev_sequence.shape
        
        # Random temporal frame dropping
        if self.config.temporal_drop and random.random() < self.config.temporal_drop_prob:
            # Zero out random frames (simulating missing data)
            drop_idx = random.randint(0, time_steps - 1)
            bev_sequence[:, drop_idx] = 0
        
        if squeeze_output:
            bev_sequence = bev_sequence.squeeze(0)
        
        return bev_sequence


class BEVAugmentation:
    """Combined BEV augmentation pipeline."""
    
    def __init__(
        self,
        config: Optional[BEVAugmentationConfig] = None,
        is_training: bool = True,
    ):
        if config is None:
            config = BEVAugmentationConfig()
        
        self.config = config
        self.is_training = is_training
        
        if is_training:
            self.spatial_aug = BEVSpatialAugmentation(config)
            self.occlusion_aug = BEVOcclusionAugmentation(config)
            self.noise_aug = BEVNoiseAugmentation(config)
    
    def __call__(
        self,
        bev: torch.Tensor,
        temporal_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply full augmentation pipeline.
        
        Args:
            bev: Current BEV tensor (C, H, W) or (B, C, H, W)
            temporal_bev: Temporal BEV for positive pair (T, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Tuple of (augmented bev, augmented temporal_bev or None)
        """
        if not self.is_training:
            return bev, temporal_bev
        
        # Apply spatial augmentations
        if bev.dim() == 3:
            bev = self.spatial_aug(bev)
        else:
            bev = self.spatial_aug(bev)
        
        # Apply occlusion augmentations
        bev = self.occlusion_aug(bev)
        
        # Apply noise augmentations
        bev = self.noise_aug(bev)
        
        # Apply temporal augmentations if provided
        if temporal_bev is not None:
            temporal_aug = BEVTemporalAugmentation(self.config)
            temporal_bev = temporal_aug(temporal_bev)
        
        return bev, temporal_bev


def build_bev_augmentation(
    config: Optional[BEVAugmentationConfig] = None,
    is_training: bool = True,
) -> BEVAugmentation:
    """Build BEV augmentation pipeline.
    
    Args:
        config: BEV augmentation configuration
        is_training: Whether to apply training augmentations
        
    Returns:
        BEVAugmentation callable
    """
    return BEVAugmentation(config=config, is_training=is_training)


def bev_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for BEV SSL dataset.
    
    Args:
        batch: List of (bev, temporal_bev) tuples
        
    Returns:
        Tuple of batched tensors
    """
    bev_list = []
    temporal_list = []
    
    for bev, temporal_bev in batch:
        bev_list.append(bev)
        temporal_list.append(temporal_bev)
    
    bev_batch = torch.stack(bev_list, dim=0)
    temporal_batch = torch.stack(temporal_list, dim=0)
    
    return bev_batch, temporal_batch


# Export all components
__all__ = [
    'BEVAugmentationConfig',
    'BEVSpatialAugmentation',
    'BEVOcclusionAugmentation',
    'BEVNoiseAugmentation',
    'BEVTemporalAugmentation',
    'BEVAugmentation',
    'build_bev_augmentation',
    'bev_collate_fn',
]

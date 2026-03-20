"""Image augmentations for Waymo SSL pretraining.

This module provides SSL-compatible augmentations for temporal contrastive learning:
- View augmentation: strong color/spatial transforms for both views
- Temporal consistency: same augmentation applied to anchor and positive

Based on best practices from MoCo, SimCLR, and BYOL.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torchvision import transforms  # type: ignore


@dataclass
class SSLAugmentationConfig:
    """Configuration for SSL augmentations."""
    # Spatial transforms
    crop_size: int = 224
    crop_scale: Tuple[float, float] = (0.2, 1.0)
    horizontal_flip: bool = True
    
    # Color transforms
    color_jitter: bool = True
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.1
    
    # Noise/blur transforms
    gaussian_blur: bool = True
    blur_prob: float = 0.5
    blur_kernel_size: int = 23
    blur_sigma: Tuple[float, float] = (0.1, 2.0)
    
    # Solarization
    solarize: bool = True
    solarize_threshold: float = 128
    solarize_prob: float = 0.2
    
    # Grayscale
    to_grayscale: bool = True
    grayscale_prob: float = 0.2


def build_ssl_augmentation(
    config: Optional[SSLAugmentationConfig] = None,
    is_training: bool = True,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Build SSL augmentation pipeline for temporal pairs.

    Creates augmentation transforms that will be applied identically to both
    anchor and positive frames (temporal consistency).

    Args:
        config: Augmentation configuration (uses defaults if None)
        is_training: Whether to apply training-time augmentations

    Returns:
        Function that takes (anchor_image, positive_image) and returns
        augmented (view1, view2) tensors
    """
    if config is None:
        config = SSLAugmentationConfig()

    if not is_training:
        # Inference: just resize and normalize
        return build_inference_transform(config.crop_size)

    # Build training augmentation pipeline
    aug_list = []

    # Random resized crop
    aug_list.append(
        transforms.RandomResizedCrop(
            config.crop_size,
            scale=config.crop_scale,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
    )

    # Random horizontal flip
    if config.horizontal_flip:
        aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # Color jitter (strong)
    if config.color_jitter:
        aug_list.append(
            transforms.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
                hue=config.hue,
            )
        )

    # Random grayscale
    if config.to_grayscale:
        aug_list.append(
            transforms.RandomGrayscale(p=config.grayscale_prob)
        )

    # Gaussian blur
    if config.gaussian_blur:
        aug_list.append(
            GaussianBlur(
                kernel_size=config.blur_kernel_size,
                sigma=config.blur_sigma,
                p=config.blur_prob,
            )
        )

    # Solarization
    if config.solarize:
        aug_list.append(
            transforms.RandomApply(
                [transforms.Lambda(lambda x: solarize(x, config.solarize_threshold))],
                p=config.solarize_prob,
            )
        )

    # Convert to tensor and normalize
    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    base_transform = transforms.Compose(aug_list)

    def apply_augmentation(
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply same augmentation to both frames for temporal consistency.

        Args:
            anchor: Anchor image tensor (C, H, W) or (H, W, C)
            positive: Positive image tensor

        Returns:
            Augmented (view1, view2) tensors
        """
        # Handle both tensor and PIL-like inputs
        # Convert to PIL-like for transforms if needed
        if isinstance(anchor, torch.Tensor):
            if anchor.dim() == 3 and anchor.shape[0] in [1, 3]:  # (C, H, W)
                anchor = anchor.permute(1, 2, 0)  # (H, W, C)
            if anchor.max() <= 1.0:
                anchor = (anchor * 255).to(torch.uint8)

        if isinstance(positive, torch.Tensor):
            if positive.dim() == 3 and positive.shape[0] in [1, 3]:
                positive = positive.permute(1, 2, 0)
            if positive.max() <= 1.0:
                positive = (positive * 255).to(torch.uint8)

        # Apply same transform to both (temporal consistency)
        view1 = base_transform(anchor)
        view2 = base_transform(positive)

        return view1, view2

    return apply_augmentation


def build_inference_transform(crop_size: int = 224) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build inference-time transform (no augmentation).

    Args:
        crop_size: Target crop size

    Returns:
        Transform function
    """
    transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def apply_transform(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            image = image.permute(1, 2, 0)
        if image.max() <= 1.0:
            image = (image * 255).to(torch.uint8)
        return transform(image)

    return apply_transform


class GaussianBlur:
    """Gaussian blur augmentation."""

    def __init__(
        self,
        kernel_size: int = 23,
        sigma: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        # Apply Gaussian blur using torchvision
        from torchvision.transforms import functional as TF  # type: ignore
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return TF.gaussian_blur(img, self.kernel_size, [sigma, sigma])


def solarize(img: torch.Tensor, threshold: float = 128) -> torch.Tensor:
    """Solarization: invert all values above threshold."""
    # For tensor images
    if isinstance(img, torch.Tensor):
        return torch.where(img > threshold, 255 - img, img)
    return img


class TemporalConsistentAugmentation:
    """Wrapper that ensures same augmentation for temporal pairs.

    This is the main class to use for SSL training with temporal pairs.
    It applies identical augmentation to anchor and positive frames
    to maintain temporal consistency.
    """

    def __init__(
        self,
        config: Optional[SSLAugmentationConfig] = None,
        is_training: bool = True,
    ):
        self.augment = build_ssl_augmentation(config, is_training)
        self.is_training = is_training

    def __call__(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporally consistent augmentation.

        Args:
            anchor: Anchor frame
            positive: Positive frame (t + delta_t)

        Returns:
            Two augmented views
        """
        return self.augment(anchor, positive)


# Factory function for common configurations
def build_moco_augmentation(crop_size: int = 224) -> Callable:
    """Build MoCo-style augmentations (strong)."""
    config = SSLAugmentationConfig(
        crop_size=crop_size,
        crop_scale=(0.2, 1.0),
        color_jitter=True,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
        gaussian_blur=True,
        blur_prob=0.5,
        solarize=True,
        solarize_prob=0.2,
    )
    return build_ssl_augmentation(config, is_training=True)


def build_simclr_augmentation(crop_size: int = 224) -> Callable:
    """Build SimCLR-style augmentations (strong color + weak spatial)."""
    config = SSLAugmentationConfig(
        crop_size=crop_size,
        crop_scale=(0.08, 1.0),  # SimCLR uses smaller min scale
        color_jitter=True,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
        gaussian_blur=False,  # SimCLR doesn't use blur by default
        solarize=False,
    )
    return build_ssl_augmentation(config, is_training=True)


def build_light_augmentation(crop_size: int = 224) -> Callable:
    """Build light augmentations for faster training."""
    config = SSLAugmentationConfig(
        crop_size=crop_size,
        crop_scale=(0.7, 1.0),  # Less aggressive crop
        color_jitter=True,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        gaussian_blur=False,
        solarize=False,
        to_grayscale=False,
    )
    return build_ssl_augmentation(config, is_training=True)

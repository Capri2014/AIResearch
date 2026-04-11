#!/usr/bin/env python3
"""
Data utilities for waypoint BC training.

Provides data augmentation transforms, collators, and preprocessing utilities
for waypoint prediction from Waymo episodes.

Usage:
    from training.sft.data_utils import (
        augment_waypoints,
        transform_image,
        collate_waypoint_batch,
        WaypointBCHybridCollator,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Tensor = Any


@dataclass
class AugmentConfig:
    """Configuration for data augmentation."""
    
    # Image augmentation
    image_flip_horizontal: bool = True
    image_flip_vertical: bool = False
    image_brightness: float = 0.1  # additive delta range
    image_contrast: float = 0.1  # multiplicative range
    image_rotate_deg: float = 5.0  # rotation range
    
    # Waypoint augmentation
    waypoint_noise_std: float = 0.05  # meters per waypoint
    waypoint_rotate: bool = True  # match image rotation
    waypoint_scale: float = 0.05  # multiplicative scale range


def augment_waypoints(
    waypoints: torch.Tensor,
    angle: float,
    scale: float = 1.0,
    translation: Tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """
    Apply geometric augmentation to waypoints.
    
    Args:
        waypoints: (N, 2) tensor of waypoints in ego frame
        angle: rotation angle in radians
        scale: multiplicative scale factor
        translation: (dx, dy) translation
    
    Returns:
        Augmented waypoints (N, 2)
    """
    if waypoints.numel() == 0:
        return waypoints
    
    # Rotation matrix
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = torch.tensor([
        [cos_a, -sin_a],
        [sin_a, cos_a],
    ], dtype=waypoints.dtype, device=waypoints.device)
    
    # Apply rotation, scale, translation
    augmented = waypoints * scale
    augmented = augmented @ rot.T
    augmented = augmented + torch.tensor(translation, device=waypoints.device)
    
    return augmented


def transform_image(
    image: torch.Tensor,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    brightness: float = 0.0,
    contrast: float = 1.0,
    rotate_deg: float = 0.0,
) -> torch.Tensor:
    """
    Apply image augmentation transforms.
    
    Args:
        image: (C, H, W) or (B, C, H, W) tensor
        flip_horizontal: whether to flip horizontally
        flip_vertical: whether to flip vertically
        brightness: additive brightness delta
        contrast: multiplicative contrast factor
        rotate_deg: rotation angle in degrees
    
    Returns:
        Transformed image
    """
    result = image.clone()
    
    # Horizontal flip (flip x coordinate -> flip along vertical axis)
    if flip_horizontal:
        result = result.flip(-1)
    
    # Vertical flip (flip y coordinate -> flip along horizontal axis)
    if flip_vertical:
        result = result.flip(-2)
    
    # Brightness
    if brightness != 0.0:
        result = result + brightness
    
    # Contrast
    if contrast != 1.0:
        mean = result.mean(dim=(-2, -1), keepdim=True)
        result = (result - mean) * contrast + mean
    
    return result.clamp_(0.0, 1.0) if result.dtype == torch.float else result


def random_augmentation(
    cfg: AugmentConfig,
    rng: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate random augmentation parameters.
    
    Args:
        cfg: Augmentation configuration
        rng: Random number generator (numpy.random if None)
    
    Returns:
        Dict of augmentation parameters for transform_image/augment_waypoints
    """
    if rng is None:
        rng = np.random
    
    params = {}
    
    # Image flips
    if cfg.image_flip_horizontal:
        params["flip_horizontal"] = rng.rand() > 0.5
    if cfg.image_flip_vertical:
        params["flip_vertical"] = rng.rand() > 0.5
    
    # Brightness
    if cfg.image_brightness > 0:
        params["brightness"] = rng.uniform(-cfg.image_brightness, cfg.image_brightness)
    
    # Contrast
    if cfg.image_contrast > 0:
        params["contrast"] = rng.uniform(1.0 - cfg.image_contrast, 1.0 + cfg.image_contrast)
    
    # Rotation (degrees -> radians)
    if cfg.image_rotate_deg > 0:
        angle_rad = rng.uniform(-cfg.image_rotate_deg, cfg.image_rotate_deg) * math.pi / 180.0
        params["rotate_deg"] = angle_rad * 180.0 / math.pi
    
    # Waypoint augmentation
    if cfg.waypoint_rotate and "rotate_deg" in params:
        params["angle"] = params["rotate_deg"] * math.pi / 180.0
    else:
        params["angle"] = 0.0
    
    if cfg.waypoint_scale > 0:
        params["scale"] = rng.uniform(1.0 - cfg.waypoint_scale, 1.0 + cfg.waypoint_scale)
    else:
        params["scale"] = 1.0
    
    return params


class WaypointBCHybridCollator:
    """
    Hybrid collator for waypoint BC that handles multiple input types.
    
    Supports:
    - Single camera (front)
    - Multi-camera (front, left, right, rear)
    - With/without speed input
    - With/without route/command input
    """
    
    def __init__(
        self,
        cameras: List[str] = None,
        include_speed: bool = True,
        include_route: bool = False,
        augment_config: Optional[AugmentConfig] = None,
    ):
        self.cameras = cameras or ["front"]
        self.include_speed = include_speed
        self.include_route = include_route
        self.augment_config = augment_config
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of waypoint BC samples.
        
        Args:
            batch: List of sample dicts with keys:
                - images: {cam: (C, H, W)} or (C, H, W)
                - waypoints: (N, 2) tensor
                - speed: float (optional)
                - route: int (optional)
        
        Returns:
            Collated batch dict
        """
        if not batch:
            return {}
        
        # Collect all keys
        result = {k: [] for k in batch[0].keys()}
        for sample in batch:
            for k, v in sample.items():
                result[k].append(v)
        
        # Stack tensors
        for key, values in result.items():
            if not values:
                continue
            
            first = values[0]
            if isinstance(first, torch.Tensor):
                try:
                    result[key] = torch.stack(values)
                except Exception:
                    pass  # Keep as list if can't stack
        
        # Apply augmentation if configured
        if self.augment_config is not None and len(batch) > 0:
            result["_aug_params"] = random_augmentation(self.augment_config)
        
        return result


def collate_waypoint_bc_simple(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Simple collator for waypoint BC that stacks everything.
    
    Args:
        batch: List of dicts with tensor values
    
    Returns:
        Collated dict with stacked tensors
    """
    if not batch:
        return {}
    
    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        
        if all(isinstance(v, torch.Tensor) for v in values):
            try:
                result[key] = torch.stack(values)
            except Exception:
                # Handle variable-size tensors
                result[key] = values
        else:
            result[key] = values
    
    return result


def compute_waypoint_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute waypoint prediction metrics.
    
    Args:
        predictions: (B, N, 2) predicted waypoints
        targets: (B, N, 2) ground truth waypoints
        valid_mask: (B, N) boolean mask for valid waypoints
    
    Returns:
        Dict with ade, fde, miss_rate metrics
    """
    if predictions.numel() == 0 or targets.numel() == 0:
        return {"ade": float("nan"), "fde": float("nan"), "miss_rate": float("nan")}
    
    # Compute L2 distances
    diff = predictions - targets  # (B, N, 2)
    distances = torch.norm(diff, dim=-1)  # (B, N)
    
    # Apply valid mask
    if valid_mask is not None:
        distances = distances * valid_mask.float()
        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            return {"ade": float("nan"), "fde": float("nan"), "miss_rate": float("nan")}
    else:
        num_valid = distances.numel()
    
    # ADE: average displacement error
    ade = distances.sum().item() / num_valid
    
    # FDE: final displacement error (last waypoint)
    fde = distances[:, -1].mean().item()
    
    # Miss rate: fraction beyond threshold (2.0m default)
    threshold = 2.0
    miss_rate = (distances[:, -1] > threshold).float().mean().item()
    
    return {
        "ade": ade,
        "fde": fde,
        "miss_rate": miss_rate,
    }


def normalize_waypoints(
    waypoints: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize waypoints to zero mean, unit variance.
    
    Args:
        waypoints: (..., N, 2) tensor
        mean: Optional precomputed mean
        std: Optional precomputed std
    
    Returns:
        Tuple of (normalized, mean, std)
    """
    if mean is None:
        mean = waypoints.mean(dim=(-2, -1), keepdim=True)
    if std is None:
        std = waypoints.std(dim=(-2, -1), keepdim=True)
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
    
    normalized = (waypoints - mean) / std
    return normalized, mean, std


def denormalize_waypoints(
    waypoints: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Denormalize waypoints from zero mean, unit variance.
    
    Args:
        waypoints: (..., N, 2) normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized waypoints
    """
    return waypoints * std + mean


# ============================================================================
# CLI for data inspection
# ============================================================================

def main() -> None:
    """CLI for inspecting waypoint BC data."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Waypoint BC data utilities")
    parser.add_argument("--glob", type=str, help="Glob for episodes")
    parser.add_argument("--output", type=Path, help="Output stats JSON")
    a = parser.parse_args()
    
    if a.glob:
        # Try to load and inspect episodes
        from pathlib import Path
        from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset
        
        ds = EpisodesWaypointBCDataset(a.glob)
        stats = {
            "num_episodes": len(ds.episodes) if hasattr(ds, "episodes") else "unknown",
            "num_frames": len(ds),
        }
        print(f"[data_utils] Loaded {stats}")
        
        if a.output:
            Path(a.output).write_text(json.dumps(stats, indent=2))
    else:
        print("[data_utils] No --glob specified, showing config defaults")
        cfg = AugmentConfig()
        print(f"[data_utils] AugmentConfig: {cfg}")
        print(random_augmentation(cfg))


if __name__ == "__main__":
    main()
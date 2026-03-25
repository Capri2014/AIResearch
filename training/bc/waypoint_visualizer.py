"""
Waypoint Prediction Visualizer for Debugging and Analysis

Provides visualization utilities for waypoint predictions:
- Overhead BEV view with waypoints
- Comparison of predicted vs target waypoints
- Per-waypoint error metrics
- BEV feature heatmaps
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class WaypointVisConfig:
    """Configuration for waypoint visualization."""
    image_size: int = 256
    waypoint_scale: float = 10.0
    num_waypoints: int = 8
    show_timestamps: bool = True
    color_pred: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_target: Tuple[int, int, int] = (255, 0, 0)  # Red
    line_width: int = 2


def waypoints_to_image(
    waypoints: torch.Tensor,
    config: Optional[WaypointVisConfig] = None
) -> np.ndarray:
    """
    Render waypoints as overhead BEV view.
    
    Args:
        waypoints: Waypoint tensor [num_waypoints, 2] or [B, num_waypoints, 2]
        config: Visualization config
        
    Returns:
        RGB image as numpy array [H, W, 3]
    """
    if config is None:
        config = WaypointVisConfig()
    
    # Handle batch
    if waypoints.dim() == 3:
        waypoints = waypoints[0]  # Take first in batch
    
    # Convert to numpy
    waypoints = waypoints.detach().cpu().numpy()
    
    # Create blank image
    image = np.ones((config.image_size, config.image_size, 3), dtype=np.uint8) * 255
    
    # Scale waypoints to image coordinates
    scale = config.image_size / config.waypoint_scale
    center = config.image_size // 2
    
    # Draw waypoints
    for i, (x, y) in enumerate(waypoints):
        px = int(center + x * scale)
        py = int(center - y * scale)  # Flip Y for BEV view
        
        # Draw point
        radius = max(3, 8 - i)  # Smaller for further waypoints
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < config.image_size and 0 <= ny < config.image_size:
                        image[ny, nx] = config.color_pred
        
        # Draw connecting line to next waypoint
        if i < len(waypoints) - 1:
            next_x, next_y = waypoints[i + 1]
            next_px = int(center + next_x * scale)
            next_py = int(center - next_y * scale)
            
            # Simple line drawing
            for t in np.linspace(0, 1, 20):
                lx = int(px + (next_px - px) * t)
                ly = int(py + (next_py - py) * t)
                if 0 <= lx < config.image_size and 0 <= ly < config.image_size:
                    image[ly, lx] = config.color_pred
    
    return image


def visualize_prediction(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    config: Optional[WaypointVisConfig] = None
) -> np.ndarray:
    """
    Compare predicted vs target waypoints side by side.
    
    Args:
        pred_waypoints: Predicted waypoints [num_waypoints, 2] or [B, num_waypoints, 2]
        target_waypoints: Target waypoints [num_waypoints, 2] or [B, num_waypoints, 2]
        config: Visualization config
        
    Returns:
        Combined visualization image
    """
    if config is None:
        config = WaypointVisConfig()
    
    # Handle batch
    if pred_waypoints.dim() == 3:
        pred_waypoints = pred_waypoints[0]
    if target_waypoints.dim() == 3:
        target_waypoints = target_waypoints[0]
    
    # Render individual images
    pred_img = waypoints_to_image(pred_waypoints, config)
    target_img = waypoints_to_image(target_waypoints, config)
    
    # Add labels
    label_h = 30
    combined = np.ones((config.image_size + label_h, config.image_size * 2 + 10, 3), dtype=np.uint8) * 255
    
    # Place images
    combined[label_h:label_h + config.image_size, :config.image_size] = pred_img
    combined[label_h:label_h + config.image_size, config.image_size + 10:] = target_img
    
    # Add text labels using numpy (simple approach)
    for i in range(label_h):
        combined[i, :] = int(255 - i * 5)  # Gradient for label background
    
    return combined


def compute_metrics(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor
) -> Dict[str, float]:
    """
    Compute error metrics for waypoint predictions.
    
    Args:
        pred_waypoints: Predicted waypoints [num_waypoints, 2] or [B, num_waypoints, 2]
        target_waypoints: Target waypoints [num_waypoints, 2] or [B, num_waypoints, 2]
        
    Returns:
        Dictionary with ADE, FDE, and per-waypoint errors
    """
    # Handle batch
    if pred_waypoints.dim() == 3:
        pred_waypoints = pred_waypoints[0]
    if target_waypoints.dim() == 3:
        target_waypoints = target_waypoints[0]
    
    # Compute Euclidean distances
    diff = pred_waypoints - target_waypoints
    distances = torch.norm(diff, dim=1)
    
    # ADE: Average Displacement Error
    ade = distances.mean().item()
    
    # FDE: Final Displacement Error (last waypoint)
    fde = distances[-1].item()
    
    # Per-waypoint errors
    per_waypoint = distances.detach().cpu().numpy().tolist()
    
    return {
        'ade': ade,
        'fde': fde,
        'per_waypoint': per_waypoint,
        'max_error': max(per_waypoint),
        'min_error': min(per_waypoint)
    }


def visualize_bev_features(
    bev_features: torch.Tensor,
    config: Optional[WaypointVisConfig] = None,
    max_channels: int = 8
) -> np.ndarray:
    """
    Visualize BEV feature activation maps as heatmap.
    
    Args:
        bev_features: BEV features [C, H, W] or [B, C, H, W]
        config: Visualization config
        max_channels: Maximum number of channels to visualize
        
    Returns:
        Heatmap visualization
    """
    if config is None:
        config = WaypointVisConfig()
    
    # Handle batch
    if bev_features.dim() == 4:
        bev_features = bev_features[0]
    
    # Take first N channels
    bev_features = bev_features[:max_channels]
    
    # Average across channels
    feature_map = bev_features.mean(0).detach().cpu().numpy()
    
    # Normalize to 0-255
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    feature_map = (feature_map * 255).astype(np.uint8)
    
    # Apply colormap (simple heatmap)
    heatmap = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)
    
    # Simple blue-red colormap
    h, w = feature_map.shape
    for i in range(h):
        for j in range(w):
            val = feature_map[i, j]
            heatmap[i, j] = (
                val,  # R
                0,    # G
                255 - val  # B
            )
    
    return heatmap


def visualize_trajectory(
    positions: torch.Tensor,
    waypoints: Optional[torch.Tensor] = None,
    config: Optional[WaypointVisConfig] = None
) -> np.ndarray:
    """
    Visualize vehicle trajectory with waypoints.
    
    Args:
        positions: Vehicle positions [T, 2]
        waypoints: Optional waypoints to overlay [num_waypoints, 2]
        config: Visualization config
        
    Returns:
        Trajectory visualization
    """
    if config is None:
        config = WaypointVisConfig()
    
    # Convert to numpy
    positions = positions.detach().cpu().numpy()
    
    # Create blank image
    image = np.ones((config.image_size, config.image_size, 3), dtype=np.uint8) * 255
    
    # Scale to image coordinates
    scale = config.image_size / config.waypoint_scale
    center = config.image_size // 2
    
    # Draw trajectory using numpy (simple line algorithm)
    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        
        px1 = int(center + x1 * scale)
        py1 = int(center - y1 * scale)
        px2 = int(center + x2 * scale)
        py2 = int(center - y2 * scale)
        
        # Simple line drawing with Bresenham-like approach
        dx = abs(px2 - px1)
        dy = abs(py2 - py1)
        steps = max(dx, dy)
        if steps == 0:
            steps = 1
        
        for t in range(steps + 1):
            t_norm = t / max(steps, 1)
            px = int(px1 + (px2 - px1) * t_norm)
            py = int(py1 + (py2 - py1) * t_norm)
            
            if 0 <= px < config.image_size and 0 <= py < config.image_size:
                # Draw thicker line
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < config.image_size and 0 <= ny < config.image_size:
                            image[ny, nx] = (100, 100, 100)
    
    # Draw current position
    if len(positions) > 0:
        px = int(center + positions[-1, 0] * scale)
        py = int(center - positions[-1, 1] * scale)
        # Draw circle manually
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dx * dx + dy * dy <= 25:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < config.image_size and 0 <= ny < config.image_size:
                        image[ny, nx] = (0, 0, 255)
    
    # Draw waypoints if provided
    if waypoints is not None:
        waypoints = waypoints.detach().cpu().numpy()
        for i, (x, y) in enumerate(waypoints):
            px = int(center + x * scale)
            py = int(center - y * scale)
            radius = max(2, 6 - i)
            # Draw filled circle manually
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < config.image_size and 0 <= ny < config.image_size:
                            image[ny, nx] = config.color_pred
    
    return image


def create_waypoint_summary(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prefix: str = ""
) -> str:
    """
    Create text summary of waypoint prediction quality.
    
    Args:
        predictions: Predicted waypoints
        targets: Target waypoints
        prefix: Prefix for output lines
        
    Returns:
        Formatted summary string
    """
    metrics = compute_metrics(predictions, targets)
    
    lines = [
        f"{prefix}ADE: {metrics['ade']:.3f}m",
        f"{prefix}FDE: {metrics['fde']:.3f}m",
        f"{prefix}Max Error: {metrics['max_error']:.3f}m",
        f"{prefix}Min Error: {metrics['min_error']:.3f}m",
    ]
    
    # Add per-waypoint errors
    lines.append(f"{prefix}Per-waypoint (m):")
    for i, err in enumerate(metrics['per_waypoint']):
        lines.append(f"  {prefix}  WP{i}: {err:.3f}")
    
    return "\n".join(lines)


class WaypointVisualizer:
    """
    Unified waypoint visualization class.
    
    Provides high-level interface for visualizing waypoint predictions
    and computing metrics.
    """
    
    def __init__(self, config: Optional[WaypointVisConfig] = None):
        if config is None:
            config = WaypointVisConfig()
        self.config = config
    
    def visualize(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor
    ) -> np.ndarray:
        """Visualize predicted vs target waypoints side by side."""
        return visualize_prediction(pred_waypoints, target_waypoints, self.config)
    
    def compute_metrics(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor
    ) -> Dict[str, float]:
        """Compute ADE, FDE, and per-waypoint errors."""
        return compute_metrics(pred_waypoints, target_waypoints)
    
    def render_waypoints(self, waypoints: torch.Tensor) -> np.ndarray:
        """Render waypoints as BEV view."""
        return waypoints_to_image(waypoints, self.config)
    
    def render_trajectory(
        self,
        positions: torch.Tensor,
        waypoints: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Render trajectory with waypoints."""
        return visualize_trajectory(positions, waypoints, self.config)
    
    def render_bev_features(self, bev_features: torch.Tensor) -> np.ndarray:
        """Render BEV feature activation heatmap."""
        return visualize_bev_features(bev_features, self.config)
    
    def summary(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
        prefix: str = ""
    ) -> str:
        """Create text summary of prediction quality."""
        return create_waypoint_summary(pred_waypoints, target_waypoints, prefix)


def create_waypoint_visualizer(
    image_size: int = 256,
    waypoint_scale: float = 10.0,
    num_waypoints: int = 8,
    **kwargs
) -> WaypointVisualizer:
    """
    Factory function to create WaypointVisualizer.
    
    Args:
        image_size: Size of rendered images
        waypoint_scale: Scale for waypoint coordinates
        num_waypoints: Number of waypoints
        **kwargs: Additional config parameters
        
    Returns:
        WaypointVisualizer instance
    """
    config = WaypointVisConfig(
        image_size=image_size,
        waypoint_scale=waypoint_scale,
        num_waypoints=num_waypoints,
        **kwargs
    )
    return WaypointVisualizer(config)


# Test function
def test_visualizer():
    """Test waypoint visualizer functionality."""
    import cv2
    
    print("Testing Waypoint Visualizer...")
    
    # Create test waypoints
    pred = torch.tensor([
        [1.0, 0.0],
        [2.0, 0.5],
        [3.0, 1.0],
        [4.0, 1.5],
        [5.0, 2.0],
        [6.0, 2.5],
        [7.0, 3.0],
        [8.0, 3.5]
    ])
    
    target = torch.tensor([
        [1.0, 0.0],
        [2.1, 0.4],
        [3.2, 0.9],
        [4.1, 1.4],
        [5.2, 1.9],
        [6.1, 2.4],
        [7.2, 2.9],
        [8.1, 3.4]
    ])
    
    # Test metrics
    metrics = compute_metrics(pred, target)
    print(f"  ADE: {metrics['ade']:.3f}m")
    print(f"  FDE: {metrics['fde']:.3f}m")
    
    # Test visualization
    config = WaypointVisConfig()
    img = waypoints_to_image(pred, config)
    print(f"  Image shape: {img.shape}")
    
    # Test trajectory
    positions = torch.randn(50, 2) * 5
    traj_img = visualize_trajectory(positions, pred, config)
    print(f"  Trajectory image shape: {traj_img.shape}")
    
    # Test summary
    summary = create_waypoint_summary(pred, target)
    print(f"  Summary:\n{summary}")
    
    print("✓ Waypoint Visualizer tests passed!")
    return True


if __name__ == "__main__":
    test_visualizer()

"""
Waypoint Visualization and Diagnostics Module.

Provides tools for visualizing waypoint predictions, BEV features,
and model attention for debugging and analysis.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class WaypointVisConfig:
    """Configuration for waypoint visualization."""
    bev_height: int = 200
    bev_width: int = 200
    bev_resolution: float = 0.1  # meters per pixel
    num_waypoints: int = 8
    waypoint_timestep: float = 0.5  # seconds between waypoints
    road_width: float = 3.5  # meters
    save_dir: str = "out/waypoint_vis"


class WaypointVisualizer:
    """
    Visualizer for waypoint predictions and BEV features.
    """
    
    def __init__(self, config: Optional[WaypointVisConfig] = None):
        self.config = config or WaypointVisConfig()
    
    def waypoints_to_image(
        self,
        waypoints: torch.Tensor,
        current_position: Tuple[float, float] = (0.0, 0.0),
        current_yaw: float = 0.0,
    ) -> np.ndarray:
        """
        Render waypoints as an overhead view image.
        
        Args:
            waypoints: [N, 2] waypoints in world coordinates (x, y)
            current_position: (x, y) of current vehicle position
            current_yaw: current heading in radians
            
        Returns:
            RGB image as numpy array [H, W, 3]
        """
        H, W = self.config.bev_height, self.config.bev_width
        resolution = self.config.bev_resolution
        image = np.zeros((H, W, 3), dtype=np.uint8)
        
        # World to image coordinate transform
        cx, cy = current_position
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        def world_to_img(x, y):
            # Translate
            dx = x - cx
            dy = y - cy
            # Rotate
            rx = dx * cos_yaw + dy * sin_yaw
            ry = -dx * sin_yaw + dy * cos_yaw
            # Scale to image coords (center at bottom-middle)
            ix = int(W // 2 + rx / resolution)
            iy = int(H // 4 + ry / resolution)
            return ix, iy
        
        # Draw road (gray)
        for i in range(W):
            for j in range(H // 2, H):
                image[j, i] = [40, 40, 40]
        
        # Draw waypoints with colors
        colors = [
            (255, 0, 0),      # Red - closest
            (255, 128, 0),    # Orange
            (255, 255, 0),    # Yellow
            (128, 255, 0),    # Lime
            (0, 255, 0),      # Green
            (0, 255, 128),    # Teal
            (0, 255, 255),    # Cyan
            (0, 128, 255),    # Light Blue - farthest
        ]
        
        for i, wp in enumerate(waypoints):
            if isinstance(wp, torch.Tensor):
                wp = wp.detach().cpu().numpy()
            wx, wy = wp[0], wp[1]
            ix, iy = world_to_img(wx, wy)
            
            # Draw waypoint marker
            if 0 <= ix < W and 0 <= iy < H:
                color = colors[i % len(colors)]
                # Draw crosshair
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if abs(dx) <= 1 or abs(dy) <= 1:
                            px, py = ix + dx, iy + dy
                            if 0 <= px < W and 0 <= py < H:
                                image[py, px] = color
        
        # Draw trajectory lines between waypoints
        prev_ix, prev_iy = None, None
        for wp in waypoints:
            if isinstance(wp, torch.Tensor):
                wp = wp.detach().cpu().numpy()
            wx, wy = wp[0], wp[1]
            ix, iy = world_to_img(wx, wy)
            
            if prev_ix is not None:
                # Draw line
                for t in np.linspace(0, 1, 20):
                    lx = int(prev_ix + (ix - prev_ix) * t)
                    ly = int(prev_iy + (iy - prev_iy) * t)
                    if 0 <= lx < W and 0 <= ly < H:
                        image[ly, lx] = [255, 255, 255]
            
            prev_ix, prev_iy = ix, iy
        
        return image
    
    def visualize_prediction(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: Optional[torch.Tensor] = None,
        current_position: Tuple[float, float] = (0.0, 0.0),
        current_yaw: float = 0.0,
    ) -> np.ndarray:
        """
        Visualize predicted waypoints vs targets.
        
        Args:
            pred_waypoints: [N, 2] predicted waypoints
            target_waypoints: [N, 2] target waypoints (optional)
            current_position: current vehicle position
            current_yaw: current heading
            
        Returns:
            RGB image
        """
        if isinstance(pred_waypoints, torch.Tensor):
            pred_waypoints = pred_waypoints.detach().cpu()
        
        # Draw predicted waypoints
        image = self.waypoints_to_image(
            pred_waypoints,
            current_position,
            current_yaw,
        )
        
        # Overlay target waypoints if provided
        if target_waypoints is not None:
            if isinstance(target_waypoints, torch.Tensor):
                target_waypoints = target_waypoints.detach().cpu()
            
            H, W = self.config.bev_height, self.config.bev_width
            resolution = self.config.bev_resolution
            cx, cy = current_position
            cos_yaw = np.cos(current_yaw)
            sin_yaw = np.sin(current_yaw)
            
            def world_to_img(x, y):
                dx = x - cx
                dy = y - cy
                rx = dx * cos_yaw + dy * sin_yaw
                ry = -dx * sin_yaw + dy * cos_yaw
                ix = int(W // 2 + rx / resolution)
                iy = int(H // 4 + ry / resolution)
                return ix, iy
            
            # Draw target waypoints as circles
            for i, wp in enumerate(target_waypoints):
                wx, wy = wp[0].item(), wp[1].item()
                ix, iy = world_to_img(wx, wy)
                
                if 0 <= ix < W and 0 <= iy < H:
                    # Draw circle
                    for dx in range(-4, 5):
                        for dy in range(-4, 5):
                            if dx*dx + dy*dy <= 16:
                                px, py = ix + dx, iy + dy
                                if 0 <= px < W and 0 <= py < H:
                                    # Purple for targets
                                    image[py, px] = [128, 0, 128]
        
        return image
    
    def compute_metrics(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
    ) -> dict:
        """
        Compute waypoint prediction metrics.
        
        Args:
            pred_waypoints: [B, N, 2] predictions
            target_waypoints: [B, N, 2] targets
            
        Returns:
            Dictionary of metrics
        """
        if isinstance(pred_waypoints, torch.Tensor):
            pred_waypoints = pred_waypoints.detach().cpu()
        if isinstance(target_waypoints, torch.Tensor):
            target_waypoints = target_waypoints.detach().cpu()
        
        # ADE (Average Displacement Error)
        ade = torch.norm(pred_waypoints - target_waypoints, dim=-1).mean().item()
        
        # FDE (Final Displacement Error) - distance to last waypoint
        fde = torch.norm(
            pred_waypoints[:, -1] - target_waypoints[:, -1],
            dim=-1
        ).mean().item()
        
        # Per-waypoint errors
        per_waypoint_errors = torch.norm(
            pred_waypoints - target_waypoints,
            dim=-1
        ).mean(dim=0).numpy().tolist()
        
        return {
            'ade': ade,
            'fde': fde,
            'per_waypoint_errors': per_waypoint_errors,
        }
    
    def visualize_bev_features(
        self,
        bev_features: torch.Tensor,
        channel_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Visualize BEV feature activation maps.
        
        Args:
            bev_features: [B, C, H, W] BEV features
            channel_indices: Which channels to visualize (default: first 3 or mean)
            
        Returns:
            RGB visualization [H, W, 3]
        """
        if isinstance(bev_features, torch.Tensor):
            bev_features = bev_features.detach().cpu()
        
        # Average across batch and channels
        if bev_features.dim() == 4:
            bev_vis = bev_features[0].mean(dim=0).numpy()
        else:
            bev_vis = bev_features.mean(dim=0).numpy()
        
        # Normalize to 0-255
        bev_vis = (bev_vis - bev_vis.min()) / (bev_vis.max() - bev_vis.min() + 1e-8)
        bev_vis = (bev_vis * 255).astype(np.uint8)
        
        # Convert to RGB (heatmap style)
        H, W = bev_vis.shape
        image = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Simple colormap: blue -> green -> red
        for i in range(H):
            for j in range(W):
                v = bev_vis[i, j] / 255.0
                if v < 0.5:
                    # Blue to green
                    g = v * 2 * 255
                    b = (1 - v * 2) * 255
                    image[i, j] = [0, int(g), int(b)]
                else:
                    # Green to red
                    r = (v - 0.5) * 2 * 255
                    g = (1 - (v - 0.5) * 2) * 255
                    image[i, j] = [int(r), int(g), 0]
        
        return image


def create_waypoint_visualizer(
    bev_height: int = 200,
    bev_width: int = 200,
    bev_resolution: float = 0.1,
    num_waypoints: int = 8,
) -> WaypointVisualizer:
    """
    Factory function to create a WaypointVisualizer.
    
    Args:
        bev_height: BEV height in pixels
        bev_width: BEV width in pixels
        bev_resolution: Resolution in meters per pixel
        num_waypoints: Number of waypoints
        
    Returns:
        WaypointVisualizer instance
    """
    config = WaypointVisConfig(
        bev_height=bev_height,
        bev_width=bev_width,
        bev_resolution=bev_resolution,
        num_waypoints=num_waypoints,
    )
    return WaypointVisualizer(config)


# Demo
if __name__ == "__main__":
    # Create visualizer
    viz = create_waypoint_visualizer()
    
    # Generate dummy waypoints
    pred_waypoints = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.1],
        [6.0, 0.2],
        [8.0, 0.3],
        [10.0, 0.4],
        [12.0, 0.5],
        [14.0, 0.6],
        [16.0, 0.7],
    ])
    
    target_waypoints = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [6.0, 0.0],
        [8.0, 0.0],
        [10.0, 0.0],
        [12.0, 0.0],
        [14.0, 0.0],
        [16.0, 0.0],
    ])
    
    # Compute metrics
    metrics = viz.compute_metrics(
        pred_waypoints.unsqueeze(0),
        target_waypoints.unsqueeze(0),
    )
    print("Metrics:", metrics)
    
    # Visualize
    image = viz.visualize_prediction(pred_waypoints, target_waypoints)
    print(f"Visualization shape: {image.shape}")
    
    # Test BEV visualization
    bev_features = torch.randn(4, 16, 200, 200)
    bev_image = viz.visualize_bev_features(bev_features)
    print(f"BEV visualization shape: {bev_image.shape}")

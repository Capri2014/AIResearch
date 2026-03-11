"""
Waypoint Visualization Module for CARLA Debugging.

Provides utilities to visualize predicted waypoints on BEV images
and in CARLA world for debugging the perception→planning→control pipeline.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# OpenCV import (lazy)
_cv2 = None

def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as __cv2
        _cv2 = __cv2
    return _cv2


@dataclass
class VisualizationConfig:
    """Configuration for waypoint visualization."""
    # Image settings
    image_width: int = 400
    image_height: int = 400
    meters_per_pixel: float = 0.5
    
    # Colors (BGR for OpenCV)
    waypoint_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    path_color: Tuple[int, int, int] = (255, 0, 0)      # Blue
    vehicle_color: Tuple[int, int, int] = (0, 0, 255)   # Red
    trajectory_color: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    
    # Visualization options
    show_waypoint_numbers: bool = True
    show_speed_predictions: bool = True
    show_heading_arrows: bool = True
    waypoint_radius: int = 4
    line_thickness: int = 2
    
    # CARLA debug settings
    debug_line_duration: float = 0.1
    debug_point_size: float = 0.3


class WaypointVisualizer:
    """
    Visualizer for waypoints in the driving pipeline.
    
    Supports both 2D image visualization and CARLA 3D world visualization.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def create_bev_image(self, 
                         waypoints: np.ndarray,
                         vehicle_position: Optional[Tuple[float, float, float]] = None,
                         vehicle_heading: float = 0.0,
                         predicted_speed: Optional[float] = None,
                         reference_path: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a BEV image with waypoints visualization.
        
        Args:
            waypoints: Array of shape (N, 2) or (N, 3) in vehicle frame.
                       If 2D: [x, y] where x=forward, y=left
            vehicle_position: (x, y, z) in meters, optional
            vehicle_heading: Vehicle heading in radians (0 = forward)
            predicted_speed: Predicted speed in m/s
            reference_path: Optional reference path to visualize (N, 2)
        
        Returns:
            BEV image as numpy array (H, W, 3) in BGR format
        """
        cv2 = _get_cv2()
        cfg = self.config
        h, w = cfg.image_height, cfg.image_width
        mpp = cfg.meters_per_pixel
        
        # Create blank image (white background)
        image = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Center of image = vehicle position
        cx, cy = w // 2, h // 2
        
        def vehicle_to_image(x: float, y: float) -> Tuple[int, int]:
            """Convert vehicle-frame coordinates to image pixels."""
            # In vehicle frame: x=forward, y=left
            # In image: y increases downward, x increases rightward
            ix = int(cx + x / mpp)
            iy = int(cy - y / mpp)
            return ix, iy
        
        # Draw reference path if provided
        if reference_path is not None and len(reference_path) > 0:
            for i in range(len(reference_path) - 1):
                p1 = vehicle_to_image(reference_path[i, 0], reference_path[i, 1])
                p2 = vehicle_to_image(reference_path[i+1, 0], reference_path[i+1, 1])
                cv2.line(image, p1, p2, cfg.path_color, cfg.line_thickness)
        
        # Draw waypoints
        if waypoints is not None and len(waypoints) > 0:
            # Ensure waypoints is 2D
            if waypoints.ndim == 1:
                waypoints = waypoints.reshape(-1, 2)
            
            # Draw path through waypoints
            for i in range(len(waypoints) - 1):
                p1 = vehicle_to_image(waypoints[i, 0], waypoints[i, 1])
                p2 = vehicle_to_image(waypoints[i+1, 0], waypoints[i+1, 1])
                cv2.line(image, p1, p2, cfg.waypoint_color, cfg.line_thickness)
            
            # Draw waypoint markers
            for i, wp in enumerate(waypoints):
                px, py = vehicle_to_image(wp[0], wp[1])
                
                # Draw circle
                cv2.circle(image, (px, py), cfg.waypoint_radius, cfg.waypoint_color, -1)
                
                # Draw number
                if cfg.show_waypoint_numbers:
                    cv2.putText(image, str(i), (px + 6, py - 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Draw heading arrow for each waypoint (if 3D)
                if len(wp) >= 3 and cfg.show_heading_arrows:
                    hx = wp[2]
                    arrow_len = 15
                    ax = int(px + arrow_len * np.cos(hx))
                    ay = int(py - arrow_len * np.sin(hx))
                    cv2.arrowedLine(image, (px, py), (ax, ay), 
                                  cfg.waypoint_color, 2, tipLength=0.3)
        
        # Draw vehicle
        cv2.circle(image, (cx, cy), cfg.waypoint_radius * 2, cfg.vehicle_color, -1)
        
        # Draw vehicle heading
        if cfg.show_heading_arrows:
            arrow_len = 20
            ax = int(cx + arrow_len * np.sin(vehicle_heading))
            ay = int(cy - arrow_len * np.cos(vehicle_heading))
            cv2.arrowedLine(image, (cx, cy), (ax, ay), 
                          cfg.vehicle_color, 2, tipLength=0.3)
        
        # Add speed text
        if predicted_speed is not None:
            cv2.putText(image, f"Speed: {predicted_speed:.1f} m/s", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image
    
    def draw_waypoints_carla(self,
                            carla_world,
                            waypoints: np.ndarray,
                            vehicle_transform,
                            color: Optional[Tuple[int, int, int]] = None,
                            duration: float = 0.0) -> None:
        """
        Draw waypoints in CARLA 3D world.
        
        Args:
            carla_world: CARLA world object
            waypoints: Array of shape (N, 2) or (N, 3) in vehicle frame
            vehicle_transform: CARLA transform of the vehicle
            color: Optional RGB color tuple (0-255)
            duration: Debug line duration in seconds
        """
        if waypoints is None or len(waypoints) == 0:
            return
        
        cfg = self.config
        color = color or cfg.waypoint_color
        
        # Get vehicle rotation
        yaw = np.radians(vehicle_transform.rotation.yaw)
        
        # Transform waypoints to world frame
        for i, wp in enumerate(waypoints):
            if wp.ndim == 1:
                wp = wp.reshape(1, -1)
            
            wx = wp[0, 0]  # forward
            wy = wp[0, 1]  # left
            
            # Rotate to world frame
            world_x = vehicle_transform.location.x + wx * np.cos(yaw) - wy * np.sin(yaw)
            world_y = vehicle_transform.location.y + wx * np.sin(yaw) + wy * np.cos(yaw)
            world_z = vehicle_transform.location.z + 0.5  # Slightly above ground
            
            # Draw point
            location = carla_world.map.get_waypoint(
                carla.Location(x=world_x, y=world_y, z=world_z)
            ).transform.location
            
            debug = carla_world.debug
            debug.draw_point(
                location,
                size=cfg.debug_point_size,
                color=carla.Color(color[2], color[1], color[0]),  # RGB -> BGR
                life_time=duration or cfg.debug_line_duration
            )
            
            # Draw line from previous waypoint
            if i > 0:
                prev_wp = waypoints[i-1]
                pw_x = prev_wp[0]
                pw_y = prev_wp[1]
                prev_world_x = vehicle_transform.location.x + pw_x * np.cos(yaw) - pw_y * np.sin(yaw)
                prev_world_y = vehicle_transform.location.y + pw_x * np.sin(yaw) + pw_y * np.cos(yaw)
                prev_z = vehicle_transform.location.z + 0.5
                
                debug.draw_line(
                    carla.Location(x=prev_world_x, y=prev_world_y, z=prev_z),
                    location,
                    thickness=0.1,
                    color=carla.Color(color[2], color[1], color[0]),
                    life_time=duration or cfg.debug_line_duration
                )
    
    def draw_trajectory_carla(self,
                             carla_world,
                             trajectory: np.ndarray,
                             color: Optional[Tuple[int, int, int]] = None,
                             duration: float = 0.0) -> None:
        """
        Draw a full trajectory in CARLA 3D world.
        
        Args:
            carla_world: CARLA world object
            trajectory: Array of shape (N, 3) - [x, y, z] in world frame
            color: Optional RGB color tuple
            duration: Debug line duration in seconds
        """
        if trajectory is None or len(trajectory) < 2:
            return
        
        cfg = self.config
        color = color or cfg.trajectory_color
        debug = carla_world.debug
        
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            
            debug.draw_line(
                carla.Location(x=p1[0], y=p1[1], z=p1[2]),
                carla.Location(x=p2[0], y=p2[1], z=p2[2]),
                thickness=0.05,
                color=carla.Color(color[2], color[1], color[0]),
                life_time=duration or cfg.debug_line_duration
            )


def create_visualizer(**kwargs) -> WaypointVisualizer:
    """Factory function to create a WaypointVisualizer."""
    config = VisualizationConfig(**kwargs)
    return WaypointVisualizer(config)


# Export public API
__all__ = [
    'WaypointVisualizer',
    'VisualizationConfig',
    'create_visualizer',
]

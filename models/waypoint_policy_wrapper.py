"""Waypoint Policy Wrapper for Camera-Based Inference.

This module provides a WaypointPolicyWrapper that bridges camera-based input
with waypoint prediction, integrating with the unified CARLA evaluator.

Pipeline: CameraSensorManager → WaypointPolicyWrapper → VehicleControl
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

import numpy as np

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required for waypoint prediction") from e
    return torch


@dataclass
class WaypointConfig:
    """Configuration for waypoint prediction."""
    # Model
    encoder_out_dim: int = 128
    waypoint_hidden_dim: int = 256
    num_waypoints: int = 20
    horizon_seconds: float = 2.0
    
    # Inference
    device: str = "cuda"  # "cuda" or "cpu"
    checkpoint: Optional[str] = None
    
    # Camera
    camera_names: List[str] = field(default_factory=lambda: ["front"])
    image_size: Tuple[int, int] = (360, 640)
    
    # Control
    speed_limit: float = 10.0  # m/s
    target_speed: float = 5.0  # m/s
    steering_tolerance: float = 1e-3


class WaypointPolicyWrapper:
    """
    Wraps encoder + waypoint head for camera-based closed-loop control.
    
    Provides act() method that:
    1. Reads latest camera frame(s)
    2. Runs encoder forward pass
    3. Predicts waypoints
    4. Converts to VehicleControl (throttle, steer, brake)
    """
    
    def __init__(self, config: WaypointConfig):
        self.config = config
        self._encoder = None
        self._waypoint_head = None
        self._device = None
        self._initialized = False
        self._latest_frame: Dict[str, np.ndarray] = {}
        self._use_combined_encoder = True
        
        # Runtime state
        self._current_speed: float = 0.0
        self._current_location: Optional[np.ndarray] = None
        self._current_rotation: Optional[np.ndarray] = None
        
    def initialize(self) -> "WaypointPolicyWrapper":
        """Initialize model components and load checkpoint."""
        if self._initialized:
            return self
            
        torch = _require_torch()
        self._device = torch.device(self.config.device)
        
        # Import encoder and waypoint head
        try:
            from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
            from models.encoders.waypoint_prediction_head import WaypointPredictionEncoder
            self._use_combined_encoder = True
        except ImportError:
            from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
            from models.encoders.waypoint_prediction_head import WaypointPredictionHead
            self._use_combined_encoder = False
        
        if self._use_combined_encoder:
            # Combined encoder + waypoint head
            self._encoder = WaypointPredictionEncoder(
                encoder_out_dim=self.config.encoder_out_dim,
                num_waypoints=self.config.num_waypoints,
            )
        else:
            # Separate encoder + head
            self._encoder = TinyMultiCamEncoder(out_dim=self.config.encoder_out_dim)
            self._waypoint_head = WaypointPredictionHead(
                encoder_out_dim=self.config.encoder_out_dim,
                num_waypoints=self.config.num_waypoints,
                hidden_dim=self.config.waypoint_hidden_dim,
            )
        
        self._encoder.to(self._device)
        self._encoder.eval()
        
        if not self._use_combined_encoder:
            self._waypoint_head.to(self._device)
            self._waypoint_head.eval()
        
        # Load checkpoint if provided
        if self.config.checkpoint:
            self._load_checkpoint(self.config.checkpoint)
        
        self._initialized = True
        logger.info(f"WaypointPolicyWrapper initialized (device={self._device})")
        return self
    
    def _load_checkpoint(self, path: str):
        """Load model weights from checkpoint."""
        torch = _require_torch()
        
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {path}, using random weights")
            return
        
        try:
            # Try to load as full checkpoint with metadata
            ckpt = torch.load(path, map_location=self._device, weights_only=False)
            
            if isinstance(ckpt, dict):
                # Checkpoint format from train_ssl_waypoint_v0.py
                if self._use_combined_encoder:
                    # Combined encoder + waypoint head
                    if "model_state" in ckpt:
                        self._encoder.load_state_dict(ckpt["model_state"])
                        logger.info("Loaded combined encoder state from checkpoint")
                    elif "encoder_state" in ckpt:
                        self._encoder.load_state_dict(ckpt["encoder_state"])
                        logger.info("Loaded encoder state from checkpoint")
                else:
                    # Separate encoder + waypoint head
                    if "encoder_state" in ckpt:
                        self._encoder.load_state_dict(ckpt["encoder_state"])
                        logger.info("Loaded encoder state from checkpoint")
                    if "waypoint_head_state" in ckpt:
                        self._waypoint_head.load_state_dict(ckpt["waypoint_head_state"])
                        logger.info("Loaded waypoint head state from checkpoint")
                    elif "model_state" in ckpt:
                        # Try to load model_state into encoder
                        self._encoder.load_state_dict(ckpt["model_state"])
                        logger.info("Loaded model state into encoder")
            else:
                # Direct state dict
                self._encoder.load_state_dict(ckpt)
                logger.info("Loaded encoder state dict")
                
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}, using random weights")
    
    def update_frame(self, camera_name: str, frame: np.ndarray):
        """Update latest frame for a camera.
        
        Args:
            camera_name: Name of camera (e.g., "front")
            frame: RGB frame as numpy array (H, W, 3), uint8
        """
        # Validate frame shape
        if frame.shape != (*self.config.image_size, 3):
            # Resize if needed
            import cv2
            frame = cv2.resize(frame, (self.config.image_size[1], self.config.image_size[0]))
        
        self._latest_frame[camera_name] = frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert RGB frame to tensor for model.
        
        Args:
            frame: (H, W, 3) RGB uint8
            
        Returns:
            tensor: (1, 3, H, W) normalized to [-1, 1]
        """
        torch = _require_torch()
        
        # Convert to float and normalize to [-1, 1]
        tensor = frame.astype(np.float32) / 255.0
        tensor = tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # HWC → CHW
        tensor = np.transpose(tensor, (2, 0, 1))
        
        # Add batch dimension
        tensor = np.expand_dims(tensor, axis=0)
        
        return torch.from_numpy(tensor).to(self._device)
    
    def predict_waypoints(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict waypoints from camera frames.
        
        Args:
            frames: Dict mapping camera name to RGB frame
            
        Returns:
            waypoints: (num_waypoints, 2) in ego coordinates (x forward, y left)
        """
        if not self._initialized:
            self.initialize()
        
        torch = _require_torch()
        
        # Build camera inputs
        images_by_cam = {}
        for cam_name in self.config.camera_names:
            frame = frames.get(cam_name)
            if frame is not None:
                images_by_cam[cam_name] = self._preprocess_frame(frame)
        
        if not images_by_cam:
            # No frames available, return stub waypoints
            return self._get_stub_waypoints()
        
        with torch.no_grad():
            if self._use_combined_encoder:
                # Combined encoder returns (embeddings, waypoints)
                embeddings, waypoints = self._encoder(images_by_cam)
            else:
                # Separate: encode first, then predict
                embeddings = self._encoder(images_by_cam)
                waypoints = self._waypoint_head(embeddings)
            
            # Convert to numpy
            waypoints = waypoints.cpu().numpy()[0]  # (num_waypoints, 2)
        
        return waypoints
    
    def _get_stub_waypoints(self) -> np.ndarray:
        """Return stub waypoints when no camera input available."""
        # Straight line ahead at constant spacing
        x = np.linspace(1.0, self.config.num_waypoints * 1.0, self.config.num_waypoints)
        y = np.zeros(self.config.num_waypoints)
        return np.stack([x, y], axis=1)
    
    def waypoints_to_control(
        self,
        waypoints: np.ndarray,
        current_speed: float,
    ) -> Tuple[float, float, float]:
        """
        Convert waypoints to vehicle control commands.
        
        Args:
            waypoints: (num_waypoints, 2) in ego coords
            current_speed: Current vehicle speed in m/s
            
        Returns:
            Tuple of (throttle, steer, brake) in [-1, 1]
        """
        if len(waypoints) == 0:
            return 0.0, 0.0, 1.0  # Brake when no waypoints
        
        # Target: first waypoint
        target = waypoints[0]
        target_distance = np.linalg.norm(target)
        target_angle = np.arctan2(target[1], target[0])
        
        # Speed control
        target_speed = self.config.target_speed
        speed_error = target_speed - current_speed
        
        if speed_error > 0:
            throttle = min(speed_error / 5.0, 1.0)  # P-controller
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-speed_error / 5.0, 1.0)
        
        # Steering control (simple P-controller)
        steer = np.clip(target_angle / 0.5, -1.0, 1.0)  # Max 0.5 rad steering
        
        return throttle, steer, brake
    
    def act(
        self,
        camera_frames: Optional[Dict[str, np.ndarray]] = None,
        vehicle_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main inference method: camera → waypoints → control.
        
        Args:
            camera_frames: Dict of camera name → RGB frame
            vehicle_state: Optional dict with current vehicle state
            
        Returns:
            Action dict with waypoints, throttle, steer, brake
        """
        # Update frames if provided
        if camera_frames:
            for name, frame in camera_frames.items():
                self.update_frame(name, frame)
        
        # Update vehicle state
        if vehicle_state:
            self._current_speed = vehicle_state.get("speed", 0.0)
            self._current_location = vehicle_state.get("location")
            self._current_rotation = vehicle_state.get("rotation")
        
        # Get latest frames
        frames = self._latest_frame.copy()
        
        # Predict waypoints
        waypoints = self.predict_waypoints(frames)
        
        # Convert to control
        throttle, steer, brake = self.waypoints_to_control(waypoints, self._current_speed)
        
        return {
            "waypoints": waypoints.tolist(),
            "throttle": float(throttle),
            "steer": float(steer),
            "brake": float(brake),
            "speed": float(self._current_speed),
        }
    
    def reset(self):
        """Reset internal state."""
        self._latest_frame.clear()
        self._current_speed = 0.0
        self._current_location = None
        self._current_rotation = None


def create_policy_wrapper(
    checkpoint: Optional[str] = None,
    device: str = "cuda",
    num_waypoints: int = 20,
) -> WaypointPolicyWrapper:
    """Factory function to create a WaypointPolicyWrapper."""
    config = WaypointConfig(
        checkpoint=checkpoint,
        device=device,
        num_waypoints=num_waypoints,
    )
    return WaypointPolicyWrapper(config).initialize()


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WaypointPolicyWrapper smoke test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda/cpu)")
    parser.add_argument("--num-waypoints", type=int, default=20, help="Number of waypoints")
    args = parser.parse_args()
    
    # Create wrapper
    wrapper = create_policy_wrapper(
        checkpoint=args.checkpoint,
        device=args.device,
        num_waypoints=args.num_waypoints,
    )
    
    # Generate dummy camera frame
    dummy_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    
    # Run inference
    result = wrapper.act(
        camera_frames={"front": dummy_frame},
        vehicle_state={"speed": 2.0},
    )
    
    print(f"Waypoints shape: {len(result['waypoints'])} waypoints")
    print(f"First waypoint: {result['waypoints'][0]}")
    print(f"Control: throttle={result['throttle']:.3f}, steer={result['steer']:.3f}, brake={result['brake']:.3f}")
    print("Smoke test PASSED")

"""
Waypoint Inference Module

Loads trained BC checkpoint and runs inference to produce waypoint predictions.
Bridges waypoint BC → waypoint controller → CARLA control.

Usage:
    from sim.driving.carla_srunner.waypoint_inference import WaypointInference
    
    inference = WaypointInference(checkpoint_path="out/waypoint_bc/run_XXXXXX/best.pt")
    waypoints = inference.predict(bev_image)  # [num_waypoints, 2] (x, y) in meters
    speed = inference.predict_speed(bev_image)  # scalar in m/s
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path
import json


@dataclass
class InferenceConfig:
    """Configuration for waypoint inference."""
    # Model architecture (must match training)
    encoder_dim: int = 256
    hidden_dim: int = 512
    num_waypoints: int = 8
    future_len: float = 4.0  # seconds
    
    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    
    # Output
    waypoint_interval: float = 0.5  # seconds between waypoints


class WaypointInference:
    """
    Waypoint inference module that loads trained BC checkpoint and produces predictions.
    """
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[InferenceConfig] = None,
        ssl_encoder_path: Optional[str] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        
        # Load checkpoint
        self._load_checkpoint(ssl_encoder_path)
        
    def _load_checkpoint(self, ssl_encoder_path: Optional[str] = None):
        """Load trained BC checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load config from checkpoint
        if "config" in checkpoint:
            saved_config = checkpoint["config"]
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Build model
        self.model = self._build_model()
        
        # Load weights - handle multiple checkpoint formats
        state_dict = None
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle minor mismatches
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded checkpoint from {self.checkpoint_path}")
        print(f"Model: {self.config.num_waypoints} waypoints, {self.config.future_len}s horizon")
        
    def _build_model(self) -> nn.Module:
        """Build the BC model architecture."""
        # Simplified model for inference (matches training architecture)
        class InferenceBCModel(nn.Module):
            def __init__(self, config: InferenceConfig):
                super().__init__()
                self.config = config
                
                # SSL Encoder (mock - in production loads real encoder)
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(128 * 16 * 16, config.encoder_dim),
                    nn.ReLU(),
                )
                
                # Waypoint head
                self.waypoint_head = nn.Sequential(
                    nn.Linear(config.encoder_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.num_waypoints * 2),  # x, y
                )
                
                # Speed head
                self.speed_head = nn.Sequential(
                    nn.Linear(config.encoder_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                )
                
            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                features = self.encoder(x)
                waypoints = self.waypoint_head(features)
                waypoints = waypoints.view(-1, self.config.num_waypoints, 2)
                speed = self.speed_head(features)
                return waypoints, speed
        
        return InferenceBCModel(self.config)
    
    @torch.no_grad()
    def predict(
        self,
        bev_image: np.ndarray,
        return_features: bool = False
    ) -> np.ndarray:
        """
        Predict waypoints from BEV image.
        
        Args:
            bev_image: [H, W, 3] or [3, H, W] numpy array, uint8 or float32
            return_features: If True, also return encoder features
            
        Returns:
            waypoints: [num_waypoints, 2] numpy array of (x, y) positions in meters
            features: [encoder_dim] (optional)
        """
        # Preprocess
        if bev_image.ndim == 3 and bev_image.shape[2] == 3:
            bev_image = bev_image.transpose(2, 0, 1)  # [3, H, W]
        
        bev_tensor = torch.from_numpy(bev_image).float()
        if bev_tensor.max() > 1.0:
            bev_tensor = bev_tensor / 255.0
        
        bev_tensor = bev_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        # Forward pass
        waypoints, speed = self.model(bev_tensor)
        
        waypoints = waypoints.squeeze(0).cpu().numpy()  # [num_waypoints, 2]
        
        if return_features:
            features = self.model.encoder(bev_tensor).squeeze(0).cpu().numpy()
            return waypoints, features
        
        return waypoints
    
    @torch.no_grad()
    def predict_speed(self, bev_image: np.ndarray) -> float:
        """
        Predict vehicle speed from BEV image.
        
        Args:
            bev_image: [H, W, 3] or [3, H, W] numpy array
            
        Returns:
            speed: float in m/s
        """
        # Preprocess
        if bev_image.ndim == 3 and bev_image.shape[2] == 3:
            bev_image = bev_image.transpose(2, 0, 1)
        
        bev_tensor = torch.from_numpy(bev_image).float()
        if bev_tensor.max() > 1.0:
            bev_tensor = bev_tensor / 255.0
        
        bev_tensor = bev_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        waypoints, speed = self.model(bev_tensor)
        
        speed = speed.squeeze().item()
        speed = max(0.0, speed)  # Clamp to non-negative
        
        return speed
    
    @torch.no_grad()
    def predict_batch(
        self,
        bev_images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict waypoints and speed for a batch of BEV images.
        
        Args:
            bev_images: List of [H, W, 3] or [3, H, W] numpy arrays
            
        Returns:
            waypoints: [batch, num_waypoints, 2] numpy array
            speeds: [batch] numpy array of speeds in m/s
        """
        # Preprocess batch
        processed = []
        for bev in bev_images:
            if bev.ndim == 3 and bev.shape[2] == 3:
                bev = bev.transpose(2, 0, 1)
            bev = torch.from_numpy(bev).float()
            if bev.max() > 1.0:
                bev = bev / 255.0
            processed.append(bev)
        
        batch = torch.stack(processed).to(self.device)
        
        # Forward pass
        waypoints, speeds = self.model(batch)
        
        waypoints = waypoints.cpu().numpy()
        speeds = speeds.squeeze(-1).cpu().numpy()
        speeds = np.maximum(0.0, speeds)  # Clamp to non-negative
        
        return waypoints, speeds
    
    def get_waypoint_times(self) -> np.ndarray:
        """
        Get the timestamps for each waypoint.
        
        Returns:
            times: [num_waypoints] array of times in seconds from now
        """
        num_wp = self.config.num_waypoints
        interval = self.config.waypoint_interval
        # E.g., [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        return np.arange(interval, (num_wp + 1) * interval, interval)
    
    def waypoints_to_carla_format(
        self,
        waypoints: np.ndarray,
        current_position: Tuple[float, float, float] = (0, 0, 0)
    ) -> List[Tuple[float, float, float]]:
        """
        Convert waypoints to CARLA transform format.
        
        Args:
            waypoints: [num_waypoints, 2] (x, y) positions in meters
            current_position: (x, y, yaw) current vehicle state
            
        Returns:
            transforms: [(x, y, z)] list, z=0 for flat plane
        """
        curr_x, curr_y, curr_yaw = current_position
        
        transforms = []
        for wx, wy in waypoints:
            # Apply rotation (simplified - assumes waypoints in vehicle frame)
            cos_yaw = np.cos(curr_yaw)
            sin_yaw = np.sin(curr_yaw)
            
            global_x = curr_x + wx * cos_yaw - wy * sin_yaw
            global_y = curr_y + wx * sin_yaw + wy * cos_yaw
            
            transforms.append((global_x, global_y, 0.0))
        
        return transforms


def create_inference(
    checkpoint_path: str,
    device: Optional[str] = None
) -> WaypointInference:
    """
    Factory function to create WaypointInference instance.
    
    Args:
        checkpoint_path: Path to trained BC checkpoint
        device: Device to run inference on ("cuda" or "cpu")
        
    Returns:
        WaypointInference instance
    """
    config = InferenceConfig()
    if device:
        config.device = device
    
    return WaypointInference(checkpoint_path=checkpoint_path, config=config)


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Waypoint Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to BC checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to BEV image (optional, generates mock if not provided)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Create inference
    inference = create_inference(args.checkpoint, device=args.device)
    
    # Generate or load test image
    if args.image:
        from PIL import Image
        bev = np.array(Image.open(args.image))
    else:
        # Mock BEV image (random)
        print("Generating mock BEV image...")
        bev = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Predict
    waypoints = inference.predict(bev)
    speed = inference.predict_speed(bev)
    
    print(f"\nPredicted waypoints (meters from vehicle):")
    print(f"  Shape: {waypoints.shape}")
    print(f"  Waypoints:\n{waypoints}")
    print(f"\nPredicted speed: {speed:.2f} m/s ({speed * 3.6:.1f} km/h)")
    
    times = inference.get_waypoint_times()
    print(f"\nWaypoint times: {times}")
    
    # Convert to CARLA format
    transforms = inference.waypoints_to_carla_format(waypoints, (0, 0, 0))
    print(f"\nCARLA transforms (from origin):")
    for i, t in enumerate(transforms):
        print(f"  WP{i}: {t}")

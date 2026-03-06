"""
Temporal Waypoint Policy Wrapper for CARLA Integration

This module provides a policy wrapper for temporal waypoint BC models,
enabling them to be evaluated in CARLA closed-loop scenarios.

Usage with ScenarioRunner:
    from sim.driving.carla_srunner.temporal_policy_wrapper import TemporalWaypointPolicyWrapper
    
    policy = TemporalWaypointPolicyWrapper(checkpoint="path/to/temporal_model.pt")
    waypoints = policy.predict_temporal(frames)  # frames: list of images
    control = policy.waypoints_to_control(waypoints)
    return control
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TemporalPolicyConfig:
    """Configuration for temporal waypoint policy."""
    checkpoint: Path
    camera_name: str = "front"
    horizon_steps: int = 20
    sequence_length: int = 4
    hidden_dim: int = 256
    num_rnn_layers: int = 2
    device: str = "auto"
    encoder_name: str = "resnet34"


class TemporalEncoder(nn.Module):
    """LSTM-based temporal encoder for waypoint prediction."""
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN encoder (pretrained)
        if encoder_name == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(backbone.children())[:-1])
            self.encoder_dim = 512
        elif encoder_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(backbone.children())[:-1])
            self.encoder_dim = 512
        elif encoder_name == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(backbone.children())[:-1])
            self.encoder_dim = 1280
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Project CNN features to hidden dim
        self.project = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM for temporal aggregation
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2 * horizon_steps),  # (x, y) for each step
        )
        
        self._initialized = False
    
    def forward(
        self,
        frames: torch.Tensor,
        sequence_length: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal encoder.
        
        Args:
            frames: (B, T, C, H, W) tensor of frame sequences
        
        Returns:
            waypoints: (B, horizon_steps, 2) predicted waypoints
            temporal_embedding: (B, hidden_dim) temporal context
        """
        B, T, C, H, W = frames.shape
        
        # Process each frame through CNN
        frames_flat = frames.view(B * T, C, H, W)
        features = self.cnn(frames_flat)  # (B*T, encoder_dim, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*T, encoder_dim)
        features = self.project(features)  # (B*T, hidden_dim)
        
        # Reshape for LSTM: (B, T, hidden_dim)
        features = features.view(B, T, -1)
        
        # LSTM temporal aggregation
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use final hidden state as temporal embedding
        temporal_embedding = h_n[-1]  # (B, hidden_dim)
        
        # Predict waypoints
        waypoint_flat = self.waypoint_head(temporal_embedding)  # (B, 2*horizon_steps)
        waypoints = waypoint_flat.view(B, -1, 2)  # (B, horizon_steps, 2)
        
        return waypoints, temporal_embedding


class TemporalWaypointPolicyWrapper:
    """Wrapper for temporal waypoint BC models in CARLA.
    
    This class provides:
    - Loading temporal waypoint BC checkpoints
    - Predicting waypoints from sequences of camera frames
    - Converting waypoints to CARLA vehicle commands
    
    The temporal model uses LSTM to aggregate context across multiple frames,
    enabling better waypoint predictions through motion understanding.
    """
    
    def __init__(self, cfg: TemporalPolicyConfig | None = None):
        cfg = cfg or TemporalPolicyConfig(checkpoint=Path("."))
        self.cfg = cfg
        self._model: Optional[TemporalEncoder] = None
        self._initialized = False
        self._device = self._get_device()
        
        # Buffer for maintaining temporal context
        self._frame_buffer: List[np.ndarray] = []
        self._max_buffer_size = cfg.sequence_length
    
    def _get_device(self) -> torch.device:
        """Determine device for model inference."""
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.cfg.device)
    
    def initialize(self) -> bool:
        """Load model from checkpoint."""
        if self._initialized:
            return True
        
        try:
            # Create model
            self._model = TemporalEncoder(
                encoder_name=self.cfg.encoder_name,
                hidden_dim=self.cfg.hidden_dim,
                num_layers=self.cfg.num_rnn_layers,
            )
            
            # Load checkpoint
            checkpoint_path = Path(self.cfg.checkpoint)
            if checkpoint_path.exists() and checkpoint_path.suffix in ['.pt', '.pth']:
                state_dict = torch.load(checkpoint_path, map_location=self._device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    self._model.load_state_dict(state_dict['model_state_dict'])
                elif 'state_dict' in state_dict:
                    self._model.load_state_dict(state_dict['state_dict'])
                else:
                    self._model.load_state_dict(state_dict)
                    
                print(f"[temporal_policy_wrapper] Loaded checkpoint: {checkpoint_path}")
            else:
                print(f"[temporal_policy_wrapper] No checkpoint found at {checkpoint_path}, using random init")
            
            self._model.to(self._device)
            self._model.eval()
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"[temporal_policy_wrapper] Failed to initialize: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def reset(self):
        """Reset frame buffer for new episode."""
        self._frame_buffer.clear()
    
    def add_frame(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Add a frame to the temporal buffer.
        
        Args:
            image: (H, W, C) numpy array in RGB format
        
        Returns:
            Current buffer of frames
        """
        # Convert to RGB if needed
        if image.shape[2] == 4:
            image = image[:, :, :3]
        
        self._frame_buffer.append(image)
        
        # Maintain max buffer size (keep most recent frames)
        if len(self._frame_buffer) > self._max_buffer_size:
            self._frame_buffer = self._frame_buffer[-self._max_buffer_size:]
        
        return self._frame_buffer
    
    def predict_temporal(self, frames: List[np.ndarray] | None = None) -> np.ndarray:
        """
        Predict waypoints from temporal sequence of frames.
        
        Args:
            frames: Optional list of frames. If None, uses internal buffer.
        
        Returns:
            waypoints: (horizon_steps, 2) numpy array in ego frame (x forward, y left)
        """
        if not self._initialized:
            raise RuntimeError("Policy not initialized. Call initialize() first.")
        
        # Use provided frames or buffer
        if frames is None:
            frames = self._frame_buffer
        
        if len(frames) == 0:
            raise ValueError("No frames available for prediction")
        
        # Pad frames if needed
        while len(frames) < self.cfg.sequence_length:
            frames.append(frames[-1])  # Pad with last frame
        
        # Convert to tensor
        frames_tensor = self._preprocess_frames(frames)
        frames_tensor = frames_tensor.to(self._device)
        
        # Get sequence length (may be less than configured if not enough frames)
        seq_len = min(len(frames), self.cfg.sequence_length)
        
        # Predict
        with torch.no_grad():
            waypoints, _ = self._model(frames_tensor, sequence_length=seq_len)
            waypoints = waypoints[0].cpu().numpy()  # (horizon_steps, 2)
        
        return waypoints
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for model input."""
        import torchvision.transforms as transforms
        
        # Standard preprocessing
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
        ])
        
        # Process each frame
        tensors = []
        for frame in frames[:self.cfg.sequence_length]:
            tensor = preprocess(frame)
            tensors.append(tensor)
        
        # Stack into (1, T, C, H, W)
        return torch.stack(tensors).unsqueeze(0)
    
    def waypoints_to_control(
        self,
        waypoints: np.ndarray,
        current_speed: float = 0.0,
        dt: float = 0.05,
    ) -> Dict[str, float]:
        """
        Convert waypoints to CARLA vehicle control commands.
        
        Args:
            waypoints: (H, 2) array in ego frame (x forward, y left)
            current_speed: Current vehicle speed in m/s
            dt: Time step for command duration
        
        Returns:
            control: Dict with throttle, steer, brake for CARLA VehicleControl
        """
        if len(waypoints) == 0:
            return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
        
        # Target: first waypoint within lookahead
        target = waypoints[0]
        target_distance = np.linalg.norm(target)
        
        # Steering: angle to target point
        steer = np.arctan2(target[1], target[0])
        
        # Clamp steering to reasonable range
        steer = float(np.clip(steer, -0.7, 0.7))  # ~40 degrees
        
        # Throttle/brake based on distance and angle
        if target_distance < 2.0:
            # Close to target: slow down
            throttle = 0.0
            brake = 0.3
        elif abs(steer) > 0.5:
            # Sharp turn: slow down
            throttle = 0.2
            brake = 0.1
        else:
            # Normal driving
            throttle = 0.4
            brake = 0.0
        
        return {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
        }
    
    def predict_and_control(
        self,
        image: np.ndarray,
        current_speed: float = 0.0,
    ) -> Dict[str, float]:
        """
        Add frame to buffer and get control command in one call.
        
        Convenience method for real-time inference.
        
        Args:
            image: Current camera frame (H, W, C)
            current_speed: Current vehicle speed
        
        Returns:
            control: Dict with throttle, steer, brake
        """
        self.add_frame(image)
        waypoints = self.predict_temporal()
        return self.waypoints_to_control(waypoints, current_speed)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration as dict."""
        return {
            "checkpoint": str(self.cfg.checkpoint),
            "camera_name": self.cfg.camera_name,
            "horizon_steps": self.cfg.horizon_steps,
            "sequence_length": self.cfg.sequence_length,
            "hidden_dim": self.cfg.hidden_dim,
            "num_rnn_layers": self.cfg.num_rnn_layers,
            "device": str(self._device),
            "encoder_name": self.cfg.encoder_name,
        }


def main():
    """CLI for testing temporal policy wrapper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Temporal Waypoint Policy Wrapper")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder name")
    parser.add_argument("--sequence-length", type=int, default=4, help="Sequence length")
    parser.add_argument("--horizon-steps", type=int, default=20, help="Waypoint horizon")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dim")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()
    
    # Create config
    cfg = TemporalPolicyConfig(
        checkpoint=args.checkpoint,
        encoder_name=args.encoder,
        sequence_length=args.sequence_length,
        horizon_steps=args.horizon_steps,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    
    # Create and initialize wrapper
    policy = TemporalWaypointPolicyWrapper(cfg)
    
    if policy.initialize():
        print("✓ Policy initialized successfully")
        print(f"Config: {policy.get_config()}")
    else:
        print("✗ Failed to initialize policy")


if __name__ == "__main__":
    main()

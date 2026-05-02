#!/usr/bin/env python3
"""
Waypoint Prediction Inference API.

Unified interface for running inference with BC/RL checkpoints to produce 
waypoint predictions compatible with CARLA's native waypoint API.

This module provides:
- Single observation inference
- Batch inference 
- Streaming inference
- CARLA waypoint format conversion
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# Optional torch import with graceful fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object


@dataclass
class InferenceConfig:
    """Configuration for waypoint inference."""
    # Model settings
    checkpoint_path: str = ""
    model_type: str = "bc"  # "bc" or "rl"
    encoder_dim: int = 256
    hidden_dim: int = 256
    num_waypoints: int = 8
    
    # Inference settings
    device: str = "cpu"  # Default to CPU since CUDA may not be available
    batch_size: int = 1
    use_fp16: bool = False
    
    # Waypoint settings
    horizon_seconds: float = 3.0
    sampling_rate_hz: float = 2.0
    
    # CARLA integration
    carla_coordinate_system: bool = True  # Convert to CARLA coordinates


@dataclass
class WaypointPrediction:
    """Single waypoint prediction result."""
    # Waypoints in meters (relative to ego) or CARLA coordinates
    waypoints: np.ndarray  # (num_waypoints, 2) or (num_waypoints, 3)
    
    # Speeds in m/s for each waypoint
    speeds: Optional[np.ndarray] = None  # (num_waypoints,)
    
    # Progress through episode (0-1)
    progress: Optional[float] = None
    
    # Confidence score (0-1)
    confidence: float = 1.0
    
    # Timing (ms)
    inference_time_ms: float = 0.0
    
    # Metadata
    frame_id: int = 0
    episode_id: str = ""
    

@dataclass  
class BatchPrediction:
    """Batch inference results."""
    predictions: List[WaypointPrediction]
    total_time_ms: float
    avg_time_per_sample_ms: float
    
    # Statistics
    mean_confidence: float = 1.0


def coordinate_transform_to_carla(waypoints: np.ndarray, heading: float = 0.0) -> np.ndarray:
    """
    Transform waypoints from OpenCV to CARLA coordinate system.
    
    OpenCV: X right, Y down (image coordinates)
    CARLA: X forward, Y right, Z up
    
    Args:
        waypoints: (N, 2) or (N, 3) waypoints in OpenCV frame
        heading: Vehicle heading in radians
        
    Returns:
        Transformed waypoints in CARLA frame
    """
    if waypoints.shape[-1] == 2:
        # 2D waypoints (x, y) -> rotate by heading, swap to CARLA
        # CARLA: forward = X, right = -Y (when heading=0)
        transformed = np.zeros_like(waypoints)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        # Rotation + coordinate flip
        transformed[:, 0] = cos_h * waypoints[:, 0] - sin_h * waypoints[:, 1]  # forward
        transformed[:, 1] = sin_h * waypoints[:, 0] + cos_h * waypoints[:, 1]  # right
        
        return transformed
    else:
        # 3D waypoints - just copy (assumed already in CARLA frame)
        return waypoints


def coordinate_transform_from_carla(waypoints: np.ndarray, heading: float = 0.0) -> np.ndarray:
    """
    Transform waypoints from CARLA to OpenCV coordinate system.
    
    Args:
        waypoints: (N, 2) or (N, 3) waypoints in CARLA frame
        heading: Vehicle heading in radians
        
    Returns:
        Transformed waypoints in OpenCV frame
    """
    if waypoints.shape[-1] == 2:
        # Inverse of coordinate_transform_to_carla
        transformed = np.zeros_like(waypoints)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        transformed[:, 0] = cos_h * waypoints[:, 0] + sin_h * waypoints[:, 1]
        transformed[:, 1] = -sin_h * waypoints[:, 0] + cos_h * waypoints[:, 1]
        
        return transformed
    else:
        return waypoints


class ResidualWaypointMLP(nn.Module if TORCH_AVAILABLE else object):
    """MLP for waypoint prediction with progress conditioning."""
    
    def __init__(self, obs_dim: int = 4, hidden_dim: int = 256, num_waypoints: int = 8):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model")
            
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Observation encoding
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )
        
        # Progress encoding
        self.progress_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_waypoints * 2),  # (x, y) per waypoint
        )
        
        # Speed prediction head  
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_waypoints),
        )
    
    def forward(self, obs: torch.Tensor, progress: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim) - [pos_x, pos_y, speed, heading]
            progress: (batch, 1) - episode progress 0-1
            
        Returns:
            waypoints: (batch, num_waypoints, 2)
            speeds: (batch, num_waypoints)
        """
        obs_emb = self.obs_encoder(obs)
        prog_emb = self.progress_encoder(progress)
        
        # Concatenate embeddings
        combined = torch.cat([obs_emb, prog_emb], dim=-1)
        
        # Predict waypoints
        waypoints = self.waypoint_head(combined)
        waypoints = waypoints.view(-1, self.num_waypoints, 2)
        
        # Predict speeds
        speeds = self.speed_head(combined)
        
        return waypoints, speeds


class RLRefinedWaypointModel(nn.Module if TORCH_AVAILABLE else object):
    """BC + Delta head for RL-refined waypoint prediction."""
    
    def __init__(self, bc_checkpoint_path: str = "", encoder_dim: int = 256, 
                 hidden_dim: int = 256, num_waypoints: int = 8, delta_scale: float = 1.0):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model")
            
        super().__init__()
        self.delta_scale = delta_scale
        self.num_waypoints = num_waypoints
        
        # Base BC model (frozen)
        self.bc_model = ResidualWaypointMLP(
            obs_dim=4, hidden_dim=hidden_dim, num_waypoints=num_waypoints
        )
        for param in self.bc_model.parameters():
            param.requires_grad = False
            
        # Residual delta head (trainable)
        self.delta_head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_waypoints * 2),
        )
        
        # Load BC checkpoint if provided
        if bc_checkpoint_path and Path(bc_checkpoint_path).exists():
            self._load_checkpoint(bc_checkpoint_path)
    
    def _load_checkpoint(self, path: str):
        """Load BC checkpoint weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                self.bc_model.load_state_dict(state_dict['model_state_dict'])
            elif 'state_dict' in state_dict:
                self.bc_model.load_state_dict(state_dict['state_dict'])
        except Exception as e:
            print(f"Warning: Could not load checkpoint {path}: {e}")
            
    def forward(self, obs: torch.Tensor, progress: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim)
            progress: (batch, 1)
            
        Returns:
            refined_waypoints: (batch, num_waypoints, 2) - BC + delta
            speeds: (batch, num_waypoints)
        """
        # Get BC predictions (frozen)
        with torch.no_grad():
            bc_waypoints, bc_speeds = self.bc_model(obs, progress)
            
        # Compute delta
        delta = self.delta_head(obs)
        delta = delta.view(-1, self.num_waypoints, 2)
        
        # Combine
        refined_waypoints = bc_waypoints + self.delta_scale * delta
        
        return refined_waypoints, bc_speeds


class WaypointInferenceAPI:
    """Main API for waypoint prediction inference."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.device = None
        
        if TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint."""
        if not TORCH_AVAILABLE:
            self.model = None
            return
        
        # Default to CPU to avoid CUDA initialization issues
        self.device = torch.device('cpu')
        
        # Create model based on type
        if self.config.model_type == "bc":
            self.model = ResidualWaypointMLP(
                obs_dim=4,
                hidden_dim=self.config.hidden_dim,
                num_waypoints=self.config.num_waypoints
            )
        else:  # rl
            self.model = RLRefinedWaypointModel(
                bc_checkpoint_path=self.config.checkpoint_path,
                encoder_dim=self.config.encoder_dim,
                hidden_dim=self.config.hidden_dim,
                num_waypoints=self.config.num_waypoints
            )
        
        # Load checkpoint if provided
        if self.config.checkpoint_path:
            try:
                checkpoint = torch.load(
                    self.config.checkpoint_path, 
                    map_location=self.device
                )
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded checkpoint: {self.config.checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # FP16 not supported on CPU
        # if self.config.use_fp16 and self.device.type == 'cuda':
    
    def predict_single(
        self, 
        obs: Dict[str, Any],
        heading: float = 0.0
    ) -> WaypointPrediction:
        """Run inference on single observation."""
        start_time = time.perf_counter()
        
        # Extract observation components
        pos_x = obs.get('pos_x', 0.0)
        pos_y = obs.get('pos_y', 0.0)
        speed = obs.get('speed', 0.0)
        obs_heading = obs.get('heading', 0.0)
        progress = obs.get('progress', 0.0)
        
        if TORCH_AVAILABLE and self.model is not None:
            # Convert to tensor
            obs_tensor = torch.tensor(
                [[pos_x, pos_y, speed, obs_heading]],
                dtype=torch.float32
            ).to(self.device)
            
            progress_tensor = torch.tensor(
                [[progress]],
                dtype=torch.float32
            ).to(self.device)
            
            # Run inference - FP16 not supported on CPU
            with torch.no_grad():
                # if self.config.use_fp16:
                #     obs_tensor = obs_tensor.half()
                #     progress_tensor = progress_tensor.half()
                    
                waypoints, speeds = self.model(obs_tensor, progress_tensor)
                
                # Convert to numpy
                waypoints = waypoints.cpu().numpy()[0]
                speeds = speeds.cpu().numpy()[0]
        else:
            # Fallback: generate synthetic waypoints
            waypoints = np.zeros((self.config.num_waypoints, 2))
            t = np.linspace(0, self.config.horizon_seconds, self.config.num_waypoints)
            for i in range(self.config.num_waypoints):
                waypoints[i, 0] = speed * t[i] * np.cos(obs_heading)
                waypoints[i, 1] = speed * t[i] * np.sin(obs_heading)
            speeds = np.full(self.config.num_waypoints, speed)
        
        # Transform to CARLA coordinates if requested
        if self.config.carla_coordinate_system:
            waypoints = coordinate_transform_to_carla(waypoints, heading)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return WaypointPrediction(
            waypoints=waypoints,
            speeds=speeds,
            progress=progress,
            confidence=1.0,
            inference_time_ms=inference_time,
            frame_id=obs.get('frame_id', 0),
            episode_id=obs.get('episode_id', '')
        )
    
    def predict_batch(
        self,
        observations: List[Dict[str, Any]]
    ) -> BatchPrediction:
        """Run batch inference on multiple observations."""
        start_time = time.perf_counter()
        
        predictions = []
        for obs in observations:
            pred = self.predict_single(obs)
            predictions.append(pred)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return BatchPrediction(
            predictions=predictions,
            total_time_ms=total_time,
            avg_time_per_sample_ms=total_time / len(observations) if observations else 0,
            mean_confidence=np.mean([p.confidence for p in predictions]) if predictions else 1.0
        )
    
    def to_carla_waypoints(
        self,
        prediction: WaypointPrediction,
        vehicle_location: Tuple[float, float, float] = (0, 0, 0)
    ) -> List[Dict[str, float]]:
        """
        Convert prediction to CARLA waypoint format.
        
        Args:
            prediction: WaypointPrediction result
            vehicle_location: (x, y, z) in CARLA coordinates
            
        Returns:
            List of waypoint dicts with x, y, z coordinates
        """
        waypoints = []
        for i, (wx, wy) in enumerate(prediction.waypoints):
            waypoint = {
                'x': vehicle_location[0] + wx,
                'y': vehicle_location[1] + wy,
                'z': vehicle_location[2],
                'speed': prediction.speeds[i] if prediction.speeds is not None else 0.0,
                'progress': prediction.progress or (i / len(prediction.waypoints))
            }
            waypoints.append(waypoint)
        
        return waypoints


def load_observation_from_dict(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Load observation from dictionary format."""
    return obs_dict


def load_observation_from_json(json_path: str) -> Dict[str, Any]:
    """Load observation from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_predictions(
    predictions: List[WaypointPrediction],
    output_path: str,
    format: str = "json"
):
    """Save predictions to file."""
    if format == "json":
        data = []
        for pred in predictions:
            entry = {
                'waypoints': pred.waypoints.tolist(),
                'speeds': pred.speeds.tolist() if pred.speeds is not None else None,
                'progress': pred.progress,
                'confidence': pred.confidence,
                'inference_time_ms': pred.inference_time_ms,
                'frame_id': pred.frame_id,
                'episode_id': pred.episode_id
            }
            data.append(entry)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_smoke_test_observation() -> Dict[str, Any]:
    """Create a dummy observation for smoke testing."""
    return {
        'pos_x': 0.0,
        'pos_y': 0.0,
        'speed': 5.0,  # 5 m/s
        'heading': 0.0,  # radians
        'progress': 0.5,  # halfway through episode
        'frame_id': 0,
        'episode_id': 'smoke_test'
    }


def main():
    """CLI for waypoint inference."""
    parser = argparse.ArgumentParser(description="Waypoint Prediction Inference API")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, default='', help='Model checkpoint path')
    infer_parser.add_argument('--model-type', type=str, default='bc', choices=['bc', 'rl'])
    infer_parser.add_argument('--num-waypoints', type=int, default=8)
    infer_parser.add_argument('--hidden-dim', type=int, default=256)
    infer_parser.add_argument('--device', type=str, default='cuda')
    infer_parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    infer_parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    infer_parser.add_argument('--format', type=str, default='json', choices=['json'])
    infer_parser.add_argument('--to-carla', action='store_true', help='Convert to CARLA waypoint format')
    infer_parser.add_argument('--smoke-test', action='store_true', help='Run smoke test')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available checkpoints')
    list_parser.add_argument('--checkpoints-dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    if args.command == 'infer':
        # Load config
        config = InferenceConfig(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            num_waypoints=args.num_waypoints,
            hidden_dim=args.hidden_dim,
            device=args.device
        )
        
        # Create API
        api = WaypointInferenceAPI(config)
        
        # Load observations
        if args.smoke_test:
            observations = [create_smoke_test_observation()]
        else:
            with open(args.input, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    observations = data
                else:
                    observations = [data]
        
        # Run inference
        if len(observations) == 1:
            prediction = api.predict_single(observations[0])
            predictions = [prediction]
        else:
            result = api.predict_batch(observations)
            predictions = result.predictions
        
        # Print summary
        print(f"Ran inference on {len(predictions)} observations")
        if predictions:
            avg_time = np.mean([p.inference_time_ms for p in predictions])
            print(f"Average inference time: {avg_time:.2f}ms")
            
            # Print first prediction
            first = predictions[0]
            print(f"First prediction: {first.waypoints.shape} waypoints")
            print(f"  Sample waypoint: {first.waypoints[0]}")
        
        # Convert to CARLA format if requested
        if args.to_carla:
            carla_waypoints = api.to_carla_waypoints(predictions[0])
            print(f"CARLA waypoints: {len(carla_waypoints)} waypoints")
            print(f"  First CARLA waypoint: {carla_waypoints[0]}")
        
        # Save predictions
        save_predictions(predictions, args.output, args.format)
        print(f"Saved to {args.output}")
        
    elif args.command == 'list':
        checkpoints_dir = Path(args.checkpoints_dir)
        if not checkpoints_dir.exists():
            print(f"Checkpoints directory not found: {checkpoints_dir}")
            return
            
        print(f"Available checkpoints in {checkpoints_dir}:")
        for checkpoint in checkpoints_dir.rglob('*.pt'):
            print(f"  {checkpoint.relative_to(checkpoints_dir)}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
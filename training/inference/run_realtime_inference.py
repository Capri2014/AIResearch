#!/usr/bin/env python3
"""
Real-time Inference Pipeline - End-to-end inference for driving models.

Connects the full pipeline: SSL encoder -> BC waypoint prediction -> RL refinement.
Designed for real-time inference on observation streams (e.g., from CARLA or sensors).

Usage:
    # CLI
    python training/inference/run_realtime_inference.py \
        --ssl-checkpoint out/ssl_train/checkpoints/final.pt \
        --bc-checkpoint out/waypoint_bc/final.pt \
        --rl-checkpoint out/ppo_smoke/final.pt \
        --input observation.json \
        --output predictions.json

    # As library
    from training.inference.run_realtime_inference import RealtimeInferencePipeline
    
    pipeline = RealtimeInferencePipeline(
        ssl_checkpoint="out/ssl_train/checkpoints/final.pt",
        bc_checkpoint="out/waypoint_bc/final.pt",
        rl_checkpoint="out/ppo_smoke/final.pt"
    )
    waypoints = pipeline.run(observation)  # (num_waypoints, 2)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.pretrain.run_ssl_trainer import (
    SSLTrainingConfig,
    ConvEncoder,
    TemporalTransformerEncoder,
    SSLModel,
)
from training.sft.train_waypoint_bc import WaypointBCConfig, WaypointBCModel
from training.rl.ppo_waypoint_delta_gae import PPOWaypointDeltaConfig, PPOWaypointAgent


@dataclass
class RealtimeInferenceConfig:
    """Configuration for real-time inference pipeline."""
    
    # Checkpoint paths
    ssl_checkpoint: str = ""
    bc_checkpoint: str = ""
    rl_checkpoint: str = ""
    
    # Model architecture (must match training)
    num_waypoints: int = 8
    encoder_dim: int = 256
    waypoint_dim: int = 2
    hidden_dim: int = 256
    num_layers: int = 4
    
    # RL refinement
    delta_scale: float = 0.5
    use_rl_refinement: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Inference options
    batch_size: int = 1
    temperature: float = 1.0


@dataclass
class Observation:
    """Single observation from the environment."""
    
    # Current RGB image (H, W, 3)
    image: np.ndarray
    
    # Agent state: (x, y, theta, speed)
    agent_state: np.ndarray
    
    # Previous waypoints (for temporal context)
    prev_waypoints: Optional[np.ndarray] = None
    
    # Timestamp
    timestamp: float = 0.0


@dataclass
class WaypointPrediction:
    """Waypoint prediction result."""
    
    # Shape: (num_waypoints, 2)
    waypoints: np.ndarray
    
    # Shape: (num_waypoints,)
    speeds: np.ndarray
    
    # Confidence score [0, 1]
    confidence: float = 1.0
    
    # Inference timing (ms)
    inference_time_ms: float = 0.0
    
    # Stage breakdown
    ssl_time_ms: float = 0.0
    bc_time_ms: float = 0.0
    rl_time_ms: float = 0.0


class ConvEncoderWrapper(nn.Module):
    """Wrapper for SSL ConvEncoder to match BC input requirements."""
    
    def __init__(self, encoder: ConvEncoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - RGB images
        Returns:
            (B, encoder_dim) - Encoded features
        """
        return self.encoder(x)


class RealtimeInferencePipeline:
    """End-to-end inference pipeline for driving models."""
    
    def __init__(self, config: RealtimeInferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.ssl_encoder: Optional[ConvEncoderWrapper] = None
        self.bc_model: Optional[WaypointBCModel] = None
        self.rl_agent: Optional[PPOWaypointAgent] = None
        
        # Load checkpoints
        self._load_models()
        
    def _load_models(self):
        """Load all model checkpoints."""
        
        # Load SSL encoder
        if self.config.ssl_checkpoint and Path(self.config.ssl_checkpoint).exists():
            print(f"Loading SSL encoder from {self.config.ssl_checkpoint}")
            ssl_config = SSLTrainingConfig()
            ssl_config.encoder_dim = self.config.encoder_dim
            
            # Create encoder
            conv_encoder = ConvEncoder(
                in_channels=3,
                encoder_dim=self.config.encoder_dim
            )
            temporal_encoder = TemporalTransformerEncoder(
                encoder_dim=self.config.encoder_dim,
                num_heads=4,
                num_layers=2
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.config.ssl_checkpoint, map_location=self.device)
            if 'model_state' in checkpoint:
                conv_encoder.load_state_dict(checkpoint['model_state']['conv_encoder'])
            
            # Wrap for BC
            self.ssl_encoder = ConvEncoderWrapper(conv_encoder)
            self.ssl_encoder.to(self.device)
            self.ssl_encoder.eval()
            
        # Load BC waypoint model
        if self.config.bc_checkpoint and Path(self.config.bc_checkpoint).exists():
            print(f"Loading BC model from {self.config.bc_checkpoint}")
            bc_config = WaypointBCConfig()
            bc_config.encoder_dim = self.config.encoder_dim
            bc_config.hidden_dim = self.config.hidden_dim
            bc_config.num_waypoints = self.config.num_waypoints
            bc_config.waypoint_dim = self.config.waypoint_dim
            bc_config.num_layers = self.config.num_layers
            
            self.bc_model = WaypointBCModel(bc_config)
            checkpoint = torch.load(self.config.bc_checkpoint, map_location=self.device)
            if 'model_state' in checkpoint:
                self.bc_model.load_state_dict(checkpoint['model_state'])
            elif 'state_dict' in checkpoint:
                self.bc_model.load_state_dict(checkpoint['state_dict'])
            
            self.bc_model.to(self.device)
            self.bc_model.eval()
            
        # Load RL refinement model
        if self.config.rl_checkpoint and Path(self.config.rl_checkpoint).exists() and self.config.use_rl_refinement:
            print(f"Loading RL agent from {self.config.rl_checkpoint}")
            rl_config = PPOWaypointDeltaConfig()
            rl_config.hidden_dim = self.config.hidden_dim
            rl_config.waypoint_dim = self.config.waypoint_dim
            rl_config.num_waypoints = self.config.num_waypoints
            rl_config.delta_scale = self.config.delta_scale
            
            self.rl_agent = PPOWaypointAgent(rl_config)
            checkpoint = torch.load(self.config.rl_checkpoint, map_location=self.device)
            if 'model_state' in checkpoint:
                self.rl_agent.load_state_dict(checkpoint['model_state'])
            
            self.rl_agent.to(self.device)
            self.rl_agent.eval()
            
    def _preprocess_observation(self, obs: Observation) -> Dict[str, torch.Tensor]:
        """Preprocess observation to model inputs."""
        
        # Image: (H, W, 3) -> (1, 3, H, W)
        image = torch.from_numpy(obs.image).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Agent state: (4,) -> (1, 4)
        agent_state = torch.from_numpy(obs.agent_state).float().unsqueeze(0)
        
        # Previous waypoints: (num_waypoints, 2) -> (1, num_waypoints, 2)
        prev_waypoints = None
        if obs.prev_waypoints is not None:
            prev_waypoints = torch.from_numpy(obs.prev_waypoints).float().unsqueeze(0)
        
        return {
            'image': image.to(self.device),
            'agent_state': agent_state.to(self.device),
            'prev_waypoints': prev_waypoints.to(self.device) if prev_waypoints is not None else None
        }
        
    def _postprocess_waypoints(self, waypoints: torch.Tensor) -> np.ndarray:
        """Convert waypoint tensor to numpy array."""
        return waypoints.detach().cpu().numpy()[0]  # (num_waypoints, 2)
        
    def run(self, obs: Observation) -> WaypointPrediction:
        """
        Run inference on a single observation.
        
        Args:
            obs: Observation from environment
            
        Returns:
            WaypointPrediction with waypoints and metadata
        """
        total_start = time.perf_counter()
        
        # Preprocess
        inputs = self._preprocess_observation(obs)
        
        # Stage 1: SSL encoding (if available)
        ssl_start = time.perf_counter()
        if self.ssl_encoder is not None:
            with torch.no_grad():
                encoding = self.ssl_encoder(inputs['image'])
            # Expand to sequence format for BC
            encoding = encoding.unsqueeze(1).expand(-1, self.config.num_waypoints, -1)
        else:
            # Fallback: use image as direct input
            encoding = inputs['image']
        ssl_time = (time.perf_counter() - ssl_start) * 1000
        
        # Stage 2: BC waypoint prediction
        bc_start = time.perf_counter()
        if self.bc_model is not None:
            with torch.no_grad():
                # BC expects: encoding, agent_state, prev_waypoints
                bc_output = self.bc_model(
                    encoding=encoding,
                    agent_state=inputs['agent_state'],
                    prev_waypoints=inputs.get('prev_waypoints')
                )
                bc_waypoints = bc_output['waypoints']  # (1, num_waypoints, 2)
        else:
            # Fallback: zeros
            bc_waypoints = torch.zeros(1, self.config.num_waypoints, self.config.waypoint_dim).to(self.device)
        bc_time = (time.perf_counter() - bc_start) * 1000
        
        # Stage 3: RL refinement (if available)
        rl_start = time.perf_counter()
        if self.rl_agent is not None and self.config.use_rl_refinement:
            with torch.no_grad():
                # Compute delta corrections
                delta_action, _ = self.rl_agent.get_action(
                    obs_encoding=encoding,
                    current_waypoints=bc_waypoints,
                    agent_state=inputs['agent_state']
                )
                # Apply delta
                refined_waypoints = bc_waypoints + delta_action * self.config.delta_scale
        else:
            refined_waypoints = bc_waypoints
        rl_time = (time.perf_counter() - rl_start) * 1000
        
        # Postprocess
        final_waypoints = self._postprocess_waypoints(refined_waypoints)
        
        # Extract speeds (placeholder - could be from another head)
        speeds = np.ones(self.config.num_waypoints) * 5.0  # m/s
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        return WaypointPrediction(
            waypoints=final_waypoints,
            speeds=speeds,
            confidence=1.0,
            inference_time_ms=total_time,
            ssl_time_ms=ssl_time,
            bc_time_ms=bc_time,
            rl_time_ms=rl_time
        )
        
    def run_batch(self, obs_batch: List[Observation]) -> List[WaypointPrediction]:
        """Run inference on a batch of observations."""
        predictions = []
        for obs in obs_batch:
            pred = self.run(obs)
            predictions.append(pred)
        return predictions


def load_observation_from_json(path: str) -> Observation:
    """Load observation from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    return Observation(
        image=np.array(data['image']),
        agent_state=np.array(data['agent_state']),
        prev_waypoints=np.array(data.get('prev_waypoints')) if 'prev_waypoints' in data else None,
        timestamp=data.get('timestamp', 0.0)
    )


def save_predictions(pred: WaypointPrediction, path: str):
    """Save predictions to JSON file."""
    data = {
        'waypoints': pred.waypoints.tolist(),
        'speeds': pred.speeds.tolist(),
        'confidence': pred.confidence,
        'inference_time_ms': pred.inference_time_ms,
        'timing': {
            'ssl_ms': pred.ssl_time_ms,
            'bc_ms': pred.bc_time_ms,
            'rl_ms': pred.rl_time_ms
        }
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def create_smoke_test_observation() -> Observation:
    """Create a dummy observation for smoke testing."""
    # Dummy 64x64 RGB image
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Agent state: (x, y, theta, speed)
    agent_state = np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float32)
    
    # Previous waypoints (optional)
    prev_waypoints = np.array([
        [0.0, 5.0],
        [1.0, 10.0],
        [2.0, 15.0]
    ], dtype=np.float32)
    
    return Observation(
        image=image,
        agent_state=agent_state,
        prev_waypoints=prev_waypoints,
        timestamp=0.0
    )


def main():
    parser = argparse.ArgumentParser(description="Real-time Inference Pipeline")
    parser.add_argument('--ssl-checkpoint', type=str, default='', help='SSL encoder checkpoint')
    parser.add_argument('--bc-checkpoint', type=str, default='', help='BC waypoint model checkpoint')
    parser.add_argument('--rl-checkpoint', type=str, default='', help='RL refinement checkpoint')
    parser.add_argument('--input', type=str, default='', help='Input observation JSON')
    parser.add_argument('--output', type=str, default='predictions.json', help='Output predictions JSON')
    parser.add_argument('--num-waypoints', type=int, default=8, help='Number of waypoints')
    parser.add_argument('--encoder-dim', type=int, default=256, help='Encoder dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--delta-scale', type=float, default=0.5, help='RL delta scale')
    parser.add_argument('--no-rl', action='store_true', help='Disable RL refinement')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke-test', action='store_true', help='Run smoke test')
    args = parser.parse_args()
    
    # Create config
    config = RealtimeInferenceConfig(
        ssl_checkpoint=args.ssl_checkpoint,
        bc_checkpoint=args.bc_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        num_waypoints=args.num_waypoints,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        delta_scale=args.delta_scale,
        use_rl_refinement=not args.no_rl,
        device=args.device
    )
    
    # Create pipeline
    print("Initializing Realtime Inference Pipeline...")
    pipeline = RealtimeInferencePipeline(config)
    
    # Load or create observation
    if args.smoke_test:
        print("Running smoke test...")
        obs = create_smoke_test_observation()
    elif args.input:
        print(f"Loading observation from {args.input}")
        obs = load_observation_from_json(args.input)
    else:
        print("No input specified, running smoke test")
        obs = create_smoke_test_observation()
    
    # Run inference
    print("Running inference...")
    prediction = pipeline.run(obs)
    
    # Print results
    print(f"\nResults:")
    print(f"  Waypoints shape: {prediction.waypoints.shape}")
    print(f"  Inference time: {prediction.inference_time_ms:.2f}ms")
    print(f"    SSL: {prediction.ssl_time_ms:.2f}ms")
    print(f"    BC: {prediction.bc_time_ms:.2f}ms")
    print(f"    RL: {prediction.rl_time_ms:.2f}ms")
    print(f"  Confidence: {prediction.confidence:.2f}")
    
    # Save predictions
    print(f"\nSaving predictions to {args.output}")
    save_predictions(prediction, args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()
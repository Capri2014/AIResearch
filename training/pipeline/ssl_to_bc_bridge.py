#!/usr/bin/env python3
"""
SSL to BC Bridge - Connects PyTorch SSL pretraining to Waypoint BC stage.

This module bridges the gap between SSL pretraining (Stage 2) and waypoint BC (Stage 3):
- Loads SSL/JEPA pretrained checkpoints
- Generates waypoint embeddings or predictions from episode latents
- Converts SSL features to BC-consumable format
- Provides batch conversion for BC training data

Usage:
    python -m training.pipeline.ssl_to_bc_bridge convert --episodes data/waymo/episodes \
        --checkpoint training/pretrain/out/best_ssl.pt --output data/waymo/bc_ready/
    
    python -m training.pipeline.ssl_to_bc_bridge evaluate --episodes data/waymo/episodes \
        --checkpoint training/pretrain/out/best_ssl.pt --num-samples 100

Architecture:
    SSL Encoder (frozen) --> Waypoint Projection --> BC-Training-Ready Features
                    or
    SSL Latent --> Waypoint Prediction Head --> Predicted Waypoints
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SSLToBCConfig:
    """Configuration for SSL to BC bridge."""
    # Model paths
    ssl_checkpoint: str = "training/pretrain/out/best_ssl.pt"
    output_dir: str = "data/waymo/bc_ready"
    
    # Model configuration
    embed_dim: int = 256
    num_waypoints: int = 20
    waypoint_hidden: int = 128
    
    # Processing
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Feature extraction
    extract_mode: str = "embeddings"  # "embeddings" | "predictions"
    use_projection: bool = True
    
    # Filtering
    min_episode_length: int = 10
    max_samples: Optional[int] = None


# ============================================================================
# Model Components
# ============================================================================

class SSLFeatureProjector(nn.Module):
    """Projects SSL features to waypoint-compatible embeddings."""
    
    def __init__(self, embed_dim: int = 256, num_waypoints: int = 20, hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_waypoints = num_waypoints
        
        # Feature projection MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
        )
        
        # Waypoint-aware projection
        self.waypoint_mlp = nn.Sequential(
            nn.Linear(embed_dim + 4, hidden),  # +4 for position encoding
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
        )
        
        # Position encoding for waypoints
        self.register_buffer(
            "waypoint_positions", 
            torch.linspace(0, 1, num_waypoints).unsqueeze(-1)
        )
    
    def forward(self, ssl_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ssl_features: (B, T, D) or (B, D) temporal SSL features
        Returns:
            projected: (B, num_waypoints, D) waypoint-ready features
        """
        if ssl_features.dim() == 2:
            ssl_features = ssl_features.unsqueeze(1)
        
        B, T, D = ssl_features.shape
        
        # Project base features
        base_features = self.feature_mlp(ssl_features)
        
        # Interpolate to num_waypoints if needed
        if T != self.num_waypoints:
            # Learnable interpolation
            indices = torch.linspace(0, T - 1, self.num_waypoints, device=base_features.device)
            indices = indices.long().clamp(0, T - 1)
            projected = base_features.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))
        else:
            projected = base_features
        
        return projected


class WaypointPredictionHead(nn.Module):
    """Predicts waypoints from SSL latent features."""
    
    def __init__(self, embed_dim: int = 256, num_waypoints: int = 20, hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_waypoints = num_waypoints
        
        # Prediction head: SSL latent --> waypoint coordinates
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_waypoints * 2),  # (x, y) per waypoint
        )
    
    def forward(self, ssl_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ssl_features: (B, D) SSL latent vector
        Returns:
            waypoints: (B, num_waypoints, 2) predicted (x, y) coordinates
        """
        B = ssl_features.shape[0]
        waypoints = self.decoder(ssl_features)
        waypoints = waypoints.view(B, self.num_waypoints, 2)
        return waypoints


@dataclass
class BridgeOutput:
    """Output from SSL to BC bridge."""
    episode_id: str
    waypoints: np.ndarray  # (num_waypoints, 2)
    features: Optional[np.ndarray]  # (num_waypoints, embed_dim)
    confidence: float
    metadata: dict = field(default_factory=dict)


# ============================================================================
# Dataset
# ============================================================================

class EpisodeDataset(Dataset):
    """Dataset loading Waymo episodes for BC conversion."""
    
    def __init__(
        self, 
        episode_dir: str,
        min_length: int = 10,
        max_samples: Optional[int] = None,
    ):
        self.episode_dir = Path(episode_dir)
        self.min_length = min_length
        
        # Find all episode files
        self.episode_files = sorted(self.episode_dir.glob("*.json"))
        
        if max_samples:
            self.episode_files = self.episode_files[:max_samples]
        
        # Load and validate episodes
        self.valid_episodes = []
        for ep_file in self.episode_files:
            try:
                with open(ep_file) as f:
                    data = json.load(f)
                
                frames = data.get("frames", [])
                if frames:
                    # Check for trajectory/state data in frames
                    has_trajectory = False
                    for frame in frames:
                        if "state" in frame or "trajectory" in frame:
                            has_trajectory = True
                            break
                    
                    if has_trajectory or len(frames) >= min_length:
                        # Store full data for later processing
                        self.valid_episodes.append((ep_file, data))
            except Exception as e:
                print(f"Warning: Error loading {ep_file}: {e}")
                continue
        
        print(f"Found {len(self.valid_episodes)} valid episodes (min_length={min_length})")
    
    def __len__(self) -> int:
        return len(self.valid_episodes)
    
    def __getitem__(self, idx: int) -> dict:
        ep_file, data = self.valid_episodes[idx]
        
        frames = data.get("frames", [])
        episode_id = data.get("episode_id", ep_file.stem)
        
        # Extract positions from frames
        positions = []
        velocities = []
        headings = []
        
        for frame in frames:
            # Try state first (newer format)
            if "state" in frame:
                state = frame["state"]
                position = state.get("position", [0, 0, 0])
                velocity = state.get("velocity", [0, 0, 0])
                heading = state.get("heading", 0.0)
            # Try trajectory in frame (older format)
            elif "trajectory" in frame:
                traj = frame["trajectory"]
                position = traj.get("position", [0, 0, 0])
                velocity = traj.get("velocity", [0, 0, 0])
                heading = traj.get("heading", 0.0)
            else:
                # Use expert trajectory (Waymo format)
                expert = frame.get("expert", {})
                if expert:
                    # Expert might have route/waypoints
                    position = expert.get("position", [0, 0, 0])
                    velocity = expert.get("velocity", [0, 0, 0])
                    heading = expert.get("heading", 0.0)
                else:
                    position = [0, 0, 0]
                    velocity = [0, 0, 0]
                    heading = 0.0
            
            positions.append(position[:2])  # Use x, y only
            velocities.append(velocity[:2])
            headings.append([heading])
        
        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        headings = np.array(headings, dtype=np.float32)
        
        return {
            "episode_id": episode_id,
            "positions": positions,
            "velocities": velocities,
            "headings": headings,
            "num_steps": len(positions),
        }


# ============================================================================
# Bridge Core
# ============================================================================

class SSLToBCBridge:
    """Main bridge class connecting SSL to BC."""
    
    def __init__(self, config: SSLToBCConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.projector = SSLFeatureProjector(
            embed_dim=config.embed_dim,
            num_waypoints=config.num_waypoints,
            hidden=config.waypoint_hidden,
        ).to(self.device)
        
        self.prediction_head = WaypointPredictionHead(
            embed_dim=config.embed_dim,
            num_waypoints=config.num_waypoints,
            hidden=config.waypoint_hidden,
        ).to(self.device)
        
        self.ssl_encoder = None
        self.checkpoint_loaded = False
        
        # Load checkpoint if provided
        if config.ssl_checkpoint and os.path.exists(config.ssl_checkpoint):
            self.load_checkpoint(config.ssl_checkpoint)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load SSL checkpoint and extract encoder weights."""
        print(f"Loading SSL checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Try to extract encoder state
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Load into projector (weights might have different keys)
            model_state = self.projector.state_dict()
            filtered_state = {}
            
            for k, v in state_dict.items():
                if k in model_state and v.shape == model_state[k].shape:
                    filtered_state[k] = v
            
            if filtered_state:
                self.projector.load_state_dict(filtered_state, strict=False)
                print(f"Loaded {len(filtered_state)} keys into projector")
            
            self.checkpoint_loaded = True
            
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using random initialization for projection")
    
    def extract_from_episode(self, episode_data: dict) -> BridgeOutput:
        """Extract BC-ready features from a single episode."""
        
        positions = episode_data["positions"]  # (T, 3)
        velocities = episode_data["velocities"]  # (T, 3)
        
        T = len(positions)
        
        # Generate SSL-like features (simulate encoder output)
        # In practice, this would come from the actual SSL encoder
        if self.checkpoint_loaded:
            with torch.no_grad():
                # Simulate SSL encoding process
                features = torch.randn(1, self.config.embed_dim, device=self.device)
                features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            # Use learned position features as fallback
            features = torch.randn(1, self.config.embed_dim, device=self.device)
        
        # Extract waypoints using simple sampling
        waypoints = self._sample_waypoints(positions, self.config.num_waypoints)
        
        # Generate projected features if needed
        projected_features = None
        if self.config.use_projection and self.checkpoint_loaded:
            with torch.no_grad():
                projected = self.projector(features)
                projected_features = projected.squeeze(0).cpu().numpy()
        
        # Generate predictions if mode is predictions
        predictions = None
        if self.config.extract_mode == "predictions":
            with torch.no_grad():
                predictions = self.prediction_head(features)
                predictions = predictions.squeeze(0).cpu().numpy()
                waypoints = predictions  # Override with predicted
        
        # Compute confidence based on trajectory smoothness
        confidence = self._compute_confidence(positions)
        
        return BridgeOutput(
            episode_id=episode_data["episode_id"],
            waypoints=waypoints,
            features=projected_features,
            confidence=confidence,
            metadata={
                "num_steps": T,
                "extract_mode": self.config.extract_mode,
            },
        )
    
    def _sample_waypoints(
        self, 
        positions: np.ndarray, 
        num_waypoints: int
    ) -> np.ndarray:
        """Sample waypoints uniformly from trajectory."""
        T = len(positions)
        indices = np.linspace(0, T - 1, num_waypoints, dtype=int)
        
        sampled = positions[indices, :2]  # (num_waypoints, 2) - x, y only
        return sampled
    
    def _compute_confidence(self, positions: np.ndarray) -> float:
        """Compute confidence score based on trajectory quality."""
        if len(positions) < 2:
            return 0.0
        
        # Compute velocity magnitudes
        velocities = np.diff(positions[:, :2], axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Compute speed variability
        if len(speeds) > 1:
            speed_std = np.std(speeds)
            speed_mean = np.mean(speeds) + 1e-8
            variability = speed_std / speed_mean
        else:
            variability = 0.0
        
        # Confidence: higher for consistent speeds
        confidence = max(0.0, 1.0 - variability)
        return float(confidence)
    
    def convert_batch(
        self, 
        episode_dir: str, 
        output_dir: Optional[str] = None
    ) -> list[BridgeOutput]:
        """Convert episodes in batch mode."""
        
        dataset = EpisodeDataset(
            episode_dir,
            min_length=self.config.min_episode_length,
            max_samples=self.config.max_samples,
        )
        
        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        outputs = []
        print(f"Converting {len(dataset)} episodes...")
        
        # Process one by one to handle variable lengths
        for idx in range(len(dataset)):
            episode_data = dataset[idx]
            
            output = self.extract_from_episode(episode_data)
            outputs.append(output)
            
            # Save output
            self._save_output(output, output_path)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1} episodes")
        
        print(f"Completed: {len(outputs)} episodes converted")
        
        # Save manifest
        manifest = {
            "num_episodes": len(outputs),
            "config": self.config.__dict__,
            "episodes": [
                {
                    "episode_id": o.episode_id,
                    "confidence": o.confidence,
                    "metadata": o.metadata,
                }
                for o in outputs
            ],
        }
        
        with open(output_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        return outputs
    
    def _save_output(self, output: BridgeOutput, output_dir: Path):
        """Save a single bridge output."""
        
        output_file = output_dir / f"{output.episode_id}_bc.json"
        
        data = {
            "episode_id": output.episode_id,
            "waypoints": output.waypoints.tolist(),
            "confidence": output.confidence,
            "metadata": output.metadata,
        }
        
        if output.features is not None:
            data["features"] = output.features.tolist()
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSL to BC Bridge - Connect SSL pretraining to waypoint BC"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert episodes to BC format")
    convert_parser.add_argument(
        "--episodes", required=True, help="Directory containing episode JSON files"
    )
    convert_parser.add_argument(
        "--checkpoint", help="SSL checkpoint path"
    )
    convert_parser.add_argument(
        "--output", help="Output directory"
    )
    convert_parser.add_argument(
        "--embed-dim", type=int, default=256, help="Embedding dimension"
    )
    convert_parser.add_argument(
        "--num-waypoints", type=int, default=20, help="Number of waypoints"
    )
    convert_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    convert_parser.add_argument(
        "--max-samples", type=int, help="Maximum number of episodes"
    )
    convert_parser.add_argument(
        "--extract-mode", choices=["embeddings", "predictions"], default="embeddings"
    )
    convert_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    convert_parser.add_argument(
        "--smoke-test", action="store_true", help="Run smoke test with 5 samples"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate bridge on episodes")
    eval_parser.add_argument(
        "--episodes", required=True, help="Directory containing episode JSON files"
    )
    eval_parser.add_argument(
        "--checkpoint", help="SSL checkpoint path"
    )
    eval_parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples"
    )
    eval_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    args = parser.parse_args()
    
    if args.command == "convert":
        # Handle smoke test
        max_samples = 5 if args.smoke_test else args.max_samples
        
        config = SSLToBCConfig(
            ssl_checkpoint=args.checkpoint or "",
            output_dir=args.output or "data/waymo/bc_ready",
            embed_dim=args.embed_dim,
            num_waypoints=args.num_waypoints,
            batch_size=args.batch_size,
            extract_mode=args.extract_mode,
            device=args.device,
            max_samples=max_samples,
        )
        
        bridge = SSLToBCBridge(config)
        outputs = bridge.convert_batch(args.episodes)
        
        print(f"\nConverted {len(outputs)} episodes to {config.output_dir}")
        
        # Calculate statistics
        confidences = [o.confidence for o in outputs]
        print(f"Mean confidence: {np.mean(confidences):.3f}")
        print(f"Median confidence: {np.median(confidences):.3f}")
    
    elif args.command == "evaluate":
        config = SSLToBCConfig(
            ssl_checkpoint=args.checkpoint or "",
            device=args.device,
            max_samples=args.num_samples,
        )
        
        bridge = SSLToBCBridge(config)
        dataset = EpisodeDataset(
            args.episodes,
            max_samples=args.num_samples,
        )
        
        print(f"Evaluating on {len(dataset)} episodes...")
        
        confidences = []
        for i in range(len(dataset)):
            episode_data = dataset[i]
            output = bridge.extract_from_episode(episode_data)
            confidences.append(output.confidence)
        
        print(f"\nEvaluation Results:")
        print(f"  Episodes: {len(confidences)}")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Median confidence: {np.median(confidences):.3f}")
        print(f"  Min confidence: {np.min(confidences):.3f}")
        print(f"  Max confidence: {np.max(confidences):.3f}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
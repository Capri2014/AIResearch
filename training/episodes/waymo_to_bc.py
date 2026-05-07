#!/usr/bin/env python3
"""
Waymo-to-Waypoint BC Integration

Bridges WaymoEpisode (from SSL integration) to waypoint BC training.
Pipeline Stage 3: Waymo → SSL → waypoint BC

Functions:
- WaymoToBCConverter: converts WaymoEpisode/SSLEpisode to BC training format
- WaypointBCTrainingData: PyTorch Dataset for waypoint BC training
- create_bc_dataloader(): creates DataLoader from Waymo episodes
- CLI: --count, --smoke, --run-bc

Usage:
    python3 -m training.episodes.waymo_to_bc --count
    python3 -m training.episodes.waymo_to_bc --smoke --episodes 3
    python3 -m training.episodes.waymo_to_bc --run-bc --epochs 1
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.episodes.waymo_episode_loader import (
    WaymoEpisode,
    discover_episodes,
    iter_episodes,
    load_episode,
)


@dataclass
class WaypointBCTrainingSample:
    """Single training sample for waypoint BC."""
    observation: np.ndarray  # [obs_dim] - agent observation (position, velocity, heading, etc.)
    route: np.ndarray       # [route_dim] - route/waypoints encoding
    future_waypoints: np.ndarray  # [num_waypoints, 2] - target waypoints in world frame
    speed: float          # current speed for temporal context
    heading: float       # current heading angle


@dataclass  
class WaypointBCEpisode:
    """Full episode for waypoint BC training."""
    episode_id: str
    num_samples: int
    waypoints: np.ndarray  # [T, num_waypoints, 2]
    observations: np.ndarray  # [T, obs_dim]
    speeds: np.ndarray  # [T]
    headings: np.ndarray  # [T]
    
    def to_samples(self, future_horizon: int = 8, route_dim: int = 64) -> List[WaypointBCTrainingSample]:
        """Convert episode to list of training samples."""
        samples = []
        T = self.observations.shape[0]
        num_wp = self.waypoints.shape[1] if len(self.waypoints.shape) > 1 else future_horizon
        
        for t in range(T):
            # Future waypoints starting from current timestep, padded to fixed horizon
            future_wp = self.waypoints[t:]  # [future_horizon', 2]
            
            if future_wp.shape[0] < future_horizon:
                # Pad with last waypoint
                padding = np.repeat(future_wp[-1:], future_horizon - future_wp.shape[0], axis=0)
                future_wp = np.concatenate([future_wp, padding], axis=0)
            elif future_wp.shape[0] > future_horizon:
                # Truncate
                future_wp = future_wp[:future_horizon]
                
            sample = WaypointBCTrainingSample(
                observation=self.observations[t],
                route=self._encode_route(future_wp, route_dim),
                future_waypoints=future_wp,
                speed=self.speeds[t],
                heading=self.headings[t],
            )
            samples.append(sample)
            
        return samples
    
    def _encode_route(self, waypoints: np.ndarray, route_dim: int = 64) -> np.ndarray:
        """Encode waypoints into fixed-dim route vector."""
        # Simple: flatten and interpolate to fixed route_dim
        wp_flat = waypoints.flatten()  # [2 * horizon]
        
        # Interpolate to fixed route_dim (e.g., 64)
        if len(wp_flat) >= route_dim:
            # Subsample
            indices = np.linspace(0, len(wp_flat) - 1, route_dim, dtype=int)
            wp_flat = wp_flat[indices]
        else:
            # Interpolate
            wp_flat = np.interp(
                np.linspace(0, len(wp_flat) - 1, route_dim),
                np.arange(len(wp_flat)),
                wp_flat
            )
            
        return wp_flat.astype(np.float32)


class WaymoToBCConverter:
    """Converts Waymo episodes to BC-ready format."""
    
    def __init__(
        self,
        obs_dim: int = 64,
        route_dim: int = 64,
        future_horizon: int = 8,  # Number of future waypoints to predict
    ):
        self.obs_dim = obs_dim
        self.route_dim = route_dim
        self.future_horizon = future_horizon
        
    def convert_episode(self, episode: WaymoEpisode) -> WaypointBCEpisode:
        """Convert WaymoEpisode to WaypointBCEpisode for BC training."""
        # Extract trajectory from waymo episode
        # For now, synthesize from episode metadata
        # Real implementation: parse TFRecord features
        
        T = episode.num_frames if hasattr(episode, 'num_frames') else 100
        
        # Synthesize for demo (real impl: extract from TFRecord)
        # Position + velocity + heading
        observations = np.random.randn(T, self.obs_dim).astype(np.float32)
        
        # Waypoints: [T, future_horizon, 2] in world frame
        waypoints = np.random.randn(T, self.future_horizon, 2).astype(np.float32) * 10
        
        # Speed and heading
        speeds = np.abs(np.random.randn(T)).astype(np.float32) * 5  # m/s
        headings = np.random.uniform(-np.pi, np.pi, T).astype(np.float32)
        
        return WaypointBCEpisode(
            episode_id=episode.episode_id,
            num_samples=T,
            waypoints=waypoints,
            observations=observations,
            speeds=speeds,
            headings=headings,
        )
    
    def convert_episodes(
        self, 
        episodes: List[WaymoEpisode]
    ) -> List[WaypointBCEpisode]:
        """Convert multiple episodes."""
        return [self.convert_episode(ep) for ep in episodes]


class WaypointBCTrainingDataset:
    """PyTorch Dataset for waypoint BC training."""
    
    def __init__(self, episodes: List[WaypointBCEpisode], split: str = "train", future_horizon: int = 8):
        self.episodes = episodes
        self.split = split
        self.future_horizon = future_horizon
        self._build_index()
        
    def _build_index(self):
        """Build flat index of all samples with cached data."""
        self.sample_index = []  # (episode_idx, sample_idx)
        self._cached_samples = {}  # ep_idx -> list of samples
        route_dim = 64
        future_horizon = self.future_horizon
        
        for ep_idx, ep in enumerate(self.episodes):
            # Cache samples at build time to ensure deterministic output
            samples = ep.to_samples(future_horizon, route_dim)
            self._cached_samples[ep_idx] = samples
            for sam_idx in range(len(samples)):
                self.sample_index.append((ep_idx, sam_idx))
                
    def __len__(self) -> int:
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Dict:
        ep_idx, sam_idx = self.sample_index[idx]
        sample = self._cached_samples[ep_idx][sam_idx]
        ep = self.episodes[ep_idx]
        
        return {
            "observation": sample.observation,
            "route": sample.route,
            "future_waypoints": sample.future_waypoints,
            "speed": sample.speed,
            "heading": sample.heading,
            "episode_id": ep.episode_id,
        }


def create_bc_dataloader(
    episodes: List[WaymoEpisode],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    future_horizon: int = 8,
) -> "torch.utils.data.DataLoader":
    """Create DataLoader for waypoint BC training."""
    try:
        import torch
        from torch.utils.data import DataLoader as TorchDataLoader
    except ImportError:
        print("PyTorch not available - returning mock loader")
        return None
    
    # Convert to BC format
    converter = WaymoToBCConverter()
    bc_episodes = converter.convert_episodes(episodes)
    
    # Create dataset with fixed future horizon
    dataset = WaypointBCTrainingDataset(bc_episodes, future_horizon=future_horizon)
    
    # Custom collate to handle variable-size tensors
    def collate_fn(batch):
        import torch
        return {
            "observation": torch.tensor(np.array([b["observation"] for b in batch])),
            "route": torch.tensor(np.array([b["route"] for b in batch])),
            "future_waypoints": torch.tensor(np.array([b["future_waypoints"] for b in batch])),
            "speed": torch.tensor([b["speed"] for b in batch]),
            "heading": torch.tensor([b["heading"] for b in batch]),
            "episode_id": [b["episode_id"] for b in batch],
        }
    
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def run_bc_training(
    episodes: List[WaymoEpisode],
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: str = "out/waypoint_bc",
) -> Dict:
    """Run waypoint BC training on Waymo episodes."""
    try:
        import torch
        import torch.nn as nn
        from torch.optim import Adam
    except ImportError:
        return {
            "status": "skipped",
            "reason": "PyTorch not available",
            "episodes": len(episodes),
        }
    
    # Create dataloader
    loader = create_bc_dataloader(episodes, batch_size=batch_size)
    
    if loader is None:
        return {"status": "error", "reason": "Failed to create dataloader"}
    
    # Simple BC model
    class SimpleWaypointBC(nn.Module):
        def __init__(self, obs_dim=64, route_dim=64, future_horizon=8, num_waypoints=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + route_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, future_horizon * 2),
            )
            self.future_horizon = future_horizon
            
        def forward(self, obs, route):
            x = torch.cat([obs, route], dim=-1)
            return self.net(x).view(-1, self.future_horizon, 2)
    
    model = SimpleWaypointBC(future_horizon=8)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop - simplified: predict NEXT waypoint only
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in loader:
            obs = batch["observation"].detach().clone()
            route = batch["route"].detach().clone()
            targets = batch["future_waypoints"].detach().clone()
            
            # Take only the next waypoint (first in sequence)
            # targets: [batch, horizon, 2] -> [batch, 1, 2] 
            targets_next = targets[:, 0, :]  # [batch, 2]
            
            optimizer.zero_grad()
            preds = model(obs, route)
            loss = criterion(preds, targets_next)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "bc_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    
    return {
        "status": "success",
        "epochs": epochs,
        "final_loss": losses[-1] if losses else None,
        "checkpoint": checkpoint_path,
        "episodes": len(episodes),
    }


def discover_bc_episodes() -> List[WaymoEpisode]:
    """Discover episodes available for BC training."""
    return list(iter_episodes())


# CLI
def main():
    parser = argparse.ArgumentParser(
        description="Waymo-to-Waypoint BC Integration"
    )
    parser.add_argument("--count", action="store_true", help="Count available episodes")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for smoke test")
    parser.add_argument("--run-bc", action="store_true", help="Run BC training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of BC epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="out/waypoint_bc", help="Output directory")
    
    args = parser.parse_args()
    
    if args.count:
        episodes = discover_bc_episodes()
        print(f"Found {len(episodes)} episodes for BC training")
        return
        
    if args.smoke:
        episodes = discover_bc_episodes()
        ep_limited = episodes[:args.episodes] if episodes else []
        
        if not ep_limited:
            print("No episodes found - using synthetic fallback")
            ep_limited = [WaymoEpisode(
                episode_id=f"dummy_{i}",
                tfrecord_path="",
                num_frames=100,
                duration_s=10.0,
                distance_m=100.0,
                has_lidar=True,
                has_camera=False,
                cameras=[],
            ) for i in range(args.episodes)]
        
        # Convert to BC format
        converter = WaymoToBCConverter()
        bc_episodes = converter.convert_episodes(ep_limited)
        
        print(f"Converted {len(bc_episodes)} episodes to BC format")
        for ep in bc_episodes:
            print(f"  - {ep.episode_id}: {ep.num_samples} samples")
            
        # Test dataset
        dataset = WaypointBCTrainingDataset(bc_episodes)
        print(f"Dataset: {len(dataset)} total samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            
        return
        
    if args.run_bc:
        episodes = discover_bc_episodes()
        
        if not episodes:
            print("No episodes found - using synthetic fallback")
            episodes = [WaymoEpisode(
                episode_id=f"dummy_{i}",
                tfrecord_path="",
                num_frames=100,
                duration_s=10.0,
                distance_m=100.0,
                has_lidar=True,
                has_camera=False,
                cameras=[],
            ) for i in range(args.episodes)]
        
        result = run_bc_training(
            episodes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=args.output_dir,
        )
        
        print(f"BC training result: {json.dumps(result, indent=2)}")
        return


if __name__ == "__main__":
    main()
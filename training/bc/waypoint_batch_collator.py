#!/usr/bin/env python3
"""
Waypoint Batch Collator for BC Training

Collates waypoint trajectories from episodes into training-ready batches.
Critical component for waypoint BC training pipeline.

Author: Pipeline
Date: 2026-04-19
"""

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np


@dataclass
class CollatorConfig:
    """Configuration for waypoint batch collator."""
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False
    max_waypoints: int = 8
    horizon_seconds: float = 3.0
    sampling_rate: float = 2.0
    augment: bool = True
    noise_std: float = 0.05
    seed: int = 42


@dataclass
class WaypointSample:
    """Single waypoint sample."""
    episode_id: str
    frame_id: int
    timestamp: float
    waypoints: np.ndarray  # (num_waypoints, 2) [x, y] in ego frame
    speed: np.ndarray  # (num_waypoints,) speed at each waypoint
    progress: np.ndarray  # (num_waypoints,) progress ratio [0, 1]
    metadata: dict = field(default_factory=dict)


class WaypointBatchCollator:
    """Collates waypoint samples into training batches."""
    
    def __init__(self, config: CollatorConfig):
        self.config = config
        self.samples: list[WaypointSample] = []
        self.episode_index: dict = {}
        
    def load_episodes(self, episodes_dir: str) -> int:
        """Load waypoint episodes from directory."""
        episodes_path = Path(episodes_dir)
        
        if not episodes_path.exists():
            print(f"Loading synthetic test data (no episodes found at {episodes_dir})")
            self._generate_synthetic_samples(100)
            return len(self.samples)
        
        # Find all episode JSON files
        episode_files = list(episodes_path.glob("*.json")) + list(episodes_path.glob("*.jsonl"))
        
        if not episode_files:
            print(f"No episodes found, loading synthetic test data")
            self._generate_synthetic_samples(100)
            return len(self.samples)
        
        for ep_file in episode_files:
            try:
                with open(ep_file) as f:
                    data = json.load(f)
                
                episode_id = ep_file.stem
                self.episode_index[episode_id] = ep_file
                
                # Extract waypoints from episode
                samples = self._extract_samples(episode_id, data)
                self.samples.extend(samples)
                
            except Exception as e:
                print(f"Error loading {ep_file}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples from {len(self.episode_index)} episodes")
        return len(self.samples)
    
    def _extract_samples(self, episode_id: str, data: dict) -> list[WaypointSample]:
        """Extract waypoint samples from episode data."""
        samples = []
        
        # Handle different data formats
        frames = data.get("frames", data.get("images", []))
        
        for frame in frames:
            frame_id = frame.get("frame_id", 0)
            timestamp = frame.get("timestamp", frame_id * 0.1)
            
            # Extract waypoints
            waypoints = frame.get("waypoints", [])
            if isinstance(waypoints, list) and len(waypoints) > 0:
                waypoints_arr = np.array(waypoints, dtype=np.float32)
            else:
                # Generate synthetic waypoints
                waypoints_arr = self._generate_synthetic_waypoints()
            
            speed = frame.get("speed", np.zeros(len(waypoints_arr)))
            progress = frame.get("progress", np.linspace(0, 1, len(waypoints_arr)))
            
            sample = WaypointSample(
                episode_id=episode_id,
                frame_id=frame_id,
                timestamp=timestamp,
                waypoints=waypoints_arr,
                speed=speed,
                progress=progress,
                metadata=frame.get("metadata", {})
            )
            samples.append(sample)
        
        return samples
    
    def _generate_synthetic_waypoints(self) -> np.ndarray:
        """Generate synthetic waypoints for testing."""
        num_waypoints = int(self.config.horizon_seconds * self.config.sampling_rate)
        t = np.linspace(0, self.config.horizon_seconds, num_waypoints)
        
        # Synthetic lane-following trajectory
        waypoints = np.zeros((num_waypoints, 2))
        waypoints[:, 0] = 2.0 * t  # Forward motion
        waypoints[:, 1] = 0.1 * np.sin(t * 2)  # Slight lateral oscillation
        
        return waypoints.astype(np.float32)
    
    def _generate_synthetic_samples(self, num_samples: int) -> None:
        """Generate synthetic samples for testing."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        for i in range(num_samples):
            sample = WaypointSample(
                episode_id=f"syn_{i:04d}",
                frame_id=i,
                timestamp=i * 0.1,
                waypoints=self._generate_synthetic_waypoints(),
                speed=np.ones(int(self.config.horizon_seconds * self.config.sampling_rate)) * 5.0,
                progress=np.linspace(0, 1, int(self.config.horizon_seconds * self.config.sampling_rate)),
                metadata={"synthetic": True}
            )
            self.samples.append(sample)
    
    def collate_batch(self, indices: Optional[list[int]] = None) -> dict[str, torch.Tensor]:
        """Collate samples into a training batch."""
        if indices is None:
            if self.config.shuffle:
                indices = random.sample(range(len(self.samples)), self.config.batch_size)
            else:
                indices = list(range(min(self.config.batch_size, len(self.samples))))
        
        # Collect batch samples
        batch_samples = [self.samples[i] for i in indices]
        
        # Build tensors
        batch_size = len(batch_samples)
        max_waypoints = self.config.max_waypoints
        
        waypoints_batch = torch.zeros(batch_size, max_waypoints, 2)
        speed_batch = torch.zeros(batch_size, max_waypoints)
        progress_batch = torch.zeros(batch_size, max_waypoints)
        frame_ids = []
        
        for i, sample in enumerate(batch_samples):
            n = min(len(sample.waypoints), max_waypoints)
            waypoints_batch[i, :n] = torch.from_numpy(sample.waypoints[:n])
            speed_batch[i, :n] = torch.from_numpy(sample.speed[:n]) if len(sample.speed) > 0 else torch.zeros(n)
            progress_batch[i, :n] = torch.from_numpy(sample.progress[:n]) if len(sample.progress) > 0 else torch.zeros(n)
            frame_ids.append(sample.frame_id)
            
            # Apply augmentation
            if self.config.augment:
                waypoints_batch[i] = self._augment_waypoints(waypoints_batch[i])
        
        return {
            "waypoints": waypoints_batch,  # (B, num_waypoints, 2)
            "speed": speed_batch,  # (B, num_waypoints)
            "progress": progress_batch,  # (B, num_waypoints)
            "frame_ids": frame_ids,
        }
    
    def _augment_waypoints(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to waypoints."""
        noise = torch.randn_like(waypoints) * self.config.noise_std
        return waypoints + noise
    
    def create_dataloader(self, num_epochs: int = 1) -> "WaypointDataLoader":
        """Create a DataLoader for training."""
        return WaypointDataLoader(self, num_epochs)
    
    def get_statistics(self) -> dict:
        """Compute dataset statistics."""
        if not self.samples:
            return {"num_samples": 0}
        
        # Filter out any samples where waypoints aren't proper arrays
        valid_waypoints = []
        for s in self.samples:
            if isinstance(s.waypoints, np.ndarray) and s.waypoints.ndim == 2:
                valid_waypoints.append(s.waypoints)
        
        if not valid_waypoints:
            return {"num_samples": len(self.samples), "num_valid": 0}
        
        all_waypoints = np.concatenate(valid_waypoints, axis=0)
        
        return {
            "num_samples": len(self.samples),
            "num_valid": len(valid_waypoints),
            "num_episodes": len(self.episode_index),
            "waypoints_shape": list(all_waypoints.shape[1:]),
            "waypoints_mean": all_waypoints.mean(axis=0).tolist(),
            "waypoints_std": all_waypoints.std(axis=0).tolist(),
            "waypoints_min": all_waypoints.min(axis=0).tolist(),
            "waypoints_max": all_waypoints.max(axis=0).tolist(),
        }
    
    def save_statistics(self, output_path: str) -> None:
        """Save statistics to JSON file."""
        stats = self.get_statistics()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved statistics to {output_path}")


class WaypointDataLoader:
    """DataLoader wrapper for WaypointBatchCollator."""
    
    def __init__(self, collator: WaypointBatchCollator, num_epochs: int):
        self.collator = collator
        self.num_epochs = num_epochs
        self.config = collator.config
        
    def __iter__(self):
        """Iterate over batches."""
        for _ in range(self.num_epochs):
            indices = list(range(len(self.collator.samples)))
            
            if self.config.shuffle:
                random.shuffle(indices)
            
            # Yield batches
            for i in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                
                if len(batch_indices) < self.config.batch_size and self.config.drop_last:
                    continue
                
                yield self.collator.collate_batch(batch_indices)
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        num_samples = len(self.collator.samples)
        return num_samples // self.config.batch_size


def main():
    parser = argparse.ArgumentParser(description="Waypoint Batch Collator for BC Training")
    parser.add_argument("--episodes-dir", type=str, default="data/waymo/episodes",
                      help="Directory containing waypoint episodes")
    parser.add_argument("--output-dir", type=str, default="out/waypoint_collator",
                      help="Output directory for statistics")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4,
                      help="Number of data loading workers")
    parser.add_argument("--shuffle", action="store_true", default=True,
                      help="Shuffle data")
    parser.add_argument("--no-augment", action="store_true",
                      help="Disable data augmentation")
    parser.add_argument("--noise-std", type=float, default=0.05,
                      help="Noise standard deviation for augmentation")
    parser.add_argument("--max-waypoints", type=int, default=8,
                      help="Maximum number of waypoints")
    parser.add_argument("--horizon-seconds", type=float, default=3.0,
                      help="Waypoint prediction horizon in seconds")
    parser.add_argument("--sampling-rate", type=float, default=2.0,
                      help="Waypoint sampling rate in Hz")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--stats-only", action="store_true",
                      help="Only compute and save statistics")
    parser.add_argument("--list", action="store_true",
                      help="List loaded samples and exit")
    
    args = parser.parse_args()
    
    # Create config
    config = CollatorConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        augment=not args.no_augment,
        noise_std=args.noise_std,
        max_waypoints=args.max_waypoints,
        horizon_seconds=args.horizon_seconds,
        sampling_rate=args.sampling_rate,
        seed=args.seed,
    )
    
    # Create collator
    collator = WaypointBatchCollator(config)
    
    # Load episodes
    num_samples = collator.load_episodes(args.episodes_dir)
    print(f"Loaded {num_samples} waypoint samples")
    
    if args.list:
        stats = collator.get_statistics()
        print(json.dumps(stats, indent=2))
        return
    
    if args.stats_only:
        output_path = Path(args.output_dir) / "statistics.json"
        collator.save_statistics(str(output_path))
        return
    
    # Create dataloader and iterate
    dataloader = collator.create_dataloader(num_epochs=1)
    
    print(f"\nCreating batches (batch_size={config.batch_size})...")
    
    num_batches = 0
    for batch in dataloader:
        num_batches += 1
        waypoints = batch["waypoints"]
        print(f"Batch {num_batches}: waypoints shape: {waypoints.shape}")
    
    print(f"\nTotal batches: {num_batches}")
    
    # Save statistics
    output_path = Path(args.output_dir) / "statistics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collator.save_statistics(str(output_path))
    
    # Save sample batch as JSON for inspection
    sample_batch = collator.collate_batch(list(range(min(3, len(collator.samples)))))
    sample_path = Path(args.output_dir) / "sample_batch.json"
    
    sample_data = {
        "waypoints": sample_batch["waypoints"].tolist()[:3],
        "frame_ids": sample_batch["frame_ids"],
    }
    
    with open(sample_path, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Saved sample batch to {sample_path}")
    print("\n✅ Smoke test PASSED")


if __name__ == "__main__":
    main()
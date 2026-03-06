"""
Waymo Episode Dataset for Driving-First Pipeline.

This module provides utilities for loading and processing Waymo motion data
for use in SSL pretraining and waypoint behavior cloning.

Usage:
    from training.data.waymo_episode_dataset import WaymoEpisodeDataset
    
    dataset = WaymoEpisodeDataset(
        episode_paths=["out/episodes/*.json"],
        sequence_length=4,
        sample_rate=1
    )
"""

import json
import glob
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset


class WaymoEpisode:
    """Represents a single Waymo driving episode."""
    
    def __init__(self, episode_path: str):
        self.path = episode_path
        with open(episode_path, 'r') as f:
            self.data = json.load(f)
        
        self.episode_id = self.data.get('episode_id', Path(episode_path).stem)
        self.frames = self.data.get('frames', [])
        self.metadata = self.data.get('metadata', {})
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    def get_frame(self, idx: int) -> Dict:
        """Get frame at index."""
        if 0 <= idx < len(self.frames):
            return self.frames[idx]
        raise IndexError(f"Frame index {idx} out of range [0, {len(self.frames)})")
    
    def get_sequence(self, start_idx: int, length: int) -> List[Dict]:
        """Get consecutive frames starting at start_idx."""
        end_idx = start_idx + length
        if end_idx > len(self.frames):
            return []
        return self.frames[start_idx:end_idx]
    
    def get_waypoints(self, frame_idx: int, horizon: int = 8) -> np.ndarray:
        """Get future waypoints from a frame.
        
        Args:
            frame_idx: Current frame index
            horizon: Number of future waypoints
            
        Returns:
            Array of shape (horizon, 2) with (x, y) waypoints
        """
        frame = self.get_frame(frame_idx)
        waypoints = frame.get('waypoints', [])
        if len(waypoints) >= horizon:
            return np.array(waypoints[:horizon])
        # Pad if needed
        result = np.zeros((horizon, 2))
        for i, wp in enumerate(waypoints):
            if i < horizon:
                result[i] = wp
        return result
    
    def get_observation(self, frame_idx: int) -> Dict:
        """Get observation dict for a frame."""
        frame = self.get_frame(frame_idx)
        return {
            'image': frame.get('image'),
            'speed': frame.get('speed', 0.0),
            'steering': frame.get('steering', 0.0),
            'timestamp': frame.get('timestamp', 0.0),
        }


class WaymoEpisodeDataset(Dataset):
    """PyTorch Dataset for Waymo episodes.
    
    Supports:
    - Temporal sequence sampling
    - Multiple episode glob patterns
    - Filtering by episode metadata
    
    Args:
        episode_paths: Glob patterns for episode JSON files
        sequence_length: Number of frames in temporal context
        sample_rate: Sample every N frames
        min_frames: Minimum frames required per episode
        transform: Optional transform applied to observations
    """
    
    def __init__(
        self,
        episode_paths: List[str],
        sequence_length: int = 4,
        sample_rate: int = 1,
        min_frames: int = 10,
        transform: Optional[callable] = None,
    ):
        self.episode_paths = []
        for pattern in episode_paths:
            self.episode_paths.extend(glob.glob(pattern))
        
        self.episode_paths = sorted(set(self.episode_paths))
        if not self.episode_paths:
            raise ValueError(f"No episodes found for patterns: {episode_paths}")
        
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.min_frames = min_frames
        self.transform = transform
        
        # Build episode index
        self._build_index()
    
    def _build_index(self):
        """Build flat index of (episode_idx, frame_idx) pairs."""
        self.episodes: List[WaymoEpisode] = []
        self.valid_indices: List[Tuple[int, int]] = []
        
        for ep_idx, ep_path in enumerate(self.episode_paths):
            try:
                episode = WaymoEpisode(ep_path)
            except Exception as e:
                print(f"Warning: Failed to load {ep_path}: {e}")
                continue
            
            if episode.num_frames < self.min_frames:
                continue
            
            self.episodes.append(episode)
            
            # Add valid starting indices for sequences
            max_start = episode.num_frames - (self.sequence_length * self.sample_rate)
            for start_idx in range(0, max_start + 1, self.sample_rate):
                self.valid_indices.append((len(self.episodes) - 1, start_idx))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sequence sample.
        
        Returns:
            Dict with:
                - frames: List of frame observations
                - waypoints: Future waypoints from final frame (horizon, 2)
                - episode_id: Episode identifier
                - frame_idx: Starting frame index
        """
        ep_idx, start_idx = self.valid_indices[idx]
        episode = self.episodes[ep_idx]
        
        # Collect frames
        frames = []
        for i in range(self.sequence_length):
            frame_idx = start_idx + (i * self.sample_rate)
            obs = episode.get_observation(frame_idx)
            frames.append(obs)
        
        # Get future waypoints from final frame
        final_frame_idx = start_idx + ((self.sequence_length - 1) * self.sample_rate)
        waypoints = episode.get_waypoints(final_frame_idx)
        
        sample = {
            'frames': frames,
            'waypoints': torch.from_numpy(waypoints).float(),
            'episode_id': episode.episode_id,
            'frame_idx': final_frame_idx,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class WaymoEpisodeCollator:
    """Custom collator for batching Waymo episode sequences.
    
    Handles variable-length sequences and prepares tensors.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate a batch of samples.
        
        Args:
            batch: List of sample dicts from WaymoEpisodeDataset
            
        Returns:
            Batched dict with:
                - images: (B, T, C, H, W) tensor
                - speeds: (B, T) tensor
                - steerings: (B, T) tensor
                - waypoints: (B, H, 2) tensor
                - metadata: List of (episode_id, frame_idx) tuples
        """
        batch_size = len(batch)
        
        # Get dimensions from first sample
        T = len(batch[0]['frames'])
        horizon = batch[0]['waypoints'].shape[0]
        
        # Assuming images are already tensors or paths
        # For now, collect as list (actual image loading would be in dataset)
        images = [sample['frames'] for sample in batch]
        
        # Stack scalar features
        speeds = torch.stack([
            torch.stack([f.get('speed', 0.0) for f in sample['frames']])
            for sample in batch
        ])
        
        steerings = torch.stack([
            torch.stack([f.get('steering', 0.0) for f in sample['frames']])
            for sample in batch
        ])
        
        # Stack waypoints
        waypoints = torch.stack([sample['waypoints'] for sample in batch])
        
        # Collect metadata
        metadata = [
            (sample['episode_id'], sample['frame_idx'])
            for sample in batch
        ]
        
        return {
            'images': images,  # Keep as list of lists for flexible image loading
            'speeds': speeds.float(),
            'steerings': steerings.float(),
            'waypoints': waypoints.float(),
            'metadata': metadata,
        }


def load_episode_paths(
    root_dir: str,
    split: str = 'train',
    limit: Optional[int] = None,
) -> List[str]:
    """Load episode paths from a root directory.
    
    Args:
        root_dir: Root directory containing episode JSON files
        split: Data split (train/val/test)
        limit: Optional limit on number of episodes
        
    Returns:
        List of episode file paths
    """
    pattern = os.path.join(root_dir, split, '**', '*.json')
    paths = sorted(glob.glob(pattern, recursive=True))
    
    if limit:
        paths = paths[:limit]
    
    return paths


def create_waymo_dataloaders(
    train_glob: str,
    val_glob: str,
    batch_size: int = 32,
    sequence_length: int = 4,
    num_workers: int = 4,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and val dataloaders for Waymo episodes.
    
    Args:
        train_glob: Glob pattern for training episodes
        val_glob: Glob pattern for validation episodes
        batch_size: Batch size
        sequence_length: Temporal context length
        num_workers: DataLoader workers
        **dataset_kwargs: Additional args to WaymoEpisodeDataset
        
    Returns:
        (train_loader, val_loader) tuple
    """
    train_dataset = WaymoEpisodeDataset(
        episode_paths=[train_glob],
        sequence_length=sequence_length,
        **dataset_kwargs,
    )
    
    val_dataset = WaymoEpisodeDataset(
        episode_paths=[val_glob],
        sequence_length=sequence_length,
        **dataset_kwargs,
    )
    
    collator = WaymoEpisodeCollator()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    
    return train_loader, val_loader


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes-glob', type=str, required=True)
    parser.add_argument('--sequence-length', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    
    # Create dataset
    dataset = WaymoEpisodeDataset(
        episode_paths=[args.episodes_glob],
        sequence_length=args.sequence_length,
    )
    
    print(f"Loaded {len(dataset.episodes)} episodes")
    print(f"Total sequences: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample waypoints shape: {sample['waypoints'].shape}")
    print(f"Sample episode: {sample['episode_id']}")

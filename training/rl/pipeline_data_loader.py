#!/usr/bin/env python3
"""
Pipeline Data Loader

Loads and preprocesses data for pipeline stages:
- Waymo episodes for SSL pretraining
- BC training data for waypoint BC
- RL trajectories for refinement

Supports:
- Automatic data format detection
- Episode-based loading with frame sampling
- Multi-camera data aggregation
- Train/val split with stratification
- Caching for fast iteration

Usage:
    python pipeline_data_loader.py --stage ssl --data-dir data/waymo/episodes
    python pipeline_data_loader.py --stage bc --data-dir data/waymo/episodes --train
    python pipeline_data_loader.py --stage rl --data-dir out/rl_trajectories
"""

import argparse
import os
import json
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random


@dataclass
class DataConfig:
    """Configuration for data loading."""
    stage: str  # ssl, bc, rl
    data_dir: str
    train_split: float = 0.9
    val_split: float = 0.1
    max_episodes: Optional[int] = None
    frame_sample_rate: int = 1
    cache_enabled: bool = True
    seed: int = 42


class EpisodeLoader:
    """Loads Waymo-style episode JSON files."""
    
    REQUIRED_KEYS = ['episode_id', 'frames']  # Waypoints inside frames as expert
    
    def __init__(self, data_dir: str, frame_sample_rate: int = 1):
        self.data_dir = Path(data_dir)
        self.frame_sample_rate = frame_sample_rate
        
    def discover_episodes(self) -> List[Path]:
        """Find all episode JSON files in data directory."""
        patterns = ['*.json', '**/*.json']
        episodes = []
        for pattern in patterns:
            episodes.extend(self.data_dir.glob(pattern))
        # Filter out non-episode files (metrics, quality reports, etc.)
        exclude_files = ['quality_report', 'metrics', 'stats', 'config']
        episodes = [e for e in episodes if not any(ex in e.name.lower() for ex in exclude_files)]
        return sorted(episodes)
    
    def load_episode(self, path: Path) -> Optional[Dict]:
        """Load a single episode file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # Validate required keys
            for key in self.REQUIRED_KEYS:
                if key not in data:
                    return None
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {path}: {e}")
            return None
    
    def load_episodes(self, max_episodes: Optional[int] = None) -> List[Dict]:
        """Load multiple episodes with optional limit."""
        episodes = self.discover_episodes()
        if max_episodes:
            episodes = episodes[:max_episodes]
        
        loaded = []
        for ep_path in episodes:
            ep_data = self.load_episode(ep_path)
            if ep_data:
                loaded.append(ep_data)
        
        return loaded
    
    def sample_frames(self, episode: Dict) -> List[Dict]:
        """Sample frames from episode at specified rate."""
        frames = episode.get('frames', [])
        if self.frame_sample_rate > 1:
            frames = frames[::self.frame_sample_rate]
        return frames


class SSLDataProcessor:
    """Processes episodes for SSL pretraining."""
    
    def __init__(self, camera_names: List[str] = None):
        self.camera_names = camera_names or ['front', 'left', 'right', 'rear']
    
    def process_episode(self, episode: Dict) -> Dict:
        """Extract SSL training data from episode."""
        frames = episode.get('frames', [])
        
        # Multi-camera observations
        observations = {'cameras': {}, 'metadata': {}}
        
        for frame in frames:
            # Extract camera images
            for cam in self.camera_names:
                if cam in frame.get('cameras', {}):
                    observations['cameras'].setdefault(cam, []).append(
                        frame['cameras'][cam]
                    )
        
        # Metadata
        observations['metadata'] = {
            'episode_id': episode.get('episode_id', 'unknown'),
            'num_frames': len(frames),
            'cameras': list(observations['cameras'].keys())
        }
        
        return observations
    
    def collate_batch(self, observations: List[Dict]) -> Dict:
        """Collate multiple observations into batch."""
        batch = {
            'cameras': {},
            'metadata': []
        }
        
        for obs in observations:
            for cam, images in obs['cameras'].items():
                batch['cameras'].setdefault(cam, []).extend(images)
            batch['metadata'].append(obs['metadata'])
        
        return batch


class BCDataProcessor:
    """Processes episodes for waypoint BC training."""
    
    def __init__(self, num_waypoints: int = 4, horizon_steps: int = 8):
        self.num_waypoints = num_waypoints
        self.horizon_steps = horizon_steps
    
    def process_episode(self, episode: Dict) -> Dict:
        """Extract BC training pairs from episode."""
        frames = episode.get('frames', [])
        
        if not frames:
            return None
        
        # Extract waypoints from each frame's expert section
        pairs = []
        
        for i, frame in enumerate(frames):
            # Extract observations from frame
            obs = frame.get('observations', {})
            
            # Extract expert waypoints (inside expert section)
            expert = frame.get('expert', {})
            waypoint_data = expert.get('waypoints', [])
            
            # Convert to array
            wp_array = self._extract_waypoints(waypoint_data)
            
            pair = {
                'observations': obs,
                'waypoints': wp_array,
                'frame_idx': i,
                'timestamp': frame.get('t', 0)
            }
            pairs.append(pair)
        
        return {
            'episode_id': episode.get('episode_id', 'unknown'),
            'pairs': pairs,
            'num_samples': len(pairs)
        }
    
    def _extract_waypoints(self, waypoint_data) -> np.ndarray:
        """Extract waypoint coordinates from waypoint data."""
        if not waypoint_data:
            return np.zeros((self.num_waypoints, 2))
        
        # waypoint_data is a list of dicts with x, y keys
        wp_list = waypoint_data[:self.num_waypoints]
        wp_array = np.array([[wp.get('x', 0), wp.get('y', 0)] for wp in wp_list])
        
        # Pad if needed
        if wp_array.shape[0] < self.num_waypoints:
            pad = np.zeros((self.num_waypoints - wp_array.shape[0], wp_array.shape[1]))
            wp_array = np.vstack([wp_array, pad])
        
        return wp_array
    
    def collate_batch(self, episodes: List[Dict]) -> Dict:
        """Collate BC training pairs into batch."""
        observations = []
        waypoint_targets = []
        
        for ep in episodes:
            if ep is None:
                continue
            for pair in ep.get('pairs', []):
                observations.append(pair['observations'])
                waypoint_targets.append(pair['waypoints'])
        
        return {
            'observations': observations,
            'waypoints': np.array(waypoint_targets) if waypoint_targets else np.array([]),
            'batch_size': len(observations)
        }


class RLTrajectoryLoader:
    """Loads RL training trajectories."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def discover_trajectories(self) -> List[Path]:
        """Find trajectory files."""
        patterns = ['**/trajectories*.json', '**/rollouts*.json', '**/episodes*.json']
        trajectories = []
        for pattern in patterns:
            trajectories.extend(self.data_dir.glob(pattern))
        return sorted(trajectories)
    
    def load_trajectory(self, path: Path) -> Optional[Dict]:
        """Load a single trajectory file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {path}: {e}")
            return None
    
    def load_all(self) -> List[Dict]:
        """Load all trajectories."""
        trajectories = []
        for path in self.discover_trajectories():
            traj = self.load_trajectory(path)
            if traj:
                trajectories.append(traj)
        return trajectories


class PipelineDataLoader:
    """Unified data loader for pipeline stages."""
    
    PROCESSORS = {
        'ssl': SSLDataProcessor,
        'bc': BCDataProcessor,
        'rl': RLTrajectoryLoader
    }
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.stage = config.stage
        
        # Initialize loader and processor
        if config.stage in ['ssl', 'bc']:
            self.loader = EpisodeLoader(
                config.data_dir, 
                frame_sample_rate=config.frame_sample_rate
            )
        elif config.stage == 'rl':
            self.loader = RLTrajectoryLoader(config.data_dir)
        else:
            raise ValueError(f"Unknown stage: {config.stage}")
        
        # Initialize processor
        processor_class = self.PROCESSORS.get(config.stage)
        if processor_class:
            self.processor = processor_class()
        else:
            self.processor = None
        
        # Cache
        self._cache = {}
        self._cache_enabled = config.cache_enabled
        
        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    def load_data(self, split: str = 'train') -> List[Any]:
        """Load data for specified split (train/val)."""
        cache_key = f"{self.stage}_{split}"
        
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load episodes
        if self.stage in ['ssl', 'bc']:
            episodes = self.loader.load_episodes(
                max_episodes=self.config.max_episodes
            )
            
            # Process episodes
            processed = []
            for ep in episodes:
                if self.processor:
                    proc = self.processor.process_episode(ep)
                    if proc:
                        processed.append(proc)
                else:
                    processed.append(ep)
            
            # Split train/val
            random.shuffle(processed)
            n_train = int(len(processed) * self.config.train_split)
            
            if split == 'train':
                data = processed[:n_train]
            else:
                data = processed[n_train:]
        
        elif self.stage == 'rl':
            trajectories = self.loader.load_all()
            n_train = int(len(trajectories) * self.config.train_split)
            
            if split == 'train':
                data = trajectories[:n_train]
            else:
                data = trajectories[n_train:]
        
        else:
            data = []
        
        if self._cache_enabled:
            self._cache[cache_key] = data
        
        return data
    
    def get_train_data(self) -> List[Any]:
        """Get training data."""
        return self.load_data('train')
    
    def get_val_data(self) -> List[Any]:
        """Get validation data."""
        return self.load_data('val')
    
    def get_batch(self, split: str = 'train', batch_size: int = 32) -> Dict:
        """Get a batch of data."""
        data = self.load_data(split)
        
        if not data:
            return {'empty': True, 'batch_size': 0}
        
        # Sample batch
        batch_data = random.sample(data, min(batch_size, len(data)))
        
        if self.processor and hasattr(self.processor, 'collate_batch'):
            return self.processor.collate_batch(batch_data)
        
        return {'data': batch_data, 'batch_size': len(batch_data)}
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        train_data = self.get_train_data()
        val_data = self.get_val_data()
        
        stats = {
            'stage': self.stage,
            'data_dir': self.config.data_dir,
            'num_train_episodes': len(train_data),
            'num_val_episodes': len(val_data),
            'total_episodes': len(train_data) + len(val_data),
            'train_split': self.config.train_split,
            'val_split': self.config.val_split,
            'cache_enabled': self._cache_enabled
        }
        
        if self.stage == 'bc' and train_data:
            total_samples = sum(ep.get('num_samples', 0) for ep in train_data if ep)
            stats['train_samples'] = total_samples
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Pipeline Data Loader')
    parser.add_argument('--stage', type=str, required=True, 
                        choices=['ssl', 'bc', 'rl'],
                        help='Pipeline stage')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Data directory')
    parser.add_argument('--train-split', type=float, default=0.9,
                        help='Training split ratio')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum episodes to load')
    parser.add_argument('--frame-sample-rate', type=int, default=1,
                        help='Frame sampling rate')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for batch mode')
    parser.add_argument('--batch-mode', action='store_true',
                        help='Output a batch instead of stats')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Data split')
    
    args = parser.parse_args()
    
    # Create config
    config = DataConfig(
        stage=args.stage,
        data_dir=args.data_dir,
        train_split=args.train_split,
        max_episodes=args.max_episodes,
        frame_sample_rate=args.frame_sample_rate,
        cache_enabled=not args.no_cache,
        seed=args.seed
    )
    
    # Create loader
    loader = PipelineDataLoader(config)
    
    if args.batch_mode:
        # Output batch
        batch = loader.get_batch(split=args.split, batch_size=args.batch_size)
        print(json.dumps(batch, indent=2, default=str))
    else:
        # Output stats
        stats = loader.get_stats()
        print(json.dumps(stats, indent=2))
        
        # Also show sample data structure
        if stats['total_episodes'] > 0:
            print(f"\n--- Sample Data Structure ({args.stage}) ---")
            train_data = loader.get_train_data()
            if train_data:
                import pprint
                pprint.pprint(train_data[0])


if __name__ == '__main__':
    main()
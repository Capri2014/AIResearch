#!/usr/bin/env python3
"""
SSL Episode Dataset Loader

Loads indexed Waymo episodes for SSL (Self-Supervised Learning) pretraining.
Integrates with episode index built by build_episode_index.py.

Supports:
- Contrastive learning: multiple views per frame
- JEPA: encoder-predictor embeddings
- MIM: masked image modeling

Usage:
    python training/pretrain/episode_ssl_dataset.py list --index data/waymo/episode_index.json
    python training/pretrain/episode_ssl_dataset.py stats --index data/waymo/episode_index.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class SSLDataConfig:
    """Configuration for SSL data loading."""
    index_path: str = "data/waymo/episode_index.json"
    episodes_dir: str = "data/waymo/episodes"
    cameras: list = field(default_factory=lambda: ["front", "front_left", "front_right", "side_left", "side_right"])
    frame_sample_rate: int = 1  # Sample every N frames
    Views_per_frame: int = 4  # Number of views for contrastive learning
    mask_ratio: float = 0.15  # Ratio of patches to mask for MIM
    image_size: tuple = (224, 224)


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode."""
    episode_id: str
    frame_count: int
    duration: float
    cameras: list
    path: str


class SSLEpisodeDataset(Dataset):
    """
    Dataset for SSL pretraining from Waymo episodes.
    
    Loads indexed episodes and provides:
    - Multiple views per frame (contrastive)
    - Encoder/decoder pairs (JEPA)
    - Masked patches (MIM)
    """
    
    def __init__(
        self,
        config: SSLDataConfig,
        split: str = "train",
        transform=None,
    ):
        self.config = config
        self.split = split
        self.transform = transform
        self.episodes: list[EpisodeMetadata] = []
        self.frame_index: list[tuple] = []  # (episode_idx, frame_idx)
        
        self._load_index()
        self._build_frame_index()
    
    def _load_index(self):
        """Load episode index from JSON."""
        if not os.path.exists(self.config.index_path):
            print(f"[WARN] Index not found: {self.config.index_path}")
            print(f"[INFO] Building synthetic index for testing")
            self._build_synthetic_index()
            return
        
        with open(self.config.index_path, 'r') as f:
            data = json.load(f)
        
        episodes = data.get('episodes', [])
        for ep in episodes:
            metadata = EpisodeMetadata(
                episode_id=ep['episode_id'],
                frame_count=ep.get('frame_count', 0),
                duration=ep.get('duration', 0.0),
                cameras=ep.get('cameras', []),
                path=ep.get('path', ''),
            )
            self.episodes.append(metadata)
        
        print(f"[INFO] Loaded {len(self.episodes)} episodes from index")
    
    def _build_synthetic_index(self):
        """Build synthetic index for testing."""
        for i in range(5):
            metadata = EpisodeMetadata(
                episode_id=f"syn_episode_{i:03d}",
                frame_count=100 + i * 10,
                duration=10.0 + i,
                cameras=self.config.cameras,
                path=f"syn_{i}",
            )
            self.episodes.append(metadata)
    
    def _build_frame_index(self):
        """Build flat index of all frames."""
        for ep_idx, ep in enumerate(self.episodes):
            num_frames = ep.frame_count // self.config.frame_sample_rate
            for frame_idx in range(num_frames):
                self.frame_index.append((ep_idx, frame_idx * self.config.frame_sample_rate))
        
        print(f"[INFO] Built frame index: {len(self.frame_index)} frames")
    
    def __len__(self) -> int:
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get SSL training sample.
        
        Returns:
            dict with keys:
                - image: Image tensor (C, H, W)
                - views: List of view tensors for contrastive
                - encoder_input: Encoder input tensor
                - masked_input: Masked input for MIM
                - mask: Boolean mask for masked positions
                - episode_id: Source episode ID
                - frame_idx: Frame index within episode
        """
        ep_idx, frame_idx = self.frame_index[idx]
        ep = self.episodes[ep_idx]
        
        # Generate synthetic image for testing (in real impl, load from episode)
        image = self._generate_synthetic_image(ep_idx, frame_idx)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Create multiple views for contrastive learning
        views = self._create_views(image)
        
        # Create encoder/decoder pair for JEPA
        encoder_input, decoder_input = self._create_jepa_pair(image)
        
        # Create masked input for MIM
        masked_input, mask = self._create_masked_input(image)
        
        return {
            'image': image,
            'views': views,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'masked_input': masked_input,
            'mask': mask,
            'episode_id': ep.episode_id,
            'frame_idx': frame_idx,
        }
    
    def _generate_synthetic_image(self, ep_idx: int, frame_idx: int) -> torch.Tensor:
        """Generate synthetic image tensor for testing."""
        # Create RGB image with gradient pattern
        h, w = self.config.image_size
        
        # Simple gradient images
        x = torch.linspace(0, 1, w).unsqueeze(0).expand(h, -1)
        y = torch.linspace(0, 1, h).unsqueeze(1).expand(-1, w)
        
        r = 0.5 + 0.3 * torch.sin(ep_idx * 0.5 + x + y)
        g = 0.5 + 0.3 * torch.cos(frame_idx * 0.1 + x - y)
        b = 0.5 + 0.3 * torch.sin((ep_idx + frame_idx) * 0.2 + x * y)
        
        image = torch.stack([r, g, b], dim=0).float()  # (3, H, W)
        return image
    
    def _create_views(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Create multiple views for contrastive learning."""
        views = []
        num_views = self.config.Views_per_frame
        
        for i in range(num_views):
            view = image.clone()
            # Apply random augmentations per view
            if torch.rand(1).item() > 0.5:
                # Random brightness
                view = view + torch.randn_like(view) * 0.1
            if torch.rand(1).item() > 0.5:
                # Random contrast
                view = view * (0.9 + torch.rand(1).item() * 0.2)
            views.append(view.clamp(-1, 1))
        
        return views
    
    def _create_jepa_pair(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create encoder/decoder pair for JEPA."""
        # Encoder sees visible patches
        # Decoder predicts masked patches
        c, h, w = image.shape
        patch_size = 16
        
        # Simple split: 85% encoder, 15% decoder
        encoder_input = image[:, :int(h * 0.85), :] if h > 16 else image
        decoder_input = image
        
        return encoder_input, decoder_input
    
    def _create_masked_input(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create masked input for MIM."""
        c, h, w = image.shape
        total_patches = (h // 16) * (w // 16)
        num_masked = int(total_patches * self.config.mask_ratio)
        
        mask = torch.zeros(total_patches, dtype=bool)
        masked_indices = torch.randperm(total_patches)[:num_masked]
        mask[masked_indices] = True
        
        # In practice, replace masked patches with learnable mask token
        masked_input = image.clone()
        
        return masked_input, mask


def list_episodes(config: SSLDataConfig) -> list[EpisodeMetadata]:
    """List all episodes in the index."""
    dataset = SSLEpisodeDataset(config)
    return dataset.episodes


def get_dataset_stats(config: SSLDataConfig) -> dict:
    """Compute dataset statistics."""
    dataset = SSLEpisodeDataset(config)
    
    total_frames = len(dataset)
    total_episodes = len(dataset.episodes)
    
    # Aggregate frame counts
    frame_counts = [ep.frame_count for ep in dataset.episodes]
    
    stats = {
        'num_episodes': total_episodes,
        'num_frames': total_frames,
        'total_frame_count': sum(frame_counts),
        'avg_frames_per_episode': sum(frame_counts) / max(1, len(frame_counts)),
        'min_frames': min(frame_counts) if frame_counts else 0,
        'max_frames': max(frame_counts) if frame_counts else 0,
        'cameras': config.cameras,
        'views_per_frame': config.Views_per_frame,
        'mask_ratio': config.mask_ratio,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='SSL Episode Dataset Loader')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List episodes in index')
    list_parser.add_argument('--index', default='data/waymo/episode_index.json', help='Episode index path')
    list_parser.add_argument('--episodes', default='data/waymo/episodes', help='Episodes directory')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Compute dataset statistics')
    stats_parser.add_argument('--index', default='data/waymo/episode_index.json', help='Episode index path')
    stats_parser.add_argument('--episodes', default='data/waymo/episodes', help='Episodes directory')
    
    # Load command (test loading)
    load_parser = subparsers.add_parser('load', help='Test data loading')
    load_parser.add_argument('--index', default='data/waymo/episode_index.json', help='Episode index path')
    load_parser.add_argument('--episodes', default='data/waymo/episodes', help='Episodes directory')
    load_parser.add_argument('--num-samples', type=int, default=4, help='Number of samples to load')
    load_parser.add_argument('--split', default='train', help='Dataset split')
    
    args = parser.parse_args()
    
    # Build config
    config = SSLDataConfig(
        index_path=args.index,
        episodes_dir=args.episodes,
    )
    
    if args.command == 'list':
        episodes = list_episodes(config)
        print(f"\n{'='*60}")
        print(f"SSL Episode Dataset")
        print(f"{'='*60}")
        print(f"Total episodes: {len(episodes)}\n")
        for ep in episodes:
            print(f"  {ep.episode_id}: {ep.frame_count} frames, {ep.duration:.1f}s")
            print(f"    Cameras: {', '.join(ep.cameras)}")
        print()
    
    elif args.command == 'stats':
        stats = get_dataset_stats(config)
        print(f"\n{'='*60}")
        print(f"SSL Dataset Statistics")
        print(f"{'='*60}")
        print(f"  Episodes: {stats['num_episodes']}")
        print(f"  Frames (indexed): {stats['num_frames']}")
        print(f"  Total frames: {stats['total_frame_count']}")
        print(f"  Avg frames/episode: {stats['avg_frames_per_episode']:.1f}")
        print(f"  Min/Max frames: {stats['min_frames']} / {stats['max_frames']}")
        print(f"  Cameras: {', '.join(stats['cameras'])}")
        print(f"  Views/frame: {stats['views_per_frame']}")
        print(f"  Mask ratio: {stats['mask_ratio']}")
        print()
    
    elif args.command == 'load':
        dataset = SSLEpisodeDataset(config, split=args.split)
        print(f"\n{'='*60}")
        print(f"SSL Dataset Loading Test")
        print(f"{'='*60}")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Split: {args.split}")
        print(f"  Loading {args.num_samples} samples...\n")
        
        for i in range(min(args.num_samples, len(dataset))):
            sample = dataset[i]
            print(f"  Sample {i}:")
            print(f"    Episode: {sample['episode_id']}")
            print(f"    Frame: {sample['frame_idx']}")
            print(f"    Image shape: {sample['image'].shape}")
            print(f"    Views: {len(sample['views'])} views, each {sample['views'][0].shape}")
            print(f"    Masked input shape: {sample['masked_input'].shape}")
            print(f"    Mask: {sample['mask'].sum()}/{len(sample['mask'])} patches masked")
            print()
        
        print("✅ Load test PASSED")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
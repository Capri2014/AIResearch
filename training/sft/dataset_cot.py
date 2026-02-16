"""
SFT Dataset with CoT (Chain of Thought) Reasoning Traces

Creates training dataset that includes both waypoint labels and reasoning traces.

Usage:
    python -m training.sft.dataset_cot --input waymo_episodes --cot cot_traces.jsonl --output sft_cot_dataset.pt
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass
import numpy as np


@dataclass
class CoTSample:
    """Single training sample with CoT reasoning trace."""
    episode_id: str
    timestamp: float
    
    # Input features (from Waymo)
    images: List[str]  # Image paths or base64 encoded
    state_features: Dict[str, float]  # ego_speed, heading, etc.
    
    # CoT reasoning trace (text)
    cot_trace: Dict[str, str]  # perception, situation, prediction, planning, confidence
    
    # Target (waypoints)
    waypoints: List[List[float]]  # [T, 3] -> [x, y, heading]
    
    # Control command (optional)
    control: Dict[str, float]  # steering, throttle, brake
    
    def to_dict(self) -> Dict:
        return {
            'episode_id': self.episode_id,
            'timestamp': self.timestamp,
            'state_features': self.state_features,
            'cot_trace': self.cot_trace,
            'waypoints': self.waypoints,
            'control': self.control,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CoTSample':
        return cls(
            episode_id=data.get('episode_id', ''),
            timestamp=data.get('timestamp', 0.0),
            images=data.get('images', []),
            state_features=data.get('state_features', {}),
            cot_trace=data.get('cot_trace', {}),
            waypoints=data.get('waypoints', []),
            control=data.get('control', {}),
        )


class CoTDataset(Dataset):
    """
    Dataset for SFT training with CoT reasoning traces.
    
    Combines Waymo-style perception data with reasoning traces
    for training models that predict both waypoints and reasoning.
    """
    
    def __init__(
        self,
        data_dir: str,
        cot_file: Optional[str] = None,
        max_seq_len: int = 16,
        transform: Optional[callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.cot_file = cot_file
        self.max_seq_len = max_seq_len
        self.transform = transform
        
        # Load CoT traces if provided
        self.cot_traces = {}
        if cot_file and Path(cot_file).exists():
            with open(cot_file, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    key = f"{sample['episode_id']}_{sample['timestamp']}"
                    self.cot_traces[key] = sample
        
        # Load episode list
        self.episodes = self._load_episodes()
        
        # Create samples
        self.samples = self._create_samples()
        
        print(f"Loaded {len(self.samples)} samples with CoT traces")
    
    def _load_episodes(self) -> List[Path]:
        """Load list of episode directories."""
        episodes = []
        for ep_dir in sorted(self.data_dir.glob("episode_*")):
            if ep_dir.is_dir():
                episodes.append(ep_dir)
        return episodes
    
    def _create_samples(self) -> List[CoTSample]:
        """Create training samples from episodes and CoT traces."""
        samples = []
        
        for episode_dir in self.episodes:
            episode_id = episode_dir.name
            
            # Load perception data (placeholder - implement based on your format)
            frames = self._load_frames(episode_dir)
            
            for frame in frames:
                timestamp = frame.get('timestamp', 0.0)
                key = f"{episode_id}_{timestamp}"
                
                # Get CoT trace if available
                cot_trace = self.cot_traces.get(key, {})
                
                # Create sample
                sample = CoTSample(
                    episode_id=episode_id,
                    timestamp=timestamp,
                    images=frame.get('images', []),
                    state_features=frame.get('state_features', {}),
                    cot_trace=cot_trace,
                    waypoints=frame.get('waypoints', []),
                    control=frame.get('control', {}),
                )
                
                samples.append(sample)
        
        return samples
    
    def _load_frames(self, episode_dir: Path) -> List[Dict]:
        """Load frames from episode directory."""
        # Placeholder - implement based on your data format
        # This would parse Waymo TFRecords or your preferred format
        frames_file = episode_dir / 'frames.json'
        if frames_file.exists():
            with open(frames_file, 'r') as f:
                return json.load(f)
        return []
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        sample = self.samples[idx]
        
        # State features
        state = self._encode_state(sample.state_features)
        
        # Waypoints
        waypoints = self._encode_waypoints(sample.waypoints)
        
        # CoT trace (concatenated text)
        cot_text = self._encode_cot(sample.cot_trace)
        
        # Control (optional)
        control = self._encode_control(sample.control)
        
        item = {
            'episode_id': sample.episode_id,
            'timestamp': sample.timestamp,
            'state': state,
            'waypoints': waypoints,
            'cot_text': cot_text,
            'control': control,
        }
        
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def _encode_state(self, state: Dict[str, float]) -> torch.Tensor:
        """Encode state features to tensor."""
        features = [
            state.get('ego_speed', 0.0),
            state.get('ego_accel', 0.0),
            state.get('ego_heading', 0.0),
            state.get('ego_heading_rate', 0.0),
        ]
        # Pad to fixed size
        while len(features) < 16:
            features.append(0.0)
        return torch.tensor(features[:16], dtype=torch.float32)
    
    def _encode_waypoints(self, waypoints: List[List[float]]) -> torch.Tensor:
        """Encode waypoints to tensor."""
        if not waypoints:
            return torch.zeros(self.max_seq_len, 3, dtype=torch.float32)
        
        # Take first max_seq_len waypoints
        wp = waypoints[:self.max_seq_len]
        
        # Pad if necessary
        if len(wp) < self.max_seq_len:
            wp = wp + [[0.0, 0.0, 0.0]] * (self.max_seq_len - len(wp))
        
        return torch.tensor(wp, dtype=torch.float32)
    
    def _encode_cot(self, cot_trace: Dict[str, str]) -> torch.Tensor:
        """
        Encode CoT trace to tensor.
        
        Concatenate all reasoning steps and tokenize.
        For simplicity, we use a fixed-size encoding.
        """
        # Concatenate all reasoning steps
        text_parts = []
        for key in ['perception', 'situation_understanding', 'behavior_prediction', 
                    'trajectory_planning', 'confidence']:
            if key in cot_trace:
                text_parts.append(cot_trace[key])
        
        full_text = " [SEP] ".join(text_parts)
        
        # Simple hash-based encoding for now
        # In practice, would use actual tokenizer
        encoding = self._simple_encode(full_text, max_len=128)
        
        return torch.tensor(encoding, dtype=torch.long)
    
    def _simple_encode(self, text: str, max_len: int = 128) -> List[int]:
        """Simple encoding based on character ord values."""
        encoded = [ord(c) % 1000 for c in text[:max_len]]
        # Pad
        while len(encoded) < max_len:
            encoded.append(0)
        return encoded[:max_len]
    
    def _encode_control(self, control: Dict[str, float]) -> torch.Tensor:
        """Encode control command to tensor."""
        features = [
            control.get('steering', 0.0),
            control.get('throttle', 0.0),
            control.get('brake', 0.0),
        ]
        return torch.tensor(features, dtype=torch.float32)


class CoTCollator:
    """Collator for batching CoT samples."""
    
    def __init__(self, tokenizer: Optional[callable] = None, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        
        # Find max sequence lengths
        max_wp_len = max([s['waypoints'].shape[0] for s in batch])
        max_cot_len = max([s['cot_text'].shape[0] for s in batch])
        
        # Pad waypoints
        padded_waypoints = []
        for s in batch:
            wp = s['waypoints']
            if wp.shape[0] < max_wp_len:
                pad = torch.zeros(max_wp_len - wp.shape[0], 3)
                wp = torch.cat([wp, pad], dim=0)
            padded_waypoints.append(wp)
        
        # Pad CoT text
        padded_cot = []
        for s in batch:
            cot = s['cot_text']
            if cot.shape[0] < max_cot_len:
                pad = torch.zeros(max_cot_len - cot.shape[0], dtype=torch.long)
                cot = torch.cat([cot, pad], dim=0)
            padded_cot.append(cot)
        
        return {
            'state': torch.stack([s['state'] for s in batch]),
            'waypoints': torch.stack(padded_waypoints),
            'cot_text': torch.stack(padded_cot),
            'control': torch.stack([s['control'] for s in batch]),
            'episode_ids': [s['episode_id'] for s in batch],
            'timestamps': [s['timestamp'] for s in batch],
        }


def create_cot_dataset_parser() -> argparse.ArgumentParser:
    """Create argument parser for dataset creation."""
    parser = argparse.ArgumentParser(description='Create CoT SFT Dataset')
    parser.add_argument('--input', type=str, required=True, help='Input Waymo data directory')
    parser.add_argument('--cot', type=str, default=None, help='CoT traces JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output dataset file')
    parser.add_argument('--max-seq-len', type=int, default=16, help='Max waypoint sequence length')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for preview')
    return parser


def main():
    """CLI entry point for dataset creation."""
    parser = create_cot_dataset_parser()
    args = parser.parse_args()
    
    print("Creating CoT SFT Dataset")
    print("=" * 50)
    print(f"Input: {args.input}")
    print(f"CoT traces: {args.cot}")
    print(f"Output: {args.output}")
    print(f"Max seq len: {args.max_seq_len}")
    
    # Create dataset
    dataset = CoTDataset(
        data_dir=args.input,
        cot_file=args.cot,
        max_seq_len=args.max_seq_len,
    )
    
    # Save dataset
    torch.save({
        'samples': [s.to_dict() for s in dataset.samples],
        'config': {
            'data_dir': args.input,
            'cot_file': args.cot,
            'max_seq_len': args.max_seq_len,
        }
    }, args.output)
    
    print(f"\nDataset saved to: {args.output}")
    print(f"Total samples: {len(dataset)}")
    
    # Preview a batch
    collator = CoTCollator()
    batch = collator([dataset[0], dataset[1], dataset[2]])
    print(f"\nBatch preview:")
    print(f"  state shape: {batch['state'].shape}")
    print(f"  waypoints shape: {batch['waypoints'].shape}")
    print(f"  cot_text shape: {batch['cot_text'].shape}")


if __name__ == "__main__":
    main()

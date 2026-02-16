"""
SFT Dataset with CoT (Chain of Thought) Reasoning Traces

Creates training dataset that includes both waypoint labels and reasoning traces.
Uses proper tokenizers (BERT) for text encoding instead of placeholder.

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
    state_features: Dict[str, float]
    
    # CoT reasoning trace (text)
    cot_trace: Dict[str, str]
    
    # Target (waypoints)
    waypoints: List[List[float]]
    
    # Control command (optional)
    control: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            'episode_id': self.episode_id,
            'timestamp': self.timestamp,
            'state_features': self.state_features,
            'cot_trace': self.cot_trace,
            'waypoints': self.waypoints,
            'control': self.control,
        }


class CoTTokenizer:
    """
    Tokenizer for CoT reasoning traces.
    
    Uses HuggingFace transformers library for proper tokenization.
    Supports BERT, GPT-2, and other tokenizers.
    
    Example:
        >>> tokenizer = CoTTokenizer('bert-base-uncased')
        >>> tokens = tokenizer.encode_cot("I see a car ahead")
        >>> tokens.shape
        torch.Size([128])
    """
    
    def __init__(
        self,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 128,
        padding: str = 'max_length',
        truncation: bool = True,
    ):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'do_not_pad')
            truncation: Whether to truncate long sequences
        """
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Try to load tokenizer from transformers
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.is_loaded = True
            print(f"Loaded tokenizer: {tokenizer_name}")
        except ImportError:
            print("Warning: transformers not installed. Using simple tokenizer.")
            self.tokenizer = None
            self.is_loaded = False
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
            print("Using simple tokenizer.")
            self.tokenizer = None
            self.is_loaded = False
    
    def encode_cot(
        self,
        cot_trace: Dict[str, str],
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode CoT trace to token indices.
        
        Args:
            cot_trace: Dict with reasoning steps
            return_tensors: Return PyTorch tensors
            
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        if not self.is_loaded or self.tokenizer is None:
            # Fallback to simple encoding
            return self._simple_encode(cot_trace)
        
        # Concatenate reasoning steps with separator
        text_parts = []
        for key in ['perception', 'situation_understanding', 'behavior_prediction', 
                    'trajectory_planning', 'confidence']:
            if key in cot_trace and cot_trace[key]:
                text_parts.append(cot_trace[key])
        
        if not text_parts:
            text_parts = ["No reasoning provided."]
        
        full_text = " [SEP] ".join(text_parts)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors='pt' if return_tensors else None,
        )
        
        if return_tensors:
            # Convert to tensors
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            }
        else:
            return encoding
    
    def _simple_encode(
        self,
        cot_trace: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """
        Simple fallback encoding when tokenizer not available.
        
        Uses basic character encoding.
        """
        # Concatenate reasoning steps
        text_parts = []
        for key in ['perception', 'situation_understanding', 'behavior_prediction', 
                    'trajectory_planning', 'confidence']:
            if key in cot_trace and cot_trace[key]:
                text_parts.append(cot_trace[key])
        
        full_text = " ".join(text_parts)
        
        # Simple character encoding
        input_ids = []
        for char in full_text[:self.max_length - 2]:  # Reserve for special tokens
            input_ids.append(ord(char) % 1000)
        
        # Pad
        while len(input_ids) < self.max_length:
            input_ids.append(0)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if i < len([c for c in full_text[:self.max_length - 2] if c != ' ']) else 0 
                         for i in range(self.max_length)]
        
        return {
            'input_ids': torch.tensor(input_ids[:self.max_length], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def decode(self, input_ids: torch.Tensor) -> str:
        """Decode token indices back to text."""
        if self.is_loaded and self.tokenizer is not None:
            return self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return "".join([chr(i % 1000) for i in input_ids.tolist()])


class CoTDataset(Dataset):
    """
    Dataset for SFT training with CoT reasoning traces.
    
    Combines Waymo-style perception data with reasoning traces
    for training models that predict both waypoints and reasoning.
    
    Example:
        >>> dataset = CoTDataset(
        ...     data_dir='waymo_episodes',
        ...     cot_file='cot_traces.jsonl',
        ...     tokenizer_name='bert-base-uncased'
        ... )
        >>> sample = dataset[0]
        >>> sample['cot_text']['input_ids'].shape
        torch.Size([128])
    """
    
    def __init__(
        self,
        data_dir: str,
        cot_file: Optional[str] = None,
        max_seq_len: int = 16,
        max_cot_len: int = 128,
        tokenizer_name: str = 'bert-base-uncased',
        transform: Optional[callable] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory with episode data
            cot_file: JSONL file with CoT traces
            max_seq_len: Maximum waypoint sequence length
            max_cot_len: Maximum CoT token length
            tokenizer_name: HuggingFace tokenizer name
            transform: Optional transform for data augmentation
        """
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.max_cot_len = max_cot_len
        self.transform = transform
        
        # Initialize tokenizer
        self.tokenizer = CoTTokenizer(
            tokenizer_name=tokenizer_name,
            max_length=max_cot_len,
        )
        
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
            
            # Load perception data
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
        
        # CoT trace (proper tokenization)
        cot_encoding = self.tokenizer.encode_cot(sample.cot_trace)
        
        # Control (optional)
        control = self._encode_control(sample.control)
        
        item = {
            'episode_id': sample.episode_id,
            'timestamp': sample.timestamp,
            'state': state,
            'waypoints': waypoints,
            'cot_input_ids': cot_encoding['input_ids'],
            'cot_attention_mask': cot_encoding['attention_mask'],
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
    
    def _encode_control(self, control: Dict[str, float]) -> torch.Tensor:
        """Encode control command to tensor."""
        features = [
            control.get('steering', 0.0),
            control.get('throttle', 0.0),
            control.get('brake', 0.0),
        ]
        return torch.tensor(features, dtype=torch.float32)


class CoTCollator:
    """
    Collator for batching CoT samples.
    
    Handles padding for variable-length sequences.
    
    Example:
        >>> collator = CoTCollator(max_length=128)
        >>> batch = collator([dataset[0], dataset[1], dataset[2]])
        >>> batch['cot_input_ids'].shape
        torch.Size([3, 128])
    """
    
    def __init__(self, max_length: int = 128):
        """
        Initialize collator.
        
        Args:
            max_length: Maximum CoT sequence length
        """
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        
        # Find max waypoint length
        max_wp_len = max([s['waypoints'].shape[0] for s in batch])
        
        # Pad waypoints
        padded_waypoints = []
        for s in batch:
            wp = s['waypoints']
            if wp.shape[0] < max_wp_len:
                pad = torch.zeros(max_wp_len - wp.shape[0], 3)
                wp = torch.cat([wp, pad], dim=0)
            padded_waypoints.append(wp)
        
        # Stack all tensors
        return {
            'state': torch.stack([s['state'] for s in batch]),
            'waypoints': torch.stack(padded_waypoints),
            'cot_input_ids': torch.stack([s['cot_input_ids'] for s in batch]),
            'cot_attention_mask': torch.stack([s['cot_attention_mask'] for s in batch]),
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
    parser.add_argument('--max-cot-len', type=int, default=128, help='Max CoT token length')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased', 
                       help='HuggingFace tokenizer name')
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
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Max CoT len: {args.max_cot_len}")
    
    # Create dataset
    dataset = CoTDataset(
        data_dir=args.input,
        cot_file=args.cot,
        max_seq_len=args.max_seq_len,
        max_cot_len=args.max_cot_len,
        tokenizer_name=args.tokenizer,
    )
    
    # Save dataset metadata
    import pickle
    dataset_info = {
        'samples': [s.to_dict() for s in dataset.samples],
        'config': {
            'data_dir': args.input,
            'cot_file': args.cot,
            'max_seq_len': args.max_seq_len,
            'max_cot_len': args.max_cot_len,
            'tokenizer_name': args.tokenizer,
        }
    }
    
    # Save as pickle
    with open(args.output.replace('.pt', '.pkl'), 'wb') as f:
        pickle.dump(dataset_info, f)
    
    # Save tokenized CoT traces separately for efficiency
    cot_tokens = []
    for sample in dataset.samples:
        cot_encoding = dataset.tokenizer.encode_cot(sample.cot_trace)
        cot_tokens.append({
            'input_ids': cot_encoding['input_ids'].tolist(),
            'attention_mask': cot_encoding['attention_mask'].tolist(),
        })
    
    with open(args.output.replace('.pt', '_cot_tokens.pkl'), 'wb') as f:
        pickle.dump(cot_tokens, f)
    
    print(f"\nDataset saved to: {args.output}")
    print(f"Total samples: {len(dataset)}")
    
    # Preview
    sample = dataset[0]
    print(f"\nSample preview:")
    print(f"  state shape: {sample['state'].shape}")
    print(f"  waypoints shape: {sample['waypoints'].shape}")
    print(f"  cot_input_ids shape: {sample['cot_input_ids'].shape}")
    print(f"  cot_attention_mask shape: {sample['cot_attention_mask'].shape}")
    
    # Test tokenizer
    print(f"\nTokenizer test:")
    decoded = dataset.tokenizer.decode(sample['cot_input_ids'])
    print(f"  Decoded text (first 100 chars): {decoded[:100]}...")


if __name__ == "__main__":
    main()

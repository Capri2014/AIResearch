"""
Waymo Episode Dataset Loader

Loads Waymo-style episodes from npz files and provides PyTorch Dataset
for waypoint prediction training.

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class WaymoEpisodeMeta:
    """Metadata for a Waymo episode."""
    episode_id: str
    scenario_type: str  # straight, lane_change_left, lane_change_right, etc.
    num_frames: int
    has_collision: bool
    route_completion: float


class WaymoEpisode:
    """Single Waymo episode with all data."""
    
    def __init__(self, npz_path: Path):
        self.npz_path = npz_path
        self._data = None
        self._load()
        
    def _load(self):
        """Load episode from npz file."""
        self._data = np.load(self.npz_path, allow_pickle=True)
        
    @property
    def camera_images(self) -> np.ndarray:
        """Get camera images (T, H, W, C)."""
        return self._data['camera_images']
    
    @property
    def waypoints(self) -> np.ndarray:
        """Get waypoints (T, num_waypoints, 2) - x, y coordinates."""
        return self._data['waypoints']
    
    @property
    def metadata(self) -> Dict:
        """Get episode metadata."""
        return self._data['metadata'].item()
    
    @property
    def num_frames(self) -> int:
        """Number of frames in episode."""
        return len(self.waypoints)
    
    @property
    def scenario_type(self) -> str:
        """Scenario type from filename."""
        return self.npz_path.stem.split('_')[-1]
    
    @property
    def episode_id(self) -> str:
        """Episode ID from filename."""
        return self.npz_path.stem


class WaymoEpisodeDataset(Dataset):
    """
    PyTorch Dataset for Waymo episodes.
    
    Provides sliding window samples for waypoint prediction:
    - Input: current state (x, y, vx, vy, goal_x, goal_y) + history
    - Output: future waypoints
    
    Args:
        data_dir: Directory containing episode npz files
        window_size: Number of history frames to use
        prediction_horizon: Number of future waypoints to predict
        stride: Stride for sliding window (default: 1)
    """
    
    def __init__(
        self,
        data_dir: str,
        window_size: int = 5,
        prediction_horizon: int = 3,
        stride: int = 1,
        split: str = 'train',
        train_ratio: float = 0.8,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.split = split
        
        # Load all episodes
        self.episodes = self._load_episodes()
        
        # Build sample index
        self.samples = self._build_sample_index()
        
        # Split train/val
        if split == 'train':
            self.samples = self.samples[:int(len(self.samples) * train_ratio)]
        else:
            self.samples = self.samples[int(len(self.samples) * train_ratio):]
        
        print(f"Loaded {len(self.episodes)} episodes, {len(self.samples)} samples ({split})")
    
    def _load_episodes(self) -> List[WaymoEpisode]:
        """Load all episodes from data directory."""
        episodes = []
        sft_path = self.data_dir / "sft_training"
        
        if not sft_path.exists():
            raise FileNotFoundError(f"Data directory not found: {sft_path}")
        
        for npz_path in sorted(sft_path.glob("*.npz")):
            try:
                episode = WaymoEpisode(npz_path)
                episodes.append(episode)
            except Exception as e:
                print(f"Failed to load {npz_path}: {e}")
        
        return episodes
    
    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """
        Build index of (episode_idx, frame_idx) for all valid samples.
        
        A valid sample has enough history and future frames.
        """
        samples = []
        
        for ep_idx, episode in enumerate(self.episodes):
            num_frames = episode.num_frames
            min_frames = self.window_size + self.prediction_horizon
            
            # Sliding window through episode
            for frame_idx in range(0, num_frames - min_frames + 1, self.stride):
                samples.append((ep_idx, frame_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        ep_idx, frame_idx = self.samples[idx]
        episode = self.episodes[ep_idx]
        
        # Get history frames (window_size frames ending at frame_idx)
        history_start = frame_idx
        history_end = frame_idx + self.window_size
        
        # Get future waypoints (prediction_horizon frames starting from frame_idx)
        future_start = frame_idx + self.window_size
        future_end = future_start + self.prediction_horizon
        
        # Extract waypoints: (T, 2) - x, y coordinates
        waypoints = episode.waypoints  # (T, 2)
        
        # History waypoints (input): (window_size, 2)
        history_waypoints = waypoints[history_start:history_end]
        
        # Future waypoints (target): (prediction_horizon, 2)
        future_waypoints = waypoints[future_start:future_end]
        
        # Get current state from last history frame: (2,)
        current_waypoint = waypoints[history_end - 1]
        
        # Compute velocity from history: (window_size-1, 2)
        if self.window_size > 1:
            velocities = np.diff(waypoints[history_start:history_end], axis=0)
            current_velocity = velocities[-1]  # (2,)
        else:
            current_velocity = np.zeros(2, dtype=np.float32)
        
        # Compute acceleration: (2,)
        if self.window_size > 2:
            accelerations = np.diff(velocities, axis=0)
            current_acceleration = accelerations[-1]
        else:
            current_acceleration = np.zeros(2, dtype=np.float32)
        
        # Convert to tensors
        sample = {
            'history_waypoints': torch.from_numpy(history_waypoints).float(),
            'current_waypoint': torch.from_numpy(current_waypoint).float(),
            'current_velocity': torch.from_numpy(current_velocity).float(),
            'current_acceleration': torch.from_numpy(current_acceleration).float(),
            'future_waypoints': torch.from_numpy(future_waypoints).float(),
            'episode_id': episode.episode_id,
            'frame_idx': frame_idx,
        }
        
        # Add camera images if available
        if episode.camera_images is not None and len(episode.camera_images) > 0:
            # Use last history frame's camera
            cam = episode.camera_images[history_end - 1]
            if isinstance(cam, np.ndarray):
                # Convert HWC -> CHW
                if cam.ndim == 3 and cam.shape[-1] == 3:
                    cam = np.transpose(cam, (2, 0, 1))
                sample['camera'] = torch.from_numpy(cam).float() / 255.0
        
        return sample


class WaypointPredictionModel(torch.nn.Module):
    """
    Simple waypoint prediction model for testing the dataset.
    
    Architecture:
    - History encoder: RNN on waypoint sequence
    - Current state: MLP on current waypoint + velocity + acceleration
    - Decoder: Predict future waypoints
    """
    
    def __init__(
        self,
        waypoint_dim: int = 2,
        hidden_dim: int = 128,
        prediction_horizon: int = 10,
        window_size: int = 5,
    ):
        super().__init__()
        
        self.prediction_horizon = prediction_horizon
        self.window_size = window_size
        
        # History encoder (RNN for sequence)
        self.history_rnn = torch.nn.GRU(
            input_size=waypoint_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
        )
        
        # Current state encoder
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(waypoint_dim * 3, hidden_dim),  # waypoint + vel + acc
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, waypoint_dim * prediction_horizon),
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict future waypoints."""
        # History: (B, window_size, 2)
        history = batch['history_waypoints']
        B = history.shape[0]
        
        # RNN on history
        _, hidden = self.history_rnn(history)  # hidden: (num_layers, B, hidden_dim)
        history_feat = hidden[-1]  # (B, hidden_dim)
        
        # Current state
        current = batch['current_waypoint']  # (B, 2)
        velocity = batch['current_velocity']  # (B, 2)
        accel = batch['current_acceleration']  # (B, 2)
        state_flat = torch.cat([current, velocity, accel], dim=-1)
        
        state_feat = self.state_encoder(state_flat)
        
        # Combine
        combined = torch.cat([history_feat, state_feat], dim=-1)
        
        # Decode
        output = self.decoder(combined)
        
        # Reshape to (B, prediction_horizon, 2)
        output = output.view(B, self.prediction_horizon, 2)
        
        return output


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    window_size: int = 5,
    prediction_horizon: int = 3,
    num_workers: int = 4,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = WaymoEpisodeDataset(
        data_dir=data_dir,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        split='train',
        train_ratio=train_ratio,
    )
    
    val_dataset = WaymoEpisodeDataset(
        data_dir=data_dir,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        split='val',
        train_ratio=train_ratio,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def test_dataset():
    """Test the dataset loader."""
    # Use existing data
    data_dir = "data/synthetic"
    
    # Test dataset
    dataset = WaymoEpisodeDataset(
        data_dir=data_dir,
        window_size=5,
        prediction_horizon=3,
        stride=1,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print("\nSample keys:", sample.keys())
    print("History waypoints shape:", sample['history_waypoints'].shape)
    print("Future waypoints shape:", sample['future_waypoints'].shape)
    print("Episode ID:", sample['episode_id'])
    
    # Test model
    model = WaypointPredictionModel(
        prediction_horizon=sample['future_waypoints'].shape[0],
        window_size=sample['history_waypoints'].shape[0],
    )
    
    # Test batch
    batch = {
        'history_waypoints': sample['history_waypoints'].unsqueeze(0),
        'current_waypoint': sample['current_waypoint'].unsqueeze(0),
        'current_velocity': sample['current_velocity'].unsqueeze(0),
        'current_acceleration': sample['current_acceleration'].unsqueeze(0),
    }
    
    output = model(batch)
    print("\nModel output shape:", output.shape)
    
    # Test data loaders
    print("\n--- Testing Data Loaders ---")
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=4,
        window_size=5,
        prediction_horizon=3,
        num_workers=0,
    )
    
    for batch in train_loader:
        print("Batch future_waypoints shape:", batch['future_waypoints'].shape)
        break
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_dataset()

"""Unified Training Data Pipeline for Driving-First Model Training.

Bridges episode data (Waymo format) to BC/SFT/RL training with:
- EpisodeDataset: Core dataset for loading episodes
- WaypointBatchCollator: Dynamic padding/collate for variable-length episodes
- train/val/test splits by episode_id
- Schema validation

This module fills the gap between raw episodes and model training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np


# ============================================================================
# Schema Types
# ============================================================================

@dataclass
class FrameData:
    """Single frame from an episode.
    
    Attributes:
        frame_id: Unique frame identifier
        timestamp: Frame timestamp in seconds
        images: Dict of camera_name -> image data (path or base64)
        state: Vehicle state (speed_mps, throttle, steering, yaw_rad)
        expert_waypoints: Future waypoints [[x, y], ...] in meters
    """
    frame_id: str
    timestamp: float
    images: Dict[str, str] = field(default_factory=dict)
    state: Dict[str, float] = field(default_factory=dict)
    expert_waypoints: List[List[float]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrameData":
        obs = d.get("observations", {})
        return cls(
            frame_id=d.get("frame_id", ""),
            timestamp=d.get("timestamp", 0.0),
            images=obs.get("cameras", {}),
            state=obs.get("state", {}),
            expert_waypoints=d.get("expert", {}).get("waypoints", []),
        )


@dataclass 
class EpisodeData:
    """Full episode with frames.
    
    Attributes:
        episode_id: Unique episode identifier
        frames: List of frames
        metadata: Additional episode metadata
    """
    episode_id: str
    frames: List[FrameData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def duration_sec(self) -> float:
        if self.frames:
            return self.frames[-1].timestamp - self.frames[0].timestamp
        return 0.0


# ============================================================================
# Dataset
# ============================================================================

class EpisodeDataset(Dataset):
    """PyTorch Dataset for Waymo-style episodes.
    
    Supports:
    - Loading from episode JSON files
    - Train/val/test split by episode_id
    - Configurable horizon (future steps to predict)
    - History encoding (past frames to condition on)
    
    Args:
        episodes_dir: Directory containing episode JSON files
        horizon: Number of future waypoints to predict (default: 20)
        history: Number of past frames to encode (default: 4)
        split: "train", "val", "test", or None for all
        split_ratio: (train, val, test) ratios (default: 0.8, 0.1, 0.1)
        max_episodes: Maximum episodes to load (None for all)
        subsample: Subsample frames by this stride (1 = all frames)
    """
    
    def __init__(
        self,
        episodes_dir: str = "data/waymo/episodes",
        horizon: int = 20,
        history: int = 4,
        split: Optional[str] = None,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        max_episodes: Optional[int] = None,
        subsample: int = 1,
    ):
        self.episodes_dir = Path(episodes_dir)
        self.horizon = horizon
        self.history = history
        self.subsample = subsample
        
        # Load episodes
        self.episodes: List[EpisodeData] = []
        self._load_episodes(max_episodes)
        
        # Apply split if requested
        if split is not None:
            self._apply_split(split, split_ratio)
        
        print(f"EpisodeDataset: {len(self.episodes)} episodes, horizon={horizon}, history={history}")
    
    def _load_episodes(self, max_episodes: Optional[int]) -> None:
        """Load episodes from directory."""
        episode_paths = sorted(self.episodes_dir.glob("**/*.json"))
        
        if max_episodes:
            episode_paths = episode_paths[:max_episodes]
        
        for ep_path in episode_paths:
            try:
                ep = json.loads(ep_path.read_text())
                ep_id = ep.get("episode_id", ep_path.stem)
                frames = [
                    FrameData.from_dict(fr) 
                    for fr in ep.get("frames", [])
                ]
                
                if frames:
                    self.episodes.append(EpisodeData(
                        episode_id=ep_id,
                        frames=frames,
                        metadata={"path": str(ep_path)},
                    ))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid episode: {ep_path} — {e}")
    
    def _apply_split(
        self, 
        split: str, 
        split_ratio: Tuple[float, float, float]
    ) -> None:
        """Apply train/val/test split by episode_id hash."""
        # Shuffle by episode_id deterministically
        random.seed(42)
        shuffled = sorted(self.episodes, key=lambda e: e.episode_id)
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])
        
        if split == "train":
            self.episodes = shuffled[:train_end]
        elif split == "val":
            self.episodes = shuffled[train_end:val_end]
        elif split == "test":
            self.episodes = shuffled[val_end:]
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get one episode as training example.
        
        Returns:
            Dict with:
            - episode_id: str
            - frames: List of FrameData
            - num_frames: int
        """
        return {
            "episode_id": self.episodes[idx].episode_id,
            "frames": self.episodes[idx].frames,
            "num_frames": self.episodes[idx].num_frames,
        }


class WaypointBatchCollator:
    """Collates episodes into training batches with padding.
    
    Creates fixed-size tensors from variable-length episodes:
    - Images: (B, T, C, H, W) with zero-padding
    - Waypoints: (B, T, horizon, 2) with zero-padding  
    - State: (B, T, state_dim) with zero-padding
    - Mask: (B, T) indicating valid frames
    
    Args:
        horizon: Future waypoints to predict
        history: Past frames to encode
        max_frames: Max frames per episode (None = max in batch)
        image_size: (H, W) for resizing
    """
    
    def __init__(
        self,
        horizon: int = 20,
        history: int = 4,
        max_frames: Optional[int] = None,
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.horizon = horizon
        self.history = history
        self.max_frames = max_frames
        self.image_size = image_size
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate one batch.
        
        Args:
            batch: List of episode dicts from EpisodeDataset
            
        Returns:
            Dict with batched tensors:
            - episode_ids: List[str]
            - images: (B, T, C, H, W)
            - waypoints: (B, T, horizon, 2)
            - states: (B, T, state_dim)
            - valid_mask: (B, T)
        """
        B = len(batch)
        
        # Handle empty batch
        if B == 0:
            return {
                "episode_ids": [],
                "images": torch.zeros(0),
                "waypoints": torch.zeros(0, self.horizon, 2),
                "states": torch.zeros(0, 4),
                "valid_mask": torch.zeros(0, dtype=torch.bool),
            }
        
        # Determine max frames in batch
        max_t = self.max_frames
        if max_t is None:
            max_t = max(ep["num_frames"] for ep in batch)
        
        # Initialize tensors
        device = torch.device("cpu")
        images = torch.zeros(B, max_t, 3, self.image_size[0], self.image_size[1], device=device)
        waypoints = torch.zeros(B, max_t, self.horizon, 2, device=device)
        states = torch.zeros(B, max_t, 4, device=device)  # speed, throttle, steering, yaw
        valid_mask = torch.zeros(B, max_t, dtype=torch.bool, device=device)
        
        episode_ids = []
        
        for b_idx, ep in enumerate(batch):
            frames = ep["frames"]
            T = min(len(frames), max_t)
            
            episode_ids.append(ep["episode_id"])
            
            for t_idx in range(T):
                fr = frames[t_idx]
                valid_mask[b_idx, t_idx] = True
                
                # Waypoints
                wps = fr.expert_waypoints
                for h in range(min(len(wps), self.horizon)):
                    waypoints[b_idx, t_idx, h, 0] = wps[h][0]
                    waypoints[b_idx, t_idx, h, 1] = wps[h][1]
                
                # State
                state = fr.state
                states[b_idx, t_idx, 0] = state.get("speed_mps", 0.0)
                states[b_idx, t_idx, 1] = state.get("throttle", 0.0)
                states[b_idx, t_idx, 2] = state.get("steering", 0.0)
                states[b_idx, t_idx, 3] = state.get("yaw_rad", 0.0)
        
        return {
            "episode_ids": episode_ids,
            "images": images,
            "waypoints": waypoints,
            "states": states,
            "valid_mask": valid_mask,
        }


def create_train_dataloader(
    episodes_dir: str = "data/waymo/episodes",
    horizon: int = 20,
    history: int = 4,
    batch_size: int = 32,
    num_workers: int = 4,
    max_episodes: Optional[int] = None,
    subsample: int = 1,
) -> DataLoader:
    """Create training DataLoader with proper collate_fn.
    
    Args:
        episodes_dir: Directory with episode JSON files
        horizon: Future waypoints to predict
        history: Past frames to encode
        batch_size: Batch size
        num_workers: DataLoader workers
        max_episodes: Max episodes to load
        subsample: Frame subsample stride
    
    Returns:
        Configured DataLoader
    """
    dataset = EpisodeDataset(
        episodes_dir=episodes_dir,
        horizon=horizon,
        history=history,
        split="train",
        max_episodes=max_episodes,
        subsample=subsample,
    )
    
    collator = WaypointBatchCollator(horizon=horizon, history=history)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )


def create_eval_dataloader(
    episodes_dir: str = "data/waymo/episodes",
    horizon: int = 20,
    history: int = 4,
    batch_size: int = 32,
    num_workers: int = 4,
    max_episodes: Optional[int] = None,
    split: str = "val",
    subsample: int = 1,
) -> DataLoader:
    """Create evaluation DataLoader (val or test).
    
    Args:
        episodes_dir: Directory with episode JSON files
        horizon: Future waypoints to predict
        history: Past frames to encode
        batch_size: Batch size
        num_workers: DataLoader workers
        max_episodes: Max episodes to load
        split: "val" or "test"
        subsample: Frame subsample stride
    
    Returns:
        Configured DataLoader
    """
    dataset = EpisodeDataset(
        episodes_dir=episodes_dir,
        horizon=horizon,
        history=history,
        split=split,
        max_episodes=max_episodes,
        subsample=subsample,
    )
    
    collator = WaypointBatchCollator(horizon=horizon, history=history)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )


# ============================================================================
# Schema Validation
# ============================================================================

def validate_episode_schema(ep_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate episode against schema.
    
    Args:
        ep_path: Path to episode JSON
        
    Returns:
        (is_valid, error_message)
    """
    try:
        ep = json.loads(ep_path.read_text())
        
        # Required fields
        if "episode_id" not in ep:
            return False, "missing episode_id"
        if "frames" not in ep or not ep["frames"]:
            return False, "missing or empty frames"
        
        # Check frames
        for i, fr in enumerate(ep["frames"]):
            if "frame_id" not in fr:
                return False, f"frame {i} missing frame_id"
            if "timestamp" not in fr:
                return False, f"frame {i} missing timestamp"
            
            obs = fr.get("observations", {})
            state = obs.get("state", {})
            
            # Check state fields
            for field in ["speed_mps"]:
                if field not in state:
                    return False, f"frame {i} missing state.{field}"
            
            # Check waypoints
            expert = fr.get("expert", {})
            wps = expert.get("waypoints", [])
            if wps:
                for j, wp in enumerate(wps):
                    if not isinstance(wp, list) or len(wp) != 2:
                        return False, f"frame {i} waypoint {j} not [x, y]"
        
        return True, None
        
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_episodes_dir(episodes_dir: Path) -> Dict[str, Any]:
    """Validate all episodes in directory.
    
    Args:
        episodes_dir: Directory with episode JSON files
        
    Returns:
        Dict with validation results:
        - total: int
        - valid: int
        - errors: List[error_dict]
    """
    episode_paths = sorted(episodes_dir.glob("**/*.json"))
    
    results = {
        "total": len(episode_paths),
        "valid": 0,
        "errors": [],
    }
    
    for ep_path in episode_paths:
        is_valid, error = validate_episode_schema(ep_path)
        if is_valid:
            results["valid"] += 1
        else:
            results["errors"].append({
                "path": str(ep_path),
                "error": error,
            })
    
    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Training Data Pipeline")
    parser.add_argument("--episodes-dir", type=str, default="data/waymo/episodes")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default=None)
    parser.add_argument("--validate", action="store_true", help="Validate episodes")
    parser.add_argument("--smoke-test", action="store_true", help="Smoke test without data")
    args = parser.parse_args()
    
    if args.validate:
        results = validate_episodes_dir(Path(args.episodes_dir))
        print(f"Validation: {results['valid']}/{results['total']} valid")
        for err in results["errors"][:5]:
            print(f"  {err['path']}: {err['error']}")
    elif args.smoke_test:
        # Smoke test with synthetic data
        print("Smoke test (synthetic data):")
        
        # Create synthetic episode for testing
        synthetic_ep = EpisodeData(
            episode_id="smoke-test-001",
            frames=[
                FrameData(
                    frame_id=f"frame_{i}",
                    timestamp=float(i * 0.1),
                    state={"speed_mps": 5.0, "throttle": 0.3, "steering": 0.0, "yaw_rad": 0.0},
                    expert_waypoints=[[float(i + j), float(i + j)] for j in range(20)],
                )
                for i in range(10)
            ],
        )
        
        # Create collator
        collator = WaypointBatchCollator(horizon=20, history=4)
        
        # Collate
        batch = collator([{
            "episode_id": synthetic_ep.episode_id,
            "frames": synthetic_ep.frames,
            "num_frames": synthetic_ep.num_frames,
        }])
        
        print(f"  episode_ids: {batch['episode_ids']}")
        print(f"  images: {batch['images'].shape}")
        print(f"  waypoints: {batch['waypoints'].shape}")
        print(f"  states: {batch['states'].shape}")
        print(f"  valid_mask: {batch['valid_mask'].shape}")
    else:
        # Create dataset (will show 0 if no episodes available)
        dataset = EpisodeDataset(
            episodes_dir=args.episodes_dir,
            horizon=args.horizon,
            history=args.history,
            split=args.split,
            max_episodes=args.max_episodes,
        )
        print(f"Dataset: {len(dataset)} episodes")
        
        # Show split stats
        for split in ["train", "val", "test"]:
            split_ds = EpisodeDataset(
                episodes_dir=args.episodes_dir,
                horizon=args.horizon,
                history=args.history,
                split=split,
                max_episodes=args.max_episodes,
            )
            print(f"  {split}: {len(split_ds)} episodes")
    
    print("Done!")
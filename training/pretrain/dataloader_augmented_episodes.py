"""PyTorch DataLoader for Augmented Waymo Episodes.

Bridges augmented episodes (from augment_episodes.py) to SSL pretraining.

Supports:
- Episode JSON format from data/schema/episode.json
- Augmented episodes from data/waymo/episodes_augmented/
- Temporal pairs for contrastive learning
- Image waypoint pairs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from torchvision import transforms
except ImportError:
    transforms = None
from PIL import Image
import numpy as np
import io


# Episode schema (from data/schema/episode.json)
# {
#   "episode_id": str,
#   "frames": [
#     {
#       "frame_id": str,
#       "timestamp": float,
#       "observations": {
#         "image": str (base64 or path),
#         "state": {"speed_mps": float, "throttle": float, "steering": float}
#       },
#       "expert": {
#         "waypoints": [[x, y], ...]  # list of [x, y] in meters
#       }
#     },
#     ...
#   ]
# }


class WaymoEpisodeDataset(Dataset):
    """PyTorch dataset for Waymo augmented episodes.
    
    Args:
        episodes_dir: Directory containing episode JSON files
        transform: Optional transform for images
        horizon: Number of future waypoints to predict
        max_episodes: Maximum number of episodes to load (None for all)
    """
    
    def __init__(
        self,
        episodes_dir: str = "data/waymo/episodes_augmented",
        transform: Optional[torch.nn.Module] = None,
        horizon: int = 20,
        max_episodes: Optional[int] = None,
    ):
        self.episodes_dir = Path(episodes_dir)
        # Fallback to basic PIL transform if torchvision unavailable
        if transform is None:
            if transforms is not None:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            else:
                # Minimal transform
                self.transform = lambda img: torch.from_numpy(np.array(img).transpose(2, 0, 1).float()) / 255.0
        else:
            self.transform = transform
        self.horizon = horizon
        self.max_episodes = max_episodes
        
        # Load episode metadata
        self.episodes: List[Dict[str, Any]] = []
        self._load_episodes()
    
    def _load_episodes(self) -> None:
        """Load episode metadata (lazy load frames)."""
        episode_paths = sorted(self.episodes_dir.glob("**/*.json"))
        
        if self.max_episodes:
            episode_paths = episode_paths[:self.max_episodes]
        
        for ep_path in episode_paths:
            try:
                ep = json.loads(ep_path.read_text())
                # Validate schema
                if "frames" in ep and len(ep["frames"]) > 0:
                    self.episodes.append({
                        "path": ep_path,
                        "episode_id": ep.get("episode_id", ep_path.stem),
                        "num_frames": len(ep["frames"]),
                    })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid episode: {ep_path} — {e}")
        
        print(f"Loaded {len(self.episodes)} episodes from {self.episodes_dir}")
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load one episode.
        
        Returns:
            Dict with:
            - episode_id: str
            - images: Tensor of shape (T, C, H, W)
            - waypoints: Tensor of shape (T, horizon, 2) in meters
            - speeds: Tensor of shape (T,)
            - timestamps: Tensor of shape (T,)
        """
        ep_meta = self.episodes[idx]
        ep = json.loads(ep_meta["path"].read_text())
        
        frames = ep["frames"]
        T = len(frames)
        
        images = []
        waypoints = []
        speeds = []
        timestamps = []
        
        for fr in frames:
            # Load image
            img_data = fr.get("observations", {}).get("image")
            if img_data:
                # Handle both base64 and path
                if img_data.startswith("/"):
                    img = Image.open(img_data)
                elif len(img_data) > 1000:  # base64
                    import base64
                    img = Image.open(io.BytesIO(base64.b64decode(img_data)))
                else:
                    img = Image.open(img_data)
                img = self.transform(img)
                images.append(img)
            else:
                # Placeholder (black image)
                images.append(torch.zeros(3, 256, 256))
            
            # Load waypoints
            wps = fr.get("expert", {}).get("waypoints", [])
            if len(wps) >= self.horizon:
                wps_tensor = torch.tensor(wps[:self.horizon], dtype=torch.float32)
            else:
                # Pad with zeros
                wps_tensor = torch.zeros(self.horizon, 2)
                for i, wp in enumerate(wps):
                    if i < self.horizon:
                        wps_tensor[i] = torch.tensor(wp, dtype=torch.float32)
            waypoints.append(wps_tensor)
            
            # Load speed
            state = fr.get("observations", {}).get("state", {})
            speeds.append(float(state.get("speed_mps", 0.0)))
            
            # Load timestamp
            timestamps.append(float(fr.get("timestamp", 0.0)))
        
        return {
            "episode_id": ep_meta["episode_id"],
            "images": torch.stack(images) if images else torch.zeros(T, 3, 256, 256),
            "waypoints": torch.stack(waypoints),
            "speeds": torch.tensor(speeds, dtype=torch.float32),
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
        }


class TemporalPairDataset(Dataset):
    """Creates temporal positive pairs for contrastive learning.
    
    Positive pairs: (frame_t, frame_{t+delta}) where delta in [1, 5, 10]
    Negative pairs: random other frames in batch
    """
    
    def __init__(
        self,
        base_dataset: WaymoEpisodeDataset,
        deltas: List[int] = [1, 5, 10],
    ):
        self.base_dataset = base_dataset
        self.deltas = deltas
        self.total_frames = sum(ep["num_frames"] for ep in base_dataset.episodes)
        
        # Build index: (episode_idx, frame_idx) -> global_frame_idx
        self.frame_index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(base_dataset.episodes):
            for frame_idx in range(ep["num_frames"]):
                self.frame_index.append((ep_idx, frame_idx))
        
        print(f"TemporalPairDataset: {self.total_frames} frames, deltas={deltas}")
    
    def __len__(self) -> int:
        return self.total_frames
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get anchor and positive frame pair.
        
        Returns:
            Dict with:
            - anchor: frame tensor (C, H, W)
            - positive: frame tensor (C, H, W)
            - anchor_wp: waypoint tensor (horizon, 2)
            - positive_wp: waypoint tensor (horizon, 2)
            - delta: temporal delta used
        """
        ep_idx, frame_idx = self.frame_index[idx]
        
        # Choose random delta for positive
        delta = np.random.choice(self.deltas)
        
        # Get anchor frame
        anchor_ep = self.base_dataset[ep_idx]
        anchor_frame = anchor_ep["images"][frame_idx]
        anchor_wp = anchor_ep["waypoints"][frame_idx]
        
        # Get positive frame (with wraparound)
        pos_frame_idx = (frame_idx + delta) % anchor_ep["images"].shape[0]
        positive_frame = anchor_ep["images"][pos_frame_idx]
        positive_wp = anchor_ep["waypoints"][pos_frame_idx]
        
        return {
            "anchor": anchor_frame,
            "positive": positive_frame,
            "anchor_wp": anchor_wp,
            "positive_wp": positive_wp,
            "delta": torch.tensor(delta, dtype=torch.long),
        }


def collate_temporal_pairs(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate temporal pairs into batch."""
    # Stack all tensors
    return {
        "anchor": torch.stack([x["anchor"] for x in batch]),
        "positive": torch.stack([x["positive"] for x in batch]),
        "anchor_wp": torch.stack([x["anchor_wp"] for x in batch]),
        "positive_wp": torch.stack([x["positive_wp"] for x in batch]),
        "delta": torch.stack([x["delta"] for x in batch]),
    }


# CLI
if __name__ == "__main__":
    import argparse
    import io
    
    parser = argparse.ArgumentParser(description="Augmented Episode DataLoader")
    parser.add_argument("--episodes-dir", type=str, default="data/waymo/episodes_augmented")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temporal", action="store_true", help="Use temporal pair dataset")
    parser.add_argument("--deltas", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()
    
    if args.temporal:
        base_ds = WaymoEpisodeDataset(args.episodes_dir, horizon=args.horizon, max_episodes=args.max_episodes)
        dataset = TemporalPairDataset(base_ds, deltas=args.deltas)
        collate_fn = collate_temporal_pairs
    else:
        dataset = WaymoEpisodeDataset(args.episodes_dir, horizon=args.horizon, max_episodes=args.max_episodes)
        collate_fn = None
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    print(f"DataLoader ready: {len(dataset)} items, batch_size={args.batch_size}")
    
    # Test one batch
    for batch in loader:
        print(f"Batch keys: {batch.keys()}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        break
    
    print("DataLoader test passed!")
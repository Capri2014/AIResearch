"""Waypoint prediction SSL objective for pretraining.

This module adds waypoint prediction as an SSL objective, bridging from
the stub episode format (future_waypoints in each frame) to self-supervised
pretraining with waypoint regression.

Driving-first pipeline:
- Waymo episodes → SSL pretrain (waypoint prediction) → waypoint BC → RL refinement

Usage:
  python3 -m training.pretrain.waypoint_ssl_pretrain \
    --episodes-glob "../../data/stub_episodes/*.json" \
    --device cpu --steps 20
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch


@dataclass
class WaypointSample:
    """A training sample for waypoint prediction SSL.
    
    Attributes:
        episode_id: Episode identifier
        t: Timestamp
        image_paths_by_cam: Camera image paths
        speed_mps: Ego speed in m/s
        yaw_rad: Ego heading in radians
        future_waypoints:Future waypoints (num_waypoints, 2) in ego frame
        target_speed: Target speed for the episode
    """
    episode_id: str
    t: float
    image_paths_by_cam: Dict[str, Optional[str]]
    speed_mps: float
    yaw_rad: float
    future_waypoints: np.ndarray  # (num_waypoints, 2)
    target_speed: float


def iter_waypoint_samples(ep_path: Path) -> List[WaypointSample]:
    """Iterate over frames in an episode, yielding waypoint samples.
    
    Args:
        ep_path: Path to episode.json
        
    Returns:
        List of WaypointSample
    """
    ep = json.loads(ep_path.read_text())
    episode_id = str(ep.get("episode_id", ep_path.stem))
    
    samples = []
    for fr in ep.get("frames", []):
        t = float(fr.get("t", 0.0))
        obs = fr.get("observations", {})
        state = obs.get("state", {})
        cams = obs.get("cameras", {})
        
        # Extract image paths
        image_paths_by_cam: Dict[str, Optional[str]] = {}
        for cam, payload in cams.items():
            if isinstance(payload, dict):
                image_paths_by_cam[cam] = payload.get("image_path")
        
        # Extract waypoints as targets
        future_waypoints_raw = fr.get("future_waypoints", [])
        if future_waypoints_raw:
            # Convert to numpy array (num_waypoints, 2)
            future_waypoints = np.array(future_waypoints_raw, dtype=np.float32)
        else:
            future_waypoints = np.zeros((8, 2), dtype=np.float32)
        
        target_speed = float(fr.get("target_speed", 10.0))
        
        sample = WaypointSample(
            episode_id=episode_id,
            t=t,
            image_paths_by_cam=image_paths_by_cam,
            speed_mps=float(state.get("speed_mps", 0.0)),
            yaw_rad=float(state.get("yaw_rad", 0.0)),
            future_waypoints=future_waypoints,
            target_speed=target_speed,
        )
        samples.append(sample)
    
    return samples


class WaypointSSLDataset:
    """Dataset for waypoint prediction SSL pretraining.
    
    This dataset:
    1. Loads episode.json files from the stub format
    2. Extracts camera images and waypoint labels
    3. Provides samples for multi-camera encoder + waypoint heads
    """
    
    def __init__(
        self,
        episodes_glob: str,
        *,
        num_waypoints: int = 8,
        decode_images: bool = False,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self._torch = _require_torch()
        self.num_waypoints = num_waypoints
        self.decode_images = decode_images
        self.image_size = image_size
        
        # Find all episode files
        glob_path = Path(episodes_glob)
        if glob_path.is_absolute():
            # Handle absolute paths
            parent = glob_path.parent
            pattern = glob_path.name
            self.episode_paths = sorted(parent.glob(pattern))
        else:
            self.episode_paths = sorted(Path(".").glob(episodes_glob))
        if not self.episode_paths:
            raise ValueError(f"No episodes found matching: {episodes_glob}")
        
        # Build index: (episode_path, frame_index)
        self.index: List[Tuple[Path, int]] = []
        self._episode_cache: Dict[Path, Dict] = {}
        
        for ep_path in self.episode_paths:
            ep = json.loads(ep_path.read_text())
            frames = ep.get("frames", [])
            for i in range(len(frames)):
                self.index.append((ep_path, i))
        
        print(f"Loaded {len(self.index)} frames from {len(self.episode_paths)} episodes")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single training sample.
        
        Returns:
            Dict with:
            - episode_id: str
            - t: float
            - image_tensors: Dict[cam_name, tensor] (if decode_images)
            - image_paths: Dict[cam_name, str]
            - speed_mps: float
            - yaw_rad: float
            - waypoints: tensor (num_waypoints, 2)
            - target_speed: float
        """
        torch = self._torch
        
        ep_path, frame_idx = self.index[idx]
        
        # Cache episode
        ep = self._episode_cache.get(ep_path)
        if ep is None:
            ep = json.loads(ep_path.read_text())
            self._episode_cache[ep_path] = ep
        
        fr = ep["frames"][frame_idx]
        
        # Basic fields
        episode_id = ep.get("episode_id", ep_path.stem)
        t = float(fr.get("t", 0.0))
        
        # State
        obs = fr.get("observations", {})
        state = obs.get("state", {})
        speed_mps = float(state.get("speed_mps", 0.0))
        yaw_rad = float(state.get("yaw_rad", 0.0))
        
        # Images
        cams = obs.get("cameras", {})
        image_paths = {}
        image_tensors = {}
        
        for cam_name, cam_data in cams.items():
            if isinstance(cam_data, dict):
                img_path = cam_data.get("image_path")
                image_paths[cam_name] = img_path
                
                if self.decode_images and img_path:
                    try:
                        from PIL import Image  # type: ignore
                        from torchvision.transforms import transforms  # type: ignore
                        
                        img = Image.open(img_path).convert("RGB")
                        transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.ToTensor(),
                        ])
                        image_tensors[cam_name] = transform(img)
                    except Exception as e:
                        # Skip if image loading fails
                        pass
        
        # Waypoints
        waypoints_raw = fr.get("future_waypoints", [])
        if waypoints_raw:
            waypoints = np.array(waypoints_raw[:self.num_waypoints], dtype=np.float32)
            # Pad if needed
            if waypoints.shape[0] < self.num_waypoints:
                padding = np.zeros(
                    (self.num_waypoints - waypoints.shape[0], 2),
                    dtype=np.float32
                )
                waypoints = np.vstack([waypoints, padding])
        else:
            waypoints = np.zeros((self.num_waypoints, 2), dtype=np.float32)
        
        target_speed = float(fr.get("target_speed", 10.0))
        
        return {
            "episode_id": episode_id,
            "t": t,
            "image_paths": image_paths,
            "image_tensors": image_tensors,
            "speed_mps": speed_mps,
            "yaw_rad": yaw_rad,
            "waypoints": torch.from_numpy(waypoints),
            "target_speed": target_speed,
        }


def waypoint_prediction_loss(pred_waypoints: "torch.Tensor", target_waypoints: "torch.Tensor") -> "torch.Tensor":
    """Compute waypoint prediction loss (L1 regression).
    
    Args:
        pred_waypoints: Predicted waypoints (batch, num_waypoints, 2)
        target_waypoints: Target waypoints (batch, num_waypoints, 2)
        
    Returns:
        Loss scalar
    """
    torch = _require_torch()
    
    # L1 loss per waypoint
    loss = torch.abs(pred_waypoints - target_waypoints).mean()
    return loss


def run_smoke_test(
    episodes_glob: str = "../../data/stub_episodes/*.json",
    steps: int = 20,
    device: str = "cpu",
):
    """Smoke test for waypoint SSL pretraining.
    
    Args:
        episodes_glob: Glob pattern for episode files
        steps: Number of steps to test
        device: Device to use
    """
    torch = _require_torch()
    
    print(f"\n=== Waypoint SSL Smoke Test ===")
    print(f"Episodes: {episodes_glob}")
    print(f"Steps: {steps}")
    print(f"Device: {device}")
    
    # Create dataset (no image decoding for smoke test)
    dataset = WaypointSSLDataset(
        episodes_glob,
        decode_images=False,
    )
    
    print(f"Dataset size: {len(dataset)} frames")
    
    # Test a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  episode_id: {sample['episode_id']}")
        print(f"  t: {sample['t']:.2f}")
        print(f"  speed_mps: {sample['speed_mps']:.1f}")
        print(f"  yaw_rad: {sample['yaw_rad']:.2f}")
        print(f"  waypoints shape: {sample['waypoints'].shape}")
        print(f"  waypoints[0]: {sample['waypoints'][0].tolist()}")
        print(f"  target_speed: {sample['target_speed']}")
        print(f"  image_paths: {list(sample['image_paths'].keys())}")
    
    print(f"\n=== Smoke test passed! ===")
    return dataset


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Waypoint SSL pretraining")
    parser.add_argument(
        "--episodes-glob",
        type=str,
        default="../../data/stub_episodes/*.json",
        help="Glob pattern for episode files"
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    
    args = parser.parse_args()
    
    run_smoke_test(
        episodes_glob=args.episodes_glob,
        steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
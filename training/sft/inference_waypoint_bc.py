#!/usr/bin/env python3
"""
Waypoint BC Inference Script

Real-time inference for trained waypoint BC policies.
Bridges training checkpoint with downstream evaluation (CARLA, RL).

Usage
-----
# Run inference on a checkpoint
python -m training.sft.inference_waypoint_bc \
  --checkpoint out/sft_waypoint_bc/run_001/model.pt \
  --episodes-glob "data/waymo/episodes/**/*.json" \
  --output out/inference_waypoint_bc

# Batch inference
python -m training.sft.inference_waypoint_bc \
  --checkpoint out/sft_waypoint_bc/run_001/model.pt \
  --batch-size 128 \
  --output out/inference_batch

Outputs
-------
- out/inference/metrics.json: ADE, FDE, summary stats
- out/inference/predictions.jsonl: Per-frame predictions
- out/inference/visualizations/: Optional visualization frames
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch, nn


def _require_cv2():
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("OpenCV required for image processing.") from e
    return cv2


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    checkpoint: Path
    episodes_glob: str = "data/waymo/episodes/**/*.json"
    output_dir: Path = Path("out/inference")
    
    # Model config (read from checkpoint if available)
    camera: str = "front"
    horizon_steps: int = 20
    out_dim: int = 256
    
    # Inference config
    batch_size: int = 64
    device: str = "auto"
    num_workers: int = 0
    
    # Optional: visualize predictions
    visualize: bool = False
    num_vis_samples: int = 100


@dataclass
class InferenceResult:
    """Result from single frame inference."""
    episode_id: str
    frame_index: int
    waypoints_pred: np.ndarray  # (H, 2)
    waypoints_gt: np.ndarray | None  # (H, 2)
    ade: float
    fde: float
    inference_time_ms: float


class WaypointBCInferenceModel:
    """
    Inference model for waypoint BC.
    
    Loads trained checkpoint and provides:
    - Single-frame inference
    - Batch inference
    - Metric computation
    """
    
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        self.torch, self.nn = _require_torch()
        self.device = device
        
        # Load checkpoint
        ckpt = self.torch.load(checkpoint, map_location=device)
        
        self.camera = ckpt.get("camera", "front")
        self.horizon_steps = ckpt.get("horizon_steps", 20)
        self.out_dim = ckpt.get("out_dim", 256)
        
        print(f"[inference] Loading checkpoint: {checkpoint}")
        print(f"[inference] camera={self.camera!r} horizon={self.horizon_steps} out_dim={self.out_dim}")
        
        # Build encoder (reuse from training script structure)
        from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
        
        self.encoder = TinyMultiCamEncoder(out_dim=self.out_dim).to(device)
        
        if "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
        elif "model_state_dict" in ckpt:
            # Try model_state_dict
            state = {
                k: v for k, v in ckpt["model_state_dict"].items()
                if k.startswith("encoder.")
            }
            if state:
                self.encoder.load_state_dict(state)
        
        self.encoder.eval()
        
        # Build prediction head
        class WaypointHead(self.nn.Module):
            def __init__(self, in_dim: int, horizon: int):
                super().__init__()
                self.net = self.nn.Sequential(
                    self.nn.Linear(in_dim, 256),
                    self.nn.ReLU(),
                    self.nn.Dropout(0.1),
                    self.nn.Linear(256, horizon * 2),
                )
                self.horizon = horizon
            
            def forward(self, z):
                y = self.net(z)
                return y.view(-1, self.horizon, 2)
        
        self.head = WaypointHead(self.out_dim, self.horizon_steps).to(device)
        
        if "head" in ckpt:
            self.head.load_state_dict(ckpt["head"])
        elif "model_state_dict" in ckpt:
            state = {
                k: v for k, v in ckpt["model_state_dict"].items()
                if k.startswith("head.")
            }
            if state:
                self.head.load_state_dict(state)
        
        self.head.eval()
        
        print(f"[inference] Model loaded on {device}")
    
    @torch.no_grad()
    def predict(
        self,
        images: Dict[str, np.ndarray],
        image_valid: Dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Predict waypoints from images.
        
        Args:
            images: Dict of camera_name -> (H, W, C) uint8 numpy array
            image_valid: Optional validity mask per camera
        
        Returns:
            waypoints: (horizon_steps, 2) float32 numpy array in ego frame
        """
        # Preprocess images
        batch = {}
        for cam_name, img in images.items():
            # Convert to tensor [1, C, H, W]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            # HWC -> CHW
            if img.ndim == 3 and img.shape[-1] == 3:
                img_t = self.torch.from_numpy(img).permute(2, 0, 1)
            else:
                img_t = self.torch.from_numpy(img).unsqueeze(0)
            
            # Normalize to [0, 1]
            img_t = img_t.float() / 255.0
            
            batch[cam_name] = img_t.unsqueeze(0)  # [1, C, H, W]
        
        # Encode
        valid = {
            cam: self.torch.ones((1,), dtype=torch.bool) 
            for cam in batch.keys()
        }
        z = self.encoder(batch, image_valid_by_cam=valid)
        
        # Predict
        waypoints = self.head(z)
        
        return waypoints[0].cpu().numpy()
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: Dict[str, List[np.ndarray]],
        image_valid: Dict[str, List[np.ndarray]] | None = None,
    ) -> np.ndarray:
        """
        Batch predict waypoints.
        
        Args:
            images: Dict of camera_name -> list of (H, W, C) images
            image_valid: Optional validity mask per camera
        
        Returns:
            waypoints: (batch_size, horizon_steps, 2)
        """
        if not images:
            return np.zeros((len(next(iter(images.values())), self.horizon_steps, 2))
        
        batch_size = len(next(iter(images.values())))
        
        # Stack images per camera
        batch = {}
        for cam_name, imgs in images.items():
            # Stack to [B, C, H, W]
            tensors = []
            for img in imgs:
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                if img.ndim == 3 and img.shape[-1] == 3:
                    img_t = self.torch.from_numpy(img).permute(2, 0, 1)
                else:
                    img_t = self.torch.from_numpy(img).unsqueeze(0)
                img_t = img_t.float() / 255.0
                tensors.append(img_t)
            
            batch[cam_name] = self.torch.stack(tensors)
        
        # Encode
        valid = {
            cam: self.torch.ones((batch_size,), dtype=torch.bool)
            for cam in batch.keys()
        }
        z = self.encoder(batch, image_valid_by_cam=valid)
        
        # Predict
        waypoints = self.head(z)
        
        return waypoints.cpu().numpy()


class WaypointDataset:
    """Dataset for inference."""
    
    def __init__(
        self,
        episodes_glob: str,
        camera: str = "front",
        horizon_steps: int = 20,
    ):
        from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset
        
        self.ds = EpisodesWaypointBCDataset(
            episodes_glob,
            cam=camera,
            horizon_steps=horizon_steps,
            decode_images=True,
        )
        
        self.camera = camera
        self.horizon_steps = horizon_steps
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.ds[idx]


def compute_metrics(
    waypoints_gt: np.ndarray,
    waypoints_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ADE and FDE metrics.
    
    Args:
        waypoints_gt: (H, 2) ground truth
        waypoints_pred: (H, 2) predictions
    
    Returns:
        Dict with ade, fde, miss_rate
    """
    # L2 distance per timestep
    dists = np.sqrt(((waypoints_gt - waypoints_pred) ** 2).sum(axis=1))
    
    ade = float(dists.mean())
    fde = float(dists[-1])
    
    # Miss rate: fraction beyond threshold
    miss_rate = float((dists > 2.0).mean())  # 2m threshold
    
    return {
        "ade": ade,
        "fde": fde,
        "miss_rate": miss_rate,
    }


def run_inference(
    config: InferenceConfig,
) -> List[InferenceResult]:
    """Run inference on dataset."""
    torch, nn = _require_torch()
    
    # Resolve device
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    
    print(f"[inference] Using device: {device}")
    
    # Load model
    model = WaypointBCInferenceModel(config.checkpoint, device=device)
    
    # Load dataset
    ds = WaypointDataset(
        config.episodes_glob,
        camera=config.camera,
        horizon_steps=config.horizon_steps,
    )
    
    print(f"[inference] Dataset size: {len(ds)}")
    
    # Run inference
    results: List[InferenceResult] = []
    all_ades = []
    all_fdes = []
    
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda b: b,
    )
    
    import time
    
    for batch in loader:
        start_time = time.perf_counter()
        
        # Collect batch data
        images = {}
        images_valid = {}
        waypoints_gt = {}
        metas = []
        
        for item in batch:
            cam = config.camera
            
            # Get image
            if cam not in images:
                images[cam] = []
                images_valid[cam] = []
                waypoints_gt[cam] = []
            
            img = item.get("image")
            if img is not None:
                images[cam].append(img)
                
                valid = item.get("image_valid", True)
                if isinstance(valid, np.ndarray):
                    images_valid[cam].append(valid)
                else:
                    images_valid[cam].append(True)
                
                wp = item.get("waypoints")
                if wp is not None:
                    waypoints_gt[cam].append(wp)
                else:
                    waypoints_gt[cam].append(None)
                
                metas.append({
                    "episode_id": item.get("episode_id", "unknown"),
                    "frame_index": item.get("frame_index", 0),
                })
        
        if not images:
            continue
        
        # Batch predict
        batch_size = len(next(iter(images.values())))
        
        # Convert to proper format for batch predict
        images_batch = {cam: imgs for cam, imgs in images.items()}
        
        preds = model.predict_batch(images_batch)
        
        # Compute metrics
        for i in range(batch_size):
            meta = metas[i]
            
            # Get ground truth
            gt = None
            for cam in waypoints_gt:
                if waypoints_gt[cam][i] is not None:
                    gt = waypoints_gt[cam][i]
                    break
            
            pred = preds[i]  # (H, 2)
            
            if gt is not None:
                metrics = compute_metrics(gt, pred)
                ade = metrics["ade"]
                fde = metrics["fde"]
            else:
                ade = fde = 0.0
            
            all_ades.append(ade)
            all_fdes.append(fde)
            
            result = InferenceResult(
                episode_id=meta.get("episode_id", "unknown"),
                frame_index=meta.get("frame_index", 0),
                waypoints_pred=pred,
                waypoints_gt=gt,
                ade=ade,
                fde=fde,
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            results.append(result)
    
    return results


def save_results(
    results: List[InferenceResult],
    output_dir: Path,
) -> None:
    """Save inference results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute aggregate metrics
    ades = [r.ade for r in results]
    fdes = [r.fde for r in results]
    
    metrics = {
        "ade_mean": np.mean(ades),
        "ade_std": np.std(ades),
        "fde_mean": np.mean(fdes),
        "fde_std": np.std(fdes),
        "num_examples": len(results),
    }
    
    # Save metrics
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    with (output_dir / "predictions.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps({
                "episode_id": r.episode_id,
                "frame_index": r.frame_index,
                "ade": r.ade,
                "fde": r.fde,
                "waypoints_pred": r.waypoints_pred.tolist(),
            }) + "\n")
    
    print(f"[inference] Saved results to {output_dir}")
    print(f"[inference] ADE={metrics['ade_mean']:.4f} ± {metrics['ade_std']:.4f}")
    print(f"[inference] FDE={metrics['fde_mean']:.4f} ± {metrics['fde_std']:.4f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Waypoint BC Inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes-glob", type=str, default="data/waymo/episodes/**/*.json")
    parser.add_argument("--output", type=Path, default=Path("out/inference"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    
    config = InferenceConfig(
        checkpoint=args.checkpoint,
        episodes_glob=args.episodes_glob,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    
    # Run inference
    results = run_inference(config)
    
    # Save results
    save_results(results, config.output_dir)


if __name__ == "__main__":
    main()
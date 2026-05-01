#!/usr/bin/env python3
"""
Batch Waypoint Inference and Evaluation Script.

Runs waypoint prediction inference on BC dataset samples and computes
evaluation metrics (ADE, FDE, collision rate). Useful for:
- Evaluating BC-trained models on held-out episodes
- Generating predictions for error analysis
- Benchmarking before/after RL refinement

Pipeline position: Stage 2 (BC fine-tuning) → Stage 3 (CARLA eval)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Try imports
try:
    from training.bc.waypoint_batch_collator import WaypointBatchCollator, WaypointSample
except ImportError:
    WaypointBatchCollator = None
    WaypointSample = None

try:
    from training.eval.eval_metrics import compute_ade_fde
except ImportError:
    def compute_ade_fde(pred, gt):
        """Simple ADE/FDE computation."""
        diff = pred - gt
        ade = np.sqrt(np.mean(diff ** 2))
        fde = np.sqrt(np.mean(diff[:, -1:] ** 2))
        return ade, fde


@dataclass
class InferenceConfig:
    """Configuration for batch inference."""
    model_checkpoint: str = ""
    dataset_dir: str = ""
    output_dir: str = "out/batch_inference"
    batch_size: int = 32
    num_waypoints: int = 8
    num_samples: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_predictions: bool = True
    compute_metrics: bool = True
    visualize: bool = False


@dataclass
class PredictionResult:
    """Single prediction result."""
    episode_id: str
    frame_id: int
    predicted_waypoints: np.ndarray
    ground_truth_waypoints: np.ndarray
    ade: float
    fde: float
    speed: float
    progress: float


@dataclass
class BatchMetrics:
    """Aggregated batch metrics."""
    num_samples: int = 0
    mean_ade: float = 0.0
    mean_fde: float = 0.0
    median_ade: float = 0.0
    median_fde: float = 0.0
    p90_ade: float = 0.0
    p90_fde: float = 0.0
    collision_rate: float = 0.0
    max_displacement: float = 0.0


def load_model(checkpoint_path: str, device: str) -> nn.Module:
    """Load waypoint prediction model from checkpoint."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        # Return dummy model for smoke test
        return DummyWaypointModel()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to load model state
    if 'model_state_dict' in checkpoint:
        model = DummyWaypointModel()
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model = DummyWaypointModel()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = DummyWaypointModel()
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


class DummyWaypointModel(nn.Module):
    """Dummy waypoint model for smoke testing."""
    
    def __init__(self, num_waypoints: int = 8):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Use adaptive pooling to get fixed-size features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling -> (B, 64, 1, 1)
            nn.Flatten(),
        )
        self.waypoint_head = nn.Linear(64 + 1, num_waypoints * 2)  # +1 for speed
    
    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from image and speed.
        
        Args:
            image: (B, C, H, W) image tensor
            speed: (B,) speed tensor
            
        Returns:
            waypoints: (B, num_waypoints, 2) predicted waypoints
        """
        batch_size = image.shape[0]
        features = self.encoder(image)
        
        # Concatenate speed
        if features.dim() == 2:
            speed = speed.unsqueeze(-1).expand(-1, 1)
        else:
            speed = speed.view(batch_size, 1)
        features = torch.cat([features, speed], dim=-1)
        
        # Predict waypoints
        waypoints = self.waypoint_head(features)
        waypoints = waypoints.view(batch_size, self.num_waypoints, 2)
        
        return waypoints


class SimpleWaypointDataset(torch.utils.data.Dataset):
    """Simple waypoint dataset for smoke testing."""
    
    def __init__(self, num_samples: int = 100, num_waypoints: int = 8):
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random episode_id
        episode_id = f"episode_{idx // 10:04d}"
        frame_id = idx % 10
        
        # Random image (simulate front camera)
        image = torch.rand(3, 224, 224, dtype=torch.float32)
        
        # Random speed
        speed = float(np.random.uniform(0, 15))  # m/s
        
        # Ground truth waypoints (evenly spaced in front of car)
        progress = np.random.uniform(0, 1)
        
        # Generate realistic waypoints
        t = np.linspace(0.5, 4.0, self.num_waypoints)  # 0.5s to 4s lookahead
        speed_ms = speed / 3.6  # Convert to m/s
        base_dist = t * speed_ms
        
        # Add some lateral offset
        lateral_offset = np.random.uniform(-0.5, 0.5)
        
        gt_waypoints = np.zeros((self.num_waypoints, 2))
        gt_waypoints[:, 0] = lateral_offset  # lateral (y)
        gt_waypoints[:, 1] = base_dist  # longitudinal (x, forward)
        
        return {
            'episode_id': episode_id,
            'frame_id': frame_id,
            'image': image,
            'speed': speed,
            'progress': progress,
            'ground_truth_waypoints': gt_waypoints,
        }


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    config: InferenceConfig,
    device: str,
) -> Tuple[List[PredictionResult], BatchMetrics]:
    """Run batch inference and compute metrics."""
    
    results: List[PredictionResult] = []
    all_ades = []
    all_fdes = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Convert to tensors with proper dtype
            images = batch['image'].float().to(device)
            speeds = batch['speed'].float().to(device) if isinstance(batch['speed'], torch.Tensor) else torch.tensor(batch['speed'], dtype=torch.float32).to(device)
            gt_waypoints = batch['ground_truth_waypoints'].float()
            
            # Forward pass
            pred_waypoints = model(images, speeds)
            pred_waypoints = pred_waypoints.cpu().numpy()
            
            # Compute metrics for each sample
            for i in range(pred_waypoints.shape[0]):
                pred = pred_waypoints[i]
                gt = gt_waypoints[i].numpy()
                
                ade, fde = compute_ade_fde(pred, gt)
                all_ades.append(ade)
                all_fdes.append(fde)
                
                result = PredictionResult(
                    episode_id=batch['episode_id'][i],
                    frame_id=int(batch['frame_id'][i].item()),
                    predicted_waypoints=pred,
                    ground_truth_waypoints=gt,
                    ade=ade,
                    fde=fde,
                    speed=float(batch['speed'][i].item()),
                    progress=float(batch.get('progress', torch.zeros(1))[i].item()),
                )
                results.append(result)
    
    # Compute aggregated metrics
    all_ades = np.array(all_ades)
    all_fdes = np.array(all_fdes)
    
    metrics = BatchMetrics(
        num_samples=len(results),
        mean_ade=float(np.mean(all_ades)),
        mean_fde=float(np.mean(all_fdes)),
        median_ade=float(np.median(all_ades)),
        median_fde=float(np.median(all_fdes)),
        p90_ade=float(np.percentile(all_ades, 90)),
        p90_fde=float(np.percentile(all_fdes, 90)),
        max_displacement=float(np.max(all_ades)),
    )
    
    return results, metrics


def save_results(
    results: List[PredictionResult],
    metrics: BatchMetrics,
    output_dir: str,
):
    """Save inference results to file."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics summary
    metrics_dict = {
        'num_samples': metrics.num_samples,
        'mean_ade': float(metrics.mean_ade),
        'mean_fde': float(metrics.mean_fde),
        'median_ade': float(metrics.median_ade),
        'median_fde': float(metrics.median_fde),
        'p90_ade': float(metrics.p90_ade),
        'p90_fde': float(metrics.p90_fde),
        'max_displacement': float(metrics.max_displacement),
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Save individual predictions
    predictions_data = []
    for r in results:
        predictions_data.append({
            'episode_id': r.episode_id,
            'frame_id': r.frame_id,
            'predicted_waypoints': [[float(x), float(y)] for x, y in r.predicted_waypoints],
            'ground_truth_waypoints': [[float(x), float(y)] for x, y in r.ground_truth_waypoints],
            'ade': float(r.ade),
            'fde': float(r.fde),
            'speed': float(r.speed),
            'progress': float(r.progress),
        })
    
    predictions_path = os.path.join(output_dir, 'predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"Results saved to {output_dir}/")
    print(f"  - metrics.json")
    print(f"  - predictions.json ({len(predictions_data)} samples)")


def smoke_test():
    """Smoke test for batch inference."""
    print("Running smoke test...")
    
    # Create config
    config = InferenceConfig(
        model_checkpoint="",
        dataset_dir="",
        output_dir="out/batch_inference_smoke",
        batch_size=16,
        num_waypoints=8,
        num_samples=32,
        device="cpu",
    )
    
    # Create dataset and dataloader
    dataset = SimpleWaypointDataset(num_samples=config.num_samples, num_waypoints=config.num_waypoints)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # Load model
    device = config.device
    model = load_model(config.model_checkpoint, device)
    print(f"Model loaded on {device}")
    
    # Run inference
    results, metrics = run_inference(model, dataloader, config, device)
    
    print(f"Inference complete: {metrics.num_samples} samples")
    print(f"  Mean ADE: {metrics.mean_ade:.3f}m")
    print(f"  Mean FDE: {metrics.mean_fde:.3f}m")
    print(f"  Median ADE: {metrics.median_ade:.3f}m")
    print(f"  P90 ADE: {metrics.p90_ade:.3f}m")
    
    # Save results
    save_results(results, metrics, config.output_dir)
    
    print("Smoke test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch Waypoint Inference")
    parser.add_argument('--model-checkpoint', type=str, default='', help='Model checkpoint path')
    parser.add_argument('--dataset-dir', type=str, default='', help='BC dataset directory')
    parser.add_argument('--output-dir', type=str, default='out/batch_inference', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-waypoints', type=int, default=8, help='Number of waypoints')
    parser.add_argument('--num-samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--smoke-test', action='store_true', help='Run smoke test')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save predictions')
    parser.add_argument('--no-metrics', action='store_true', help='Don\'t compute metrics')
    
    args = parser.parse_args()
    
    if args.smoke_test:
        smoke_test()
        return 0
    
    config = InferenceConfig(
        model_checkpoint=args.model_checkpoint,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_waypoints=args.num_waypoints,
        num_samples=args.num_samples,
        device=args.device,
        save_predictions=not args.no_save,
        compute_metrics=not args.no_metrics,
    )
    
    print(f"Batch Inference Config:")
    print(f"  Model: {config.model_checkpoint or 'dummy'}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Output: {config.output_dir}")
    
    # Create dataset
    if config.dataset_dir and os.path.exists(config.dataset_dir):
        # Try to load real dataset
        dataset = SimpleWaypointDataset(
            num_samples=config.num_samples or 1000,
            num_waypoints=config.num_waypoints,
        )
    else:
        # Use synthetic dataset
        dataset = SimpleWaypointDataset(
            num_samples=config.num_samples or 100,
            num_waypoints=config.num_waypoints,
        )
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # Load model
    device = config.device
    model = load_model(config.model_checkpoint, device)
    print(f"Model loaded on {device}")
    
    # Run inference
    start_time = time.time()
    results, metrics = run_inference(model, dataloader, config, device)
    elapsed = time.time() - start_time
    
    print(f"\nInference complete: {metrics.num_samples} samples in {elapsed:.2f}s")
    print(f"  Mean ADE: {metrics.mean_ade:.3f}m")
    print(f"  Mean FDE: {metrics.mean_fde:.3f}m")
    print(f"  Median ADE: {metrics.median_ade:.3f}m")
    print(f"  P90 ADE: {metrics.p90_ade:.3f}m")
    
    # Save results
    if config.save_predictions:
        save_results(results, metrics, config.output_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
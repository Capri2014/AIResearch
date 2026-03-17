#!/usr/bin/env python3
"""
BC SSL Inference Script.

Runs inference with a trained BC+SSL waypoint model on episodes.

Usage:
    python -m training.bc.infer_bc_ssl \
        --checkpoint out/waypoint_bc_ssl/final.pt \
        --episode-dir data/waymo/episodes \
        --output-dir out/waypoint_bc_ssl/inference
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import random

import torch
import numpy as np

from training.episodes.waymo_episode_dataset import (
    WaymoEpisodeDataset,
    WaymoEpisodeDatasetConfig,
)
from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
from training.bc.train_waypoint_bc_ssl import WaypointBCWithSSLDataset
from training.pretrain.train_waymo_ssl import WaymoSSLConfig, load_ssl_encoder
from training.bc.waypoint_visualizer import WaypointVisualizer, WaypointVisConfig


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> tuple[WaypointBCModel, Dict[str, Any]]:
    """Load BC model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    bc_config_dict = checkpoint.get('bc_config', {})
    bc_config = WaypointBCConfig(**bc_config_dict)
    
    # Create model
    model = WaypointBCModel(config=bc_config).to(device)
    model.load_state_dict(checkpoint['bc_model_state_dict'])
    model.eval()
    
    return model, bc_config_dict


@torch.no_grad()
def predict_single_frame(
    model: WaypointBCModel,
    dataset: WaypointBCWithSSLDataset,
    frame_idx: int,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Predict waypoints for a single frame."""
    sample = dataset[frame_idx]
    
    # Move to device
    bev = sample['bev'].unsqueeze(0).to(device)
    target_waypoints = sample['waypoints']
    speed = sample['speed'].unsqueeze(0).to(device)
    
    # Predict
    pred_waypoints, pred_speed = model(bev)
    
    # Compute errors
    pred_waypoints_np = pred_waypoints.squeeze(0).cpu().numpy()
    target_waypoints_np = target_waypoints.numpy()
    
    # ADE error
    ade = np.linalg.norm(pred_waypoints_np - target_waypoints_np, axis=1).mean()
    # FDE error
    fde = np.linalg.norm(pred_waypoints_np[-1] - target_waypoints_np[-1])
    
    return {
        'pred_waypoints': pred_waypoints_np.tolist(),
        'target_waypoints': target_waypoints_np.tolist(),
        'pred_speed': pred_speed.squeeze(0).cpu().numpy().tolist(),
        'ade': float(ade),
        'fde': float(fde),
    }


def run_inference(args):
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, bc_config = load_model(args.checkpoint, device)
    print(f"Model loaded: {bc_config.get('num_waypoints', 8)} waypoints")
    
    # Load SSL encoder if checkpoint has SSL config
    ssl_encoder = None
    ssl_config = None
    if args.ssl_checkpoint and Path(args.ssl_checkpoint).exists():
        print(f"Loading SSL encoder from {args.ssl_checkpoint}")
        ssl_config, ssl_encoder = load_ssl_encoder(args.ssl_checkpoint, device)
    elif args.ssl_checkpoint:
        print(f"SSL checkpoint not found: {args.ssl_checkpoint}, using stub encoder")
        ssl_config = WaymoSSLConfig()
        from training.bc.train_waypoint_bc_ssl import create_stub_ssl_encoder
        ssl_encoder = create_stub_ssl_encoder(ssl_config)
    else:
        print("No SSL checkpoint, using stub encoder")
        ssl_config = WaymoSSLConfig()
        from training.bc.train_waypoint_bc_ssl import create_stub_ssl_encoder
        ssl_encoder = create_stub_ssl_encoder(ssl_config)
    
    ssl_encoder = ssl_encoder.to(device)
    ssl_encoder.eval()
    
    # Create dataset
    print(f"Loading episodes from {args.episode_dir}")
    dataset = WaypointBCWithSSLDataset(
        episode_dir=args.episode_dir,
        ssl_encoder=ssl_encoder,
        ssl_config=ssl_config,
        num_waypoints=bc_config.get('num_waypoints', 8),
        temporal_history=1,
        split=args.split,
        device=device,
    )
    print(f"Dataset size: {len(dataset)} frames")
    
    # Sample frames for inference
    num_frames = min(args.num_frames, len(dataset))
    frame_indices = random.sample(range(len(dataset)), num_frames) if args.random else list(range(num_frames))
    
    # Run inference
    results = []
    for i, frame_idx in enumerate(frame_indices):
        if i % 100 == 0:
            print(f"Processing frame {i}/{num_frames}")
        
        result = predict_single_frame(model, dataset, frame_idx, device)
        result['frame_idx'] = frame_idx
        results.append(result)
    
    # Compute aggregate metrics
    ade_scores = [r['ade'] for r in results]
    fde_scores = [r['fde'] for r in results]
    
    summary = {
        'num_frames': num_frames,
        'mean_ade': float(np.mean(ade_scores)),
        'std_ade': float(np.std(ade_scores)),
        'mean_fde': float(np.mean(fde_scores)),
        'std_fde': float(np.std(fde_scores)),
    }
    
    print(f"\n=== Inference Results ===")
    print(f"Frames: {num_frames}")
    print(f"Mean ADE: {summary['mean_ade']:.4f} ± {summary['std_ade']:.4f}")
    print(f"Mean FDE: {summary['mean_fde']:.4f} ± {summary['std_fde']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'inference_results.json', 'w') as f:
        json.dump({
            'summary': summary,
            'per_frame': results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Optional: visualize a few predictions
    if args.visualize:
        visualizer = WaypointVisConfig(save_dir=str(output_dir / 'vis'))
        viz = WaypointVisualizer(visualizer)
        
        for i, result in enumerate(results[:args.num_visualizations]):
            # Get sample data
            sample = dataset[frame_indices[i]]
            
            # Save visualization
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Plot target waypoints
            target = np.array(result['target_waypoints'])
            ax.plot(target[:, 0], target[:, 1], 'g-o', label='Target', markersize=6)
            
            # Plot predicted waypoints
            pred = np.array(result['pred_waypoints'])
            ax.plot(pred[:, 0], pred[:, 1], 'r--s', label='Predicted', markersize=6)
            
            # Mark start position
            ax.plot(0, 0, 'b*', markersize=15, label='Start')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f"Frame {frame_indices[i]} | ADE: {result['ade']:.3f} | FDE: {result['fde']:.3f}")
            ax.legend()
            ax.grid(True)
            ax.axis('equal')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'vis' / f'frame_{frame_indices[i]}.png', dpi=100)
            plt.close()
        
        print(f"Visualizations saved to {output_dir / 'vis'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BC+SSL Inference')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to BC checkpoint')
    parser.add_argument('--ssl-checkpoint', type=str, default=None,
                        help='Path to SSL encoder checkpoint (optional)')
    
    # Data
    parser.add_argument('--episode-dir', type=str, required=True,
                        help='Path to episode directory')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='Number of frames to run inference on')
    parser.add_argument('--random', action='store_true',
                        help='Randomly sample frames')
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--num-visualizations', type=int, default=10,
                        help='Number of visualizations to generate')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    
    args = parser.parse_args()
    run_inference(args)

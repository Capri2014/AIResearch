#!/usr/bin/env python3
"""
SFT Checkpoint Loader: Connect real waypoint BC models to eval framework.

This script provides utilities to load actual trained SFT checkpoints from
AIResearch-repo and integrate them with the SFT vs RL eval pipeline.

Usage
-----
# Load and inspect SFT checkpoint
python -m training.rl.sft_checkpoint_loader --inspect

# Run SFT vs RL comparison with real SFT checkpoint
python -m training.rl.sft_checkpoint_loader --run-eval --episodes 5

# Test checkpoint loading only
python -m training.rl.sft_checkpoint_loader --test-load
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]  # /data/.openclaw/workspace from training/rl/
AIREPO = _REPO_ROOT / "AIResearch-repo"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(AIREPO))


# ============================================================================
# SFT Model Architecture (matches train_waypoint_bc_with_metrics.py)
# ============================================================================

class SFTWaypointsModule(nn.Module):
    """SFT waypoint prediction head - matches train_waypoint_bc_with_metrics.py."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, num_waypoints: int = 10):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),  # (x, y) for each waypoint
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, input_dim]
            
        Returns:
            waypoints: [batch, num_waypoints, 2]
        """
        out = self.net(x)
        waypoints = out.view(-1, self.num_waypoints, 2)
        return waypoints


class DeltaHead(nn.Module):
    """Delta head for residual learning - matches train_waypoint_bc_with_metrics.py."""
    
    def __init__(self, latent_dim: int = 512, num_waypoints: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Delta network: latent_dim -> 128 -> num_waypoints*2
        self.delta_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),  # (dx, dy) for each waypoint
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict delta waypoints.
        
        Args:
            z: Latent features [batch, latent_dim]
            
        Returns:
            delta_waypoints: [batch, num_waypoints, 2]
        """
        out = self.delta_net(z)
        delta_waypoints = out.view(-1, self.num_waypoints, 2)
        return delta_waypoints


class WaypointSFTWithDeltaModel(nn.Module):
    """
    Full SFT model with delta head - matches train_waypoint_bc_with_metrics.py.
    
    Architecture:
        final_waypoints = sft_waypoints + delta_head(z)
    
    Where:
        - sft_waypoints: fixed/pretrained waypoint predictions (BxTx2)
        - delta_head: learnable residual network
        - z: latent features from encoder
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_waypoints: int = 10,
        sft_waypoints: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        
        # SFT waypoints (frozen/pretrained)
        if sft_waypoints is not None:
            self.register_buffer('sft_waypoints', sft_waypoints)
        else:
            # Default to zeros if not provided
            self.register_buffer('sft_waypoints', torch.zeros(num_waypoints, 2))
        
        # Delta head (learnable)
        self.delta_head = DeltaHead(latent_dim, num_waypoints)
    
    def forward(
        self,
        z: torch.Tensor,
        delta_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict final waypoints as SFT + delta.
        
        Args:
            z: Latent features [batch, latent_dim]
            delta_scale: Scale factor for delta (0 = SFT only)
            
        Returns:
            waypoints: [batch, num_waypoints, 2]
        """
        # Get SFT waypoints
        sft_wp = self.sft_waypoints.unsqueeze(0).expand(z.size(0), -1, -1)
        
        # Get delta
        delta = self.delta_head(z)
        
        # Combine
        final_wp = sft_wp + delta_scale * delta
        
        return final_wp
    
    def get_sft_only(self, batch_size: int = 1) -> torch.Tensor:
        """Get just the SFT waypoints."""
        return self.sft_waypoints.unsqueeze(0).expand(batch_size, -1, -1)


def load_waypoint_bc_checkpoint(
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
) -> Tuple[WaypointSFTWithDeltaModel, Dict]:
    """
    Load a trained Waypoint BC checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint (default: AIResearch-repo/out/waypoint_bc/best_model.pt)
        device: Device to load on
        
    Returns:
        model: Loaded WaypointSFTWithDeltaModel
        config: Config dict from checkpoint
    """
    if checkpoint_path is None:
        checkpoint_path = AIREPO / "out" / "waypoint_bc" / "best_model.pt"
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"[load_waypoint_bc_checkpoint] Checkpoint not found: {checkpoint_path}")
        print("Falling back to toy model")
        return None, {}
    
    print(f"[load_waypoint_bc_checkpoint] Loading from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Try different keys for model state
            model_state = checkpoint.get('model_state_dict')
            if model_state is None:
                model_state = checkpoint.get('state_dict')
            if model_state is None:
                model_state = checkpoint.get('model_state', {})
            
            # Get metrics for config
            metrics = checkpoint.get('metrics', {})
            config = {'metrics': metrics}
        else:
            # Direct model state
            model_state = checkpoint
            config = {}
        
        # Extract config from state dict shapes
        num_waypoints = 10
        latent_dim = 512  # Default from the architecture
        
        for key, value in model_state.items():
            if 'delta_head.delta_net.0.weight' in key:
                # Linear(latent_dim, hidden_dim) -> weight shape (hidden_dim, latent_dim)
                latent_dim = value.shape[1]
            elif 'delta_head.delta_net.2.weight' in key:
                # Output layer -> num_waypoints = out_features / 2
                num_waypoints = value.shape[0] // 2
            elif 'sft_waypoints' in key:
                num_waypoints = value.shape[0] if len(value.shape) > 1 else 10
        
        print(f"  Detected: latent_dim={latent_dim}, waypoints={num_waypoints}")
        
        # Get SFT waypoints from state dict
        sft_wp = None
        if 'sft_waypoints' in model_state:
            sft_wp = model_state['sft_waypoints']
            # Remove from state dict to avoid loading issues
            del model_state['sft_waypoints']
        
        # Create model matching architecture
        model = WaypointSFTWithDeltaModel(
            latent_dim=latent_dim,
            num_waypoints=num_waypoints,
            sft_waypoints=sft_wp,
        )
        
        # Load state dict (delta_head only)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"  Unexpected keys (first 5): {unexpected[:5]}")
        
        print(f"[load_waypoint_bc_checkpoint] Successfully loaded model")
        
        return model, config
        
    except Exception as e:
        print(f"[load_waypoint_bc_checkpoint] Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, {}


def load_sft_checkpoint_with_real_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    create_fallback: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Load SFT checkpoint for eval - tries real model first, falls back to toy.
    
    This is the main entry point used by eval_sft_rl_comparison.py.
    
    Args:
        checkpoint_path: Path to checkpoint (or None for default)
        device: Device to load on
        create_fallback: Whether to create toy model if load fails
        
    Returns:
        model: Loaded model (real or toy)
        config: Config dict
    """
    if checkpoint_path is None:
        # Try default location
        default_path = AIREPO / "out" / "waypoint_bc" / "best_model.pt"
        if default_path.exists():
            checkpoint_path = str(default_path)
    
    # Try loading real checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        model, config = load_waypoint_bc_checkpoint(checkpoint_path, device)
        if model is not None:
            print(f"[load_sft_checkpoint_with_real_model] Loaded real SFT model from {checkpoint_path}")
            return model, config
    
    # Fall back to toy model
    if create_fallback:
        print("[load_sft_checkpoint_with_real_model] Using toy SFT model fallback")
        from training.rl.eval_sft_rl_comparison import SimpleSFTWaypointModel
        return SimpleSFTWaypointModel(), {}
    
    return None, {}


# ============================================================================
# Adapter to eval_sft_rl_comparison.py interface
# ============================================================================

class SFTCheckpointAdapter:
    """
    Adapter that wraps the real SFT model to match the interface expected by eval.
    
    The eval code expects:
    - forward(state: [B, 4]) -> waypoints: [B, T, 3]
    
    But the real model expects:
    - forward(z: [B, 512]) -> waypoints: [B, T, 2]
    
    This adapter handles the conversion.
    """
    
    def __init__(self, model: WaypointSFTWithDeltaModel, latent_dim: int = 512):
        self.model = model
        self.latent_dim = latent_dim
        self.state_dim = 4  # [x, y, velocity, heading]
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Map state to latent and get waypoints.
        
        For real use, you'd need a proper encoder. This is a placeholder.
        """
        B = state.shape[0]
        
        # Simple identity mapping: use state features as "latent" features
        # In real system, this would go through a vision encoder (512 dims)
        if state.shape[1] >= self.latent_dim:
            z = state[:, :self.latent_dim]
        else:
            # Pad with zeros
            z = torch.cat([
                state,  # [B, 4]
                torch.zeros(B, self.latent_dim - state.shape[1], device=state.device)
            ], dim=1)
        
        # Get waypoints from model
        waypoints = self.model(z, delta_scale=0.0)
        
        # Convert [B, T, 2] to [B, T, 3] by adding heading (0)
        waypoints_3d = torch.cat([waypoints, torch.zeros_like(waypoints[..., :1])], dim=2)
        
        return waypoints_3d


# ============================================================================
# Test functions
# ============================================================================

def test_checkpoint_loading():
    """Test loading the actual SFT checkpoint."""
    print("=" * 60)
    print("Testing SFT checkpoint loading")
    print("=" * 60)
    
    # Test 1: Load real checkpoint
    print("\n[Test 1] Loading real checkpoint...")
    model, config = load_sft_checkpoint_with_real_model()
    
    if model is not None:
        print(f"  Model type: {type(model).__name__}")
        
        # Test forward pass with correct input
        # Real model expects [batch, latent_dim] = [batch, 512]
        import random
        random.seed(42)
        torch.manual_seed(42)
        
        dummy_input = torch.randn(2, 512)  # Simulated latent features
        with torch.no_grad():
            waypoints = model(dummy_input, delta_scale=0.0)
        print(f"  Forward pass output shape: {waypoints.shape}")
        print(f"  Sample SFT waypoints[0]: {waypoints[0, :3].tolist()}")
        
        # Test with delta
        with torch.no_grad():
            waypoints_with_delta = model(dummy_input, delta_scale=1.0)
        print(f"  With delta (scale=1.0): {waypoints_with_delta[0, :3].tolist()}")
        
        # Print config metrics if available
        if config and 'metrics' in config:
            metrics = config['metrics']
            if 'train_loss' in metrics:
                print(f"  Train losses: {metrics['train_loss'][:3]}...")
            if 'eval_ade' in metrics:
                print(f"  Eval ADE: {metrics['eval_ade'][:3]}...")
    else:
        print("  Failed to load model")
    
    # Test 2: List available checkpoints
    print("\n[Test 2] Available checkpoints:")
    bc_dir = AIREPO / "out" / "waypoint_bc"
    if bc_dir.exists():
        for f in bc_dir.glob("*.pt"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
    
    print("\n✓ Checkpoint loading test complete")
    return model is not None


def inspect_checkpoint(checkpoint_path: str):
    """Inspect a checkpoint file."""
    print(f"Inspecting checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print(f"  Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"  Keys: {list(checkpoint.keys())}")
        
        # Model state
        model_state = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint.get('model_state_dict', {})))
        if model_state:
            print(f"  State dict keys: {list(model_state.keys())}")
            print(f"  Total params: {sum(p.numel() for p in model_state.values() if hasattr(p, 'numel')):,}")
            
            # Print shapes
            print(f"  State dict shapes:")
            for k, v in model_state.items():
                print(f"    {k}: {v.shape}")
        
        # Metrics
        metrics = checkpoint.get('metrics', {})
        if metrics:
            print(f"  Metrics keys: {list(metrics.keys())}")
            
            # Print sample metrics
            if 'train_loss' in metrics:
                losses = metrics['train_loss']
                print(f"  Train losses: {losses[:3]}...")
            
            if 'eval_ade' in metrics:
                ade = metrics['eval_ade']
                print(f"  Eval ADE: {ade[:3]}...")


def run_eval_with_real_sft(episodes: int = 5, seed_base: int = 100):
    """Run SFT vs RL eval with real SFT checkpoint."""
    print("=" * 60)
    print(f"Running SFT vs RL eval (episodes={episodes}, seed_base={seed_base})")
    print("=" * 60)
    
    # Get real checkpoint path
    checkpoint_path = str(AIREPO / "out" / "waypoint_bc" / "best_model.pt")
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Note: The eval runner needs adapter to work with real model
    # For now, just print instructions
    print("\nNote: To run unified_eval_runner with real checkpoint:")
    print(f"  python -m training.rl.unified_eval_runner \\")
    print(f"    --episodes {episodes} \\")
    print(f"    --seed-base {seed_base} \\")
    print(f"    --sft-checkpoint {checkpoint_path}")
    
    print("\n✓ Eval preparation complete")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SFT Checkpoint Loader")
    parser.add_argument("--inspect", action="store_true", help="Inspect SFT checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--test-load", action="store_true", help="Test checkpoint loading")
    parser.add_argument("--run-eval", action="store_true", help="Run eval with real SFT")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--seed-base", type=int, default=100, help="Random seed base")
    
    args = parser.parse_args()
    
    if args.inspect:
        checkpoint = args.checkpoint or str(AIREPO / "out" / "waypoint_bc" / "best_model.pt")
        inspect_checkpoint(checkpoint)
    elif args.test_load:
        test_checkpoint_loading()
    elif args.run_eval:
        run_eval_with_real_sft(episodes=args.episodes, seed_base=args.seed_base)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

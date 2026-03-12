"""
BC-to-RL Bridge Module

Loads trained Waypoint BC model and provides waypoint predictions as initial
proposals for RL refinement training. Bridges the BC → RL pipeline.

Usage:
    # Load BC model and get waypoint predictions
    from training.bc.bc_to_rl_bridge import BCToRLBridge
    
    bridge = BCToRLBridge("out/waypoint_bc/run_XXX/best.pt")
    waypoints = bridge.predict_waypoints(bev_input)  # [B, num_waypoints, 2]
    
    # Use in RL environment
    env = ToyWaypointEnv()
    initial_waypoints = bridge.predict_waypoints(bridge.encode_state(env.reset()))
    env.set_proposal_waypoints(initial_waypoints)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import glob
import os

# Import BC model
from training.bc.waypoint_bc import WaypointBCModel, BCConfig


@dataclass
class BCToRLBridgeConfig:
    """Configuration for BC-to-RL bridge."""
    # Model
    bc_checkpoint_path: Optional[str] = None  # Path to BC checkpoint
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Waypoint processing
    num_waypoints: int = 8
    waypoint_spacing: float = 1.0  # meters between waypoints
    
    # RL integration
    proposal_waypoints: bool = True  # Use BC as initial proposal
    delta_learning: bool = True  # Learn residual from BC predictions


class BCToRLBridge:
    """
    Bridge module that loads trained BC model and provides predictions
    for RL refinement training.
    
    Features:
    - Loads trained BC checkpoint
    - Predicts waypoints from BEV/observation input
    - Provides proposals for RL environments
    - Supports residual learning (BC + delta)
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[BCToRLBridgeConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or BCToRLBridgeConfig()
        
        if device:
            self.config.device = device
            
        self.device = torch.device(self.config.device)
        self.model: Optional[WaypointBCModel] = None
        self.bc_checkpoint_path = checkpoint_path or self.config.bc_checkpoint_path
        
        if self.bc_checkpoint_path:
            self._load_model()
        else:
            print("Warning: No BC checkpoint provided. Bridge will not function.")
    
    def _load_model(self) -> None:
        """Load trained BC model from checkpoint."""
        if not self.bc_checkpoint_path:
            raise ValueError("No checkpoint path provided")
        
        # Handle "latest" keyword
        if self.bc_checkpoint_path == "latest":
            self.bc_checkpoint_path = self._find_latest_checkpoint()
        
        # Load checkpoint
        checkpoint = torch.load(self.bc_checkpoint_path, map_location=self.device)
        
        # Extract config
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                bc_config_dict = checkpoint['config']
                bc_config = BCConfig()
                for k, v in bc_config_dict.items():
                    if hasattr(bc_config, k):
                        setattr(bc_config, k, v)
            else:
                bc_config = BCConfig()
            
            # Load model state - check various possible keys
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            bc_config = BCConfig()
            state_dict = checkpoint
        
        # Create model
        self.model = WaypointBCModel(bc_config).to(self.device)
        
        # Load state
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        print(f"Loaded BC model from {self.bc_checkpoint_path}")
    
    def _find_latest_checkpoint(self) -> str:
        """Find the latest BC checkpoint in out/waypoint_bc/."""
        base_dir = Path("out/waypoint_bc")
        
        if not base_dir.exists():
            raise FileNotFoundError(f"BC checkpoint directory not found: {base_dir}")
        
        # Find all run directories
        run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        
        if not run_dirs:
            raise FileNotFoundError(f"No BC runs found in {base_dir}")
        
        # Sort by modification time (most recent first)
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_dir = run_dirs[0]
        
        # Find best.pt or latest.pt
        for name in ["best.pt", "latest.pt", "checkpoint.pt"]:
            ckpt = latest_dir / name
            if ckpt.exists():
                return str(ckpt)
        
        # Fall back to any .pt file
        pt_files = list(latest_dir.glob("*.pt"))
        if pt_files:
            pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(pt_files[0])
        
        raise FileNotFoundError(f"No checkpoint found in {latest_dir}")
    
    @torch.no_grad()
    def predict_waypoints(
        self,
        bev_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict waypoints from BEV input.
        
        Args:
            bev_input: [B, C, H, W] tensor
            
        Returns:
            waypoints: [B, num_waypoints, 2] tensor in ego frame (x, y in meters)
        """
        if self.model is None:
            raise RuntimeError("BC model not loaded")
        
        bev_input = bev_input.to(self.device)
        waypoints, speed = self.model(bev_input)
        
        return waypoints
    
    @torch.no_grad()
    def predict(
        self,
        bev_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict waypoints and speed from BEV input.
        
        Args:
            bev_input: [B, C, H, W] tensor
            
        Returns:
            waypoints: [B, num_waypoints, 2] tensor
            speed: [B, 1] tensor (m/s)
        """
        if self.model is None:
            raise RuntimeError("BC model not loaded")
        
        bev_input = bev_input.to(self.device)
        waypoints, speed = self.model(bev_input)
        
        return waypoints, speed
    
    def encode_state(
        self,
        state: torch.Tensor,
        bev_size: int = 128,
    ) -> torch.Tensor:
        """
        Encode environment state to BEV-like representation for BC model.
        
        This is a simplified encoder for toy environments. In practice,
        this would convert real perception output to BC input format.
        
        Args:
            state: [B, state_dim] environment state
            bev_size: BEV grid size (default 128 to match trained model)
            
        Returns:
            bev_input: [B, 3, bev_size, bev_size] tensor
        """
        # Simplified: create fake BEV from state
        # Real implementation would use perception outputs
        batch_size = state.shape[0]
        
        # Create simple BEV representation (use 128 to match trained model)
        bev = torch.zeros(batch_size, 3, bev_size, bev_size, device=self.device)
        
        # Encode position in BEV
        if state.shape[1] >= 2:
            # Normalize positions to [-1, 1]
            scale = bev_size / 2  # 128 for 256
            x = (state[:, 0] / 50.0).clamp(-1, 1)  # Assume world size 100
            y = (state[:, 1] / 50.0).clamp(-1, 1)
            
            # Convert to BEV coordinates (center at bev_size/2, bev_size/2)
            bx = ((x + 1) * scale).long().clamp(0, bev_size - 1)
            by = ((y + 1) * scale).long().clamp(0, bev_size - 1)
            
            # Set agent position (channel 0)
            for i in range(batch_size):
                bev[i, 0, by[i], bx[i]] = 1.0
        
        # Encode goal (channel 1)
        if state.shape[1] >= 4:
            gx = (state[:, 2] / 50.0).clamp(-1, 1)
            gy = (state[:, 3] / 50.0).clamp(-1, 1)
            
            bgx = ((gx + 1) * scale).long().clamp(0, bev_size - 1)
            bgy = ((gy + 1) * scale).long().clamp(0, bev_size - 1)
            
            for i in range(batch_size):
                bev[i, 1, bgy[i], bgx[i]] = 1.0
        
        # Velocity (channel 2) - encode at center
        center = bev_size // 2
        if state.shape[1] >= 6:
            vx = state[:, 4].clamp(-2, 2) / 2.0 + 0.5
            vy = state[:, 5].clamp(-2, 2) / 2.0 + 0.5
            bev[:, 2, center, center] = torch.stack([vx, vy], dim=1).mean(dim=1)
        
        return bev
    
    def get_proposal_waypoints(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get BC predictions as proposal waypoints for RL environment.
        
        Args:
            state: [B, state_dim] environment state
            
        Returns:
            waypoints: [B, num_waypoints, 2] proposal waypoints
        """
        bev = self.encode_state(state)
        return self.predict_waypoints(bev)
    
    def create_delta_head(
        self,
        input_dim: int,
        output_dim: int = 2,
    ) -> nn.Module:
        """
        Create a small delta head for residual learning.
        
        The delta head learns to correct BC predictions:
            final_waypoints = bc_waypoints + delta_head(features)
        
        Args:
            input_dim: Feature dimension from BC encoder
            output_dim: Output dimension (2 for x, y)
            
        Returns:
            Delta head module
        """
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )


def find_latest_bc_checkpoint(base_dir: str = "out/waypoint_bc") -> Optional[str]:
    """
    Find the latest BC checkpoint.
    
    Args:
        base_dir: Base directory containing BC runs
        
    Returns:
        Path to latest checkpoint, or None if not found
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return None
    
    # Find all run directories
    run_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not run_dirs:
        return None
    
    # Sort by modification time
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_dir = run_dirs[0]
    
    # Find checkpoint
    for name in ["best.pt", "latest.pt", "checkpoint.pt"]:
        ckpt = latest_dir / name
        if ckpt.exists():
            return str(ckpt)
    
    # Any .pt file
    pt_files = list(latest_dir.glob("*.pt"))
    if pt_files:
        pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(pt_files[0])
    
    return None


# === Smoke Test ===

if __name__ == "__main__":
    print("BC-to-RL Bridge Smoke Test")
    print("=" * 40)
    
    # Test with no checkpoint
    bridge = BCToRLBridge(checkpoint_path=None)
    print(f"✓ Created bridge without checkpoint")
    
    # Test state encoding
    test_state = torch.randn(2, 6)  # [x, y, goal_x, goal_y, vx, vy]
    bev = bridge.encode_state(test_state, bev_size=128)
    print(f"✓ State encoding: {test_state.shape} -> {bev.shape}")
    
    # Try loading latest checkpoint
    try:
        latest = find_latest_bc_checkpoint()
        if latest:
            bridge_loaded = BCToRLBridge(checkpoint_path=latest)
            print(f"✓ Loaded BC model from: {latest}")
            
            # Test prediction
            test_bev = torch.randn(2, 3, 128, 128)
            waypoints = bridge_loaded.predict_waypoints(test_bev)
            print(f"✓ Waypoint prediction: {waypoints.shape}")
            
            waypoints, speed = bridge_loaded.predict(test_bev)
            print(f"✓ Speed prediction: {speed.shape}")
        else:
            print("⚠ No BC checkpoint found (expected if not yet trained)")
    except Exception as e:
        print(f"⚠ Could not load BC checkpoint: {e}")
    
    # Test delta head creation
    delta_head = bridge.create_delta_head(input_dim=256)
    test_features = torch.randn(2, 256)
    delta = delta_head(test_features)
    print(f"✓ Delta head: {test_features.shape} -> {delta.shape}")
    
    print("\n" + "=" * 40)
    print("Smoke test complete!")

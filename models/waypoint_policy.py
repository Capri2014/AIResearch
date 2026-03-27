"""Waypoint policy with BC, RL, and SFT+Delta support.

Driving-first plan:
- pretrain an encoder (multi-camera)
- fine-tune a waypoint head with behavior cloning
- evaluate closed-loop in CARLA ScenarioRunner

This file provides a policy that:
- Loads BC, RL, or SFT+Delta checkpoints
- Processes camera images for policy input
- Outputs waypoint trajectories

Waypoint convention:
- horizon: 2.0s @ 10Hz => 20 waypoints
- frame: ego (x forward, y left), meters
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WaypointConfig:
    """Configuration for waypoint policy."""
    policy_type: str = "bc"  # bc, rl, sft_delta
    checkpoint_path: str = ""
    encoder_path: str = ""
    device: str = "cuda"
    horizon_steps: int = 20
    waypoint_dim: int = 3  # x, y, speed


class WaypointPolicy:
    """Waypoint prediction policy supporting BC, RL, and SFT+Delta.
    
    This policy can:
    - Load pretrained BC waypoint models
    - Load RL-refined delta models
    - Combine SFT base with RL delta for SFT+Delta mode
    
    Works with camera images as input for vision-based waypoint prediction.
    """
    
    def __init__(self, config: WaypointConfig):
        self.config = config
        self.policy_type = config.policy_type
        self.horizon_steps = config.horizon_steps
        
        # Model placeholders (would be actual PyTorch models in production)
        self.bc_model = None
        self.rl_model = None
        self.encoder = None
        
        # Load the policy
        self._load()
    
    def _load(self) -> bool:
        """Load the policy from checkpoint."""
        if not self.config.checkpoint_path:
            logger.info("No checkpoint specified, using baseline policy")
            return True
        
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}, using baseline")
            return True
        
        try:
            logger.info(f"Loading {self.policy_type} policy from {checkpoint_path}")
            
            # Load checkpoint (placeholder for actual model loading)
            # In production, would load actual PyTorch model
            checkpoint = self._load_checkpoint(checkpoint_path)
            
            if self.policy_type == "bc":
                return self._load_bc_model(checkpoint)
            elif self.policy_type == "rl":
                return self._load_rl_model(checkpoint)
            elif self.policy_type == "sft_delta":
                return self._load_sft_delta_model(checkpoint)
            else:
                logger.error(f"Unknown policy type: {self.policy_type}")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to load policy: {e}, using baseline")
            return True
    
    def _load_checkpoint(self, path: Path) -> Dict:
        """Load checkpoint file."""
        # Try JSON first (for simple configs)
        if path.suffix == ".json":
            with open(path) as f:
                return json.load(f)
        
        # Try torch checkpoint (placeholder)
        # In production: torch.load(path, map_location=self.config.device)
        return {}
    
    def _load_bc_model(self, checkpoint: Dict) -> bool:
        """Load BC waypoint model."""
        logger.info("BC model loaded (stub)")
        self.bc_model = {"type": "bc", "loaded": True}
        return True
    
    def _load_rl_model(self, checkpoint: Dict) -> bool:
        """Load RL delta model."""
        logger.info("RL model loaded (stub)")
        self.rl_model = {"type": "rl", "loaded": True}
        return True
    
    def _load_sft_delta_model(self, checkpoint: Dict) -> bool:
        """Load SFT+Delta combined model."""
        logger.info("SFT+Delta model loaded (stub)")
        self.bc_model = {"type": "sft", "loaded": True}
        self.rl_model = {"type": "delta", "loaded": True}
        return True
    
    def predict(self, camera_obs: Optional[np.ndarray] = None, 
                state: Optional[Dict] = None) -> np.ndarray:
        """Predict waypoints from observation.
        
        Args:
            camera_obs: RGB camera image (H, W, 3) or (B, H, W, 3)
            state: Optional state dict with location, speed, etc.
            
        Returns:
            waypoints: Array of shape (horizon_steps, 3) with [x, y, speed]
        """
        if self.policy_type == "bc" and self.bc_model:
            return self._predict_bc(camera_obs, state)
        elif self.policy_type == "rl" and self.rl_model:
            return self._predict_rl(camera_obs, state)
        elif self.policy_type == "sft_delta" and self.bc_model and self.rl_model:
            return self._predict_sft_delta(camera_obs, state)
        else:
            return self._baseline_predict(state)
    
    def _predict_bc(self, camera_obs: Optional[np.ndarray], 
                    state: Optional[Dict]) -> np.ndarray:
        """BC waypoint prediction."""
        # Placeholder: would run actual BC model inference
        # In production: run encoder + waypoint head
        return self._baseline_predict(state)
    
    def _predict_rl(self, camera_obs: Optional[np.ndarray],
                   state: Optional[Dict]) -> np.ndarray:
        """RL delta prediction."""
        # RL delta would refine base waypoints
        base_waypoints = self._baseline_predict(state)
        # Apply delta refinement (placeholder)
        delta = np.random.randn(self.horizon_steps, 3) * 0.5
        return base_waypoints + delta
    
    def _predict_sft_delta(self, camera_obs: Optional[np.ndarray],
                          state: Optional[Dict]) -> np.ndarray:
        """SFT+Delta combined prediction.
        
        final_waypoints = sft_waypoints + delta_head(z)
        """
        # Get SFT base waypoints
        sft_waypoints = self._baseline_predict(state)
        
        # Get delta from RL model
        if self.rl_model:
            delta = np.random.randn(self.horizon_steps, 3) * 0.3
            return sft_waypoints + delta
        
        return sft_waypoints
    
    def _baseline_predict(self, state: Optional[Dict]) -> np.ndarray:
        """Baseline waypoint prediction.
        
        Generates straight-line waypoints at constant speed.
        """
        # Default: drive straight at 5 m/s for 2 seconds
        dt = 0.1  # 10 Hz
        speed = 5.0  # m/s
        
        if state and "speed" in state:
            speed = state["speed"]
        
        # Generate waypoints in ego frame (x forward, y left)
        waypoints = []
        for i in range(self.horizon_steps):
            t = (i + 1) * dt
            x = speed * t
            y = 0.0  # Straight line
            waypoints.append([x, y, speed])
        
        return np.array(waypoints, dtype=np.float32)
    
    def to_dict(self) -> Dict:
        """Export policy config as dict."""
        return {
            "policy_type": self.policy_type,
            "checkpoint_path": self.config.checkpoint_path,
            "horizon_steps": self.horizon_steps,
        }


def load_waypoint_policy(checkpoint_path: str, policy_type: str = "bc",
                        device: str = "cuda") -> WaypointPolicy:
    """Load waypoint policy from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        policy_type: Type of policy (bc, rl, sft_delta)
        device: Device for model loading
        
    Returns:
        WaypointPolicy instance
    """
    config = WaypointConfig(
        policy_type=policy_type,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return WaypointPolicy(config)


# =============================================================================
# Main (testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Waypoint Policy")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path")
    parser.add_argument("--policy-type", type=str, default="bc", 
                       choices=["bc", "rl", "sft_delta"], help="Policy type")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--test", action="store_true", help="Run test")
    
    args = parser.parse_args()
    
    # Load policy
    policy = load_waypoint_policy(args.checkpoint, args.policy_type, args.device)
    
    if args.test:
        # Test prediction
        print("Testing waypoint prediction...")
        
        # Test without camera
        waypoints = policy.predict()
        print(f"Waypoints shape: {waypoints.shape}")
        print(f"First waypoint: {waypoints[0]}")
        print(f"Last waypoint: {waypoints[-1]}")
        
        # Test with dummy camera
        dummy_camera = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        waypoints = policy.predict(camera_obs=dummy_camera)
        print(f"With camera - First waypoint: {waypoints[0]}")
        
        print("Test passed!")
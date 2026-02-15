"""Policy wrapper for CARLA ScenarioRunner integration.

This module provides:
- WaypointPolicyWrapper: Loads trained checkpoints and serves waypoints to CARLA
- Integration with ScenarioRunner's agent interface

Usage
-----
# Load a trained waypoint policy
python -m sim.driving.carla_srunner.policy_wrapper --help
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np


@dataclass
class PolicyConfig:
    """Configuration for policy wrapper."""
    checkpoint: Path
    camera_name: str = "front"
    horizon_steps: int = 20
    device: str = "auto"


class WaypointPolicyWrapper:
    """Wrapper around trained waypoint policy for CARLA integration.
    
    This class provides a simple interface for ScenarioRunner agents to:
    - Load trained checkpoints (SFT or SFT+RL)
    - Predict waypoints from camera images
    - Convert waypoints to CARLA vehicle commands
    
    Usage with ScenarioRunner:
        policy = WaypointPolicyWrapper(checkpoint="path/to/model.pt")
        waypoints = policy.predict(image)
        control = policy.waypoints_to_control(waypoints)
        return control
    """
    
    def __init__(self, cfg: PolicyConfig | None = None):
        cfg = cfg or PolicyConfig()
        self.cfg = cfg
        self._model = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the policy from checkpoint."""
        if self._initialized:
            return True
        
        try:
            from training.rl.waypoint_policy_torch import WaypointPolicyTorch, WaypointPolicyConfig
            
            wp_cfg = WaypointPolicyConfig(
                checkpoint=self.cfg.checkpoint,
                cam=self.cfg.camera_name,
                horizon_steps=self.cfg.horizon_steps,
                device=self.cfg.device,
            )
            self._model = WaypointPolicyTorch(wp_cfg)
            self._initialized = True
            print(f"[policy_wrapper] Loaded checkpoint: {self.cfg.checkpoint}")
            return True
        except Exception as e:
            print(f"[policy_wrapper] Failed to load checkpoint: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def predict(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict waypoints from images.
        
        Args:
            images: Dict of camera_name -> (H, W, C) numpy array
        
        Returns:
            waypoints: (horizon_steps, 2) numpy array in ego frame (x, y) meters
        """
        if not self._initialized:
            raise RuntimeError("Policy not initialized. Call initialize() first.")
        
        return self._model.predict_batch(images)
    
    def predict_with_delta(
        self,
        images: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict base waypoints and delta corrections (if using RL-refined policy).
        
        Returns:
            base_waypoints: (H, 2) SFT predictions
            delta: (H, 2) corrections
        """
        if not self._initialized:
            raise RuntimeError("Policy not initialized. Call initialize() first.")
        
        raise NotImplementedError("Requires WaypointPolicyWithDelta")
    
    def waypoints_to_control(
        self,
        waypoints: np.ndarray,
        current_speed: float = 0.0,
        dt: float = 0.05,
    ) -> Dict[str, float]:
        """
        Convert waypoints to CARLA vehicle control commands.
        
        Args:
            waypoints: (H, 2) array in ego frame (x forward, y left)
            current_speed: Current vehicle speed in m/s
            dt: Time step for command duration
        
        Returns:
            control: Dict with throttle, steer, brake for CARLA VehicleControl
        """
        if len(waypoints) == 0:
            return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
        
        # Target: first waypoint within lookahead
        target = waypoints[0]
        target_distance = np.linalg.norm(target)
        
        # Steering: angle to target point
        steer = np.arctan2(target[1], target[0])
        
        # Clamp steering to reasonable range
        steer = float(np.clip(steer, -0.7, 0.7))  # ~40 degrees
        
        # Throttle/brake based on distance and angle
        if target_distance < 2.0:
            # Close to target: slow down
            throttle = 0.0
            brake = 0.3
        elif abs(steer) > 0.5:
            # Sharp turn: slow down
            throttle = 0.2
            brake = 0.1
        else:
            # Normal driving
            throttle = 0.5
            brake = 0.0
        
        return {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
        }
    
    def get_action(
        self,
        observation: Dict,
    ) -> Dict[str, float]:
        """
        Get action from a ScenarioRunner-style observation.
        
        Args:
            observation: Dict containing:
                - images: Dict[str, np.ndarray] of camera images
                - speed: Optional[float] current speed
                - state: Optional[Dict] additional state
        
        Returns:
            control: Dict with throttle, steer, brake
        """
        images = observation.get("images", {})
        speed = observation.get("speed", 0.0)
        
        if not images:
            # Fallback: no images, return stop command
            return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
        
        waypoints = self.predict(images)
        control = self.waypoints_to_control(waypoints, current_speed=speed)
        
        return control


class StubPolicyWrapper:
    """Stub policy that returns simple controls for testing."""
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.cfg = config
    
    def initialize(self) -> bool:
        return True
    
    @property
    def is_initialized(self) -> bool:
        return True
    
    def predict(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """Return dummy waypoints: straight line."""
        horizon = self.cfg.horizon_steps if self.cfg else 20
        return np.linspace([0, 0], [horizon * 0.5, 0], horizon)
    
    def waypoints_to_control(
        self,
        waypoints: np.ndarray,
        current_speed: float = 0.0,
        dt: float = 0.05,
    ) -> Dict[str, float]:
        """Simple control: go straight."""
        return {"throttle": 0.3, "steer": 0.0, "brake": 0.0}
    
    def get_action(self, observation: Dict) -> Dict[str, float]:
        """Get action from observation."""
        return self.waypoints_to_control(self.predict(observation.get("images", {})))


def load_policy(checkpoint: Path | None = None) -> WaypointPolicyWrapper | StubPolicyWrapper:
    """
    Load a policy from checkpoint, or return stub if no checkpoint.
    
    Args:
        checkpoint: Path to trained model checkpoint, or None for stub
    
    Returns:
        Policy wrapper instance
    """
    if checkpoint is None:
        print("[policy_wrapper] No checkpoint provided, using stub policy")
        return StubPolicyWrapper(PolicyConfig())
    
    cfg = PolicyConfig(checkpoint=checkpoint)
    policy = WaypointPolicyWrapper(cfg)
    
    if policy.initialize():
        return policy
    
    print(f"[policy_wrapper] Failed to load {checkpoint}, using stub")
    return StubPolicyWrapper(cfg)


def main():
    """CLI for testing policy loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Policy Wrapper for CARLA")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stub", action="store_true", help="Use stub policy")
    parser.add_argument("--test-control", action="store_true", help="Test control generation")
    args = parser.parse_args()
    
    if args.stub:
        policy = StubPolicyWrapper()
    else:
        policy = load_policy(args.checkpoint)
    
    print(f"Policy initialized: {policy.is_initialized}")
    
    if args.test_control:
        waypoints = policy.predict({}) if hasattr(policy, 'predict') else np.zeros((20, 2))
        control = policy.waypoints_to_control(waypoints)
        print(f"Waypoints shape: {waypoints.shape}")
        print(f"Control: {control}")


if __name__ == "__main__":
    main()

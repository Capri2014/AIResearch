"""Policy wrapper for CARLA ScenarioRunner integration.

This module provides:
- WaypointPolicyWrapper: Loads trained checkpoints and serves waypoints to CARLA
- Integration with ScenarioRunner's agent interface
- BEV Encoder: Unified camera + LiDAR to BEV feature conversion

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

# BEV Encoder availability flag
BEV_ENCODER_AVAILABLE = True
SSL_PRETRAINED_AVAILABLE = True

try:
    from sim.driving.carla_srunner.bev_encoder import (
        BEVEncoder,
        BEVEncoderConfig,
        create_bev_encoder,
        FusionType,
    )
except ImportError as e:
    BEV_ENCODER_AVAILABLE = False
    print(f"[policy_wrapper] BEV encoder not available: {e}")

try:
    from training.sft.ssl_pretrained_loader import (
        load_ssl_pretrained,
        SSLFeatureExtractor,
        BCWithSSLEncoder,
        create_bc_with_ssl_pretrained,
    )
except ImportError as e:
    SSL_PRETRAINED_AVAILABLE = False
    print(f"[policy_wrapper] SSL pretrained loader not available: {e}")


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
        target_speeds: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Convert waypoints to CARLA vehicle control commands.
        
        Args:
            waypoints: (H, 2) array in ego frame (x forward, y left)
            current_speed: Current vehicle speed in m/s
            dt: Time step for command duration
            target_speeds: (H,) optional target speeds in m/s for each waypoint
        
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
        
        # Get target speed if available (from speed prediction)
        if target_speeds is not None and len(target_speeds) > 0:
            target_speed = float(target_speeds[0])
        else:
            # Default speed profile: slower for turns, faster for straight
            # Compute curvature from waypoints
            curvature = 0.0
            if len(waypoints) >= 3:
                # Simple curvature proxy: angle change between segments
                v1 = waypoints[1] - waypoints[0]
                v2 = waypoints[2] - waypoints[1]
                if np.linalg.norm(v1) > 0.1 and np.linalg.norm(v2) > 0.1:
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))
                    curvature = angle
            
            # Speed: slower for curves, faster for straight
            if curvature > 0.3:
                target_speed = 3.0  # ~10 km/h for sharp turns
            elif curvature > 0.1:
                target_speed = 6.0  # ~20 km/h for gentle curves
            else:
                target_speed = 10.0  # ~35 km/h for straight
        
        # Speed control: throttle/brake based on current vs target speed
        speed_error = target_speed - current_speed
        
        if target_distance < 2.0:
            # Close to target: slow down
            throttle = 0.0
            brake = 0.5
        elif speed_error < -1.0:
            # Going too fast: brake
            throttle = 0.0
            brake = min(0.3, -speed_error / 10.0)
        elif speed_error > 2.0:
            # Need to speed up
            if abs(steer) > 0.5:
                # Sharp turn: moderate throttle
                throttle = 0.3
                brake = 0.0
            else:
                throttle = 0.6
                brake = 0.0
        elif abs(steer) > 0.5:
            # Sharp turn: slow down
            throttle = 0.2
            brake = 0.1
        else:
            # Maintain speed
            throttle = 0.4 if speed_error > 0 else 0.0
            brake = 0.1 if speed_error < 0 else 0.0
        
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
    
    def predict_with_speed(
        self,
        images: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict waypoints and speeds from images.
        
        Requires speed prediction-enabled model. Falls back to
        default speed profile if not available.
        
        Args:
            images: Dict of camera_name -> (H, W, C) numpy array
        
        Returns:
            waypoints: (H, 2) numpy array in ego frame
            speeds: (H,) numpy array of target speeds in m/s
        """
        if not self._initialized:
            raise RuntimeError("Policy not initialized. Call initialize() first.")
        
        # Default: predict waypoints, use default speed profile
        waypoints = self.predict(images)
        
        # Compute speed profile based on waypoint geometry
        speeds = self._compute_speed_profile(waypoints)
        
        return waypoints, speeds
    
    def _compute_speed_profile(self, waypoints: np.ndarray) -> np.ndarray:
        """Compute default speed profile from waypoints.
        
        Uses curvature to determine appropriate speeds:
        - Straight segments: higher speeds
        - Curved segments: lower speeds
        """
        H = len(waypoints)
        speeds = np.ones(H) * 10.0  # Default 10 m/s
        
        if H < 2:
            return speeds
        
        # Compute direction vectors
        diffs = np.diff(waypoints, axis=0)  # (H-1, 2)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)  # (H-1, 1)
        norms = np.maximum(norms, 0.1)  # Avoid division by zero
        
        directions = diffs / norms  # (H-1, 2)
        
        # Compute curvature (direction changes)
        dir_diffs = np.diff(directions, axis=0)  # (H-2, 2)
        curvatures = np.linalg.norm(dir_diffs, axis=1)  # (H-2,)
        
        # Set speeds based on curvature
        for i in range(len(curvatures)):
            if curvatures[i] > 0.5:
                speeds[i] = 3.0  # Sharp turn
            elif curvatures[i] > 0.2:
                speeds[i] = 6.0  # Gentle curve
            else:
                speeds[i] = 10.0  # Straight
        
        # First/last waypoints get similar speeds to neighbors
        if len(speeds) > 0:
            speeds[0] = speeds[1] if len(speeds) > 1 else 5.0
            speeds[-1] = speeds[-2] if len(speeds) > 1 else 5.0
        
        return speeds


class SSLWaypointPolicyWrapper:
    """Waypoint policy using SSL pretrained encoder.
    
    This class provides a waypoint policy that uses an SSL pretrained encoder
    for feature extraction, then predicts waypoints for CARLA control.
    
    Usage:
        from sim.driving.carla_srunner.policy_wrapper import SSLWaypointPolicyWrapper, SSLPolicyConfig
        
        cfg = SSLPolicyConfig(
            ssl_checkpoint="path/to/ssl/model.pt",
            num_waypoints=8,
            horizon_steps=20
        )
        policy = SSLWaypointPolicyWrapper(cfg)
        waypoints = policy.predict(images)
        control = policy.waypoints_to_control(waypoints)
    """
    
    @dataclass
    class SSLPolicyConfig:
        """Configuration for SSL waypoint policy."""
        ssl_checkpoint: Optional[Path] = None  # Path to SSL pretrained checkpoint
        model_type: str = "jepa"  # jepa, contrastive, temporal_contrastive
        num_waypoints: int = 8
        horizon_steps: int = 20
        hidden_dim: int = 256
        feature_dim: int = 256
        device: str = "auto"
    
    def __init__(self, cfg: SSLPolicyConfig | None = None):
        cfg = cfg or self.SSLPolicyConfig()
        self.cfg = cfg
        self._model = None
        self._initialized = False
        
        # Auto-detect device
        if cfg.device == "auto":
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = cfg.device
    
    def initialize(self) -> bool:
        """Initialize the SSL waypoint policy."""
        if self._initialized:
            return True
        
        if not SSL_PRETRAINED_AVAILABLE:
            print("[SSLWaypointPolicy] SSL pretrained loader not available")
            return False
        
        try:
            import torch
            from training.sft.ssl_pretrained_loader import (
                create_bc_with_ssl_pretrained,
            )
            
            # Create BC model with SSL pretrained encoder
            self._model = create_bc_with_ssl_pretrained(
                checkpoint_path=str(self.cfg.ssl_checkpoint) if self.cfg.ssl_checkpoint else None,
                model_type=self.cfg.model_type,
                num_waypoints=self.cfg.num_waypoints,
                device=self._device
            )
            self._model.eval()
            
            self._initialized = True
            print(f"[SSLWaypointPolicy] Loaded SSL checkpoint: {self.cfg.ssl_checkpoint}")
            return True
        except Exception as e:
            print(f"[SSLWaypointPolicy] Failed to initialize: {e}")
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
        
        import torch
        
        # Get front camera image
        front_image = images.get("front")
        if front_image is None:
            # Use first available camera
            front_image = list(images.values())[0]
        
        # Convert to tensor (B=1, C, H, W)
        if front_image.dtype != np.float32:
            front_image = front_image.astype(np.float32) / 255.0
        
        # Handle (H, W, C) -> (C, H, W)
        if front_image.ndim == 3 and front_image.shape[-1] == 3:
            front_image = np.transpose(front_image, (2, 0, 1))
        
        # Add batch dimension and normalize to [-1, 1]
        image_tensor = torch.from_numpy(front_image).unsqueeze(0).to(self._device)
        image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        
        # Predict waypoints
        with torch.no_grad():
            waypoints = self._model(image_tensor)
        
        # Convert to numpy (num_waypoints, 2)
        waypoints_np = waypoints.cpu().numpy()[0]
        
        # Ensure we have the right number of waypoints
        if len(waypoints_np) < self.cfg.horizon_steps:
            # Repeat last waypoint to fill
            last = waypoints_np[-1:]
            waypoints_np = np.vstack([waypoints_np] + [last] * (self.cfg.horizon_steps - len(waypoints_np)))
        elif len(waypoints_np) > self.cfg.horizon_steps:
            waypoints_np = waypoints_np[:self.cfg.horizon_steps]
        
        return waypoints_np
    
    def waypoints_to_control(
        self,
        waypoints: np.ndarray,
        current_speed: float = 0.0,
        dt: float = 0.05,
        target_speeds: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Convert waypoints to CARLA vehicle control commands."""
        # Reuse the method from WaypointPolicyWrapper
        wrapper = WaypointPolicyWrapper(PolicyConfig(horizon_steps=len(waypoints)))
        return wrapper.waypoints_to_control(waypoints, current_speed, dt, target_speeds)


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

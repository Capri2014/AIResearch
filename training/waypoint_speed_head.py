"""Speed prediction head for waypoint BC model.

This module adds speed prediction capability to the waypoint policy,
predicting appropriate speeds for each waypoint in the trajectory.

Usage:
    # Add to existing waypoint policy
    from training.waypoint_speed_head import SpeedHead, SpeedHeadConfig
    
    speed_head = SpeedHead(config).to(device)
    speeds = speed_head(z)  # (B, horizon_steps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SpeedHeadConfig:
    """Configuration for speed prediction head."""
    in_dim: int = 128  # Encoder output dimension
    horizon_steps: int = 20  # Number of waypoint timesteps
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    min_speed: float = 0.0  # m/s
    max_speed: float = 15.0  # m/s (~54 km/h)
    use_tanh: bool = True  # Use tanh for bounded output


class SpeedHead(nn.Module):
    """MLP head for predicting speed at each waypoint timestep.
    
    Architecture:
        - MLP with hidden layers
        - Optional layer normalization
        - Output activation for bounded speeds
    """
    
    def __init__(self, config: SpeedHeadConfig):
        super().__init__()
        self.config = config
        
        # Build MLP
        layers = []
        in_dim = config.in_dim
        for i in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(config.hidden_dim, config.horizon_steps)
        
        # Output activation
        if config.use_tanh:
            # Tanh outputs [-1, 1], scale to [min, max]
            self.activation = nn.Tanh()
        else:
            # ReLU for non-negative speeds
            self.activation = nn.ReLU()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict speeds from latent features.
        
        Args:
            z: (B, in_dim) latent features from encoder
        
        Returns:
            speeds: (B, horizon_steps) speed in m/s for each timestep
        """
        h = self.mlp(z)  # (B, hidden_dim)
        out = self.output(h)  # (B, horizon_steps)
        
        # Apply activation and scale to [min_speed, max_speed]
        out = self.activation(out)  # (B, horizon_steps)
        
        # Scale from [-1, 1] or [0, 1] to [min_speed, max_speed]
        if self.config.use_tanh:
            # Tanh: [-1, 1] -> [min, max]
            speeds = (out + 1) / 2 * (self.config.max_speed - self.config.min_speed) + self.config.min_speed
        else:
            # ReLU: [0, inf] -> [min, max] (clamped)
            speeds = torch.clamp(out, 0, 1) * self.config.max_speed
        
        return speeds
    
    def forward_with_waypoints(
        self,
        z: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict speeds conditioned on waypoint positions.
        
        Uses both latent features and waypoint geometry to predict speeds.
        Faster movement expected for straighter segments.
        
        Args:
            z: (B, in_dim) latent features from encoder
            waypoints: (B, horizon_steps, 2) waypoint positions
        
        Returns:
            speeds: (B, horizon_steps) speed in m/s
        """
        # Base speed from latent
        base_speed = self.forward(z)  # (B, H)
        B, H = base_speed.shape
        
        # Compute waypoint curvature / direction changes
        if waypoints.shape[-1] == 2:
            # Direction vectors between consecutive waypoints
            diffs = waypoints[:, 1:, :] - waypoints[:, :-1, :]  # (B, H-1, 2)
            diff_norms = torch.norm(diffs, dim=-1, keepdim=True)  # (B, H-1, 1)
            
            # Direction changes (curvature proxy)
            directions = diffs / (diff_norms + 1e-6)  # (B, H-1, 2)
            dir_diffs = directions[:, 1:, :] - directions[:, :-1, :]  # (B, H-2, 2)
            curvature = torch.norm(dir_diffs, dim=-1)  # (B, H-2)
            
            # Pad to match horizon (pad both ends to get H values)
            curvature = torch.nn.functional.pad(curvature, (1, 1), value=0)  # (B, H)
            
            # Reduce speed where curvature is high
            speed_factor = 1.0 / (1.0 + curvature * 2.0)  # (B, H)
            speeds = base_speed * speed_factor
        else:
            speeds = base_speed
        
        return torch.clamp(speeds, self.config.min_speed, self.config.max_speed)


class SpeedBCDataset(torch.utils.data.Dataset):
    """Dataset for speed prediction training.
    
    Expects episodes with speed labels (from Waymo open-scale or synthetic).
    """
    
    def __init__(
        self,
        episodes_glob: str,
        cam: str = "front",
        horizon_steps: int = 20,
        decode_images: bool = True,
    ):
        from pathlib import Path
        import glob
        import numpy as np
        
        self.episodes_glob = episodes_glob
        self.cam = cam
        self.horizon_steps = horizon_steps
        self.decode_images = decode_images
        
        # Load episode paths
        self.episode_paths = sorted(Path(p).resolve() for p in glob.glob(episodes_glob))
        if not self.episode_paths:
            raise ValueError(f"No episodes found: {episodes_glob}")
        
        self.episodes = []
        self._load_episodes()
    
    def _load_episodes(self):
        """Lazy load episodes."""
        import numpy as np
        
        for ep_path in self.episode_paths:
            try:
                data = np.load(ep_path, allow_pickle=True)
                self.episodes.append((ep_path, data))
            except Exception as e:
                print(f"Warning: failed to load {ep_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.episodes) * 100  # Sample multiple frames per episode
    
    def __getitem__(self, idx: int):
        ep_idx = idx // 100
        frame_idx = idx % 100
        
        ep_path, data = self.episodes[ep_idx]
        
        # Extract features
        images = data.get("images", {})
        speeds = data.get("speeds", np.zeros(100))  # Speed labels
        
        if self.cam not in images:
            # Fallback to first available camera
            self.cam = list(images.keys())[0]
        
        img = images[self.cam][frame_idx]
        speed = speeds[frame_idx]
        
        return {
            "image": img,
            "speed": speed,
            "episode": str(ep_path),
            "frame": frame_idx,
        }


def create_speed_head(
    in_dim: int = 128,
    horizon_steps: int = 20,
    **kwargs,
) -> SpeedHead:
    """Factory function to create speed head."""
    config = SpeedHeadConfig(
        in_dim=in_dim,
        horizon_steps=horizon_steps,
        **kwargs,
    )
    return SpeedHead(config)


def speed_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 loss for speed prediction.
    
    Args:
        predictions: (B, H) predicted speeds
        targets: (B, H) target speeds
        valid_mask: (B, H) optional mask for valid timesteps
    
    Returns:
        loss: scalar loss
    """
    loss = torch.abs(predictions - targets)  # (B, H)
    
    if valid_mask is not None:
        loss = loss * valid_mask
        loss = loss.sum() / (valid_mask.sum() + 1e-6)
    else:
        loss = loss.mean()
    
    return loss


def speed_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE loss for speed prediction.
    
    Args:
        predictions: (B, H) predicted speeds
        targets: (B, H) target speeds
        valid_mask: (B, H) optional mask for valid timesteps
    
    Returns:
        loss: scalar loss
    """
    loss = (predictions - targets) ** 2  # (B, H)
    
    if valid_mask is not None:
        loss = loss * valid_mask
        loss = loss.sum() / (valid_mask.sum() + 1e-6)
    else:
        loss = loss.mean()
    
    return loss


# =============================================================================
# Integration with WaypointPolicyTorch
# =============================================================================

class WaypointSpeedPolicyConfig:
    """Combined config for waypoint + speed prediction."""
    
    def __init__(
        self,
        checkpoint: "Path" = None,
        cam: str = "front",
        horizon_steps: int = 20,
        device: str = "auto",
        speed_min: float = 0.0,
        speed_max: float = 15.0,
    ):
        self.checkpoint = checkpoint
        self.cam = cam
        self.horizon_steps = horizon_steps
        self.device = device
        self.speed_min = speed_min
        self.speed_max = speed_max


class WaypointSpeedPolicy:
    """Combined waypoint + speed prediction policy.
    
    This combines the existing waypoint BC model with a speed head
    for joint prediction of trajectory and speed profile.
    """
    
    def __init__(
        self,
        cfg: WaypointSpeedPolicyConfig,
        waypoint_policy: "WaypointPolicyTorch" = None,
    ):
        self.cfg = cfg
        self.waypoint_policy = waypoint_policy
        
        # Create speed head
        self.speed_config = SpeedHeadConfig(
            in_dim=128,  # Match encoder output
            horizon_steps=cfg.horizon_steps,
            min_speed=cfg.speed_min,
            max_speed=cfg.speed_max,
        )
        self.speed_head = SpeedHead(self.speed_config)
        
        # Device
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        
        self.speed_head.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        images: dict,
        image_valid: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict waypoints and speeds.
        
        Args:
            images: Dict of camera_name -> (B, C, H, W) tensor
            image_valid: Optional dict of camera_name -> (B,) bool tensor
        
        Returns:
            waypoints: (B, horizon_steps, 2) in world coords
            speeds: (B, horizon_steps) in m/s
        """
        if image_valid is None:
            image_valid = {
                k: torch.ones(v.shape[0], dtype=torch.bool, device=self.device)
                for k, v in images.items()
            }
        
        # Get latent features
        z = self.waypoint_policy.encoder(images, image_valid_by_cam=image_valid)
        
        # Predict waypoints
        waypoints = self.waypoint_policy.head(z)
        
        # Predict speeds (conditioned on waypoints)
        speeds = self.speed_head.forward_with_waypoints(z, waypoints)
        
        return waypoints, speeds
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: "np.ndarray",
        image_valid: Optional["np.ndarray"] = None,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Predict from numpy images (B, H, W, C)."""
        # Convert to tensor
        x = torch.from_numpy(images).float().to(self.device) / 255.0
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=self.device)
        if image_valid is not None:
            valid = torch.from_numpy(image_valid).bool().to(self.device)
        
        waypoints, speeds = self.predict(
            {self.cfg.cam: x},
            image_valid={self.cfg.cam: valid},
        )
        
        return waypoints.cpu().numpy(), speeds.cpu().numpy()


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for speed prediction."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Speed Prediction Head")
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    if args.test:
        # Test forward pass
        config = SpeedHeadConfig(
            in_dim=args.in_dim,
            horizon_steps=args.horizon,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
        head = SpeedHead(config)
        
        # Dummy input
        z = torch.randn(4, args.in_dim)
        waypoints = torch.randn(4, args.horizon, 2)
        
        # Test both forward modes
        speeds = head(z)
        print(f"Base speed output shape: {speeds.shape}")
        
        speeds_cond = head.forward_with_waypoints(z, waypoints)
        print(f"Conditioned speed output shape: {speeds_cond.shape}")
        
        # Test loss functions
        targets = torch.rand(4, args.horizon) * 10  # 0-10 m/s
        l1 = speed_l1_loss(speeds, targets)
        mse = speed_mse_loss(speeds, targets)
        print(f"L1 loss: {l1.item():.4f}")
        print(f"MSE loss: {mse.item():.4f}")
        
        print("\n✓ Speed head test passed!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

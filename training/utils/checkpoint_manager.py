"""
Unified Checkpoint Manager

Provides unified interface for loading any checkpoint type:
- BC waypoint models (with/without SSL encoder)
- RL delta-waypoint models (PPO, GRPO)
- SSL encoder checkpoints

Usage:
    from training.utils.checkpoint_manager import CheckpointManager, CheckpointType
    
    # Auto-detect and load
    manager = CheckpointManager(device="cuda")
    model, config = manager.load_checkpoint("out/bc/final.pt")
    
    # Checkpoint info without loading
    info = manager.inspect("out/ppo_delta/final.pt")
    print(f"Type: {info['type']}, Epoch: {info.get('epoch')}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json

import torch
import torch.nn as nn


class CheckpointType(Enum):
    """Types of checkpoints supported by the pipeline."""
    BC_WAYPOINT = "bc_waypoint"           # Waypoint BC model
    BC_WAYPOINT_SSL = "bc_waypoint_ssl"   # Waypoint BC with SSL encoder
    RL_PPO_DELTA = "rl_ppo_delta"         # PPO delta-waypoint model
    RL_GRPO_DELTA = "rl_grpo_delta"       # GRPO delta-waypoint model
    SSL_ENCODER = "ssl_encoder"           # SSL pretrained encoder
    UNKNOWN = "unknown"


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: Path
    type: CheckpointType
    size_mb: float
    config: Dict[str, Any] = field(default_factory=dict)
    epoch: Optional[int] = None
    step: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    has_optimizer: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "type": self.type.value,
            "size_mb": self.size_mb,
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "has_optimizer": self.has_optimizer,
        }


# Known key patterns for checkpoint type detection
CHECKPOINT_PATTERNS = {
    CheckpointType.BC_WAYPOINT: [
        "encoder", "decoder", "waypoint_head", "mlp_head",
    ],
    CheckpointType.BC_WAYPOINT_SSL: [
        "ssl_encoder", "encoder.backbone", "waypoint_mlp",
    ],
    CheckpointType.RL_PPO_DELTA: [
        "actor", "critic", "policy", "ppo", "delta_head",
    ],
    CheckpointType.RL_GRPO_DELTA: [
        "grpo", "group_reward", "delta_head",
    ],
    CheckpointType.SSL_ENCODER: [
        "backbone", "projection_head", "encoder",
    ],
}


class CheckpointManager:
    """
    Unified checkpoint manager for loading any pipeline checkpoint.
    
    Supports automatic type detection and appropriate loading.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    def detect_type(self, checkpoint: Dict[str, Any]) -> CheckpointType:
        """
        Automatically detect checkpoint type from its contents.
        
        Args:
            checkpoint: Loaded checkpoint dict
            
        Returns:
            Detected CheckpointType
        """
        # Check for explicit type in config
        if "config" in checkpoint:
            config = checkpoint["config"]
            if isinstance(config, dict):
                if config.get("model_type"):
                    model_type = config["model_type"]
                    if "SSL" in model_type:
                        return CheckpointType.BC_WAYPOINT_SSL
                    elif "Waypoint" in model_type:
                        return CheckpointType.BC_WAYPOINT
                if config.get("algorithm") == "PPO":
                    return CheckpointType.RL_PPO_DELTA
                if config.get("algorithm") == "GRPO":
                    return CheckpointType.RL_GRPO_DELTA
        
        # Check for explicit type field
        if "type" in checkpoint:
            type_str = checkpoint["type"]
            try:
                return CheckpointType(type_str)
            except ValueError:
                pass
        
        # Fall back to key pattern matching
        keys = set()
        if isinstance(checkpoint, dict):
            keys = set(checkpoint.keys())
        
        # Check for RL patterns
        if any(p in str(keys).lower() for p in ["actor", "policy", "ppo"]):
            return CheckpointType.RL_PPO_DELTA
        
        # Check for BC patterns
        if any(p in str(keys).lower() for p in ["waypoint", "encoder", "mlp"]):
            if "ssl" in str(keys).lower():
                return CheckpointType.BC_WAYPOINT_SSL
            return CheckpointType.BC_WAYPOINT
        
        # Check for SSL patterns
        if "backbone" in keys or "projection_head" in keys:
            return CheckpointType.SSL_ENCODER
        
        return CheckpointType.UNKNOWN
    
    def inspect(self, checkpoint_path: Union[str, Path]) -> CheckpointInfo:
        """
        Inspect a checkpoint without loading the full model.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            CheckpointInfo object
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Get basic info
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        # Detect type
        checkpoint_type = self.detect_type(ckpt)
        
        # Extract metadata
        config = {}
        epoch = None
        step = None
        metrics = {}
        has_optimizer = False
        
        if isinstance(ckpt, dict):
            config = ckpt.get("config", ckpt.get("cfg", {}))
            epoch = ckpt.get("epoch")
            step = ckpt.get("step", ckpt.get("global_step"))
            metrics = ckpt.get("metrics", ckpt.get("train_metrics", {}))
            has_optimizer = "optimizer" in ckpt or "optimizer_state_dict" in ckpt
        
        return CheckpointInfo(
            path=checkpoint_path,
            type=checkpoint_type,
            size_mb=size_mb,
            config=config if isinstance(config, dict) else {},
            epoch=epoch,
            step=step,
            metrics=metrics if isinstance(metrics, dict) else {},
            has_optimizer=has_optimizer,
        )
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = False,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a checkpoint with automatic type detection.
        
        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to also load optimizer state
            
        Returns:
            Tuple of (model, config_dict)
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Inspect first
        info = self.inspect(checkpoint_path)
        print(f"Loading {info.type.value} checkpoint from {checkpoint_path.name}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if not isinstance(ckpt, dict):
            # Raw state dict
            return self._load_state_dict(ckpt, info.type)
        
        # Extract config and state
        config = ckpt.get("config", ckpt.get("cfg", {}))
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        
        # Load based on type
        if info.type == CheckpointType.BC_WAYPOINT:
            return self._load_bc_waypoint(state_dict, config)
        elif info.type == CheckpointType.BC_WAYPOINT_SSL:
            return self._load_bc_waypoint_ssl(state_dict, config)
        elif info.type == CheckpointType.RL_PPO_DELTA:
            return self._load_rl_ppo_delta(state_dict, config)
        elif info.type == CheckpointType.RL_GRPO_DELTA:
            return self._load_rl_grpo_delta(state_dict, config)
        elif info.type == CheckpointType.SSL_ENCODER:
            return self._load_ssl_encoder(state_dict, config)
        else:
            # Try as raw state dict
            return self._load_state_dict(state_dict, info.type)
    
    def _load_bc_waypoint(self, state_dict: Dict, config: Dict) -> Tuple[nn.Module, Dict]:
        """Load BC waypoint model."""
        from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
        
        bc_config = WaypointBCConfig(
            bev_feature_dim=config.get("bev_feature_dim", 256),
            bev_height=config.get("bev_height", 200),
            bev_width=config.get("bev_width", 200),
            num_waypoints=config.get("num_waypoints", 8),
            waypoint_dim=config.get("waypoint_dim", 2),
            predict_speed=config.get("predict_speed", True),
            use_temporal=config.get("use_temporal", True),
            temporal_history=config.get("temporal_history", 3),
        )
        
        model = WaypointBCModel(bc_config)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def _load_bc_waypoint_ssl(self, state_dict: Dict, config: Dict) -> Tuple[nn.Module, Dict]:
        """Load BC waypoint model with SSL encoder."""
        from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
        from training.pretrain.train_waymo_ssl import SimpleEncoder, WaymoSSLConfig
        
        ssl_config = WaymoSSLConfig(
            backbone=config.get("ssl_encoder_type", "resnet34"),
            feature_dim=config.get("ssl_feature_dim", 256),
        )
        ssl_encoder = SimpleEncoder(ssl_config)
        
        bc_config = WaypointBCConfig(
            bev_feature_dim=config.get("bev_feature_dim", 256),
            num_waypoints=config.get("num_waypoints", 8),
            waypoint_dim=config.get("waypoint_dim", 2),
            predict_speed=config.get("predict_speed", True),
        )
        
        model = WaypointBCModel(bc_config, ssl_encoder=ssl_encoder)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def _load_rl_ppo_delta(self, state_dict: Dict, config: Dict) -> Tuple[nn.Module, Dict]:
        """Load PPO delta-waypoint model."""
        from training.rl.ppo_delta_waypoint_trainer import PPODeltaPolicy, PPODeltaConfig
        
        rl_config = PPODeltaConfig(
            bev_feature_dim=config.get("bev_feature_dim", 256),
            num_waypoints=config.get("num_waypoints", 8),
            waypoint_dim=config.get("waypoint_dim", 2),
            hidden_dim=config.get("hidden_dim", 512),
            lr=config.get("learning_rate", 3e-4),
        )
        
        model = PPODeltaPolicy(rl_config)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def _load_rl_grpo_delta(self, state_dict: Dict, config: Dict) -> Tuple[nn.Module, Dict]:
        """Load GRPO delta-waypoint model."""
        # Similar to PPO but with GRPO-specific heads
        from training.rl.ppo_delta_waypoint_trainer import PPODeltaPolicy, PPODeltaConfig
        
        rl_config = PPODeltaConfig(
            bev_feature_dim=config.get("bev_feature_dim", 256),
            num_waypoints=config.get("num_waypoints", 8),
            waypoint_dim=config.get("waypoint_dim", 2),
        )
        
        model = PPODeltaPolicy(rl_config)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def _load_ssl_encoder(self, state_dict: Dict, config: Dict) -> Tuple[nn.Module, Dict]:
        """Load SSL encoder."""
        from training.pretrain.train_waymo_ssl import SimpleEncoder, WaymoSSLConfig
        
        ssl_config = WaymoSSLConfig(
            backbone=config.get("backbone", "resnet34"),
            feature_dim=config.get("feature_dim", 256),
        )
        
        model = SimpleEncoder(ssl_config)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def _load_state_dict(self, state_dict: Dict, checkpoint_type: CheckpointType) -> Tuple[nn.Module, Dict]:
        """Load as raw state dict (fallback)."""
        # Return a simple wrapper
        class DummyModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                self._state_dict = state_dict
                for k, v in state_dict.items():
                    self.register_buffer(k, v)
        
        model = DummyModel(state_dict)
        model.to(self.device)
        model.eval()
        
        return model, {"type": checkpoint_type.value}


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified checkpoint manager")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to load to")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    manager = CheckpointManager(device=args.device)
    
    # Inspect
    info = manager.inspect(args.checkpoint)
    
    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
    else:
        print(f"Checkpoint: {info.path.name}")
        print(f"Type: {info.type.value}")
        print(f"Size: {info.size_mb:.1f} MB")
        if info.epoch is not None:
            print(f"Epoch: {info.epoch}")
        if info.step is not None:
            print(f"Step: {info.step}")
        if info.config:
            print(f"Config: {info.config}")


if __name__ == "__main__":
    main()

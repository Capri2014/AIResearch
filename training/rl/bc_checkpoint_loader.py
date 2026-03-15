"""
BC Checkpoint Loader for Waypoint Prediction Models.

Loads trained BC waypoint checkpoints (with or without SSL encoder) 
for inference, RL refinement, or CARLA evaluation.

Supports:
- WaypointBCModel checkpoints (from train_waypoint_bc.py)
- WaypointBCWithSSL checkpoints (from train_waypoint_bc_ssl.py)
- Both have standardized checkpoint formats with config + state_dict

Usage:
    from training.rl.bc_checkpoint_loader import load_bc_waypoint_model
    
    # Load with SSL encoder
    model, cfg = load_bc_waypoint_model(
        checkpoint_path="out/bc_ssl/model.pt",
        ssl_encoder=encoder,  # Optional pretrained encoder
        device="cuda"
    )
    
    # Load without encoder (for inference only)
    model, cfg = load_bc_waypoint_model(
        checkpoint_path="out/bc/model.pt",
        device="cpu"
    )
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

import torch
import torch.nn as nn


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BCCheckpointConfig:
    """Configuration stored in BC checkpoint."""
    # Model architecture
    model_type: str = "WaypointBCModel"  # or "WaypointBCWithSSL"
    bev_feature_dim: int = 256
    bev_height: int = 200
    bev_width: int = 200
    num_waypoints: int = 8
    waypoint_dim: int = 2
    predict_speed: bool = True
    use_temporal: bool = True
    temporal_history: int = 3
    
    # Training config
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    
    # SSL encoder config (if applicable)
    ssl_encoder_type: Optional[str] = None
    ssl_feature_dim: Optional[int] = None
    
    # Metadata
    checkpoint_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BCCheckpointConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BCLoadConfig:
    """Configuration for loading BC model."""
    checkpoint: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    strict: bool = True  # Whether to strictly check state_dict keys
    eval_mode: bool = True  # Set to eval() after loading


# ============================================================================
# Model Loading
# ============================================================================

def load_bc_waypoint_model(
    checkpoint: Path | str,
    ssl_encoder: Optional[nn.Module] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_mode: bool = True,
    strict: bool = True,
) -> Tuple[nn.Module, BCCheckpointConfig]:
    """
    Load a BC waypoint model from checkpoint.
    
    Args:
        checkpoint: Path to checkpoint file (.pt)
        ssl_encoder: Optional pretrained SSL encoder to use
                    If provided, will use SSL encoder instead of loading from checkpoint
        device: Device to load model to
        eval_mode: Whether to set model to eval mode
        strict: Whether to strictly enforce state_dict matching
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    
    # Extract config
    if isinstance(ckpt, dict):
        if "config" in ckpt:
            config = BCCheckpointConfig.from_dict(ckpt["config"])
        elif "cfg" in ckpt:
            config = BCCheckpointConfig.from_dict(ckpt["cfg"])
        else:
            print("Warning: No config found in checkpoint, using defaults")
            config = BCCheckpointConfig()
        
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        # Legacy: checkpoint is just the state_dict
        state_dict = ckpt
        config = BCCheckpointConfig()
    
    # Determine model type
    model_type = config.model_type
    
    # Build model based on type
    if model_type == "WaypointBCWithSSL" or ssl_encoder is not None:
        model = _build_model_with_ssl(config, ssl_encoder)
    else:
        model = _build_standard_model(config)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        print(f"Warning: Failed to load state dict strictly: {e}")
        if not strict:
            raise
    
    # Move to device and set eval mode
    model = model.to(device)
    if eval_mode:
        model.eval()
    
    print(f"Loaded {model_type} with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, config


def _build_standard_model(config: BCCheckpointConfig) -> nn.Module:
    """Build standard WaypointBCModel."""
    from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
    
    bc_config = WaypointBCConfig(
        bev_feature_dim=config.bev_feature_dim,
        bev_height=config.bev_height,
        bev_width=config.bev_width,
        num_waypoints=config.num_waypoints,
        waypoint_dim=config.waypoint_dim,
        predict_speed=config.predict_speed,
        use_temporal=config.use_temporal,
        temporal_history=config.temporal_history,
    )
    
    return WaypointBCModel(bc_config)


def _build_model_with_ssl(
    config: BCCheckpointConfig,
    ssl_encoder: Optional[nn.Module] = None,
) -> nn.Module:
    """Build WaypointBCModel with SSL encoder."""
    from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
    from training.pretrain.train_waymo_ssl import SimpleEncoder, WaymoSSLConfig
    
    # Build SSL encoder if not provided
    if ssl_encoder is None:
        ssl_config = WaymoSSLConfig(
            backbone=config.ssl_encoder_type or "resnet34",
            feature_dim=config.ssl_feature_dim or 256,
        )
        ssl_encoder = SimpleEncoder(ssl_config)
    
    bc_config = WaypointBCConfig(
        bev_feature_dim=config.bev_feature_dim,
        bev_height=config.bev_height,
        bev_width=config.bev_width,
        num_waypoints=config.num_waypoints,
        waypoint_dim=config.waypoint_dim,
        predict_speed=config.predict_speed,
        use_temporal=config.use_temporal,
        temporal_history=config.temporal_history,
    )
    
    return WaypointBCModel(bc_config, ssl_encoder=ssl_encoder)


# ============================================================================
# Checkpoint Info Utility
# ============================================================================

def inspect_checkpoint(checkpoint: Path | str) -> Dict[str, Any]:
    """
    Inspect a checkpoint without loading the full model.
    
    Returns:
        Dict with keys, config, and metadata
    """
    checkpoint = Path(checkpoint)
    
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    
    info = {
        "path": str(checkpoint),
        "size_mb": checkpoint.stat().st_size / (1024 * 1024),
    }
    
    if isinstance(ckpt, dict):
        info["keys"] = list(ckpt.keys())
        if "config" in ckpt:
            info["config"] = ckpt["config"]
        if "epoch" in ckpt:
            info["epoch"] = ckpt["epoch"]
        if "step" in ckpt:
            info["step"] = ckpt["step"]
        if "metrics" in ckpt:
            info["metrics"] = ckpt["metrics"]
    else:
        info["type"] = "raw_state_dict"
        info["keys"] = list(ckpt.keys())[:10]  # First 10 keys
    
    return info


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load BC waypoint model from checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    parser.add_argument("--device", default="cuda", help="Device to load to")
    parser.add_argument("--inspect", action="store_true", help="Only inspect checkpoint")
    parser.add_argument("--output", type=Path, help="Save inspection to JSON")
    
    args = parser.parse_args()
    
    if args.inspect:
        info = inspect_checkpoint(args.checkpoint)
        print(json.dumps(info, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(info, f, indent=2)
        return
    
    model, config = load_bc_waypoint_model(args.checkpoint, device=args.device)
    print(f"Model loaded successfully!")
    print(f"Config: {config}")
    

if __name__ == "__main__":
    main()

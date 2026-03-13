"""
Checkpoint Utilities for Driving Pipeline

Utilities for loading, validating, and converting training checkpoints
for use across the driving-first pipeline (BC → RL → CARLA eval).

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Checkpoint type detection
class CheckpointType:
    """Enum-like for checkpoint types."""
    WAYPOINT_BC = "waypoint_bc"
    WAYPOINT_RL = "waypoint_rl"
    SSL_PRETRAIN = "ssl_pretrain"
    UNKNOWN = "unknown"


def detect_checkpoint_type(checkpoint_path: Union[str, Path]) -> CheckpointType:
    """
    Detect the type of checkpoint from its metadata or structure.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        CheckpointType enum value
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
        return CheckpointType.UNKNOWN
    
    # Detect type from checkpoint keys
    keys = checkpoint.keys() if isinstance(checkpoint, dict) else []
    
    # Waypoint BC indicators
    if "model_state" in keys or "state_dict" in keys:
        state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", {}))
        if any("waypoint" in k.lower() for k in state_dict.keys()):
            if "actor_critic" in str(state_dict.keys()) or "policy" in str(state_dict.keys()):
                return CheckpointType.WAYPOINT_RL
            return CheckpointType.WAYPOINT_BC
    
    # RL indicators
    if "actor_critic" in keys or "policy" in keys or "reward" in str(checkpoint.get("training", "")):
        return CheckpointType.WAYPOINT_RL
    
    # SSL pretrain indicators
    if "encoder" in keys or "backbone" in keys:
        return CheckpointType.SSL_PRETRAIN
    
    # Checkpoint path/name hints
    name_lower = checkpoint_path.name.lower()
    if "rl" in name_lower or "ppo" in name_lower or "grpo" in name_lower:
        return CheckpointType.WAYPOINT_RL
    if "bc" in name_lower or "sft" in name_lower or "waypoint" in name_lower:
        return CheckpointType.WAYPOINT_BC
    if "ssl" in name_lower or "pretrain" in name_lower or "contrastive" in name_lower:
        return CheckpointType.SSL_PRETRAIN
    
    return CheckpointType.UNKNOWN


def load_checkpoint_metadata(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metadata from a checkpoint if available.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Metadata dict (empty if not found)
    """
    checkpoint_path = Path(checkpoint_path)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
        return {}
    
    # Extract metadata
    metadata = {}
    for key in ["epoch", "step", "training_step", "config", "hyperparameters", 
                "metrics", "git_info", "timestamp", "version"]:
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    
    return metadata


def validate_checkpoint_for_eval(checkpoint_path: Union[str, Path]) -> tuple[bool, str]:
    """
    Validate that a checkpoint can be used for CARLA evaluation.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        (is_valid, message) tuple
    """
    checkpoint_type = detect_checkpoint_type(checkpoint_path)
    
    if checkpoint_type == CheckpointType.UNKNOWN:
        return False, f"Unknown checkpoint type: {checkpoint_path}"
    
    if checkpoint_type == CheckpointType.SSL_PRETRAIN:
        return False, "SSL pretrained checkpoints require BC fine-tuning before eval"
    
    # Load and check structure
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"Failed to load checkpoint: {e}"
    
    if not isinstance(checkpoint, dict):
        return False, "Checkpoint is not a dict (may be old format)"
    
    # Check for required keys
    if checkpoint_type == CheckpointType.WAYPOINT_BC:
        if "model_state" not in checkpoint and "state_dict" not in checkpoint:
            return False, "BC checkpoint missing model_state or state_dict"
    
    if checkpoint_type == CheckpointType.WAYPOINT_RL:
        required = ["model_state", "policy_state"]
        if not any(k in checkpoint for k in required):
            return False, f"RL checkpoint missing required keys: {required}"
    
    # Check for metrics if available
    metadata = load_checkpoint_metadata(checkpoint_path)
    if "metrics" in metadata:
        logger.info(f"Checkpoint metrics: {metadata['metrics']}")
    
    return True, f"Valid {checkpoint_type} checkpoint"


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive info about a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Dict with type, metadata, validation status
    """
    checkpoint_path = Path(checkpoint_path)
    
    info = {
        "path": str(checkpoint_path),
        "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        "type": detect_checkpoint_type(checkpoint_path),
        "metadata": load_checkpoint_metadata(checkpoint_path),
    }
    
    is_valid, message = validate_checkpoint_for_eval(checkpoint_path)
    info["eval_valid"] = is_valid
    info["eval_message"] = message
    
    return info


def print_checkpoint_info(checkpoint_path: Union[str, Path]) -> None:
    """Pretty-print checkpoint information."""
    info = get_checkpoint_info(checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"Checkpoint: {info['path']}")
    print(f"{'='*60}")
    print(f"Type: {info['type']}")
    print(f"Size: {info['size_mb']:.1f} MB")
    print(f"Eval Valid: {'✓' if info['eval_valid'] else '✗'}")
    print(f"  → {info['eval_message']}")
    
    if info['metadata']:
        print("\nMetadata:")
        for key, value in info['metadata'].items():
            if key == "config" and isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect driving pipeline checkpoints")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    if args.json:
        info = get_checkpoint_info(args.checkpoint)
        print(json.dumps(info, indent=2, default=str))
    else:
        print_checkpoint_info(args.checkpoint)

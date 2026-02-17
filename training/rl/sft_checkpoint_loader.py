#!/usr/bin/env python3
"""
SFT Checkpoint Loader for RL Pipeline Integration

This module provides robust SFT checkpoint loading for the RL delta-waypoint
training pipeline. It handles multiple checkpoint formats and provides
integration utilities for chaining SFT → RL training.

Checkpoint Format Support
-------------------------
- `out/waypoint_bc/model.pt` - WaypointBCModel format (encoder + head)
- `out/waypoint_bc/best_model.pt` - Best checkpoint variant
- `out/sft_waypoint_bc/model.pt` - SFT-specific format
- Legacy formats with `model_state`, `state_dict` keys

Integration Pattern
------------------
    sft_ckpt = load_sft_checkpoint("out/waypoint_bc/model.pt")
    rl_trainer = RLDeltaTrainer(sft_checkpoint=sft_ckpt)
    # RL trains delta_head that learns corrections to frozen SFT output

Usage
-----
# Load and inspect checkpoint
python -m training.rl.sft_checkpoint_loader --checkpoint out/waypoint_bc/model.pt --inspect

# Run smoke test (no real checkpoint required)
python -m training.rl.sft_checkpoint_loader --smoke

# Validate checkpoint format
python -m training.rl.sft_checkpoint_loader --checkpoint out/waypoint_bc/model.pt --validate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
import sys

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTCheckpointMetadata:
    """Metadata extracted from SFT checkpoint."""
    checkpoint_path: str
    format_version: str = "unknown"
    horizon_steps: int = 0
    out_dim: int = 0
    encoder_type: str = "unknown"
    has_encoder: bool = False
    has_head: bool = False
    has_waypoints: bool = False
    extra_keys: List[str] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"SFTCheckpointMetadata(\n"
            f"  path={self.checkpoint_path},\n"
            f"  format={self.format_version},\n"
            f"  horizon={self.horizon_steps},\n"
            f"  out_dim={self.out_dim},\n"
            f"  encoder={self.encoder_type} (has={self.has_encoder}),\n"
            f"  head={self.has_head},\n"
            f"  waypoints={self.has_waypoints}\n"
            f")"
        )


@dataclass
class LoadedSFTModel:
    """Container for loaded SFT model components."""
    metadata: SFTCheckpointMetadata
    encoder: Optional[nn.Module] = None
    head: Optional[nn.Module] = None
    waypoints: Optional[torch.Tensor] = None
    config: Optional[Dict] = None
    raw_checkpoint: Dict = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"LoadedSFTModel(",
            f"  format={self.metadata.format_version},",
            f"  horizon={self.metadata.horizon_steps},",
        ]
        if self.encoder is not None:
            parts.append(f"  encoder={type(self.encoder).__name__}")
        if self.head is not None:
            parts.append(f"  head={type(self.head).__name__}")
        if self.waypoints is not None:
            parts.append(f"  waypoints={tuple(self.waypoints.shape)}")
        parts.append(")")
        return "\n".join(parts)


# === Checkpoint Format Detection ===

CHECKPOINT_FORMATS = {
    "WaypointBCModel": {
        "required_keys": ["encoder", "head"],
        "optional_keys": ["config", "cam", "horizon_steps", "out_dim"],
        "version": "1.0",
    },
    "SFTWaypointModel": {
        "required_keys": ["sft_waypoints"],
        "optional_keys": ["config", "model_state", "state_dict"],
        "version": "1.1",
    },
    "LegacyFormat": {
        "required_keys": ["model_state", "state_dict"],
        "optional_keys": [],
        "version": "legacy",
    },
    "MinimalFormat": {
        "required_keys": [],
        "optional_keys": [],
        "version": "minimal",
    },
}


def detect_checkpoint_format(ckpt: Dict) -> Tuple[str, Dict]:
    """Detect the format of a checkpoint dictionary.
    
    Returns:
        Tuple of (format_name, format_info)
    """
    # Check for WaypointBCModel format
    if "encoder" in ckpt and "head" in ckpt:
        return "WaypointBCModel", CHECKPOINT_FORMATS["WaypointBCModel"]
    
    # Check for SFTWaypointModel format
    if "sft_waypoints" in ckpt:
        return "SFTWaypointModel", CHECKPOINT_FORMATS["SFTWaypointModel"]
    
    # Check for legacy format with state dict
    if "model_state" in ckpt or "state_dict" in ckpt:
        return "LegacyFormat", CHECKPOINT_FORMATS["LegacyFormat"]
    
    # Minimal format (may just contain config)
    return "MinimalFormat", CHECKPOINT_FORMATS["MinimalFormat"]


def extract_checkpoint_metadata(ckpt: Dict, checkpoint_path: str) -> SFTCheckpointMetadata:
    """Extract metadata from a checkpoint."""
    format_name, format_info = detect_checkpoint_format(ckpt)
    
    metadata = SFTCheckpointMetadata(
        checkpoint_path=checkpoint_path,
        format_version=format_info["version"],
    )
    
    # Extract config if present
    if "config" in ckpt:
        config = ckpt["config"]
        if isinstance(config, dict):
            metadata.horizon_steps = config.get("num_waypoints", config.get("horizon_steps", 0))
            metadata.out_dim = config.get("out_dim", 0)
            metadata.encoder_type = config.get("encoder_type", "unknown")
            metadata.config = config
    
    # Extract from WaypointBCModel format
    if format_name == "WaypointBCModel":
        metadata.has_encoder = "encoder" in ckpt
        metadata.has_head = "head" in ckpt
        metadata.horizon_steps = ckpt.get("horizon_steps", metadata.horizon_steps)
        metadata.out_dim = ckpt.get("out_dim", metadata.out_dim)
        if metadata.horizon_steps == 0 and metadata.has_head:
            # Try to infer from head shape
            head_state = ckpt.get("head", {})
            if isinstance(head_state, nn.Module):
                pass  # Can't inspect unloaded module
    
    # Extract from SFTWaypointModel format
    if format_name == "SFTWaypointModel":
        metadata.has_waypoints = "sft_waypoints" in ckpt
        if metadata.has_waypoints:
            wp = ckpt["sft_waypoints"]
            if isinstance(wp, torch.Tensor):
                metadata.horizon_steps = wp.shape[0] if len(wp.shape) >= 1 else 0
    
    # Track extra and missing keys
    all_known_keys = set()
    for fmt_info in CHECKPOINT_FORMATS.values():
        all_known_keys.update(fmt_info["required_keys"])
        all_known_keys.update(fmt_info["optional_keys"])
    
    metadata.extra_keys = [k for k in ckpt.keys() if k not in all_known_keys]
    required = set(format_info["required_keys"])
    metadata.missing_keys = [k for k in required if k not in ckpt]
    
    return metadata


# === Encoder Loading ===

def load_encoder_from_checkpoint(
    ckpt: Dict,
    out_dim: Optional[int] = None,
    device: str = "cpu",
) -> Optional[nn.Module]:
    """Load encoder module from checkpoint.
    
    Args:
        ckpt: Checkpoint dictionary
        out_dim: Override output dimension
        device: Device to load weights to
    
    Returns:
        Loaded encoder module or None if not available
    """
    if "encoder" not in ckpt:
        logger.warning("No encoder found in checkpoint")
        return None
    
    # Try to determine encoder type from checkpoint
    encoder_type = ckpt.get("config", {}).get("encoder_type", "unknown")
    
    # Import the actual encoder class
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
        
        # Determine out_dim
        if out_dim is None:
            out_dim = ckpt.get("out_dim", 128)
            if out_dim == 0:
                # Try to infer from checkpoint
                encoder_state = ckpt["encoder"]
                if isinstance(encoder_state, dict):
                    # Find first Linear layer to infer dim
                    for key, value in encoder_state.items():
                        if isinstance(value, torch.Tensor) and "weight" in key:
                            out_dim = value.shape[0] if "weight" in key else 128
                            break
        
        encoder = TinyMultiCamEncoder(out_dim=out_dim)
        
        # Load state dict
        if isinstance(ckpt["encoder"], dict):
            encoder.load_state_dict(ckpt["encoder"])
        else:
            logger.warning("Encoder state is not a dict, cannot load")
            return None
        
        encoder.to(device)
        encoder.eval()
        
        logger.info(f"Loaded TinyMultiCamEncoder with out_dim={out_dim}")
        return encoder
        
    except ImportError as e:
        logger.warning(f"Could not import encoder: {e}")
        return None


def load_head_from_checkpoint(
    ckpt: Dict,
    in_dim: Optional[int] = None,
    horizon_steps: Optional[int] = None,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> Optional[nn.Module]:
    """Load waypoint prediction head from checkpoint.
    
    Args:
        ckpt: Checkpoint dictionary
        in_dim: Input dimension (latent dim)
        horizon_steps: Number of waypoints to predict
        hidden_dim: Hidden dimension
        device: Device to load weights to
    
    Returns:
        Loaded head module or None if not available
    """
    if "head" not in ckpt:
        logger.warning("No head found in checkpoint")
        return None
    
    # Determine dimensions
    if horizon_steps is None:
        horizon_steps = ckpt.get("horizon_steps", 4)
    
    if in_dim is None:
        in_dim = ckpt.get("out_dim", 128)
        if in_dim == 0:
            # Try to infer from head state
            head_state = ckpt["head"]
            if isinstance(head_state, dict):
                for key, value in head_state.items():
                    if "weight" in key and isinstance(value, torch.Tensor):
                        # Linear layer weight: (out_dim, in_dim)
                        in_dim = value.shape[1] if len(value.shape) == 2 else 128
                        break
    
    # Create appropriate head based on format
    format_name, _ = detect_checkpoint_format(ckpt)
    
    if format_name == "WaypointBCModel":
        # This was likely a TinyMulticamEncoder + WaypointHead
        class WaypointHead(nn.Module):
            def __init__(self, in_dim, horizon):
                super().__init__()
                self.horizon = horizon
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, horizon * 2),
                )
            
            def forward(self, z):
                y = self.net(z)
                return y.view(-1, self.horizon, 2)
        
        head = WaypointHead(in_dim, horizon_steps)
        
        if isinstance(ckpt["head"], dict):
            head.load_state_dict(ckpt["head"])
        
        head.to(device)
        head.eval()
        
        logger.info(f"Loaded WaypointHead with in_dim={in_dim}, horizon={horizon_steps}")
        return head
    
    else:
        logger.warning(f"Unknown head format for {format_name}")
        return None


# === Main Loading Function ===

def load_sft_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    load_encoder: bool = True,
    load_head: bool = True,
) -> LoadedSFTModel:
    """Load an SFT checkpoint for use in RL training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to
        load_encoder: Whether to load encoder weights
        load_head: Whether to load head weights
    
    Returns:
        LoadedSFTModel containing all loaded components
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading SFT checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,  # Our checkpoints are trusted
        )
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
    
    # Extract metadata
    metadata = extract_checkpoint_metadata(ckpt, str(checkpoint_path))
    logger.info(f"Detected format: {metadata.format_version}")
    
    # Load components
    encoder = None
    head = None
    waypoints = None
    
    if load_encoder and metadata.has_encoder:
        encoder = load_encoder_from_checkpoint(ckpt, metadata.out_dim, device)
    
    if load_head and metadata.has_head:
        head = load_head_from_checkpoint(
            ckpt,
            in_dim=metadata.out_dim,
            horizon_steps=metadata.horizon_steps,
            device=device,
        )
    
    # Load waypoints for SFTWaypointModel format
    if "sft_waypoints" in ckpt:
        waypoints = ckpt["sft_waypoints"]
        if isinstance(waypoints, torch.Tensor):
            waypoints = waypoints.to(device)
            waypoints.requires_grad = False
    
    return LoadedSFTModel(
        metadata=metadata,
        encoder=encoder,
        head=head,
        waypoints=waypoints,
        config=metadata.config,
        raw_checkpoint=ckpt,
    )


# === Validation Functions ===

def validate_checkpoint(checkpoint_path: Path) -> Tuple[bool, List[str]]:
    """Validate a checkpoint for use in the RL pipeline.
    
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    is_valid = True
    
    if not checkpoint_path.exists():
        return False, [f"Checkpoint not found: {checkpoint_path}"]
    
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        return False, [f"Failed to load checkpoint: {e}"]
    
    format_name, format_info = detect_checkpoint_format(ckpt)
    logger.info(f"Detected format: {format_name}")
    
    # Check required keys
    for key in format_info["required_keys"]:
        if key not in ckpt:
            issues.append(f"Missing required key: {key}")
            is_valid = False
    
    # Check for common issues
    if format_name == "WaypointBCModel":
        if "encoder" in ckpt and not isinstance(ckpt["encoder"], dict):
            issues.append("Encoder is not a state dict")
            is_valid = False
        if "head" in ckpt and not isinstance(ckpt["head"], dict):
            issues.append("Head is not a state dict")
            is_valid = False
    
    if format_name == "SFTWaypointModel":
        if "sft_waypoints" in ckpt:
            wp = ckpt["sft_waypoints"]
            if not isinstance(wp, torch.Tensor):
                issues.append("sft_waypoints is not a tensor")
                is_valid = False
            elif len(wp.shape) < 2:
                issues.append(f"sft_waypoints has unexpected shape: {wp.shape}")
                is_valid = False
    
    return is_valid, issues


def inspect_checkpoint(checkpoint_path: Path) -> Dict:
    """Get detailed inspection of a checkpoint."""
    if not checkpoint_path.exists():
        return {"error": f"Checkpoint not found: {checkpoint_path}"}
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    metadata = extract_checkpoint_metadata(ckpt, str(checkpoint_path))
    
    inspection = {
        "path": str(checkpoint_path),
        "file_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        "format": metadata.format_version,
        "keys": list(ckpt.keys()),
        "metadata": {
            "horizon_steps": metadata.horizon_steps,
            "out_dim": metadata.out_dim,
            "has_encoder": metadata.has_encoder,
            "has_head": metadata.has_head,
            "has_waypoints": metadata.has_waypoints,
        },
    }
    
    # Add shape info for tensors
    tensor_info = {}
    for key, value in ckpt.items():
        if isinstance(value, torch.Tensor):
            tensor_info[key] = {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
            }
    inspection["tensors"] = tensor_info
    
    return inspection


# === Inference Utility ===

def run_sft_inference(
    model: LoadedSFTModel,
    images: Optional[Dict[str, torch.Tensor]] = None,
    waypoints: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run SFT model inference.
    
    Args:
        model: Loaded SFT model
        images: Input images (required if encoder is loaded)
        waypoints: Input waypoints (required if no encoder)
    
    Returns:
        Predicted waypoints (H, 2) tensor
    
    Raises:
        ValueError: If required inputs are missing
    """
    if model.encoder is not None and images is not None:
        # Full encoder-based inference
        with torch.no_grad():
            if hasattr(model.encoder, 'encode'):
                z = model.encoder.encode(images)
            else:
                z = model.encoder(images)
            if model.head is not None:
                waypoints = model.head(z)
            else:
                # Fallback to stored waypoints
                if model.waypoints is not None:
                    waypoints = model.waypoints.unsqueeze(0)
                else:
                    raise ValueError("No waypoints available for inference")
        return waypoints.squeeze(0)
    
    elif model.waypoints is not None:
        # Return stored waypoints
        return model.waypoints
    
    else:
        raise ValueError("Cannot run inference: no encoder/images or waypoints available")


# === Comparison Utilities ===

def compare_sft_vs_rl(
    sft_model: LoadedSFTModel,
    rl_delta_head: nn.Module,
    test_waypoints: torch.Tensor,
) -> Dict[str, float]:
    """Compare SFT-only predictions vs SFT+RL predictions.
    
    Args:
        sft_model: Loaded SFT model
        rl_delta_head: Trained RL delta head
        test_waypoints: Test waypoints to compare on
    
    Returns:
        Dictionary of comparison metrics
    """
    sft_wp = sft_model.waypoints
    if sft_wp is None:
        return {"error": "No SFT waypoints available"}
    
    with torch.no_grad():
        # SFT-only
        sft_output = sft_wp.unsqueeze(0) if sft_wp.dim() == 1 else sft_wp
        
        # RL delta
        if hasattr(rl_delta_head, 'net'):
            # DeltaHead format
            delta = rl_delta_head.net(test_waypoints)
            delta = delta.view(-1, sft_output.shape[-2], 2)
        else:
            delta = rl_delta_head(test_waypoints)
        
        # Combined
        combined = sft_output + delta
    
    # Compute metrics
    sft_l2 = torch.norm(sft_output, dim=-1).mean()
    delta_l2 = torch.norm(delta, dim=-1).mean()
    combined_l2 = torch.norm(combined, dim=-1).mean()
    
    return {
        "sft_l2_mean": sft_l2.item(),
        "delta_l2_mean": delta_l2.item(),
        "combined_l2_mean": combined_l2.item(),
        "delta_ratio": (delta_l2 / (sft_l2 + 1e-6)).item(),
    }


# === Smoke Test ===

def run_smoke_test():
    """Run smoke test without real checkpoint."""
    print("=" * 60)
    print("SMOKE TEST: SFT Checkpoint Loader")
    print("=" * 60)
    
    # Test 1: Checkpoint format detection
    print("\n[Test 1] Checkpoint format detection")
    
    # Mock WaypointBCModel format
    mock_ckpt_waypoint_bc = {
        "encoder": {"layer1.weight": torch.randn(64, 128)},
        "head": {"net.0.weight": torch.randn(256, 64)},
        "horizon_steps": 4,
        "out_dim": 64,
    }
    fmt, _ = detect_checkpoint_format(mock_ckpt_waypoint_bc)
    assert fmt == "WaypointBCModel", f"Expected WaypointBCModel, got {fmt}"
    print("  ✓ WaypointBCModel format detected")
    
    # Mock SFTWaypointModel format
    mock_ckpt_sft = {
        "sft_waypoints": torch.randn(4, 2),
        "config": {"num_waypoints": 4},
    }
    fmt, _ = detect_checkpoint_format(mock_ckpt_sft)
    assert fmt == "SFTWaypointModel", f"Expected SFTWaypointModel, got {fmt}"
    print("  ✓ SFTWaypointModel format detected")
    
    # Mock legacy format
    mock_ckpt_legacy = {
        "model_state": {"layer1.weight": torch.randn(64, 128)},
    }
    fmt, _ = detect_checkpoint_format(mock_ckpt_legacy)
    assert fmt == "LegacyFormat", f"Expected LegacyFormat, got {fmt}"
    print("  ✓ LegacyFormat detected")
    
    # Test 2: Metadata extraction
    print("\n[Test 2] Metadata extraction")
    metadata = extract_checkpoint_metadata(mock_ckpt_waypoint_bc, "test.pt")
    assert metadata.format_version == "1.0"
    assert metadata.has_encoder is True
    assert metadata.has_head is True
    print(f"  ✓ Metadata extracted: {metadata.format_version}")
    
    # Test 3: SFTWaypointModel metadata
    metadata_sft = extract_checkpoint_metadata(mock_ckpt_sft, "test_sft.pt")
    assert metadata_sft.has_waypoints is True
    assert metadata_sft.horizon_steps == 4
    print(f"  ✓ SFT metadata: horizon={metadata_sft.horizon_steps}")
    
    # Test 4: Validation
    print("\n[Test 3] Checkpoint validation")
    
    # Create a minimal valid checkpoint
    valid_ckpt = {
        "encoder": {"layer1.weight": torch.randn(64, 128)},
        "head": {"net.0.weight": torch.randn(256, 64)},
    }
    # Note: validate_checkpoint requires a file, so we skip full test
    
    # Test 5: LoadedSFTModel creation
    print("\n[Test 4] LoadedSFTModel creation")
    loaded = LoadedSFTModel(
        metadata=metadata,
        encoder=None,  # Skip actual loading in smoke test
        head=None,
        waypoints=torch.randn(4, 2),
    )
    assert loaded.waypoints is not None
    print(f"  ✓ LoadedSFTModel created with waypoints shape {loaded.waypoints.shape}")
    
    # Test 6: Compare utilities
    print("\n[Test 5] Comparison utilities")
    
    class MockDeltaHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(128, 8)  # 4 waypoints * 2 coords
    
    delta_head = MockDeltaHead()
    test_wp = torch.randn(1, 128)
    metrics = compare_sft_vs_rl(loaded, delta_head, test_wp)
    assert "sft_l2_mean" in metrics
    print(f"  ✓ Comparison metrics computed: sft_l2={metrics['sft_l2_mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    print("\nUsage:")
    print("  # Load checkpoint")
    print("  sft_ckpt = load_sft_checkpoint('out/waypoint_bc/model.pt')")
    print("  print(sft_ckpt.summary())")
    print()
    print("  # Inspect checkpoint")
    print("  python -m training.rl.sft_checkpoint_loader --inspect --checkpoint <path>")
    print()
    print("  # Validate checkpoint")
    print("  python -m training.rl.sft_checkpoint_loader --validate --checkpoint <path>")


# === Main Entry Point ===

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SFT Checkpoint Loader for RL Pipeline Integration"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to SFT checkpoint file",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect checkpoint structure and exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate checkpoint for RL pipeline compatibility",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test without checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load checkpoint to",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    
    args = parser.parse_args()
    
    if args.smoke:
        run_smoke_test()
        return
    
    if not args.checkpoint:
        parser.error("--checkpoint required (or use --smoke)")
    
    if args.inspect:
        inspection = inspect_checkpoint(args.checkpoint)
        if args.json:
            print(json.dumps(inspection, indent=2))
        else:
            print(json.dumps(inspection, indent=2))  # Always JSON for inspection
        return
    
    if args.validate:
        is_valid, issues = validate_checkpoint(args.checkpoint)
        if args.json:
            print(json.dumps({"valid": is_valid, "issues": issues}, indent=2))
        else:
            if is_valid:
                print(f"✓ Checkpoint {args.checkpoint} is valid")
            else:
                print(f"✗ Checkpoint {args.checkpoint} has issues:")
                for issue in issues:
                    print(f"  - {issue}")
        return
    
    # Load checkpoint
    try:
        sft_model = load_sft_checkpoint(
            args.checkpoint,
            device=args.device,
        )
        
        if args.json:
            print(json.dumps({
                "metadata": {
                    "format": sft_model.metadata.format_version,
                    "horizon_steps": sft_model.metadata.horizon_steps,
                    "out_dim": sft_model.metadata.out_dim,
                    "has_encoder": sft_model.encoder is not None,
                    "has_head": sft_model.head is not None,
                    "has_waypoints": sft_model.waypoints is not None,
                }
            }, indent=2))
        else:
            print(sft_model.summary())
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

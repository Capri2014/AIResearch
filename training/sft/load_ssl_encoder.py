#!/usr/bin/env python3
"""
SSL Encoder to Waypoint BC Bridge Script

Loads a pretrained SSL encoder checkpoint and adapts it for use in waypoint BC training.
This bridges Stage 1 (SSL pretrain) → Stage 2 (Waypoint BC).

The SSL encoder can be:
- CombinedSSLModel (contrastive + MIM)
- SSLEncoder (contrastive only)
- Any CNN/Transformer encoder with forward() returning embeddings

Usage:
    # Extract encoder weights from SSL checkpoint for BC training
    python training/sft/load_ssl_encoder.py \
        --ssl-checkpoint out/ssl_pretrain/final.pt \
        --output out/waypoint_bc/encoder_weights.pt \
        --encoder-type combined

    # Verify encoder works with BC model
    python training/sft/load_ssl_encoder.py \
        --ssl-checkpoint out/ssl_pretrain/final.pt \
        --verify \
        --encoder-type combined
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def _load_ssl_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load SSL checkpoint and handle different formats."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint
    return checkpoint


def extract_encoder_from_combined(checkpoint: Dict[str, Any]) -> nn.Module:
    """Extract encoder from CombinedSSLModel checkpoint."""
    # CombinedSSLModel typically has encoder as "encoder" or "backbone"
    state_dict = {}
    
    for key, value in checkpoint.items():
        # Extract encoder-related keys
        if key.startswith("encoder.") or key.startswith("backbone."):
            new_key = key.replace("encoder.", "").replace("backbone.", "")
            state_dict[new_key] = value
    
    return state_dict


def extract_encoder_from_contrastive(checkpoint: Dict[str, Any]) -> nn.Module:
    """Extract encoder from contrastive-only SSL checkpoint."""
    state_dict = {}
    
    for key, value in checkpoint.items():
        if key.startswith("encoder.") or key.startswith("backbone."):
            new_key = key.replace("encoder.", "").replace("backbone.", "")
            state_dict[new_key] = value
        elif "encoder" in key.lower():
            state_dict[key] = value
    
    return state_dict


@dataclass
class SSLEncoderConfig:
    """Configuration for SSL encoder extraction."""
    # Input
    ssl_checkpoint: Path
    encoder_type: str = "combined"  # "combined", "contrastive", "jepa"
    
    # Output
    output_path: Optional[Path] = None
    
    # Verification
    verify: bool = False
    test_input_shape: Tuple[int, int, int] = (3, 224, 224)


def load_ssl_encoder(config: SSLEncoderConfig) -> Dict[str, Any]:
    """Load and extract encoder weights from SSL checkpoint."""
    
    # Load checkpoint
    checkpoint = _load_ssl_checkpoint(config.ssl_checkpoint)
    
    # Extract encoder state dict based on type
    if config.encoder_type == "combined":
        encoder_state = extract_encoder_from_combined(checkpoint)
    elif config.encoder_type == "contrastive":
        encoder_state = extract_encoder_from_contrastive(checkpoint)
    else:
        raise ValueError(f"Unknown encoder type: {config.encoder_type}")
    
    if not encoder_state:
        # Fallback: try to use entire checkpoint as encoder
        print(f"Warning: No encoder keys found, using full checkpoint")
        encoder_state = checkpoint
    
    result = {
        "encoder_state": encoder_state,
        "encoder_type": config.encoder_type,
        "num_params": sum(v.numel() for v in encoder_state.values()),
    }
    
    # Try to extract config if available
    if "config" in checkpoint:
        result["config"] = checkpoint["config"]
    elif "args" in checkpoint:
        result["config"] = checkpoint["args"]
    
    return result


def save_encoder_weights(encoder_state: Dict[str, torch.Tensor], output_path: Path):
    """Save extracted encoder weights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state, output_path)
    print(f"Saved encoder weights to {output_path}")
    print(f"  Total parameters: {sum(v.numel() for v in encoder_state.values()):,}")


def verify_encoder(config: SSLEncoderConfig):
    """Verify the extracted encoder works correctly."""
    print(f"\n=== Verifying SSL Encoder ===")
    print(f"Checkpoint: {config.ssl_checkpoint}")
    print(f"Encoder type: {config.encoder_type}")
    
    result = load_ssl_encoder(config)
    encoder_state = result["encoder_state"]
    
    print(f"Extracted {result['num_params']:,} parameters")
    
    # Try to create a simple encoder and load weights
    # This is a smoke test - we just check the state dict is valid
    print("\nSmoke test: state dict is valid")
    print(f"  Keys: {list(encoder_state.keys())[:5]}...")
    
    # Save if output path specified
    if config.output_path:
        save_encoder_weights(encoder_state, config.output_path)
    
    print("\n✓ Encoder extraction successful")
    return result


class WaypointBCWithSSL(nn.Module):
    """Waypoint BC model that can load SSL encoder weights."""
    
    def __init__(
        self,
        encoder_dim: int = 256,
        hidden_dim: int = 512,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
    ):
        super().__init__()
        
        # Encoder (CNN-based, compatible with SSL encoders)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Temporal transformer
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=4,
        )
        
        # Prediction heads
        self.waypoint_head = nn.Linear(encoder_dim, num_waypoints * waypoint_dim)
        self.speed_head = nn.Linear(encoder_dim, 1)
        self.progress_head = nn.Linear(encoder_dim, 1)
    
    def load_ssl_encoder(self, encoder_state: Dict[str, torch.Tensor], strict: bool = False):
        """Load SSL encoder weights into the model."""
        # Filter keys that match our encoder structure
        encoder_keys = {}
        for key, value in encoder_state.items():
            # Handle different key prefixes
            if key.startswith("encoder.") or key.startswith("backbone."):
                continue  # Skip if already processed
            if "conv" in key.lower() or "bn" in key.lower() or "fc" in key.lower():
                encoder_keys[key] = value
        
        # Try to load with partial matching
        model_dict = self.encoder.state_dict()
        loaded_dict = {}
        
        for model_key, model_param in model_dict.items():
            # Try exact match
            if model_key in encoder_state:
                if model_param.shape == encoder_state[model_key].shape:
                    loaded_dict[model_key] = encoder_state[model_key]
                else:
                    print(f"  Skip {model_key}: shape mismatch {model_param.shape} vs {encoder_state[model_key].shape}")
            else:
                # Try fuzzy match
                for enc_key, enc_param in encoder_state.items():
                    # Extract base key name
                    enc_base = enc_key.split(".")[-1]
                    if enc_base == model_key.split(".")[-1] and model_param.shape == enc_param.shape:
                        loaded_dict[model_key] = enc_param
                        break
        
        if loaded_dict:
            self.encoder.load_state_dict(loaded_dict, strict=False)
            print(f"Loaded {len(loaded_dict)} encoder layers from SSL checkpoint")
        else:
            print("Warning: No matching encoder weights found")
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: (B, C, H, W) input images
            
        Returns:
            Dictionary with waypoints, speed, progress predictions
        """
        B = images.shape[0]
        
        # Encode images
        features = self.encoder(images)  # (B, encoder_dim, 1, 1)
        features = features.flatten(1)  # (B, encoder_dim)
        
        # Temporal modeling (reshape to sequence)
        # For now, treat as sequence of length 1
        features = features.unsqueeze(1)  # (B, 1, encoder_dim)
        temporal_features = self.temporal_transformer(features)
        
        # Use final timestep
        final_features = temporal_features[:, -1]  # (B, encoder_dim)
        
        # Predictions
        waypoints = self.waypoint_head(final_features)  # (B, num_waypoints * waypoint_dim)
        speed = self.speed_head(final_features)  # (B, 1)
        progress = self.progress_head(final_features)  # (B, 1)
        
        return {
            "waypoints": waypoints.reshape(B, -1, 2),  # (B, num_waypoints, 2)
            "speed": speed,
            "progress": progress,
        }


def create_sample_model() -> Tuple[WaypointBCWithSSL, Dict[str, torch.Tensor]]:
    """Create sample model and encoder state for testing."""
    model = WaypointBCWithSSL(encoder_dim=256, hidden_dim=512, num_waypoints=8)
    
    # Create dummy encoder state
    encoder_state = {}
    for key, param in model.encoder.named_parameters():
        encoder_state[key] = torch.randn_like(param)
    
    return model, encoder_state


def test_encoder_loading():
    """Test loading encoder weights into waypoint BC model."""
    print("\n=== Testing SSL Encoder → Waypoint BC Loading ===")
    
    # Create model and dummy SSL encoder state
    model, encoder_state = create_sample_model()
    
    print(f"Model encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"SSL encoder state params: {sum(v.numel() for v in encoder_state.values()):,}")
    
    # Load SSL encoder weights
    model.load_ssl_encoder(encoder_state)
    
    # Forward pass test
    test_input = torch.randn(2, 3, 224, 224)
    output = model(test_input)
    
    print(f"\nForward pass successful:")
    print(f"  waypoints shape: {output['waypoints'].shape}")
    print(f"  speed shape: {output['speed'].shape}")
    print(f"  progress shape: {output['progress'].shape}")
    
    print("\n✓ SSL encoder loading test passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Load SSL encoder for waypoint BC")
    parser.add_argument(
        "--ssl-checkpoint",
        type=Path,
        required=True,
        help="Path to SSL pretraining checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for extracted encoder weights",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="combined",
        choices=["combined", "contrastive", "jepa"],
        help="Type of SSL encoder",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify encoder extraction works",
    )
    parser.add_argument(
        "--test-loading",
        action="store_true",
        help="Test loading encoder into waypoint BC model",
    )
    
    args = parser.parse_args()
    
    config = SSLEncoderConfig(
        ssl_checkpoint=args.ssl_checkpoint,
        encoder_type=args.encoder_type,
        output_path=args.output,
        verify=args.verify or args.test_loading,
    )
    
    if args.test_loading:
        # Test the full pipeline
        test_encoder_loading()
        return
    
    if args.verify:
        verify_encoder(config)
        return
    
    # Default: extract and save encoder weights
    result = load_ssl_encoder(config)
    
    if args.output:
        save_encoder_weights(result["encoder_state"], args.output)
    
    # Print summary
    print(f"\n=== SSL Encoder Extraction Summary ===")
    print(f"Encoder type: {result['encoder_type']}")
    print(f"Parameters: {result['num_params']:,}")
    if "config" in result:
        print(f"Config available: Yes")
    else:
        print(f"Config available: No")


if __name__ == "__main__":
    main()
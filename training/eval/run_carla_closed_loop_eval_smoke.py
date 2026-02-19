#!/usr/bin/env python3
"""Smoke test for CARLA evaluation components without requiring CARLA installation.

Tests the waypoint policy wrapper and related components.
"""

from __future__ import annotations

import sys
import torch
import numpy as np
from pathlib import Path

# Test 1: Import module components
print("=" * 60)
print("Smoke Test: Waypoint Policy Components")
print("=" * 60)
print()

print("[1/5] Testing imports...")
try:
    # Import the waypoint head classes directly
    from training.eval.run_carla_closed_loop_eval import (
        WaypointHead,
        DeltaWaypointHead,
        WaypointPolicyWrapper,
        WaypointBCModelWrapper,
    )
    print("  ✓ All classes imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test WaypointHead
print("[2/5] Testing WaypointHead...")
try:
    head = WaypointHead(torch=torch, in_dim=128, horizon_steps=20)
    
    # Test forward pass
    z = torch.randn(2, 128)  # batch of 2
    out = head(z)
    assert out.shape == (2, 20, 2), f"Expected shape (2, 20, 2), got {out.shape}"
    
    # Test state_dict/load_state_dict
    sd = head.state_dict()
    head.load_state_dict(sd)
    
    print(f"  ✓ WaypointHead: forward pass OK, shape={out.shape}")
except Exception as e:
    print(f"  ✗ WaypointHead test failed: {e}")
    sys.exit(1)

# Test 3: Test DeltaWaypointHead
print("[3/5] Testing DeltaWaypointHead...")
try:
    delta_head = DeltaWaypointHead(torch=torch, in_dim=128, horizon_steps=20, hidden_dim=64)
    
    # Test forward pass
    z = torch.randn(2, 128)
    delta = delta_head(z)
    assert delta.shape == (2, 20, 2), f"Expected shape (2, 20, 2), got {delta.shape}"
    
    # Test state_dict/load_state_dict
    sd = delta_head.state_dict()
    delta_head.load_state_dict(sd)
    
    print(f"  ✓ DeltaWaypointHead: forward pass OK, shape={delta.shape}")
except Exception as e:
    print(f"  ✗ DeltaWaypointHead test failed: {e}")
    sys.exit(1)

# Test 4: Test WaypointPolicyWrapper with mock checkpoint
print("[4/5] Testing WaypointPolicyWrapper (mock checkpoint)...")
try:
    import tempfile
    from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
    
    # Create a mock checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "mock_model.pt"
        
        # Build a real encoder and head, save their state dicts
        encoder = TinyMultiCamEncoder(out_dim=128)
        encoder_state = encoder.state_dict()
        
        head = WaypointHead(torch=torch, in_dim=128, horizon_steps=20)
        head_state = head.state_dict()
        
        torch.save({
            "encoder": encoder_state,
            "head": head_state,
            "cam": "front",
            "horizon_steps": 20,
            "out_dim": 128,
        }, ckpt_path)
        
        # Load with the wrapper
        wrapper = WaypointPolicyWrapper(str(ckpt_path), device="cpu")
        assert wrapper.cam == "front"
        assert wrapper.horizon_steps == 20
        
        # Test prediction with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        waypoints = wrapper.predict(dummy_image)
        assert waypoints.shape == (20, 2), f"Expected shape (20, 2), got {waypoints.shape}"
        
        print(f"  ✓ WaypointPolicyWrapper: checkpoint loading OK, predict shape={waypoints.shape}")
except Exception as e:
    print(f"  ✗ WaypointPolicyWrapper test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test backward compatibility
print("[5/5] Testing backward compatibility (WaypointBCModelWrapper)...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "mock_model.pt"
        
        encoder = TinyMultiCamEncoder(out_dim=128)
        encoder_state = encoder.state_dict()
        
        head = WaypointHead(torch=torch, in_dim=128, horizon_steps=20)
        head_state = head.state_dict()
        
        torch.save({
            "encoder": encoder_state,
            "head": head_state,
            "cam": "front",
            "horizon_steps": 20,
            "out_dim": 128,
        }, ckpt_path)
        
        # Load with backward-compatible wrapper
        wrapper = WaypointBCModelWrapper(str(ckpt_path), device="cpu")
        
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        waypoints = wrapper.predict(dummy_image)
        assert waypoints.shape == (20, 2)
        
        print(f"  ✓ WaypointBCModelWrapper: backward compatibility OK")
except Exception as e:
    print(f"  ✗ Backward compatibility test failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("All smoke tests passed! ✓")
print("=" * 60)
print()
print("Summary:")
print("  - WaypointHead: SFT waypoint prediction network")
print("  - DeltaWaypointHead: RL delta correction network")
print("  - WaypointPolicyWrapper: Unified wrapper for SFT + RL models")
print("  - WaypointBCModelWrapper: Backward compatible wrapper")
print()
print("To run full CARLA evaluation:")
print("  python -m training.eval.run_carla_closed_loop_eval \\")
print("    --checkpoint out/sft_waypoint_bc/model.pt \\")
print("    --output-dir out/carla_eval")
print()
print("For RL models (SFT + delta):")
print("  python -m training.eval.run_carla_closed_loop_eval \\")
print("    --checkpoint out/rl_delta/model.pt \\")
print("    --output-dir out/carla_rl_eval \\")
print("    --rl-mode")

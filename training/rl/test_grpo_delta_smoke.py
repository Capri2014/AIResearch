#!/usr/bin/env python3
"""
Smoke test for GRPO delta-waypoint training.

Validates the training pipeline without full training.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.train_grpo_delta_waypoint import (
    GRPODeltaWaypointConfig,
    DeltaWaypointHead,
    GRPODeltaWaypointModel,
)
from training.rl.sft_checkpoint_loader import SFTModelWrapper
from training.rl.toy_waypoint_env import ToyWaypointEnv, ToyWaypointConfig


def test_delta_head():
    """Test delta head forward pass."""
    print("Testing DeltaWaypointHead...")
    
    feature_dim = 256
    waypoint_dim = 3
    n_waypoints = 10
    hidden_dim = 128
    
    delta_head = DeltaWaypointHead(
        feature_dim=feature_dim,
        waypoint_dim=waypoint_dim,
        n_waypoints=n_waypoints,
        hidden_dim=hidden_dim,
    )
    
    # Test forward pass
    batch_size = 4
    features = torch.randn(batch_size, feature_dim)
    
    delta, uncertainty = delta_head(features)
    
    assert delta.shape == (batch_size, n_waypoints, waypoint_dim), \
        f"Expected shape ({batch_size}, {n_waypoints}, {waypoint_dim}), got {delta.shape}"
    assert uncertainty.shape == delta.shape, \
        f"Uncertainty shape mismatch: {uncertainty.shape} vs {delta.shape}"
    
    print(f"  ✓ Delta shape: {delta.shape}")
    print(f"  ✓ Uncertainty shape: {uncertainty.shape}")
    print("  DeltaHead forward pass: PASSED")


def test_sft_model_wrapper():
    """Test SFT model wrapper."""
    print("\nTesting SFTModelWrapper...")
    
    # Create mock SFT model
    mock_model = torch.nn.Linear(256, 30)  # 10 waypoints * 3
    config = {"hidden_dim": 256, "waypoint_dim": 3}
    
    sft_wrapper = SFTModelWrapper(mock_model, config)
    
    # Test forward pass
    features = torch.randn(4, 256)
    waypoints = sft_wrapper(features)
    
    assert waypoints.shape[0] == 4, f"Expected batch size 4, got {waypoints.shape[0]}"
    
    print(f"  ✓ Waypoints shape: {waypoints.shape}")
    print("  SFTModelWrapper forward pass: PASSED")


def test_combined_model():
    """Test combined SFT + Delta model."""
    print("\nTesting GRPODeltaWaypointModel...")
    
    # Create mock SFT model
    mock_sft = torch.nn.Linear(256, 30)
    sft_wrapper = SFTModelWrapper(mock_sft, {"hidden_dim": 256, "waypoint_dim": 3})
    
    # Create delta head
    delta_head = DeltaWaypointHead(
        feature_dim=256,
        waypoint_dim=3,
        n_waypoints=10,
        hidden_dim=128,
    )
    
    # Create combined model
    model = GRPODeltaWaypointModel(sft_wrapper, delta_head)
    
    # Test forward pass
    features = torch.randn(4, 256)
    final_waypoints, sft_waypoints = model(features, return_sft=True)
    
    assert final_waypoints.shape == (4, 10, 3), \
        f"Expected final waypoints shape (4, 10, 3), got {final_waypoints.shape}"
    assert sft_waypoints.shape == (4, 10, 3), \
        f"Expected SFT waypoints shape (4, 10, 3), got {sft_waypoints.shape}"
    
    # Verify SFT is frozen
    assert not any(p.requires_grad for p in sft_wrapper.parameters()), \
        "SFT model should be frozen"
    
    # Verify delta head is trainable
    assert any(p.requires_grad for p in delta_head.parameters()), \
        "Delta head should be trainable"
    
    print(f"  ✓ Final waypoints shape: {final_waypoints.shape}")
    print(f"  ✓ SFT waypoints shape: {sft_waypoints.shape}")
    print("  Combined model: PASSED")


def test_environment():
    """Test toy waypoint environment."""
    print("\nTesting ToyWaypointEnv...")
    
    config = ToyWaypointConfig(
        horizon=16,
        n_waypoints=10,
        noise_scale=0.5,
        seed=42,
    )
    
    env = ToyWaypointEnv(config)
    
    # Test reset
    state, info = env.reset()
    assert state.shape[0] == config.horizon * config.n_waypoints * config.waypoint_dim, \
        f"Expected state shape {config.horizon * config.n_waypoints * config.waypoint_dim}, got {state.shape[0]}"
    
    # Test step
    action = np.random.randn(config.n_waypoints * config.waypoint_dim)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"
    assert isinstance(terminated, bool), f"Terminated should be bool, got {type(terminated)}"
    
    print(f"  ✓ State shape: {state.shape}")
    print(f"  ✓ Action shape: {action.shape}")
    print(f"  ✓ Reward: {reward:.3f}")
    print("  ToyWaypointEnv: PASSED")


def test_training_config():
    """Test training configuration."""
    print("\nTesting GRPODeltaWaypointConfig...")
    
    config = GRPODeltaWaypointConfig(
        sft_checkpoint="out/waypoint_bc/test/model.pt",
        delta_hidden_dim=128,
        grpo_group_size=4,
        num_episodes=10,
        output_dir="out/grpo_delta_test",
    )
    
    assert config.delta_hidden_dim == 128
    assert config.grpo_group_size == 4
    assert config.num_episodes == 10
    
    print(f"  ✓ Config: delta_hidden_dim={config.delta_hidden_dim}")
    print(f"  ✓ Config: grpo_group_size={config.grpo_group_size}")
    print("  Configuration: PASSED")


def main():
    """Run all smoke tests."""
    print("="*60)
    print("GRPO Delta-Waypoint Training - Smoke Tests")
    print("="*60)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_training_config()
        test_delta_head()
        test_sft_model_wrapper()
        test_combined_model()
        test_environment()
        
        print("\n" + "="*60)
        print("All smoke tests PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

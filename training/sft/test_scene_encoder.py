"""
Test script for SceneTransformerEncoder with dummy data.
Verifies forward pass, gradient flow, and output shapes.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/data/.openclaw/workspace')

from training.sft.scene_encoder import (
    SceneTransformerEncoder,
    SceneTransformerWithWaypointHead,
    SceneEncoderConfig,
)


def create_dummy_data(batch_size=2, num_agents=8, num_history=20, num_map_points=20, points_per_polyline=10):
    """Create dummy inputs for testing."""
    # (B, A, T, 7) - agent history: x, y, heading, speed, type, length, width
    agent_history = torch.randn(batch_size, num_agents, num_history, 7)
    # Mark first 3 agents as valid
    agent_masks = torch.zeros(batch_size, num_agents, num_history, dtype=torch.bool)
    agent_masks[:, :3, :] = True
    
    # (B, P, M, 3) - map polylines: x, y, is_endpoint
    map_polylines = torch.randn(batch_size, num_map_points, points_per_polyline, 3)
    # Mark first 10 polylines as valid lanes
    polyline_masks = torch.zeros(batch_size, num_map_points, points_per_polyline, dtype=torch.bool)
    polyline_masks[:, :10, :] = True
    
    return agent_history, map_polylines, agent_masks, polyline_masks


def test_encoder_forward():
    """Test encoder forward pass."""
    print("=" * 60)
    print("Testing SceneTransformerEncoder forward pass...")
    print("=" * 60)
    
    config = SceneEncoderConfig(
        num_agents=8,
        num_map_points=20,
        num_history=20,
        hidden_dim=128,
        output_dim=128,
        num_heads=4,
        num_layers=2,
    )
    
    encoder = SceneTransformerEncoder(config)
    encoder.eval()
    
    agent_history, map_polylines, agent_masks, polyline_masks = create_dummy_data()
    
    with torch.no_grad():
        outputs = encoder(agent_history, map_polylines, agent_masks, polyline_masks)
    
    print(f"Agent embeddings shape: {outputs['agent_embeddings'].shape}")
    print(f"Map embeddings shape: {outputs['map_embeddings'].shape}")
    print(f"Scene embedding shape: {outputs['scene_embedding'].shape}")
    
    assert outputs['agent_embeddings'].shape == (2, 8, 128), "Wrong agent embeddings shape"
    assert outputs['scene_embedding'].shape == (2, 128), "Wrong scene embedding shape"
    
    print("✓ Encoder forward pass passed!\n")
    return True


def test_full_model_forward():
    """Test full model with waypoint head."""
    print("=" * 60)
    print("Testing SceneTransformerWithWaypointHead forward pass...")
    print("=" * 60)
    
    # Use 512 hidden_dim to match proposal head expectations
    config = SceneEncoderConfig(
        num_agents=8,
        num_map_points=20,
        num_history=20,
        hidden_dim=512,
        output_dim=256,
        num_heads=8,
        num_layers=2,
    )
    
    model = SceneTransformerWithWaypointHead(
        config, 
        use_proposal_head=True,
        num_proposals=5,
        horizon_steps=20
    )
    model.eval()
    
    agent_history, map_polylines, agent_masks, polyline_masks = create_dummy_data()
    
    with torch.no_grad():
        outputs = model(agent_history, map_polylines, agent_masks, polyline_masks)
    
    print(f"Proposals shape: {outputs['proposals'].shape}")
    print(f"Scores shape: {outputs['scores'].shape}")
    
    # Proposals: (B, K, H, 2) where K=5 modes, H=20 horizon
    assert outputs['proposals'].shape == (2, 5, 20, 2), f"Wrong proposals shape: {outputs['proposals'].shape}"
    assert outputs['scores'].shape == (2, 5), "Wrong scores shape"
    
    print("✓ Full model forward pass passed!\n")
    return True


def test_gradient_flow():
    """Test gradient flow through the model."""
    print("=" * 60)
    print("Testing gradient flow...")
    print("=" * 60)
    
    config = SceneEncoderConfig(
        num_agents=8,
        num_map_points=20,
        num_history=10,
        hidden_dim=256,
        output_dim=128,
        num_heads=4,
        num_layers=1,
    )
    
    model = SceneTransformerWithWaypointHead(
        config, 
        use_proposal_head=True,
        num_proposals=3,
        horizon_steps=10
    )
    
    agent_history, map_polylines, agent_masks, polyline_masks = create_dummy_data(
        batch_size=2, num_agents=8, num_history=10, num_map_points=20, points_per_polyline=8
    )
    
    outputs = model(agent_history, map_polylines, agent_masks, polyline_masks)
    
    # Compute dummy loss
    loss = outputs['scores'].mean() + outputs['proposals'].mean()
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"  {name}: grad norm = {param.grad.norm().item():.4f}")
    
    assert has_grad, "No gradients computed!"
    print("✓ Gradient flow passed!\n")
    return True


def test_regression_mode():
    """Test model in regression mode (no proposal head)."""
    print("=" * 60)
    print("Testing regression mode...")
    print("=" * 60)
    
    config = SceneEncoderConfig(
        num_agents=8,
        num_map_points=20,
        num_history=20,
        hidden_dim=128,
        output_dim=128,
    )
    
    model = SceneTransformerWithWaypointHead(config, use_proposal_head=False)
    model.eval()
    
    agent_history, map_polylines, agent_masks, polyline_masks = create_dummy_data()
    
    with torch.no_grad():
        outputs = model(agent_history, map_polylines, agent_masks, polyline_masks)
    
    print(f"Waypoints shape: {outputs['waypoints'].shape}")
    
    # Waypoints: (B, H, 2) where H=20 horizon
    assert outputs['waypoints'].shape == (2, 20, 2), "Wrong waypoints shape"
    
    print("✓ Regression mode passed!\n")
    return True


def test_config_defaults():
    """Test config defaults."""
    print("=" * 60)
    print("Testing config defaults...")
    print("=" * 60)
    
    config = SceneEncoderConfig()
    print(f"  num_agents: {config.num_agents}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  output_dim: {config.output_dim}")
    
    assert config.num_agents == 32
    assert config.hidden_dim == 256
    assert config.num_heads == 8
    
    print("✓ Config defaults passed!\n")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Scene Transformer Encoder Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_config_defaults,
        test_encoder_forward,
        test_full_model_forward,
        test_gradient_flow,
        test_regression_mode,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

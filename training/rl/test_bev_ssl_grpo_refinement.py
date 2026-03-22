"""Smoke test for BEV SSL GRPO refinement module."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from training.rl.bev_ssl_grpo_refinement import (
    BEVSSLGRPOConfig,
    BEVSSLDeltaWaypointHead,
    BEVSSLGRPOAgent,
    BEVSSLGRPOTrainer,
    WaypointRefinementEnv,
)


def test_delta_head():
    """Test delta waypoint head."""
    print("Testing BEVSSLDeltaWaypointHead...")
    
    head = BEVSSLDeltaWaypointHead(
        bev_feature_dim=128,
        num_waypoints=8,
        waypoint_dim=2,
        hidden_dim=64,
        delta_scale=3.0,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in head.parameters())
    print(f"  Delta head params: {num_params:,}")
    
    # Forward pass
    bev_features = torch.randn(4, 128)
    deltas = head(bev_features)
    
    assert deltas.shape == (4, 8, 2), f"Expected (4, 8, 2), got {deltas.shape}"
    assert deltas.abs().max() <= 3.0, "Deltas exceed delta_scale"
    
    print("  Forward pass: ✅")
    print()


def test_bev_ssl_grpo_agent():
    """Test combined BC + delta agent."""
    print("Testing BEVSSLGRPOAgent...")
    
    from training.pretrain.bev_encoder import BEVEncoder, BEVConfig, create_bev_encoder
    
    # Create BEV encoder
    bev_config = BEVConfig(
        encoder_dim=128,
        bev_channels=128,
        fusion_method="concat",
        num_cameras=6,
    )
    bev_encoder = create_bev_encoder(bev_config)
    
    # Create agent
    delta_head = BEVSSLDeltaWaypointHead(
        bev_feature_dim=128,
        num_waypoints=8,
        waypoint_dim=2,
    )
    
    agent = BEVSSLGRPOAgent(
        bev_encoder=bev_encoder,
        bc_model=None,
        delta_head=delta_head,
        freeze_bc=True,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"  Agent params: {num_params:,}")
    
    # Forward pass with dummy data
    batch_size = 4
    num_cameras = 6
    H, W = 256, 256
    
    images = torch.randn(batch_size, num_cameras, 3, H, W)
    lidars = torch.randn(batch_size, 100, 3)  # Dummy LiDAR points
    bc_waypoints = torch.randn(batch_size, 8, 2)
    
    final_waypoints, deltas = agent(images, lidars=lidars, bc_waypoints=bc_waypoints)
    
    assert final_waypoints.shape == (batch_size, 8, 2)
    assert deltas.shape == (batch_size, 8, 2)
    
    print("  Forward pass: ✅")
    print()


def test_config():
    """Test configuration."""
    print("Testing BEVSSLGRPOConfig...")
    
    config = BEVSSLGRPOConfig(
        bev_encoder_type="concat",
        bev_hidden_dim=128,
        episodes=100,
        batch_size=32,
        output_dir="out/test_grpo",
    )
    
    assert config.bev_encoder_type == "concat"
    assert config.bev_hidden_dim == 128
    assert config.episodes == 100
    
    print("  Config: ✅")
    print()


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("BEV SSL GRPO Refinement Smoke Test")
    print("=" * 60)
    print()
    
    test_config()
    test_delta_head()
    test_bev_ssl_grpo_agent()
    
    print("=" * 60)
    print("All smoke tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()

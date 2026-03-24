#!/usr/bin/env python3
"""Smoke test for BEV SSL PPO refinement module."""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rl.bev_ssl_ppo_refinement import (
    BEVSSLPPORefineConfig,
    StubWaypointPredictor,
    WaypointPolicyHead,
    ValueNetwork,
    KinematicWaypointEnv,
    PPORefineAgent,
    train_bev_ssl_ppo_refinement,
)


def test_config():
    """Test config creation."""
    print("Testing config creation...")
    config = BEVSSLPPORefineConfig(
        output_dir="out/test_ppo_refine",
        episodes=5,
    )
    assert config.num_waypoints == 8
    assert config.gamma == 0.99
    print("  ✓ Config creation")


def test_stub_predictor():
    """Test stub BC model."""
    print("Testing stub predictor...")
    config = BEVSSLPPORefineConfig()
    predictor = StubWaypointPredictor(config)
    
    state = [0.0, 0.0, 0.0, 5.0]  # x, y, heading, speed
    waypoints = predictor.predict_waypoints(state)
    
    assert waypoints.shape == (config.num_waypoints, 2)
    print(f"  ✓ Stub predictor: {waypoints.shape}")


def test_policy_head():
    """Test waypoint policy head."""
    print("Testing policy head...")
    import torch
    
    policy_head = WaypointPolicyHead(
        state_dim=21,
        hidden_dims=[64, 32],
    )
    
    # Test forward
    state = torch.randn(2, 21)
    action, std = policy_head(state)
    
    assert action.shape == (2, 2)
    print(f"  ✓ Policy head forward: {action.shape}")
    
    # Test get_action
    sampled, log_prob = policy_head.get_action(state)
    assert sampled.shape == (2, 2)
    print(f"  ✓ Policy head action: {sampled.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in policy_head.parameters())
    print(f"  ✓ Policy head params: {num_params}")


def test_value_network():
    """Test value network."""
    print("Testing value network...")
    import torch
    
    value_net = ValueNetwork(state_dim=21)
    state = torch.randn(4, 21)
    value = value_net(state)
    
    assert value.shape == (4, 1)
    print(f"  ✓ Value network: {value.shape}")
    
    num_params = sum(p.numel() for p in value_net.parameters())
    print(f"  ✓ Value network params: {num_params}")


def test_environment():
    """Test kinematic environment."""
    print("Testing environment...")
    import numpy as np
    
    config = BEVSSLPPORefineConfig()
    bc_model = StubWaypointPredictor(config)
    env = KinematicWaypointEnv(config, bc_model)
    
    # Reset
    state = env.reset()
    assert state.shape == (config.num_waypoints * 2 + 5,)
    print(f"  ✓ Environment reset: {state.shape}")
    
    # Step
    state, reward, done, info = env.step(0.1, 0.5)
    assert state.shape == (config.num_waypoints * 2 + 5,)
    print(f"  ✓ Environment step: reward={reward:.2f}, done={done}")
    
    # Run episode
    state = env.reset()
    total_reward = 0
    for _ in range(10):
        action = [0.1, 0.5]  # [steer, throttle]
        state, reward, done, info = env.step(action[0], action[1])
        total_reward += reward
        if done:
            break
    
    print(f"  ✓ Episode: reward={total_reward:.2f}, ade={info.get('ade', 0):.2f}")


def test_agent():
    """Test PPO agent."""
    print("Testing PPO agent...")
    import numpy as np
    import torch
    
    config = BEVSSLPPORefineConfig(episodes=3, batch_size=10)
    bc_model = StubWaypointPredictor(config)
    agent = PPORefineAgent(config, bc_model)
    
    # Get action
    state = np.zeros(config.num_waypoints * 2 + 5)
    action, log_prob, value = agent.get_action(state)
    
    assert action.shape == (2,)
    print(f"  ✓ Agent action: {action}")
    
    # Store transition
    agent.store_transition(state, action, log_prob, 0.0, False, value)
    assert len(agent.states) == 1
    print(f"  ✓ Agent memory: {len(agent.states)} transitions")
    
    # Run short training
    env = KinematicWaypointEnv(config, bc_model)
    state = env.reset()
    
    for _ in range(15):
        action, log_prob, value = agent.get_action(state)
        next_state, reward, done, info = env.step(action[0], action[1])
        agent.store_transition(state, action, log_prob, reward, done, value)
        state = next_state
        if done:
            break
    
    # Update
    if len(agent.states) >= config.batch_size:
        update_metrics = agent.update()
        print(f"  ✓ Agent update: policy_loss={update_metrics.get('policy_loss', 0):.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.policy_head.parameters())
    total_params += sum(p.numel() for p in agent.value_net.parameters())
    print(f"  ✓ Agent total params: {total_params}")


def test_training():
    """Test full training loop."""
    print("Testing training loop...")
    
    config = BEVSSLPPORefineConfig(
        output_dir="out/test_ppo_refine",
        episodes=10,
        eval_interval=5,
        save_interval=10,
    )
    
    summary = train_bev_ssl_ppo_refinement(config, test_mode=True)
    
    assert os.path.exists(config.output_dir)
    assert os.path.exists(os.path.join(config.output_dir, "final.pt"))
    assert os.path.exists(os.path.join(config.output_dir, "metrics.json"))
    
    print(f"  ✓ Training complete: {summary}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("BEV SSL PPO Refinement Smoke Test")
    print("="*60 + "\n")
    
    try:
        test_config()
        test_stub_predictor()
        test_policy_head()
        test_value_network()
        test_environment()
        test_agent()
        test_training()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

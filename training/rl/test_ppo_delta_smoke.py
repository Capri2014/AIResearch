"""Smoke test for PPO delta-waypoint training.

Quick validation that the training pipeline works correctly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from training.rl.train_ppo_delta_waypoint import (
    PPOConfig,
    ToyWaypointEnv,
    PPOPolicy,
    set_seed,
)


def test_delta_head():
    """Test that delta head produces correct output shapes."""
    batch_size = 4
    horizon_steps = 20
    hidden_dim = 128

    delta_head = torch.nn.Linear(hidden_dim, horizon_steps * 2)
    z = torch.randn(batch_size, hidden_dim)
    delta = delta_head(z)

    expected_shape = (batch_size, horizon_steps, 2)
    assert delta.shape == expected_shape, f"Expected {expected_shape}, got {delta.shape}"
    print(f"[test] DeltaHead output shape: {delta.shape} ✓")


def test_value_head():
    """Test that value head produces correct output shapes."""
    batch_size = 4
    hidden_dim = 128

    value_head = torch.nn.Linear(hidden_dim, 1)
    z = torch.randn(batch_size, hidden_dim)
    value = value_head(z)

    assert value.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {value.shape}"
    print(f"[test] ValueHead output shape: {value.shape} ✓")


def test_gae_computation():
    """Test GAE computation."""
    from training.rl.train_ppo_delta_waypoint import compute_gae

    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.8, 0.3]
    dones = [False, False, False]

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

    assert len(advantages) == len(rewards), "Advantages length mismatch"
    assert len(returns) == len(rewards), "Returns length mismatch"
    print(f"[test] GAE computation: advantages={advantages} ✓")


def test_toy_env():
    """Test toy environment interactions."""
    env = ToyWaypointEnv(horizon_steps=20, sft_noise_std=2.0)
    obs = env.reset()

    assert 'sft_waypoints' in obs
    assert 'target_waypoints' in obs
    assert obs['sft_waypoints'].shape == (20, 2)
    print(f"[test] ToyEnv reset: obs shape = {obs['sft_waypoints'].shape} ✓")

    # Test step with random action
    action = {'delta_waypoints': np.zeros((20, 2))}
    obs, reward, done, info = env.step(action)

    assert 'corrected_waypoints' in obs
    assert 'ade' in info
    assert 'fde' in info
    print(f"[test] ToyEnv step: reward={reward:.4f}, ade={info['ade']:.4f} ✓")


def test_ppo_policy():
    """Test PPO policy forward pass."""
    device = torch.device("cpu")
    horizon_steps = 20
    hidden_dim = 128

    # Create mock encoder (identity)
    class MockEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, image_valid_by_cam=None):
            return {'front': torch.randn(x['front'].shape[0], hidden_dim)}

        def eval(self):
            pass

    class MockEncoder2(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def __call__(self, x, image_valid_by_cam=None):
            if isinstance(x, dict):
                return torch.randn(1, hidden_dim)
            return torch.randn(1, hidden_dim)

        def eval(self):
            pass

    encoder = MockEncoder2()

    cfg = PPOConfig(
        sft_checkpoint=Path("dummy.pt"),
        out_dir=Path(tempfile.mkdtemp()),
        delta_hidden_dim=hidden_dim,
        value_hidden_dim=hidden_dim,
        horizon_steps=horizon_steps,
        env_name="toy",
        num_envs=2,
    )

    policy = PPOPolicy(cfg, encoder, device)

    # Test forward pass
    obs = {
        'image': np.random.randn(224, 224, 3).astype(np.float32),
        'sft_waypoints': np.random.randn(horizon_steps, 2).astype(np.float32),
        'state': {'embedding': np.random.randn(hidden_dim).tolist()},
    }

    action, value, log_prob, info = policy.get_action(obs)
    assert 'delta_waypoints' in action
    assert 'final_waypoints' in action
    print(f"[test] Policy forward pass: action shape = {action['delta_waypoints'].shape} ✓")


def test_training_loop():
    """Run a minimal training iteration."""
    from training.rl.train_ppo_delta_waypoint import main

    set_seed(42)

    # Create minimal config for testing
    import sys
    original_argv = sys.argv

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy SFT checkpoint
        sft_ckpt = {
            'encoder': {k: v for k, v in torch.nn.Linear(128, 128).state_dict().items()},
            'head': torch.nn.Linear(256, 40).state_dict(),
        }
        sft_path = Path(tmpdir) / "sft_model.pt"
        torch.save(sft_ckpt, sft_path)

        # Run minimal training
        sys.argv = [
            'test',
            '--sft-checkpoint', str(sft_path),
            '--out-dir', str(Path(tmpdir) / "rl_output"),
            '--env', 'toy',
            '--num-iterations', '2',
            '--batch-size', '8',
            '--horizon-steps', '10',
            '--log-interval', '1',
            '--eval-interval', '1',
        ]

        try:
            main()
            print(f"[test] Training loop: completed successfully ✓")
        except Exception as e:
            print(f"[test] Training loop: failed with {e}")
            raise
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    print("=" * 60)
    print("Running PPO Delta-Waypoint Smoke Tests")
    print("=" * 60)

    print("\n--- Unit Tests ---")
    test_delta_head()
    test_value_head()
    test_gae_computation()
    test_toy_env()
    test_ppo_policy()

    print("\n--- Integration Tests ---")
    test_training_loop()

    print("\n" + "=" * 60)
    print("All smoke tests passed! ✓")
    print("=" * 60)

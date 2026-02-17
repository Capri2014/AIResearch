#!/usr/bin/env python3
"""Smoke test for RL delta-waypoint training.

This script verifies that the RL refinement pipeline works correctly:
1. Environment initialization
2. PPO agent interaction
3. Training loop execution
4. Metrics and checkpoint saving
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import shutil

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.train_rl_delta_waypoint import (
    TrainingConfig,
    WaypointEnvConfig,
    PPOConfig,
    RLDeltaTrainer,
)


def run_smoke_test():
    """Run a quick smoke test of the RL delta-waypoint training pipeline."""
    print("[smoke] Starting RL delta-waypoint smoke test...")
    
    # Create temp output directory
    out_dir = Path(tempfile.mkdtemp(prefix="rl_delta_smoke_"))
    
    try:
        # Configuration for smoke test
        env_cfg = WaypointEnvConfig(
            world_size=50.0,
            horizon_steps=5,  # Small for speed
            waypoint_spacing=5.0,
            max_episode_steps=20,  # Short episodes
            target_reach_radius=3.0,
        )
        
        ppo_cfg = PPOConfig(
            encoder_out_dim=1 + 5 * 2,  # speed + 5 waypoints * 2 coords
            horizon_steps=5,
            hidden_dim=32,
            episodes=10,  # Very short
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            clip_ratio=0.2,
            update_epochs=2,
            batch_size=32,
            eval_interval=5,
            save_interval=10,
            device="cpu",
            seed=42,
        )
        
        cfg = TrainingConfig(
            out_dir=out_dir,
            env=env_cfg,
            ppo=ppo_cfg,
        )
        
        # Run training
        trainer = RLDeltaTrainer(cfg)
        result = trainer.train()
        
        # Verify outputs
        print("[smoke] Verifying outputs...")
        
        # Check config
        assert (out_dir / "config.json").exists(), "config.json not found"
        print("✓ config.json created")
        
        # Check metrics
        assert (out_dir / "metrics.json").exists(), "metrics.json not found"
        with open(out_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert len(metrics) > 0, "metrics.json is empty"
        print(f"✓ metrics.json created ({len(metrics)} entries)")
        
        # Check train summary
        assert (out_dir / "train_metrics.json").exists(), "train_metrics.json not found"
        with open(out_dir / "train_metrics.json") as f:
            summary = json.load(f)
        assert "rewards" in summary, "train_metrics.json missing rewards"
        assert len(summary["rewards"]) == ppo_cfg.episodes, "reward count mismatch"
        print(f"✓ train_metrics.json created ({len(summary['rewards'])} episodes)")
        
        # Check final checkpoint
        assert (out_dir / "final.pt").exists(), "final.pt not found"
        print("✓ final.pt checkpoint created")
        
        # Check checkpoints directory
        checkpoints_dir = out_dir / "checkpoints"
        assert checkpoints_dir.exists(), "checkpoints directory not found"
        print(f"✓ checkpoints directory created ({len(list(checkpoints_dir.glob('*.pt')))} checkpoints)")
        
        # Verify metrics content
        latest_metrics = metrics[-1]
        assert "episode" in latest_metrics, "metrics missing episode"
        assert "mean_reward" in latest_metrics, "metrics missing mean_reward"
        assert "update" in latest_metrics, "metrics missing update info"
        print(f"✓ metrics contain episode={latest_metrics['episode']}, reward={latest_metrics['mean_reward']:.2f}")
        
        # Print final results
        print("\n" + "="*50)
        print("SMOKE TEST PASSED ✓")
        print("="*50)
        print(f"Output directory: {out_dir}")
        print(f"Episodes trained: {len(summary['rewards'])}")
        print(f"Final avg reward: {summary['final_metrics']['mean_reward_100ep']:.2f}")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n[smoke] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print(f"\n[smoke] Cleaning up temp directory: {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

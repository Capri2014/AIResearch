#!/usr/bin/env python3
"""
Quick Evaluation Runner for RL After SFT Pipeline.

Runs deterministic evaluation comparing SFT-only vs RL-refined policy
on the toy waypoint environment. Auto-detects latest checkpoints.
Integrates with AdvancedCheckpointSelector for smart checkpoint selection.

Usage:
    python eval_quick.py                    # Auto-find latest checkpoints
    python eval_quick.py --smoke             # Quick smoke test (5 episodes)
    python eval_quick.py --episodes 20       # Full evaluation (20 episodes)
    python eval_quick.py --checkpoint <path> # Use specific checkpoint
    python eval_quick.py --best-ade          # Select best checkpoint by ADE
    python eval_quick.py --best-composite    # Select best by composite score
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add training/rl to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_metrics import find_latest_checkpoint, find_latest_sft_checkpoint
from checkpoint_selector_advanced import AdvancedCheckpointSelector


def find_best_checkpoint(criterion: str = 'ade') -> str:
    """
    Find best checkpoint based on evaluation metrics.
    
    Args:
        criterion: 'ade', 'fde', 'success', or 'composite'
    
    Returns:
        Path to best checkpoint
    """
    selector = AdvancedCheckpointSelector('out/')
    
    if criterion == 'ade':
        best = selector.select_by_ade()
    elif criterion == 'fde':
        best = selector.select_by_fde()
    elif criterion == 'success':
        best = selector.select_by_success()
    elif criterion == 'composite':
        best = selector.select_composite()
    else:
        return None
    
    if best and best[0].checkpoint_path:
        return best[0].checkpoint_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Quick evaluation runner for RL after SFT'
    )
    parser.add_argument('--smoke', action='store_true',
                        help='Run smoke test with 5 episodes')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to RL checkpoint (auto-detected if not specified)')
    parser.add_argument('--sft-checkpoint', type=str, default=None,
                        help='Path to SFT checkpoint (auto-detected if not specified)')
    parser.add_argument('--output-dir', type=str, default='out/eval',
                        help='Output directory')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Waypoint horizon')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--best-ade', action='store_true',
                        help='Select best checkpoint by ADE (requires eval metrics)')
    parser.add_argument('--best-fde', action='store_true',
                        help='Select best checkpoint by FDE')
    parser.add_argument('--best-success', action='store_true',
                        help='Select best checkpoint by success rate')
    parser.add_argument('--best-composite', action='store_true',
                        help='Select best checkpoint by composite score')
    parser.add_argument('--show-checkpoints', action='store_true',
                        help='Show available checkpoints with metrics')
    args = parser.parse_args()
    
    # Show available checkpoints
    if args.show_checkpoints:
        selector = AdvancedCheckpointSelector('out/')
        checkpoints = selector.get_all_checkpoints()
        print(f"\nAvailable checkpoints with evaluation metrics ({len(checkpoints)}):\n")
        for c in checkpoints[:10]:
            m = c.eval_metrics
            if m:
                print(f"  {c.run_id}")
                print(f"    ADE: {m.ade_mean:.3f}m, FDE: {m.fde_mean:.3f}m, Success: {m.success_rate:.1%}")
                if c.checkpoint_path:
                    print(f"    Path: {c.checkpoint_path}")
                print()
        return 0
    
    # Smart checkpoint selection based on eval metrics
    if args.best_ade or args.best_fde or args.best_success or args.best_composite:
        criterion = 'ade' if args.best_ade else ('fde' if args.best_fde else ('success' if args.best_success else 'composite'))
        print(f"Finding best checkpoint by {criterion.upper()}...")
        best_path = find_best_checkpoint(criterion)
        if best_path:
            args.checkpoint = best_path
            print(f"  Selected: {best_path}")
        else:
            print("  No evaluated checkpoint found, falling back to auto-detect")
    
    # Auto-detect checkpoint if not specified
    if args.checkpoint is None:
        print("Auto-detecting latest RL checkpoint...")
        ckpt, meta = find_latest_checkpoint()
        if ckpt:
            args.checkpoint = ckpt
            print(f"  Found: {ckpt}")
            print(f"  Modified: {meta.get('modified', 'unknown')}")
            if 'metrics' in meta:
                m = meta['metrics']
                print(f"  RL metrics: avg_reward={m.get('rl_metrics', {}).get('final_avg_reward', 'N/A'):.2f}")
        else:
            print("No RL checkpoint found, using random weights")
    
    # Build command
    cmd = [
        sys.executable, '-m', 'training.rl.eval_waypoint_rl',
        '--output-dir', args.output_dir,
        '--seed-base', str(args.seed_base),
        '--horizon', str(args.horizon)
    ]
    
    if args.smoke:
        cmd.append('--smoke')
    else:
        cmd.extend(['--num-episodes', str(args.episodes)])
    
    if args.checkpoint:
        cmd.extend(['--checkpoint', args.checkpoint])
    
    if args.verbose:
        cmd.append('--verbose')
    
    print(f"\nRunning evaluation...")
    print(f"  Command: {' '.join(cmd)}\n")
    
    # Run evaluation
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    return result.returncode


if __name__ == '__main__':
    exit(main())

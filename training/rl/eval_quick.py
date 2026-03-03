#!/usr/bin/env python3
"""
Quick Evaluation Runner for RL After SFT Pipeline.

Runs deterministic evaluation comparing SFT-only vs RL-refined policy
on the toy waypoint environment. Auto-detects latest checkpoints.

Usage:
    python eval_quick.py                    # Auto-find latest checkpoints
    python eval_quick.py --smoke             # Quick smoke test (5 episodes)
    python eval_quick.py --episodes 20       # Full evaluation (20 episodes)
    python eval_quick.py --checkpoint <path> # Use specific checkpoint
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add training/rl to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_metrics import find_latest_checkpoint, find_latest_sft_checkpoint


# Get workspace root (parent of AIResearch-repo)
def get_workspace_root():
    current_file = os.path.abspath(__file__)
    # training/rl/eval_quick.py -> AIResearch-repo
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    # AIResearch-repo -> workspace
    return os.path.dirname(repo_root)


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
    args = parser.parse_args()
    
    # Auto-detect checkpoint if not specified
    if args.checkpoint is None:
        print("Auto-detecting latest RL checkpoint...")
        workspace = get_workspace_root()
        ckpt, meta = find_latest_checkpoint(base_dir=os.path.join(workspace, 'out'))
        if ckpt:
            args.checkpoint = ckpt
            print(f"  Found: {ckpt}")
            print(f"  Modified: {meta.get('modified', 'unknown')}")
            if 'metrics' in meta:
                m = meta['metrics']
                print(f"  RL metrics: avg_reward={m.get('rl_metrics', {}).get('final_avg_reward', 'N/A'):.2f}")
        else:
            print("No RL checkpoint found, using random weights")
    
    # Determine working directory (use workspace root for eval script)
    workspace = get_workspace_root()
    eval_script = os.path.join(workspace, 'training', 'rl', 'eval_waypoint_rl.py')
    work_dir = workspace
    
    # Build command
    cmd = [
        sys.executable, eval_script,
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
    print(f"  Working directory: {work_dir}")
    
    # Run evaluation
    result = subprocess.run(cmd, cwd=work_dir)
    
    return result.returncode


if __name__ == '__main__':
    exit(main())

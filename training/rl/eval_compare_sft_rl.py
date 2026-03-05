#!/usr/bin/env python3
"""
Loader script to compare SFT-only vs RL-refined policy evaluation results.

Loads metrics from eval output and prints a 3-line summary report.

Usage:
    python -m training.rl.eval_compare_sft_rl
    python -m training.rl.eval_compare_sft_rl --run toy_waypoint_eval_2026-03-04_21-33-13
    python -m training.rl.eval_compare_sft_rl --runsdir out/eval
"""
import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


def load_metrics(rundir: str) -> Dict[str, Any]:
    """Load metrics.json from a run directory."""
    metrics_path = os.path.join(rundir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def get_latest_run(runsdir: str) -> Optional[str]:
    """Get the most recent run directory."""
    runsdir = Path(runsdir)
    if not runsdir.exists():
        return None
    
    runs = [d for d in runsdir.iterdir() if d.is_dir()]
    if not runs:
        return None
    
    # Sort by modification time (newest first)
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0].name


def extract_policy_stats(metrics: Dict[str, Any], policy_prefix: str) -> Dict[str, float]:
    """Extract stats for a specific policy from metrics."""
    scenarios = metrics.get('scenarios', [])
    
    # Filter scenarios by policy type (if present)
    policy_scenarios = []
    for s in scenarios:
        # Check if scenario has policy_type or scenario_id contains the prefix
        if policy_prefix in s.get('scenario_id', ''):
            policy_scenarios.append(s)
    
    # Fallback: use summary section if available
    summary = metrics.get('summary', {})
    if policy_prefix in summary:
        return summary[policy_prefix]
    
    # Compute from scenarios
    if not policy_scenarios:
        return {}
    
    returns = [s['return'] for s in policy_scenarios]
    ades = [s['ade'] for s in policy_scenarios]
    fdes = [s['fde'] for s in policy_scenarios]
    successes = [s['success'] for s in policy_scenarios]
    
    import numpy as np
    return {
        'ade_mean': float(np.mean(ades)),
        'ade_std': float(np.std(ades)),
        'fde_mean': float(np.mean(fdes)),
        'fde_std': float(np.std(fdes)),
        'success_rate': float(np.mean(successes)),
        'return_mean': float(np.mean(returns)),
        'num_episodes': len(policy_scenarios)
    }


def print_3line_report(metrics: Dict[str, Any]):
    """Print 3-line comparison report."""
    summary = metrics.get('summary', {})
    comparison = metrics.get('comparison', {})
    
    sft = summary.get('sft', {})
    rl = summary.get('rl', {})
    
    if not sft or not rl:
        print("Error: Missing SFT or RL summary in metrics")
        return
    
    # Format: SFT: ADE=X.XXXm, FDE=X.XXXm, Success=X.X%
    sft_line = f"SFT:  ADE={sft.get('ade_mean', 0):.3f}m, FDE={sft.get('fde_mean', 0):.3f}m, Success={sft.get('success_rate', 0)*100:.1f}%"
    rl_line = f"RL:   ADE={rl.get('ade_mean', 0):.3f}m, FDE={rl.get('fde_mean', 0):.3f}m, Success={rl.get('success_rate', 0)*100:.1f}%"
    
    ade_imp = comparison.get('ade_improvement_pct', 0)
    fde_imp = comparison.get('fde_improvement_pct', 0)
    success_diff = comparison.get('success_rate_diff', 0)
    
    delta_line = f"Delta: ADE {ade_imp:+.1f}%, FDE {fde_imp:+.1f}%, Success {success_diff*100:+.1f}%"
    
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {metrics.get('run_id', 'Unknown')}")
    print("=" * 60)
    print(sft_line)
    print(rl_line)
    print(delta_line)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Compare SFT vs RL evaluation results'
    )
    parser.add_argument('--run', type=str, default=None,
                        help='Run directory name (default: latest)')
    parser.add_argument('--runsdir', type=str, default='out/eval',
                        help='Base directory for runs')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print the 3-line report')
    
    args = parser.parse_args()
    
    # Find run directory
    if args.run:
        rundir = os.path.join(args.runsdir, args.run)
    else:
        latest = get_latest_run(args.runsdir)
        if not latest:
            print("No runs found")
            return 1
        rundir = os.path.join(args.runsdir, latest)
        print(f"Using latest run: {latest}")
    
    # Load and print
    try:
        metrics = load_metrics(rundir)
        print_3line_report(metrics)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())

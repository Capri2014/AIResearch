"""
Policy Comparison Loader.

Compares SFT-only vs RL-refined policy evaluation results
and prints a 3-line summary report.
"""
import json
import argparse
import os
from typing import Dict, Any, List


def load_metrics(path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_policy_stats(metrics: Dict, prefix: str) -> Dict[str, float]:
    """Extract statistics for a specific policy (sft or rl)."""
    scenarios = [s for s in metrics['scenarios'] if s['scenario_id'].startswith(prefix)]
    
    if not scenarios:
        return {}
    
    returns = [s['return'] for s in scenarios]
    ades = [s['ade'] for s in scenarios]
    fdes = [s['fde'] for s in scenarios]
    successes = [s['success'] for s in scenarios]
    
    import numpy as np
    return {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'ade_mean': float(np.mean(ades)),
        'ade_std': float(np.std(ades)),
        'fde_mean': float(np.mean(fdes)),
        'fde_std': float(np.std(fdes)),
        'success_rate': float(np.mean(successes)),
    }


def print_comparison_report(metrics: Dict):
    """Print 3-line comparison report."""
    sft_stats = extract_policy_stats(metrics, 'eval_seed_')
    rl_stats = extract_policy_stats(metrics, 'rl_seed_')
    
    print("=" * 60)
    print("POLICY COMPARISON REPORT")
    print("=" * 60)
    print(f"Run ID: {metrics.get('run_id', 'unknown')}")
    print(f"Domain: {metrics.get('domain', 'unknown')}")
    print()
    
    # 3-line report
    sft_ade = sft_stats.get('ade_mean', 0)
    sft_fde = sft_stats.get('fde_mean', 0)
    sft_success = sft_stats.get('success_rate', 0)
    
    rl_ade = rl_stats.get('ade_mean', 0)
    rl_fde = rl_stats.get('fde_mean', 0)
    rl_success = rl_stats.get('success_rate', 0)
    
    print(f"SFT:  ADE={sft_ade:.3f}m, FDE={sft_fde:.3f}m, Success={sft_success:.1%}")
    print(f"RL:   ADE={rl_ade:.3f}m, FDE={rl_fde:.3f}m, Success={rl_success:.1%}")
    
    # Compute improvements
    if sft_ade > 0:
        ade_improvement = (sft_ade - rl_ade) / sft_ade * 100
    else:
        ade_improvement = 0
    
    if sft_fde > 0:
        fde_improvement = (sft_fde - rl_fde) / sft_fde * 100
    else:
        fde_improvement = 0
    
    success_diff = rl_success - sft_success
    
    print(f"Delta: ADE={ade_improvement:+.1f}%, FDE={fde_improvement:+.1f}%, Success={success_diff:+.1%}")
    print("=" * 60)
    
    # Also print summary if present
    if 'summary' in metrics:
        print("\nSummary from metrics:")
        summary = metrics['summary']
        print(f"  Episodes: {summary.get('num_episodes', 'N/A')}")
        print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
        print(f"  ADE mean: {summary.get('ade_mean', 0):.3f}m")
    
    if 'comparison' in metrics:
        print("\nComparison:")
        comp = metrics['comparison']
        print(f"  Baseline: {comp.get('baseline_policy', 'N/A')}")
        print(f"  Target: {comp.get('target_policy', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='Compare SFT vs RL policy metrics')
    parser.add_argument('metrics_file', type=str, help='Path to metrics.json')
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics_file):
        print(f"Error: File not found: {args.metrics_file}")
        return 1
    
    metrics = load_metrics(args.metrics_file)
    print_comparison_report(metrics)
    
    return 0


if __name__ == '__main__':
    exit(main())

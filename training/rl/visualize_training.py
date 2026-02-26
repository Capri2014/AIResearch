#!/usr/bin/env python3
"""
Training Visualization Utility

Generates matplotlib plots and HTML reports for:
- Training curves (reward, entropy, gradient norm over episodes)
- Evaluation metrics comparison (ADE, FDE, success rate across runs)
- Multi-run comparison visualizations

Usage:
    # Plot training curves from a run
    python visualize_training.py --run out/rl_delta_toy_v0/<run_id> --plot training

    # Plot evaluation metrics comparison
    python visualize_training.py --run out/eval/<eval_id> --plot eval

    # Compare multiple runs
    python visualize_training.py --compare out/rl_delta_toy_v0/<run1> out/rl_delta_toy_v0/<run2> --plot training

    # Generate HTML report
    python visualize_training.py --run out/rl_delta_toy_v0/<run_id> --output report.html
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np

# Try to import pandas for better data handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def find_metrics_file(run_path: str, pattern: str = "train_metrics.json") -> Optional[str]:
    """Find metrics file in run directory."""
    run_path = Path(run_path)
    
    # Direct match
    direct = run_path / pattern
    if direct.exists():
        return str(direct)
    
    # Search in subdirectories
    for subdir in run_path.iterdir():
        if subdir.is_dir():
            metrics_file = subdir / pattern
            if metrics_file.exists():
                return str(metrics_file)
    
    return None


def load_training_metrics(run_path: str) -> Dict[str, Any]:
    """Load training metrics from a run directory."""
    metrics_file = find_metrics_file(run_path, "train_metrics.json")
    
    if metrics_file and os.path.exists(metrics_file):
        return load_json(metrics_file)
    
    # Try alternative patterns
    for pattern in ["metrics.json", "training_metrics.json"]:
        metrics_file = find_metrics_file(run_path, pattern)
        if metrics_file and os.path.exists(metrics_file):
            data = load_json(metrics_file)
            # Check if it's training metrics
            if 'episodes' in data or 'rewards' in data:
                return data
    
    return {}


def load_eval_metrics(eval_path: str) -> Dict[str, Any]:
    """Load evaluation metrics from an eval directory."""
    eval_path = Path(eval_path)
    
    # Look for metrics.json
    metrics_file = eval_path / "metrics.json"
    if metrics_file.exists():
        return load_json(str(metrics_file))
    
    return {}


def smooth_values(values: List[float], window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to values."""
    if len(values) < window:
        return np.array(values)
    
    arr = np.array(values)
    smoothed = np.convolve(arr, np.ones(window)/window, mode='valid')
    return smoothed


def plot_training_curves(metrics: Dict, ax: plt.Axes, title: str = ""):
    """Plot training curves (reward, entropy, grad_norm)."""
    if not metrics:
        ax.text(0.5, 0.5, "No training metrics found", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Extract data
    episodes = metrics.get('episodes', [])
    rewards = metrics.get('rewards', [])
    entropies = metrics.get('entropies', [])
    grad_norms = metrics.get('grad_norms', [])
    losses = metrics.get('losses', [])
    
    # Create subplots if we have multiple metrics
    has_data = bool(episodes or rewards or entropies or grad_norms or losses)
    
    if not has_data:
        ax.text(0.5, 0.5, "No training data found", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Plot each metric
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    if rewards:
        ax.plot(episodes[:len(rewards)], rewards, alpha=0.3, color=colors[0], label='Reward')
        if len(rewards) >= 10:
            ax.plot(smooth_values(rewards), color=colors[0], label='Reward (smoothed)', linewidth=2)
    
    if entropies:
        ax.plot(episodes[:len(entropies)], entropies, alpha=0.3, color=colors[1], label='Entropy')
        if len(entropies) >= 10:
            ax.plot(smooth_values(entropies), color=colors[1], label='Entropy (smoothed)', linewidth=2)
    
    if grad_norms:
        ax.plot(episodes[:len(grad_norms)], grad_norms, alpha=0.3, color=colors[2], label='Grad Norm')
        if len(grad_norms) >= 10:
            ax.plot(smooth_values(grad_norms), color=colors[2], label='Grad Norm (smoothed)', linewidth=2)
    
    if losses:
        ax.plot(episodes[:len(losses)], losses, alpha=0.3, color=colors[3], label='Loss')
        if len(losses) >= 10:
            ax.plot(smooth_values(losses), color=colors[3], label='Loss (smoothed)', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.set_title(title or 'Training Curves')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_eval_metrics(metrics: Dict, ax: plt.Axes, title: str = ""):
    """Plot evaluation metrics (ADE, FDE, Success Rate)."""
    if not metrics or 'scenarios' not in metrics:
        ax.text(0.5, 0.5, "No evaluation metrics found", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    summary = metrics.get('summary', {})
    run_id = metrics.get('run_id', 'Unknown')
    
    # Extract metrics
    ade_mean = summary.get('ade_mean', 0)
    ade_std = summary.get('ade_std', 0)
    fde_mean = summary.get('fde_mean', 0)
    fde_std = summary.get('fde_std', 0)
    success_rate = summary.get('success_rate', 0)
    avg_return = summary.get('return_mean', summary.get('avg_return', 0))
    
    # Bar chart
    metric_names = ['ADE', 'FDE', 'Success Rate', 'Avg Return']
    metric_values = [ade_mean, fde_mean, success_rate * 100, avg_return]
    metric_stds = [ade_std, fde_std, 0, 0]
    
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, yerr=metric_stds, capsize=5)
    
    # Add value labels
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Value')
    ax.set_title(title or f'Eval Metrics: {run_id}')
    ax.grid(True, alpha=0.3, axis='y')


def plot_multi_run_comparison(runs_data: List[Dict], ax: plt.Axes, metric: str = 'ade'):
    """Plot comparison of a metric across multiple runs."""
    run_ids = [r.get('run_id', r.get('id', f'Run {idx}')) for idx, r in enumerate(runs_data)]
    
    # Extract values based on metric type
    if metric == 'ade':
        values = [r.get('summary', {}).get('ade_mean', 0) for r in runs_data]
        stds = [r.get('summary', {}).get('ade_std', 0) for r in runs_data]
        title = 'ADE Comparison (lower is better)'
    elif metric == 'fde':
        values = [r.get('summary', {}).get('fde_mean', 0) for r in runs_data]
        stds = [r.get('summary', {}).get('fde_std', 0) for r in runs_data]
        title = 'FDE Comparison (lower is better)'
    elif metric == 'success':
        values = [r.get('summary', {}).get('success_rate', 0) * 100 for r in runs_data]
        stds = [0] * len(values)
        title = 'Success Rate Comparison (higher is better)'
    elif metric == 'return':
        values = [r.get('summary', {}).get('return_mean', r.get('summary', {}).get('avg_return', 0)) for r in runs_data]
        stds = [0] * len(values)
        title = 'Average Return Comparison (higher is better)'
    else:
        values = [0] * len(runs_data)
        stds = [0] * len(runs_data)
        title = f'{metric} Comparison'
    
    # Shorten run IDs for display
    display_ids = [rid.split('_')[-1][:12] if len(rid) > 12 else rid for rid in run_ids]
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(run_ids)))
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.8, yerr=stds, capsize=5)
    
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(display_ids, rotation=45, ha='right')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)


def plot_training_multi_run(runs_data: List[Dict], ax: plt.Axes):
    """Plot training curves for multiple runs on same axes."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))
    
    for i, (run_data, color) in enumerate(zip(runs_data, colors)):
        run_id = run_data.get('run_id', f'Run {i}')
        episodes = run_data.get('episodes', [])
        rewards = run_data.get('rewards', [])
        
        if rewards:
            label = run_id.split('_')[-1][:10]
            ax.plot(episodes[:len(rewards)], rewards, alpha=0.2, color=color)
            if len(rewards) >= 10:
                ax.plot(smooth_values(rewards), color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Rewards - Multi-Run Comparison')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def generate_html_report(run_paths: List[str], output_path: str, title: str = "Training Visualization Report"):
    """Generate HTML report with embedded visualizations."""
    # Create temporary figures and save as images
    images = []
    
    # Process each run
    for run_path in run_paths:
        run_path = Path(run_path)
        
        # Try to load training metrics
        train_metrics = load_training_metrics(str(run_path))
        
        # Try to load eval metrics
        eval_metrics = load_eval_metrics(str(run_path))
        
        if train_metrics:
            # Create training plot
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_training_curves(train_metrics, ax, f"Training Curves: {run_path.name}")
            img_path = output_path.replace('.html', f'_{run_path.name}_train.png')
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close()
            images.append(('Training', img_path))
        
        if eval_metrics:
            # Create eval plot
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_eval_metrics(eval_metrics, ax, f"Eval Metrics: {run_path.name}")
            img_path = output_path.replace('.html', f'_{run_path.name}_eval.png')
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close()
            images.append(('Evaluation', img_path))
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        img {{ max-width: 100%; border-radius: 4px; }}
        .timestamp {{ color: #888; font-size: 14px; }}
        .metric-card {{ display: inline-block; background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 8px; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Overview</h2>
    <div class="card">
        <p>Analyzed {len(run_paths)} run(s):</p>
        <ul>
"""
    
    for path in run_paths:
        html += f"            <li><code>{path}</code></li>\n"
    
    html += """        </ul>
    </div>
"""
    
    # Add images
    for label, img_path in images:
        rel_path = os.path.basename(img_path)
        html += f"""
    <h2>{label}</h2>
    <div class="card">
        <img src="{rel_path}" alt="{label}">
    </div>
"""
    
    html += """
</body>
</html>"""
    
    # Write HTML
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report generated: {output_path}")
    
    # Also copy images to output directory
    output_dir = os.path.dirname(output_path) or '.'
    for _, img_path in images:
        # Images are already in the same directory
        pass
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Training Visualization Utility')
    parser.add_argument('--run', type=str, help='Path to training run or eval directory')
    parser.add_argument('--compare', nargs='+', help='Paths to multiple runs for comparison')
    parser.add_argument('--plot', type=str, choices=['training', 'eval', 'both'], default='both',
                        help='Type of plot to generate')
    parser.add_argument('--output', type=str, help='Output path for plot or HTML report')
    parser.add_argument('--title', type=str, help='Title for the plot/report')
    parser.add_argument('--metric', type=str, default='ade', help='Metric for comparison plot (ade/fde/success/return)')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'html'], default='png',
                        help='Output format')
    
    args = parser.parse_args()
    
    # Determine runs to process
    if args.compare:
        run_paths = args.compare
    elif args.run:
        run_paths = [args.run]
    else:
        print("Error: Must specify --run or --compare")
        parser.print_help()
        sys.exit(1)
    
    # Handle HTML output
    if args.format == 'html' or args.output and args.output.endswith('.html'):
        output_path = args.output or 'training_report.html'
        generate_html_report(run_paths, output_path, args.title or "Training Visualization Report")
        return
    
    # Handle single run plotting
    if len(run_paths) == 1:
        run_path = run_paths[0]
        
        if args.plot in ['training', 'both']:
            train_metrics = load_training_metrics(run_path)
            if train_metrics:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_training_curves(train_metrics, ax, args.title or f"Training Curves: {Path(run_path).name}")
                
                if args.show:
                    plt.show()
                elif args.output:
                    plt.savefig(args.output, dpi=150, bbox_inches='tight')
                else:
                    plt.savefig(f"{Path(run_path).name}_training.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        if args.plot in ['eval', 'both']:
            eval_metrics = load_eval_metrics(run_path)
            if eval_metrics:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_eval_metrics(eval_metrics, ax, args.title or f"Eval Metrics: {Path(run_path).name}")
                
                if args.show:
                    plt.show()
                elif args.output:
                    plt.savefig(args.output, dpi=150, bbox_inches='tight')
                else:
                    plt.savefig(f"{Path(run_path).name}_eval.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    # Handle multi-run comparison
    else:
        # Load all eval metrics for comparison
        eval_datas = []
        train_datas = []
        
        for run_path in run_paths:
            eval_data = load_eval_metrics(run_path)
            if eval_data:
                eval_datas.append(eval_data)
            
            train_data = load_training_metrics(run_path)
            if train_data:
                train_datas.append(train_data)
        
        if eval_datas and args.plot in ['eval', 'both']:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_multi_run_comparison(eval_datas, ax, args.metric)
            
            if args.show:
                plt.show()
            elif args.output:
                plt.savefig(args.output, dpi=150, bbox_inches='tight')
            else:
                plt.savefig(f"comparison_{args.metric}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        if train_datas and args.plot in ['training', 'both']:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_training_multi_run(train_datas, ax)
            
            if args.show:
                plt.show()
            output_train = args.output.replace('.png', '_train.png') if args.output else 'comparison_train.png'
            plt.savefig(output_train, dpi=150, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main()

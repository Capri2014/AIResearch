"""
Advanced Checkpoint Selection with ADE/FDE Metrics.

This module extends checkpoint_manager.py with driving-specific metrics
for better checkpoint selection in the driving-first pipeline.

Features:
- ADE/FDE-based checkpoint selection (critical for driving precision)
- Composite scoring with multiple metrics
- Comprehensive evaluation report generation
- Route completion and collision analysis

Usage:
    from checkpoint_selector_advanced import AdvancedCheckpointSelector, EvaluationReporter
    
    # Select best checkpoint by ADE (lower is better)
    selector = AdvancedCheckpointSelector('out/')
    best = selector.select_by_ade()
    
    # Generate comprehensive evaluation report
    reporter = EvaluationReporter('out/')
    report = reporter.generate_markdown_report()
"""
import os
import json
import glob
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class DrivingMetrics:
    """Driving-specific evaluation metrics."""
    ade_mean: float = 0.0
    ade_std: float = 0.0
    fde_mean: float = 0.0
    fde_std: float = 0.0
    success_rate: float = 0.0
    route_completion: float = 0.0
    collisions: int = 0
    red_light_violations: int = 0
    stop_sign_violations: int = 0
    avg_speed: float = 0.0
    avg_progress: float = 0.0


@dataclass  
class EvalCheckpoint:
    """Checkpoint with full evaluation metrics."""
    run_id: str
    run_path: str
    domain: str
    checkpoint_path: str
    training_metrics: Dict[str, Any]
    eval_metrics: Optional[DrivingMetrics] = None
    timestamp: str = ""
    
    @property
    def ade(self) -> float:
        """Get ADE (lower is better)."""
        if self.eval_metrics:
            return self.eval_metrics.ade_mean
        return float('inf')
    
    @property
    def fde(self) -> float:
        """Get FDE (lower is better)."""
        if self.eval_metrics:
            return self.eval_metrics.fde_mean
        return float('inf')
    
    @property
    def success(self) -> float:
        """Get success rate (higher is better)."""
        if self.eval_metrics:
            return self.eval_metrics.success_rate
        return 0.0


class AdvancedCheckpointSelector:
    """Advanced checkpoint selection for driving tasks."""
    
    def __init__(self, out_dir: str = 'out/'):
        self.out_dir = out_dir
        self.checkpoints: List[EvalCheckpoint] = []
        self._scan_checkpoints()
    
    def _scan_checkpoints(self):
        """Scan output directory for checkpoints with eval metrics."""
        # Find all eval metrics files
        eval_pattern = os.path.join(self.out_dir, '**/metrics.json')
        eval_files = glob.glob(eval_pattern, recursive=True)
        
        for eval_file in eval_files:
            try:
                run_dir = os.path.dirname(eval_file)
                checkpoint = self._load_checkpoint(run_dir)
                if checkpoint:
                    self.checkpoints.append(checkpoint)
            except Exception as e:
                print(f"Warning: Failed to load {eval_file}: {e}")
                continue
    
    def _load_checkpoint(self, run_dir: str) -> Optional[EvalCheckpoint]:
        """Load checkpoint with evaluation metrics."""
        # Load eval metrics
        eval_file = os.path.join(run_dir, 'metrics.json')
        if not os.path.exists(eval_file):
            return None
            
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
        except Exception:
            return None
        
        # Parse eval metrics
        summary = eval_data.get('summary', {})
        
        # Handle both nested and flat metric formats
        if 'sft' in summary and 'rl' in summary:
            # New format: nested sft/rl
            rl_metrics = summary.get('rl', {})
            ade = rl_metrics.get('ade_mean', rl_metrics.get('ade', float('inf')))
            fde = rl_metrics.get('fde_mean', rl_metrics.get('fde', float('inf')))
            success = rl_metrics.get('success_rate', rl_metrics.get('success', 0.0))
            route_comp = rl_metrics.get('route_completion', 0.0)
        else:
            # Old format: flat
            ade = summary.get('ade_mean', summary.get('ade', float('inf')))
            fde = summary.get('fde_mean', summary.get('fde', float('inf')))
            success = summary.get('success_rate', summary.get('success', 0.0))
            route_comp = summary.get('route_completion', 0.0)
        
        driving_metrics = DrivingMetrics(
            ade_mean=ade if ade != float('inf') else 0.0,
            ade_std=summary.get('ade_std', 0.0),
            fde_mean=fde if fde != float('inf') else 0.0,
            fde_std=summary.get('fde_std', 0.0),
            success_rate=success,
            route_completion=route_comp,
            collisions=summary.get('collisions', 0),
            red_light_violations=summary.get('red_light_violations', 0),
            stop_sign_violations=summary.get('stop_sign_violations', 0),
            avg_speed=summary.get('avg_speed', 0.0),
            avg_progress=summary.get('avg_progress', 0.0)
        )
        
        # Load training metrics
        train_file = os.path.join(run_dir, 'train_metrics.json')
        train_metrics = {}
        if os.path.exists(train_file):
            try:
                with open(train_file, 'r') as f:
                    train_metrics = json.load(f)
            except Exception:
                pass
        
        # Find checkpoint file
        checkpoint_path = None
        for name in ['checkpoint.pt', 'best_reward_checkpoint.pt', 'best_entropy_checkpoint.pt']:
            path = os.path.join(run_dir, name)
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        return EvalCheckpoint(
            run_id=train_metrics.get('run_id', os.path.basename(run_dir)),
            run_path=run_dir,
            domain=train_metrics.get('domain', 'unknown'),
            checkpoint_path=checkpoint_path or '',
            training_metrics=train_metrics,
            eval_metrics=driving_metrics,
            timestamp=train_metrics.get('timestamp', '')
        )
    
    def select_by_ade(self, top_k: int = 1) -> List[EvalCheckpoint]:
        """Select best checkpoint by ADE (lower is better)."""
        valid = [c for c in self.checkpoints if c.ade > 0]
        valid.sort(key=lambda x: x.ade)
        return valid[:top_k]
    
    def select_by_fde(self, top_k: int = 1) -> List[EvalCheckpoint]:
        """Select best checkpoint by FDE (lower is better)."""
        valid = [c for c in self.checkpoints if c.fde > 0]
        valid.sort(key=lambda x: x.fde)
        return valid[:top_k]
    
    def select_by_success(self, top_k: int = 1) -> List[EvalCheckpoint]:
        """Select best checkpoint by success rate (higher is better)."""
        valid = [c for c in self.checkpoints if c.eval_metrics]
        valid.sort(key=lambda x: x.eval_metrics.success_rate if x.eval_metrics else 0, reverse=True)
        return valid[:top_k]
    
    def select_composite(
        self,
        ade_weight: float = 0.4,
        fde_weight: float = 0.3,
        success_weight: float = 0.3,
        top_k: int = 1
    ) -> List[EvalCheckpoint]:
        """
        Select best checkpoint by composite score.
        
        Score = ade_weight * (1 - norm_ade) + fde_weight * (1 - norm_fde) + success_weight * norm_success
        
        Where norm_* are normalized to [0, 1] across available checkpoints.
        """
        valid = [c for c in self.checkpoints if c.eval_metrics]
        if not valid:
            return []
        
        # Get ranges for normalization
        ades = [c.ade for c in valid if c.ade > 0]
        fdes = [c.fde for c in valid if c.fde > 0]
        successes = [c.eval_metrics.success_rate for c in valid]
        
        ade_min, ade_max = (min(ades), max(ades)) if ades else (0, 1)
        fde_min, fde_max = (min(fdes), max(fdes)) if fdes else (0, 1)
        succ_min, succ_max = min(successes), max(successes) if successes else (0, 1)
        
        # Compute composite score
        for c in valid:
            # Normalize ADE (invert: lower is better)
            if ade_max > ade_min:
                norm_ade = 1 - (c.ade - ade_min) / (ade_max - ade_min)
            else:
                norm_ade = 1.0
            
            # Normalize FDE (invert: lower is better)
            if fde_max > fde_min:
                norm_fde = 1 - (c.fde - fde_min) / (fde_max - fde_min)
            else:
                norm_fde = 1.0
            
            # Normalize success (higher is better)
            if succ_max > succ_min:
                norm_success = (c.eval_metrics.success_rate - succ_min) / (succ_max - succ_min)
            else:
                norm_success = 1.0
            
            c.composite_score = (
                ade_weight * norm_ade + 
                fde_weight * norm_fde + 
                success_weight * norm_success
            )
        
        valid.sort(key=lambda x: getattr(x, 'composite_score', 0), reverse=True)
        return valid[:top_k]
    
    def get_all_checkpoints(self) -> List[EvalCheckpoint]:
        """Get all checkpoints with evaluation metrics."""
        return sorted(self.checkpoints, key=lambda x: x.timestamp, reverse=True)


class EvaluationReporter:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, out_dir: str = 'out/'):
        self.out_dir = out_dir
        self.selector = AdvancedCheckpointSelector(out_dir)
    
    def generate_markdown_report(self) -> str:
        """Generate markdown report of all evaluations."""
        checkpoints = self.selector.get_all_checkpoints()
        
        if not checkpoints:
            return "# Evaluation Report\n\nNo evaluation results found."
        
        lines = [
            "# Driving Pipeline Evaluation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Checkpoints Evaluated:** {len(checkpoints)}",
            "",
            "## Summary",
            "",
            "| Run ID | Domain | ADE (m) | FDE (m) | Success Rate | Route Comp |",
            "|--------|--------|---------|---------|--------------|------------|"
        ]
        
        for c in checkpoints:
            m = c.eval_metrics
            if m:
                lines.append(
                    f"| {c.run_id[:30]:30s} | {c.domain:10s} | "
                    f"{m.ade_mean:7.3f} | {m.fde_mean:7.3f} | "
                    f"{m.success_rate:11.1%} | {m.route_completion:10.1%} |"
                )
        
        # Best by each criterion
        lines.extend([
            "",
            "## Best Checkpoints",
            ""
        ])
        
        best_ade = self.selector.select_by_ade()
        if best_ade:
            b = best_ade[0]
            lines.append(f"**Best by ADE:** `{b.run_id}` ({b.ade:.3f}m)")
        
        best_fde = self.selector.select_by_fde()
        if best_fde:
            b = best_fde[0]
            lines.append(f"**Best by FDE:** `{b.run_id}` ({b.fde:.3f}m)")
        
        best_success = self.selector.select_by_success()
        if best_success:
            b = best_success[0]
            lines.append(f"**Best by Success:** `{b.run_id}` ({b.eval_metrics.success_rate:.1%})")
        
        best_composite = self.selector.select_composite()
        if best_composite:
            b = best_composite[0]
            lines.append(f"**Best Composite:** `{b.run_id}` (score: {b.composite_score:.3f})")
        
        return "\n".join(lines)
    
    def generate_comparison_table(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> str:
        """Generate comparison table for specific runs."""
        if metrics is None:
            metrics = ['ade_mean', 'fde_mean', 'success_rate', 'route_completion', 'collisions']
        
        checkpoints = [c for c in self.selector.get_all_checkpoints() if c.run_id in run_ids]
        
        if not checkpoints:
            return f"No checkpoints found for: {run_ids}"
        
        lines = [
            "# Comparison Report",
            "",
            "| Metric | " + " | ".join([c.run_id[:20] for c in checkpoints]) + " |",
            "|--------|" + "|".join(["---" for _ in checkpoints]) + "|"
        ]
        
        for metric in metrics:
            row = [metric]
            for c in checkpoints:
                if c.eval_metrics:
                    value = getattr(c.eval_metrics, metric, 'N/A')
                    if isinstance(value, float):
                        row.append(f"{value:.3f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("N/A")
            lines.append(" | ".join(row))
        
        return "\n".join(lines)
    
    def save_report(self, output_path: str = "out/evaluation_report.md"):
        """Save markdown report to file."""
        report = self.generate_markdown_report()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        return output_path


# CLI interface
def main():
    """CLI for advanced checkpoint selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Checkpoint Selection')
    parser.add_argument('--out-dir', default='out/', help='Output directory')
    parser.add_argument('--best-ade', action='store_true', help='Select best by ADE')
    parser.add_argument('--best-fde', action='store_true', help='Select best by FDE')
    parser.add_argument('--best-success', action='store_true', help='Select best by success')
    parser.add_argument('--best-composite', action='store_true', help='Select best by composite score')
    parser.add_argument('--report', action='store_true', help='Generate markdown report')
    parser.add_argument('--save-report', type=str, help='Save report to file')
    parser.add_argument('--top-k', type=int, default=3, help='Number of results to show')
    
    args = parser.parse_args()
    
    selector = AdvancedCheckpointSelector(args.out_dir)
    
    if args.best_ade:
        best = selector.select_by_ade(args.top_k)
        print(f"Best by ADE (lower is better):")
        for i, c in enumerate(best, 1):
            print(f"  {i}. {c.run_id} (ADE: {c.ade:.3f}m)")
    
    if args.best_fde:
        best = selector.select_by_fde(args.top_k)
        print(f"Best by FDE (lower is better):")
        for i, c in enumerate(best, 1):
            print(f"  {i}. {c.run_id} (FDE: {c.fde:.3f}m)")
    
    if args.best_success:
        best = selector.select_by_success(args.top_k)
        print(f"Best by Success Rate (higher is better):")
        for i, c in enumerate(best, 1):
            print(f"  {i}. {c.run_id} (Success: {c.eval_metrics.success_rate:.1%})")
    
    if args.best_composite:
        best = selector.select_composite(top_k=args.top_k)
        print(f"Best by Composite Score:")
        for i, c in enumerate(best, 1):
            print(f"  {i}. {c.run_id} (Score: {c.composite_score:.3f})")
    
    if args.report or args.save_report:
        reporter = EvaluationReporter(args.out_dir)
        report = reporter.generate_markdown_report()
        print(report)
        if args.save_report:
            path = reporter.save_report(args.save_report)
            print(f"\nReport saved to: {path}")
    
    # Default: show all with eval metrics
    if not (args.best_ade or args.best_fde or args.best_success or args.best_composite or args.report):
        checkpoints = selector.get_all_checkpoints()
        print(f"Found {len(checkpoints)} checkpoints with evaluation metrics:")
        for c in checkpoints[:10]:
            m = c.eval_metrics
            if m:
                print(f"  - {c.run_id}: ADE={m.ade_mean:.3f}, FDE={m.fde_mean:.3f}, Success={m.success_rate:.1%}")


if __name__ == '__main__':
    main()

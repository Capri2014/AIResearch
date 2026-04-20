#!/usr/bin/env python3
"""
Waypoint Evaluation Visualizer

Visualizes waypoint policy evaluation results from WaypointScenarioEvaluator.
Generates plots for ADE/FDE, collision rates, route completion, and per-scenario breakdowns.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try matplotlib, fall back to text if unavailable
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class EvaluationVisualizerConfig:
    """Configuration for evaluation visualization."""
    results_dir: str = "out/waypoint_evaluation"
    output_dir: str = "out/waypoint_evaluation/vis"
    runs: Optional[list] = None  # None = all runs
    format: str = "png"  # png, pdf, svg
    dpi: int = 150
    figsize: tuple = (10, 6)
    show: bool = False


class WaypointEvaluationVisualizer:
    """Visualizes waypoint policy evaluation results."""
    
    def __init__(self, config: Optional[EvaluationVisualizerConfig] = None):
        self.config = config or EvaluationVisualizerConfig()
        self.results_dir = Path(self.config.results_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, run_path: Path) -> dict:
        """Load evaluation results from a run directory."""
        # Try metrics.json
        metrics_file = run_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                result = json.load(f)
                # Convert "scenarios" to "scenario_results" for compatibility
                if "scenarios" in result and "scenario_results" not in result:
                    result["scenario_results"] = result["scenarios"]
                return result
        
        # Try results.json
        results_file = run_path / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                result = json.load(f)
                if "scenarios" in result and "scenario_results" not in result:
                    result["scenario_results"] = result["scenarios"]
                return result
        
        # Try basic_suite_result.json
        suite_file = run_path / "basic_suite_result.json"
        if suite_file.exists():
            with open(suite_file) as f:
                result = json.load(f)
                if "scenarios" in result and "scenario_results" not in result:
                    result["scenario_results"] = result["scenarios"]
                # Map avg_ade -> ade, avg_fde -> fde for compatibility
                if "metrics" not in result:
                    result["metrics"] = {
                        "ade": result.get("avg_ade", 0),
                        "fde": result.get("avg_fde", 0),
                        "route_completion": result.get("avg_route_completion", 0) / 100,
                        "collision_rate": result.get("total_collisions", 0) / max(result.get("num_scenarios", 1), 1)
                    }
                return result
        
        return {}
    
    def load_all_runs(self) -> list[tuple[Path, dict]]:
        """Load all evaluation runs."""
        runs = []
        if not self.results_dir.exists():
            return runs
        
        # Check if there's a direct result file in the results directory
        direct_result = self.load_results(self.results_dir)
        if direct_result:
            runs.append((self.results_dir, direct_result))
        
        # Also check subdirectories
        for run_dir in sorted(self.results_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name not in ['vis', 'scenarios']:  # Skip helper dirs
                if self.config.runs is None or run_dir.name in self.config.runs:
                    results = self.load_results(run_dir)
                    if results:
                        runs.append((run_dir, results))
        return runs
    
    def plot_ade_fde_comparison(self, runs: list[tuple[Path, dict]]) -> Optional[str]:
        """Plot ADE/FDE comparison across runs."""
        if not HAS_MPL or not runs:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        names = []
        ade_vals = []
        fde_vals = []
        
        for run_path, results in runs:
            names.append(run_path.name)
            metrics = results.get("metrics", results)
            ade_vals.append(metrics.get("ade", metrics.get("ade_m", float('nan'))))
            fde_vals.append(metrics.get("fde", metrics.get("fde_m", float('nan'))))
        
        x = range(len(names))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], ade_vals, width, label='ADE (m)', color='#2196F3')
        ax.bar([i + width/2 for i in x], fde_vals, width, label='FDE (m)', color='#FF5722')
        
        ax.set_xlabel('Run')
        ax.set_ylabel('Error (m)')
        ax.set_title('ADE/FDE Comparison Across Runs')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "ade_fde_comparison.png"
        plt.savefig(output_path, dpi=self.config.dpi)
        plt.close()
        
        return str(output_path)
    
    def plot_route_completion(self, runs: list[tuple[Path, dict]]) -> Optional[str]:
        """Plot route completion rates."""
        if not HAS_MPL or not runs:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        names = []
        completion_vals = []
        
        for run_path, results in runs:
            names.append(run_path.name)
            metrics = results.get("metrics", results)
            completion_vals.append(metrics.get("route_completion", 0) * 100)
        
        bars = ax.bar(names, completion_vals, color='#4CAF50')
        ax.axhline(y=100, color='r', linestyle='--', label='100%')
        
        ax.set_xlabel('Run')
        ax.set_ylabel('Route Completion (%)')
        ax.set_title('Route Completion Rate')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, completion_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = self.output_dir / "route_completion.png"
        plt.savefig(output_path, dpi=self.config.dpi)
        plt.close()
        
        return str(output_path)
    
    def plot_collision_rate(self, runs: list[tuple[Path, dict]]) -> Optional[str]:
        """Plot collision rates."""
        if not HAS_MPL or not runs:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        names = []
        collision_vals = []
        
        for run_path, results in runs:
            names.append(run_path.name)
            metrics = results.get("metrics", results)
            collision_vals.append(metrics.get("collision_rate", 0) * 100)
        
        bars = ax.bar(names, collision_vals, color='#f44336')
        
        ax.set_xlabel('Run')
        ax.set_ylabel('Collision Rate (%)')
        ax.set_title('Collision Rate')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, max(max(collision_vals) * 1.2, 10))
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, collision_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = self.output_dir / "collision_rate.png"
        plt.savefig(output_path, dpi=self.config.dpi)
        plt.close()
        
        return str(output_path)
    
    def plot_scenario_breakdown(self, run_path: Path, results: dict) -> Optional[str]:
        """Plot per-scenario breakdown for a single run."""
        if not HAS_MPL:
            return None
        
        scenarios = results.get("scenario_results", [])
        if not scenarios:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        names = []
        ade_vals = []
        fde_vals = []
        
        for scenario in scenarios:
            names.append(scenario.get("scenario", scenario.get("name", "unknown")))
            ade_vals.append(scenario.get("ade", float('nan')))
            fde_vals.append(scenario.get("fde", float('nan')))
        
        x = range(len(names))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], ade_vals, width, label='ADE (m)', color='#2196F3')
        ax.bar([i + width/2 for i in x], fde_vals, width, label='FDE (m)', color='#FF5722')
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Error (m)')
        ax.set_title(f'Per-Scenario Breakdown: {run_path.name}')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"scenario_breakdown_{run_path.name}.png"
        plt.savefig(output_path, dpi=self.config.dpi)
        plt.close()
        
        return str(output_path)
    
    def generate_markdown_summary(self, runs: list[tuple[Path, dict]]) -> str:
        """Generate markdown summary of evaluation results."""
        lines = ["# Waypoint Policy Evaluation Summary\n"]
        
        if not runs:
            lines.append("*No evaluation results found.*")
            return "\n".join(lines)
        
        lines.append(f"## Runs Analyzed: {len(runs)}\n")
        
        # Summary table
        lines.append("| Run | ADE (m) | FDE (m) | Route Completion | Collision Rate |")
        lines.append("|-----|--------|--------|------------------|---------------|")
        
        for run_path, results in runs:
            metrics = results.get("metrics", results)
            ade = metrics.get("ade", metrics.get("ade_m", "N/A"))
            fde = metrics.get("fde", metrics.get("fde_m", "N/A"))
            completion = metrics.get("route_completion", 0) * 100
            collision = metrics.get("collision_rate", 0) * 100
            
            lines.append(f"| {run_path.name} | {ade:.2f} | {fde:.2f} | {completion:.1f}% | {collision:.1f}% |")
        
        # Best run
        if runs:
            best_run = min(runs, key=lambda r: r[1].get("metrics", r[1]).get("ade", float('inf')))
            lines.append(f"\n## Best Run\n")
            lines.append(f"**{best_run[0].name}** with ADE: {best_run[1].get('metrics', {}).get('ade', 'N/A'):.2f}m")
        
        return "\n".join(lines)
    
    def visualize(self) -> dict:
        """Run visualization on all loaded results."""
        runs = self.load_all_runs()
        
        outputs = {}
        
        # Generate plots
        if HAS_MPL:
            ade_fde_path = self.plot_ade_fde_comparison(runs)
            if ade_fde_path:
                outputs["ade_fde_comparison"] = ade_fde_path
            
            completion_path = self.plot_route_completion(runs)
            if completion_path:
                outputs["route_completion"] = completion_path
            
            collision_path = self.plot_collision_rate(runs)
            if collision_path:
                outputs["collision_rate"] = collision_path
            
            # Per-scenario breakdowns
            for run_path, results in runs:
                scenario_path = self.plot_scenario_breakdown(run_path, results)
                if scenario_path:
                    outputs[f"scenario_{run_path.name}"] = scenario_path
        else:
            print("matplotlib not available, skipping plots")
        
        # Markdown summary
        summary = self.generate_markdown_summary(runs)
        summary_path = self.output_dir / "summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        outputs["summary"] = str(summary_path)
        
        return outputs


def main():
    parser = argparse.ArgumentParser(description="Visualize waypoint policy evaluation results")
    parser.add_argument("--results-dir", default="out/waypoint_evaluation",
                     help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="out/waypoint_evaluation/vis",
                     help="Output directory for visualizations")
    parser.add_argument("--runs", nargs="+",
                     help="Specific runs to visualize (default: all)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                     help="Output format")
    parser.add_argument("--dpi", type=int, default=150,
                     help="Output DPI")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 6],
                     help="Figure size")
    parser.add_argument("--show", action="store_true",
                     help="Show plots (requires display)")
    parser.add_argument("--list", action="store_true",
                     help="List available runs and exit")
    
    args = parser.parse_args()
    
    config = EvaluationVisualizerConfig(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        runs=args.runs,
        format=args.format,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        show=args.show,
    )
    
    viz = WaypointEvaluationVisualizer(config)
    
    # List runs if requested
    if args.list:
        runs = viz.load_all_runs()
        print(f"Found {len(runs)} evaluation runs:")
        for run_path, _ in runs:
            print(f"  - {run_path.name}")
        return
    
    # Run visualization
    print(f"Visualizing waypoint evaluation results...")
    outputs = viz.visualize()
    
    print(f"Generated {len(outputs)} outputs:")
    for name, path in outputs.items():
        print(f"  - {name}: {path}")
    
    # Print summary
    summary_path = outputs.get("summary")
    if summary_path:
        print(f"\nMarkdown summary: {summary_path}")


if __name__ == "__main__":
    main()
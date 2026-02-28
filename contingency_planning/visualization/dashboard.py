"""
Visualization Dashboard for Contingency Planning Results

Creates comparison plots and animations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    output_dir: str = "out/contingency_planning"
    figsize: tuple = (12, 8)
    dpi: int = 100


class ComparisonDashboard:
    """
    Dashboard for visualizing contingency planning comparison results.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict],
        save: bool = True,
    ):
        """
        Plot bar chart comparing metrics across approaches.
        
        Args:
            results: Dict of approach_name -> metrics dict
            save: Whether to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=self.config.figsize)
        fig.suptitle("Contingency Planning: Approach Comparison")
        
        approaches = list(results.keys())
        
        # Extract metrics
        metrics = {
            'Success Rate': [results[a].get('success_rate', 0) for a in approaches],
            'Collision Rate': [results[a].get('collision_rate', 0) for a in approaches],
            'MRC Rate': [results[a].get('mrc_rate', 0) for a in approaches],
            'Avg Steps': [results[a].get('avg_steps', 0) for a in approaches],
            'Avg Planning Time (ms)': [results[a].get('avg_planning_time_ms', 0) for a in approaches],
        }
        
        # Plot each metric
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(approaches, values, color=['#2ecc71', '#3498db', '#e74c3c'][:len(approaches)])
            ax.set_title(metric_name)
            ax.set_ylabel(metric_name)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "metrics_comparison.png", dpi=self.config.dpi)
            print(f"Saved: {self.output_dir / 'metrics_comparison.png'}")
        
        return fig
    
    def plot_trajectory_comparison(
        self,
        episode_data: Dict[str, List],
        scenario_info: Optional[Dict] = None,
        save: bool = True,
    ):
        """
        Plot trajectories from different approaches.
        
        Args:
            episode_data: Dict of approach_name -> list of states
            scenario_info: Optional scenario info (goal, obstacles)
            save: Whether to save plot
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for idx, (approach, states) in enumerate(episode_data.items()):
            # Extract positions
            positions = np.array([[s['position'][0], s['position'][1]] for s in states])
            
            if len(positions) > 0:
                ax.plot(positions[:, 0], positions[:, 1], 
                       color=colors[idx % len(colors)],
                       label=approach, linewidth=2, alpha=0.7)
                ax.scatter(positions[0, 0], positions[0, 1], 
                          color=colors[idx % len(colors)], marker='o', s=100)
                ax.scatter(positions[-1, 0], positions[-1, 1], 
                          color=colors[idx % len(colors)], marker='x', s=100)
        
        # Plot scenario info
        if scenario_info:
            # Goal
            goal = scenario_info.get('goal', [100, 0])
            ax.scatter(goal[0], goal[1], color='gold', marker='*', s=200, 
                      label='Goal', zorder=5)
            
            # Obstacles
            for obs in scenario_info.get('obstacles', []):
                circle = patches.Circle(
                    (obs.get('x', 0), obs.get('y', 0)),
                    obs.get('radius', 2),
                    color='red', alpha=0.3
                )
                ax.add_patch(circle)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "trajectory_comparison.png", dpi=self.config.dpi)
            print(f"Saved: {self.output_dir / 'trajectory_comparison.png'}")
        
        return fig
    
    def plot_belief_evolution(
        self,
        episode_data: Dict,
        save: bool = True,
    ):
        """
        Plot belief state evolution over time.
        
        Args:
            episode_data: Episode data with belief history
            save: Whether to save plot
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        beliefs = episode_data.get('beliefs', {})
        
        for hypothesis, belief_values in beliefs.items():
            ax.plot(belief_values, label=hypothesis)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Belief Probability')
        ax.set_title('Belief Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "belief_evolution.png", dpi=self.config.dpi)
            print(f"Saved: {self.output_dir / 'belief_evolution.png'}")
        
        return fig
    
    def generate_report(
        self,
        results: Dict[str, Dict],
        episode_examples: Optional[Dict[str, Dict]] = None,
    ):
        """
        Generate full HTML report.
        
        Args:
            results: Summary results
            episode_examples: Example episodes for visualization
        """
        # Create plots
        self.plot_metrics_comparison(results)
        
        if episode_examples:
            for approach, episode in episode_examples.items():
                self.plot_trajectory_comparison(
                    {approach: episode.get('states', [])},
                    scenario_info=episode.get('scenario_info')
                )
        
        # Generate HTML
        html = self._generate_html_report(results)
        
        with open(self.output_dir / "report.html", 'w') as f:
            f.write(html)
        
        print(f"Report: {self.output_dir / 'report.html'}")
    
    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Contingency Planning Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
    </style>
</head>
<body>
    <h1>Contingency Planning Benchmark Report</h1>
    
    <h2>Summary</h2>
    <table>
        <tr>
            <th>Approach</th>
            <th>Success Rate</th>
            <th>Collision Rate</th>
            <th>MRC Rate</th>
            <th>Avg Steps</th>
            <th>Avg Planning Time (ms)</th>
        </tr>
"""
        
        for approach, metrics in results.items():
            html += f"""
        <tr>
            <td><strong>{approach}</strong></td>
            <td>{metrics.get('success_rate', 0):.1%}</td>
            <td>{metrics.get('collision_rate', 0):.1%}</td>
            <td>{metrics.get('mrc_rate', 0):.1%}</td>
            <td>{metrics.get('avg_steps', 0):.1f}</td>
            <td>{metrics.get('avg_planning_time_ms', 0):.1f}</td>
        </tr>
"""
        
        html += """
    </table>
    
    <h2>Visualizations</h2>
    <img src="metrics_comparison.png" alt="Metrics Comparison">
    
    <h2>Conclusion</h2>
    <p>This report compares tree-based and model-based contingency planning approaches
    on safety-critical autonomous driving scenarios.</p>
</body>
</html>
"""
        return html


def load_results(results_dir: str) -> Dict:
    """Load results from directory."""
    results_path = Path(results_dir) / "summary.json"
    
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    
    return {}


if __name__ == "__main__":
    # Example: Load and visualize results
    results_dir = "out/contingency_benchmark"
    results = load_results(results_dir)
    
    if results:
        dashboard = ComparisonDashboard()
        dashboard.plot_metrics_comparison(results)
        print("Visualization complete!")
    else:
        print("No results found. Run benchmark first.")

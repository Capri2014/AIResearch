#!/usr/bin/env python3
"""
Pipeline Metrics Dashboard
Interactive dashboard for visualizing and analyzing pipeline metrics across all stages.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineMetricsDashboard:
    """Dashboard for pipeline metrics visualization."""
    
    def __init__(self, output_dir: str = "out/pipeline_metrics_dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_stage_metrics(self, base_dir: str = "out") -> dict:
        """Load metrics from all pipeline stages."""
        stages = {}
        base = Path(base_dir)
        
        # SSL stage metrics
        ssl_dirs = list(base.glob("ssl_train*/"))
        if ssl_dirs:
            latest = max(ssl_dirs, key=lambda p: p.stat().st_mtime)
            metrics_file = latest / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    stages["ssl"] = json.load(f)
                    
        # BC stage metrics
        bc_dirs = list(base.glob("bc_train*/")) + list(base.glob("waypoint_bc*/"))
        if bc_dirs:
            latest = max(bc_dirs, key=lambda p: p.stat().st_mtime)
            metrics_file = latest / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    stages["bc"] = json.load(f)
                    
        # RL stage metrics
        rl_dirs = list(base.glob("rl_*/")) + list(base.glob("ppo_*/"))
        if rl_dirs:
            latest = max(rl_dirs, key=lambda p: p.stat().st_mtime)
            metrics_file = latest / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    stages["rl"] = json.load(f)
                    
        # Evaluation metrics (multiple runs)
        eval_dirs = list(base.glob("eval*/"))
        if eval_dirs:
            latest = max(eval_dirs, key=lambda p: p.stat().st_mtime)
            metrics_file = latest / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    stages["eval"] = json.load(f)
                    
        return stages
    
    def generate_html_dashboard(self, stages: dict) -> str:
        """Generate HTML dashboard."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Metrics Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
              margin: 0; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 28px; }}
        .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                 gap: 20px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ margin: 0 0 15px 0; color: #333; font-size: 18px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #666; }}
        .metric-value {{ font-weight: bold; color: #333; }}
        .stage-ssl {{ border-left: 4px solid #667eea; }}
        .stage-bc {{ border-left: 4px solid #f093fb; }}
        .stage-rl {{ border-left: 4px solid #4facfe; }}
        .stage-eval {{ border-left: 4px solid #43e97b; }}
        .no-data {{ color: #999; font-style: italic; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 Pipeline Metrics Dashboard</h1>
        <p>Driving-First Pipeline Performance Overview</p>
        <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="grid">
"""
        
        # SSL metrics
        if "ssl" in stages:
            html += '<div class="card stage-ssl"><h2>🔍 SSL Pretraining</h2>'
            m = stages["ssl"]
            html += self._render_metrics(m)
            html += '</div>'
        else:
            html += '<div class="card stage-ssl"><h2>🔍 SSL Pretraining</h2><p class="no-data">No metrics available</p></div>'
            
        # BC metrics
        if "bc" in stages:
            html += '<div class="card stage-bc"><h2>🎯 Waypoint BC</h2>'
            m = stages["bc"]
            html += self._render_metrics(m)
            html += '</div>'
        else:
            html += '<div class="card stage-bc"><h2>🎯 Waypoint BC</h2><p class="no-data">No metrics available</p></div>'
            
        # RL metrics
        if "rl" in stages:
            html += '<div class="card stage-rl"><h2>⚡ RL Refinement</h2>'
            m = stages["rl"]
            html += self._render_metrics(m)
            html += '</div>'
        else:
            html += '<div class="card stage-rl"><h2>⚡ RL Refinement</h2><p class="no-data">No metrics available</p></div>'
            
        # Eval metrics
        if "eval" in stages:
            html += '<div class="card stage-eval"><h2>📊 Evaluation</h2>'
            m = stages["eval"]
            html += self._render_metrics(m)
            html += '</div>'
        else:
            html += '<div class="card stage-eval"><h2>📊 Evaluation</h2><p class="no-data">No metrics available</p></div>'
            
        html += """
    </div>
</body>
</html>
"""
        return html
    
    def _render_metrics(self, metrics: dict) -> str:
        """Render metrics dictionary to HTML."""
        html = ""
        for key, value in metrics.items():
            if isinstance(value, float):
                if "ade" in key.lower() or "fde" in key.lower():
                    display = f"{value:.2f}m"
                elif "rate" in key.lower() or "success" in key.lower():
                    display = f"{value*100:.1f}%"
                elif "loss" in key.lower():
                    display = f"{value:.4f}"
                else:
                    display = f"{value:.2f}"
            else:
                display = str(value)
            html += f'<div class="metric"><span class="metric-label">{key}</span><span class="metric-value">{display}</span></div>'
        return html
    
    def save_dashboard(self, stages: dict) -> Path:
        """Save HTML dashboard to file."""
        html = self.generate_html_dashboard(stages)
        output_file = self.output_dir / "index.html"
        with open(output_file, "w") as f:
            f.write(html)
        return output_file
    
    def generate_summary_json(self, stages: dict) -> Path:
        """Generate JSON summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        for stage, metrics in stages.items():
            summary["stages"][stage] = metrics
        output_file = self.output_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        return output_file


def main():
    parser = argparse.ArgumentParser(description="Pipeline Metrics Dashboard")
    parser.add_argument("--output-dir", type=str, default="out/pipeline_metrics_dashboard",
                     help="Output directory")
    parser.add_argument("--base-dir", type=str, default="out",
                     help="Base directory for metrics")
    parser.add_argument("--smoke-test", action="store_true",
                     help="Run smoke test with synthetic data")
    args = parser.parse_args()
    
    dashboard = PipelineMetricsDashboard(args.output_dir)
    
    if args.smoke_test:
        # Synthetic data for testing
        stages = {
            "ssl": {"loss": 0.0451, "contrastive_loss": 0.0312, "mim_loss": 0.0590},
            "bc": {"loss": 0.0234, "ade": 6.30, "fde": 6.12},
            "rl": {"reward": 125.3, "success_rate": 0.85, "value_loss": 0.0123},
            "eval": {"ade": 2.45, "fde": 3.21, "success_rate": 0.88, "route_completion": 0.92}
        }
        print("Smoke test: Using synthetic metrics")
    else:
        stages = dashboard.load_stage_metrics(args.base_dir)
        print(f"Loaded metrics from {len(stages)} stages")
    
    # Generate dashboard
    output_file = dashboard.save_dashboard(stages)
    print(f"Dashboard: {output_file}")
    
    summary_file = dashboard.generate_summary_json(stages)
    print(f"Summary: {summary_file}")
    
    print("\n✅ Pipeline Metrics Dashboard generated successfully!")
    print(f"   Open {output_file} in a browser to view the dashboard.")


if __name__ == "__main__":
    main()
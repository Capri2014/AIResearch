"""
Pipeline Run Analyzer - Analyze and compare pipeline runs.

Usage:
    python training/pipeline_run_analyzer.py --run-id <run_id>
    python training/pipeline_run_analyzer.py --compare <run1> <run2>
    python training/pipeline_run_analyzer.py --list
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunInfo:
    """Information about a pipeline run."""
    
    run_id: str
    timestamp: str
    stages_completed: List[str] = field(default_factory=list)
    total_time_seconds: float = 0.0
    
    # Stage metrics
    ssl_loss: Optional[float] = None
    bc_loss: Optional[float] = None
    rl_reward: Optional[float] = None
    eval_ade: Optional[float] = None
    eval_fde: Optional[float] = None
    
    # Checkpoint paths
    ssl_checkpoint: Optional[str] = None
    bc_checkpoint: Optional[str] = None
    rl_checkpoint: Optional[str] = None
    
    # Status
    status: str = "unknown"  # running, completed, failed


@dataclass
class StageMetrics:
    """Metrics from a single stage."""
    
    stage_name: str
    loss: Optional[float] = None
    reward: Optional[float] = None
    checkpoint_path: Optional[str] = None
    duration_seconds: float = 0.0


class PipelineRunAnalyzer:
    """Analyze pipeline runs and extract metrics."""
    
    def __init__(self, base_dir: Path = Path("out")):
        self.base_dir = base_dir
    
    def find_runs(self) -> List[str]:
        """Find all pipeline runs in output directory."""
        runs = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and "pipeline" in d.name:
                runs.append(d.name)
        return sorted(runs)
    
    def load_run_info(self, run_id: str) -> RunInfo:
        """Load run information from metadata file."""
        metadata_path = self.base_dir / f"{run_id}_metadata.json"
        
        if not metadata_path.exists():
            return RunInfo(run_id=run_id, timestamp="unknown", status="missing")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        info = RunInfo(
            run_id=run_id,
            timestamp=metadata.get("timestamp", "unknown"),
            status=metadata.get("status", "unknown"),
        )
        
        # Parse stage results
        stages = metadata.get("stages", {})
        for stage_name, stage_data in stages.items():
            info.stages_completed.append(stage_name)
            if stage_name == "ssl":
                info.ssl_loss = stage_data.get("loss")
                info.ssl_checkpoint = stage_data.get("checkpoint")
            elif stage_name == "bc":
                info.bc_loss = stage_data.get("loss")
                info.bc_checkpoint = stage_data.get("checkpoint")
            elif stage_name == "rl":
                info.rl_reward = stage_data.get("reward")
                info.rl_checkpoint = stage_data.get("checkpoint")
            elif stage_name == "eval":
                info.eval_ade = stage_data.get("ade")
                info.eval_fde = stage_data.get("fde")
        
        info.total_time_seconds = metadata.get("total_time_seconds", 0.0)
        
        return info
    
    def analyze_run(self, run_id: str) -> Dict[str, Any]:
        """Analyze a single run and return detailed analysis."""
        info = self.load_run_info(run_id)
        
        analysis = {
            "run_id": run_id,
            "timestamp": info.timestamp,
            "status": info.status,
            "stages": info.stages_completed,
            "duration_seconds": info.total_time_seconds,
            "metrics": {},
        }
        
        # Collect metrics
        if info.ssl_loss is not None:
            analysis["metrics"]["ssl_loss"] = info.ssl_loss
        if info.bc_loss is not None:
            analysis["metrics"]["bc_loss"] = info.bc_loss
        if info.rl_reward is not None:
            analysis["metrics"]["rl_reward"] = info.rl_reward
        if info.eval_ade is not None:
            analysis["metrics"]["eval_ade"] = info.eval_ade
        if info.eval_fde is not None:
            analysis["metrics"]["eval_fde"] = info.eval_fde
        
        # Check checkpoint health
        for name, path in [("ssl", info.ssl_checkpoint), 
                          ("bc", info.bc_checkpoint), 
                          ("rl", info.rl_checkpoint)]:
            if path:
                exists = Path(path).exists() if path else False
                analysis[f"{name}_checkpoint_exists"] = exists
        
        # Compute health score
        health_score = 0
        if info.stages_completed:
            health_score = len(info.stages_completed) / 4 * 100
        analysis["health_score"] = health_score
        
        return analysis
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        comparison = {"runs": [], "differences": {}}
        
        for run_id in run_ids:
            analysis = self.analyze_run(run_id)
            comparison["runs"].append(analysis)
        
        # Compute differences
        if len(run_ids) >= 2:
            metrics = ["ssl_loss", "bc_loss", "rl_reward", "eval_ade", "eval_fde"]
            for metric in metrics:
                values = [r.get("metrics", {}).get(metric) for r in comparison["runs"]]
                values = [v for v in values if v is not None]
                if len(values) >= 2:
                    diff = values[-1] - values[0]
                    comparison["differences"][metric] = diff
        
        return comparison
    
    def print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print human-readable summary."""
        print(f"\n{'='*50}")
        print(f"Pipeline Run Analysis: {analysis['run_id']}")
        print(f"{'='*50}")
        print(f"Status: {analysis['status']}")
        print(f"Timestamp: {analysis['timestamp']}")
        print(f"Stages: {', '.join(analysis['stages']) or 'none'}")
        print(f"Duration: {analysis['duration_seconds']:.1f}s")
        print(f"Health Score: {analysis['health_score']:.0f}%")
        
        metrics = analysis.get("metrics", {})
        if metrics:
            print(f"\nMetrics:")
            for k, v in metrics.items():
                if v is not None:
                    print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Run Analyzer")
    parser.add_argument("--run-id", type=str, help="Run ID to analyze")
    parser.add_argument("--compare", nargs="+", help="Compare multiple runs")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--base-dir", type=str, default="out", help="Base directory for runs")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    analyzer = PipelineRunAnalyzer(base_dir=Path(args.base_dir))
    
    if args.list:
        runs = analyzer.find_runs()
        print(f"Found {len(runs)} pipeline runs:")
        for r in runs:
            print(f"  - {r}")
    
    elif args.run_id:
        analysis = analyzer.analyze_run(args.run_id)
        analyzer.print_summary(analysis)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"\nSaved to {args.output}")
    
    elif args.compare:
        comparison = analyzer.compare_runs(args.compare)
        print(f"\nComparison of {len(args.compare)} runs:")
        for run in comparison["runs"]:
            print(f"\n{run['run_id']}:")
            for k, v in run.get("metrics", {}).items():
                if v is not None:
                    print(f"  {k}: {v:.4f}")
        if comparison["differences"]:
            print(f"\nDifferences:")
            for k, v in comparison["differences"].items():
                print(f"  {k}: {v:+.4f}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
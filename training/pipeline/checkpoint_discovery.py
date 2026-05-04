"""
Pipeline Checkpoint Discovery and Selection System.

Auto-discovers checkpoints across all pipeline stages (SSL/BC/RL),
selects best checkpoints based on metrics, and provides unified
checkpoint access for evaluation.
"""

import argparse
import json
import os
import re
import glob
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable


@dataclass
class CheckpointInfo:
    """Information about a single checkpoint."""
    path: str
    stage: str  # ssl, bc, rl, eval
    run_id: str
    epoch: Optional[int] = None
    step: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    size_mb: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None and os.path.exists(self.path):
            self.created_at = datetime.fromtimestamp(
                os.path.getmtime(self.path)
            ).isoformat()
        if self.size_mb == 0.0 and os.path.exists(self.path):
            self.size_mb = os.path.getsize(self.path) / (1024 * 1024)


@dataclass
class CheckpointSelection:
    """Selected checkpoints for pipeline evaluation."""
    ssl: Optional[CheckpointInfo] = None
    bc: Optional[CheckpointInfo] = None
    rl: Optional[CheckpointInfo] = None
    eval: Optional[CheckpointInfo] = None
    stage_order: List[str] = field(default_factory=list)
    
    def has_stage(self, stage: str) -> bool:
        """Check if checkpoint exists for stage."""
        return getattr(self, stage, None) is not None
    
    def get_stage(self, stage: str) -> Optional[CheckpointInfo]:
        """Get checkpoint for stage."""
        return getattr(self, stage, None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"stage_order": self.stage_order}
        for stage in ["ssl", "bc", "rl", "eval"]:
            ckpt = self.get_stage(stage)
            if ckpt:
                result[stage] = {
                    "path": ckpt.path,
                    "run_id": ckpt.run_id,
                    "epoch": ckpt.epoch,
                    "metrics": ckpt.metrics,
                }
        return result


class PipelineCheckpointSelector:
    """Select best checkpoints across pipeline stages."""
    
    # Priority order for checkpoint files
    CHECKPOINT_PRIORITY = [
        "final.pt",
        "best.pt", 
        "best_reward.pt",
        "best_ade.pt",
        "best_fde.pt",
        "best_success.pt",
        "checkpoint.pt",
        "epoch_*.pt",
    ]
    
    # Metric priorities for selection
    METRIC_PRIORITY = {
        "bc": ["val_ade", "val_loss", "loss"],
        "rl": ["reward", "success_rate", "episode_reward"],
        "ssl": ["loss", "val_loss"],
        "eval": ["ade", "fde", "success_rate"],
    }
    
    def __init__(self, base_dir: str = "out"):
        self.base_dir = Path(base_dir)
        
    def discover_stage_checkpoints(
        self, 
        stage: str, 
        run_id: Optional[str] = None
    ) -> List[CheckpointInfo]:
        """Discover all checkpoints for a stage."""
        checkpoints = []
        
        # Search patterns
        if run_id:
            patterns = [f"{self.base_dir}/{stage}_{run_id}*/**/*.pt"]
        else:
            patterns = [
                f"{self.base_dir}/{stage}_*/**/*.pt",
                f"{self.base_dir}/**/checkpoints/{stage}/*.pt",
            ]
        
        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                if not os.path.isfile(path):
                    continue
                    
                # Skip non-checkpoint files
                basename = os.path.basename(path)
                if basename.startswith("."):
                    continue
                    
                # Extract run_id from path
                run_id_match = re.search(f"{stage}_([^/]+)", path)
                run_id = run_id_match.group(1) if run_id_match else "unknown"
                
                # Extract epoch/step
                epoch = None
                step = None
                epoch_match = re.search(r"epoch_(\d+)", basename)
                step_match = re.search(r"step_(\d+)", basename)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                if step_match:
                    step = int(step_match.group(1))
                
                # Try to load metrics from sidecar JSON
                metrics = self._load_checkpoint_metrics(path, stage)
                
                ckpt = CheckpointInfo(
                    path=path,
                    stage=stage,
                    run_id=run_id,
                    epoch=epoch,
                    step=step,
                    metrics=metrics,
                )
                checkpoints.append(ckpt)
        
        return checkpoints
    
    def _load_checkpoint_metrics(
        self, 
        checkpoint_path: str, 
        stage: str
    ) -> Dict[str, float]:
        """Load metrics from checkpoint or sidecar JSON."""
        metrics = {}
        
        # Try sidecar metrics.json
        json_path = checkpoint_path.replace(".pt", "_metrics.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                    metrics = {
                        k: float(v) for k, v in data.items() 
                        if isinstance(v, (int, float))
                    }
            except Exception:
                pass
        
        # Try train_metrics.json in same directory
        ckpt_dir = os.path.dirname(checkpoint_path)
        train_metrics_path = os.path.join(ckpt_dir, "train_metrics.json")
        if os.path.exists(train_metrics_path) and not metrics:
            try:
                with open(train_metrics_path) as f:
                    data = json.load(f)
                    # Extract best metrics
                    if "best" in data:
                        metrics = {
                            k: float(v) for k, v in data["best"].items()
                            if isinstance(v, (int, float))
                        }
                    elif "final" in data:
                        metrics = {
                            k: float(v) for k, v in data["final"].items()
                            if isinstance(v, (int, float))
                        }
            except Exception:
                pass
        
        return metrics
    
    def select_best(
        self, 
        checkpoints: List[CheckpointInfo],
        stage: str,
        metric: Optional[str] = None,
    ) -> Optional[CheckpointInfo]:
        """Select best checkpoint from list."""
        if not checkpoints:
            return None
        
        # First, try priority order for checkpoint files
        for priority_name in self.CHECKPOINT_PRIORITY:
            for ckpt in checkpoints:
                if priority_name.replace("*", "") in ckpt.path:
                    # Verify metric if specified
                    if metric and metric in ckpt.metrics:
                        return ckpt
                    return ckpt
        
        # If no priority match, use metric
        if metric:
            for ckpt in checkpoints:
                if metric in ckpt.metrics:
                    return ckpt
        
        # Fallback to first checkpoint
        return checkpoints[0] if checkpoints else None
    
    def rank_checkpoints(
        self,
        checkpoints: List[CheckpointInfo],
        stage: str,
    ) -> List[CheckpointInfo]:
        """Rank checkpoints by metric priority."""
        if not checkpoints:
            return []
        
        priorities = self.METRIC_PRIORITY.get(stage, ["loss", "reward"])
        
        def get_rank(ckpt: CheckpointInfo) -> float:
            for i, metric in enumerate(priorities):
                if metric in ckpt.metrics:
                    # Lower is better for most metrics
                    value = ckpt.metrics[metric]
                    return i + value / 1000.0
            return len(priorities)  # Lowest priority if no metrics
        
        return sorted(checkpoints, key=get_rank)
    
    def find_latest(
        self,
        stage: str,
        run_id: Optional[str] = None,
    ) -> Optional[CheckpointInfo]:
        """Find latest checkpoint for stage."""
        checkpoints = self.discover_stage_checkpoints(stage, run_id)
        if not checkpoints:
            return None
        
        # Sort by created_at
        checkpoints.sort(
            key=lambda c: c.created_at or "",
            reverse=True,
        )
        return checkpoints[0]
    
    def find_best_by_metric(
        self,
        stage: str,
        metric: str,
        run_id: Optional[str] = None,
    ) -> Optional[CheckpointInfo]:
        """Find best checkpoint by metric."""
        checkpoints = self.discover_stage_checkpoints(stage, run_id)
        if not checkpoints:
            return None
        
        # Filter to only those with metric
        with_metric = [c for c in checkpoints if metric in c.metrics]
        if not with_metric:
            return None
        
        # Lower is better for most metrics
        lower_is_better = metric in ["loss", "ade", "fde", "error"]
        
        if lower_is_better:
            return min(with_metric, key=lambda c: c.metrics[metric])
        else:
            return max(with_metric, key=lambda c: c.metrics[metric])


class PipelineCheckpointDiscovery:
    """Comprehensive checkpoint discovery for full pipeline."""
    
    STAGES = ["ssl", "bc", "rl", "eval"]
    
    def __init__(self, base_dir: str = "out"):
        self.base_dir = Path(base_dir)
        self.selector = PipelineCheckpointSelector(base_dir)
        
    def discover_all(
        self,
        run_id: Optional[str] = None,
    ) -> CheckpointSelection:
        """Discover all available checkpoints."""
        selection = CheckpointSelection()
        
        for stage in self.STAGES:
            checkpoints = self.selector.discover_stage_checkpoints(stage, run_id)
            if checkpoints:
                best = self.selector.select_best(checkpoints, stage)
                setattr(selection, stage, best)
                selection.stage_order.append(stage)
        
        return selection
    
    def discover_by_run(
        self,
        run_id: str,
    ) -> CheckpointSelection:
        """Discover checkpoints for specific run."""
        return self.discover_all(run_id)
    
    def print_summary(
        self,
        selection: Optional[CheckpointSelection] = None,
    ) -> str:
        """Print summary of discovered checkpoints."""
        if selection is None:
            selection = self.discover_all()
        
        lines = ["Pipeline Checkpoint Summary", "=" * 40]
        
        for stage in self.STAGES:
            ckpt = selection.get_stage(stage)
            if ckpt:
                lines.append(f"\n{stage.upper()}:")
                lines.append(f"  Path: {ckpt.path}")
                lines.append(f"  Run: {ckpt.run_id}")
                if ckpt.epoch is not None:
                    lines.append(f"  Epoch: {ckpt.epoch}")
                if ckpt.metrics:
                    lines.append(f"  Metrics: {ckpt.metrics}")
            else:
                lines.append(f"\n{stage.upper()}: Not found")
        
        return "\n".join(lines)
    
    def save_summary(
        self,
        selection: CheckpointSelection,
        output_path: str,
    ) -> None:
        """Save selection to JSON."""
        with open(output_path, "w") as f:
            json.dump(selection.to_dict(), f, indent=2)


def create_smoke_test() -> bool:
    """Smoke test for checkpoint discovery."""
    print("Running smoke test...")
    
    base_dir = "out"
    if not os.path.exists(base_dir):
        print(f"  Creating synthetic test directory")
        os.makedirs(base_dir, exist_ok=True)
        
        # Create synthetic checkpoints
        for stage in ["ssl", "bc", "rl"]:
            stage_dir = os.path.join(base_dir, f"{stage}_test_run")
            os.makedirs(stage_dir, exist_ok=True)
            
            # Create dummy checkpoint
            ckpt_path = os.path.join(stage_dir, "final.pt")
            with open(ckpt_path, "w") as f:
                f.write("dummy")
            
            # Create metrics
            metrics = {"loss": 0.5, "ade": 2.5} if stage != "ssl" else {"loss": 0.5}
            metrics_path = os.path.join(stage_dir, "train_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"best": metrics, "final": metrics}, f)
    
    # Test discovery
    discovery = PipelineCheckpointDiscovery(base_dir)
    selection = discovery.discover_all()
    
    print(f"  Discovered stages: {selection.stage_order}")
    
    for stage in ["ssl", "bc", "rl"]:
        ckpt = selection.get_stage(stage)
        if ckpt:
            print(f"  {stage}: {ckpt.path}")
    
    # Test summary
    summary = discovery.print_summary(selection)
    print(f"\n{summary}")
    
    # Test save
    output_path = os.path.join(base_dir, "checkpoint_discovery_summary.json")
    discovery.save_summary(selection, output_path)
    print(f"\n  Saved to: {output_path}")
    
    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pipeline checkpoint discovery and selection"
    )
    parser.add_argument(
        "--base-dir", 
        default="out",
        help="Base directory for checkpoints"
    )
    parser.add_argument(
        "--run-id",
        help="Specific run ID to find"
    )
    parser.add_argument(
        "--stage",
        choices=PipelineCheckpointSelector.METRIC_PRIORITY.keys(),
        help="Specific stage to discover"
    )
    parser.add_argument(
        "--metric",
        help="Metric to use for selection"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discovered checkpoints"
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Show best checkpoint by metric"
    )
    parser.add_argument(
        "--output",
        help="Output JSON path"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test"
    )
    
    args = parser.parse_args()
    
    if args.smoke_test:
        success = create_smoke_test()
        print(f"Smoke test: {'PASSED' if success else 'FAILED'}")
        return
    
    discovery = PipelineCheckpointDiscovery(args.base_dir)
    
    if args.list or args.best:
        selection = discovery.discover_all(args.run_id)
        print(discovery.print_summary(selection))
        
        if args.output:
            discovery.save_summary(selection, args.output)
            print(f"\nSaved to: {args.output}")
    else:
        # Default: show summary
        selection = discovery.discover_all(args.run_id)
        print(discovery.print_summary(selection))
        
        if args.output:
            discovery.save_summary(selection, args.output)
            print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
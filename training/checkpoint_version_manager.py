#!/usr/bin/env python3
"""
Checkpoint Version Manager - Track/checkpoint versions, lineage, and performance evolution.

This module manages the version history of checkpoints across pipeline stages,
tracking which checkpoint versions were created when and what metrics they achieved.
This enables reproducible experiments and easy rollback.

Author: ClawBot
Date: 2026-04-22
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse


@dataclass
class CheckpointVersion:
    """A single checkpoint version with metadata."""
    version_id: str
    stage: str  # ssl, bc, rl
    checkpoint_path: str
    created_at: str
    parent_version_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    run_id: Optional[str] = None
    epoch: Optional[int] = None
    step: Optional[int] = None


@dataclass
class CheckpointLineage:
    """Tracks the lineage of a checkpoint (parent chain)."""
    checkpoint_path: str
    versions: List[CheckpointVersion] = field(default_factory=list)
    current_version_id: str = ""


class CheckpointVersionManager:
    """
    Manages checkpoint versions and their performance history.
    
    Tracks which checkpoint versions exist, their metrics, and lineage.
    Enables finding best checkpoints, rolling back, and reproducing experiments.
    """
    
    def __init__(self, base_dir: str = "out"):
        self.base_dir = Path(base_dir)
        self.version_db_path = self.base_dir / "checkpoint_versions.json"
        self.versions: Dict[str, CheckpointVersion] = {}
        self._load_version_db()
    
    def _load_version_db(self):
        """Load version database from disk."""
        if self.version_db_path.exists():
            try:
                with open(self.version_db_path) as f:
                    data = json.load(f)
                    for v_id, v_data in data.get("versions", {}).items():
                        self.versions[v_id] = CheckpointVersion(**v_data)
            except Exception as e:
                print(f"Warning: Could not load version db: {e}")
    
    def _save_version_db(self):
        """Save version database to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "versions": {v_id: asdict(v) for v_id, v in self.versions.items()},
            "last_updated": datetime.now().isoformat()
        }
        with open(self.version_db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_checkpoint(
        self,
        checkpoint_path: str,
        stage: str,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        notes: str = "",
        parent_version_id: Optional[str] = None,
        run_id: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> str:
        """
        Register a new checkpoint version.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            stage: Pipeline stage (ssl, bc, rl)
            metrics: Performance metrics (ADE, FDE, reward, etc.)
            config: Training configuration
            notes: Notes about this version
            parent_version_id: Parent version for lineage tracking
            run_id: Run identifier
            epoch: Training epoch
            step: Training step
            
        Returns:
            The new version_id
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{stage}_{timestamp}"
        
        version = CheckpointVersion(
            version_id=version_id,
            stage=stage,
            checkpoint_path=checkpoint_path,
            created_at=datetime.now().isoformat(),
            parent_version_id=parent_version_id,
            metrics=metrics or {},
            config=config or {},
            notes=notes,
            run_id=run_id,
            epoch=epoch,
            step=step,
        )
        
        self.versions[version_id] = version
        self._save_version_db()
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[CheckpointVersion]:
        """Get a specific version by ID."""
        return self.versions.get(version_id)
    
    def get_versions_for_stage(self, stage: str) -> List[CheckpointVersion]:
        """Get all versions for a pipeline stage."""
        return [v for v in self.versions.values() if v.stage == stage]
    
    def get_best_version(
        self,
        stage: str,
        metric: str = "ade",
        lower_is_better: bool = True,
    ) -> Optional[CheckpointVersion]:
        """
        Get the best version for a stage by metric.
        
        Args:
            stage: Pipeline stage
            metric: Metric name to optimize
            lower_is_better: Whether lower values are better
            
        Returns:
            Best version or None
        """
        stage_versions = self.get_versions_for_stage(stage)
        if not stage_versions:
            return None
        
        best_version = None
        best_value = float('inf') if lower_is_better else float('-inf')
        
        for version in stage_versions:
            metric_value = version.metrics.get(metric)
            if metric_value is None:
                continue
            
            if lower_is_better:
                if metric_value < best_value:
                    best_value = metric_value
                    best_version = version
            else:
                if metric_value > best_value:
                    best_value = metric_value
                    best_version = version
        
        return best_version
    
    def get_lineage(self, version_id: str) -> List[CheckpointVersion]:
        """
        Get the full lineage chain for a version.
        
        Returns list from oldest ancestor to current version.
        """
        lineage = []
        current = self.versions.get(version_id)
        
        while current:
            lineage.append(current)
            if current.parent_version_id:
                current = self.versions.get(current.parent_version_id)
            else:
                current = None
        
        return list(reversed(lineage))
    
    def compare_versions(
        self,
        version_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> str:
        """
        Compare multiple versions.
        
        Returns a formatted comparison table.
        """
        if metrics is None:
            metrics = ["ade", "fde", "success_rate", "reward"]
        
        lines = ["# Checkpoint Version Comparison", ""]
        lines.append(f"{'Version ID':<25} | " + " | ".join(f"{m:>12}" for m in metrics))
        lines.append("-" * len(lines[1]))
        
        for v_id in version_ids:
            version = self.versions.get(v_id)
            if not version:
                continue
            
            row = f"{version.version_id:<25}"
            for m in metrics:
                value = version.metrics.get(m)
                if value is not None:
                    row += f" | {value:>12.4f}"
                else:
                    row += f" | {'N/A':>12}"
            lines.append(row)
        
        return "\n".join(lines)
    
    def find_checkpoints_in_dir(
        self,
        directory: str,
        pattern: str = "*.pt",
    ) -> List[str]:
        """
        Find all checkpoints in a directory.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern for matching
            
        Returns:
            List of checkpoint paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        
        return [str(p) for p in dir_path.glob(pattern)]
    
    def auto_register_from_dir(
        self,
        directory: str,
        stage: str,
        run_id: Optional[str] = None,
    ) -> List[str]:
        """
        Automatically register all checkpoints in a directory.
        
        Args:
            directory: Directory containing checkpoints
            stage: Pipeline stage
            run_id: Optional run identifier
            
        Returns:
            List of registered version IDs
        """
        version_ids = []
        
        # Find common checkpoint patterns
        for pattern in ["final.pt", "best.pt", "best_reward.pt", "checkpoint.pt"]:
            checkpoints = self.find_checkpoints_in_dir(directory, pattern)
            for ckpt in checkpoints:
                # Try to extract metrics from metrics.json
                metrics = {}
                metrics_path = Path(directory) / "metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                    except:
                        pass
                
                version_id = self.register_checkpoint(
                    checkpoint_path=ckpt,
                    stage=stage,
                    metrics=metrics,
                    run_id=run_id,
                )
                version_ids.append(version_id)
        
        return version_ids
    
    def export_report(self, output_path: str) -> str:
        """
        Export a full version report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the saved report
        """
        lines = ["# Checkpoint Version Report", ""]
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total versions: {len(self.versions)}")
        lines.append("")
        
        # Group by stage
        for stage in ["ssl", "bc", "rl"]:
            versions = self.get_versions_for_stage(stage)
            if not versions:
                continue
            
            lines.append(f"## {stage.upper()} ({len(versions)} versions)")
            lines.append("")
            
            for v in sorted(versions, key=lambda x: x.created_at, reverse=True):
                lines.append(f"- **{v.version_id}**")
                lines.append(f"  - Path: `{v.checkpoint_path}`")
                lines.append(f"  - Created: {v.created_at}")
                
                if v.metrics:
                    metrics_str = ", ".join(
                        f"{k}={v.metrics[k]:.4f}" 
                        for k in sorted(v.metrics.keys())
                    )
                    lines.append(f"  - Metrics: {metrics_str}")
                
                if v.notes:
                    lines.append(f"  - Notes: {v.notes}")
                
                lines.append("")
        
        report = "\n".join(lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        return output_path
    
    def list_versions(
        self,
        stage: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """
        List recent versions.
        
        Args:
            stage: Optional stage filter
            limit: Maximum number to show
            
        Returns:
            Formatted version list
        """
        versions = list(self.versions.values())
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x.created_at, reverse=True)
        versions = versions[:limit]
        
        lines = ["Recent Checkpoint Versions:", ""]
        for v in versions:
            age = datetime.now() - datetime.fromisoformat(v.created_at)
            age_str = f"{age.total_seconds()/3600:.1f}h ago"
            
            metrics_str = ""
            if v.metrics:
                if "ade" in v.metrics:
                    metrics_str = f" ADE={v.metrics['ade']:.3f}"
                elif "reward" in v.metrics:
                    metrics_str = f" reward={v.metrics['reward']:.3f}"
            
            lines.append(
                f"- {v.version_id} ({v.stage}) [{age_str}]{metrics_str}"
            )
        
        return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Checkpoint Version Manager"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a checkpoint")
    register_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    register_parser.add_argument("--stage", required=True, help="Pipeline stage")
    register_parser.add_argument("--metrics", help="Metrics JSON")
    register_parser.add_argument("--notes", default="", help="Notes")
    register_parser.add_argument("--run-id", help="Run ID")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List versions")
    list_parser.add_argument("--stage", help="Filter by stage")
    list_parser.add_argument("--limit", type=int, default=10, help="Limit")
    
    # Best command
    best_parser = subparsers.add_parser("best", help="Find best version")
    best_parser.add_argument("--stage", required=True, help="Pipeline stage")
    best_parser.add_argument("--metric", default="ade", help="Metric")
    best_parser.add_argument("--higher-is-better", action="store_true")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare versions")
    compare_parser.add_argument("--versions", nargs="+", required=True)
    compare_parser.add_argument("--metrics", nargs="+", 
                               default=["ade", "fde", "success_rate", "reward"])
    
    # Auto-register command
    auto_parser = subparsers.add_parser("auto-register", help="Auto-register directory")
    auto_parser.add_argument("--dir", required=True, help="Directory")
    auto_parser.add_argument("--stage", required=True, help="Pipeline stage")
    auto_parser.add_argument("--run-id", help="Run ID")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export report")
    export_parser.add_argument("--output", required=True, help="Output path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = CheckpointVersionManager()
    
    if args.command == "register":
        metrics = {}
        if args.metrics:
            metrics = json.loads(args.metrics)
        
        version_id = manager.register_checkpoint(
            checkpoint_path=args.checkpoint,
            stage=args.stage,
            metrics=metrics,
            notes=args.notes,
            run_id=args.run_id,
        )
        print(f"Registered: {version_id}")
    
    elif args.command == "list":
        print(manager.list_versions(stage=args.stage, limit=args.limit))
    
    elif args.command == "best":
        version = manager.get_best_version(
            stage=args.stage,
            metric=args.metric,
            lower_is_better=not args.higher_is_better,
        )
        if version:
            print(f"Best: {version.version_id}")
            print(f"  Path: {version.checkpoint_path}")
            print(f"  {args.metric}: {version.metrics.get(args.metric)}")
        else:
            print(f"No versions found for stage: {args.stage}")
    
    elif args.command == "compare":
        print(manager.compare_versions(args.versions, args.metrics))
    
    elif args.command == "auto-register":
        version_ids = manager.auto_register_from_dir(
            args.dir, args.stage, args.run_id
        )
        print(f"Registered {len(version_ids)} checkpoints")
        for v_id in version_ids:
            print(f"  - {v_id}")
    
    elif args.command == "export":
        output = manager.export_report(args.output)
        print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
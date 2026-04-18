#!/usr/bin/env python3
"""
Pipeline Progress Tracker

Monitors and reports on the status of all pipeline stages:
- Data Loading (Waymo episodes)
- SSL Pretraining
- Waypoint BC
- RL Refinement
- CARLA Evaluation

Provides unified status view, checkpoint discovery, and progress metrics.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============== Pipeline Stages ==============

PIPELINE_STAGES = [
    "data_loading",
    "ssl_pretrain", 
    "waypoint_bc",
    "rl_refinement",
    "carla_eval"
]


# ============== Configuration ==============

@dataclass
class TrackerConfig:
    """Configuration for pipeline tracker."""
    workspace_dir: str = "/data/.openclaw/workspace"
    data_dir: str = "data"
    training_dir: str = "training"
    sim_dir: str = "sim"
    output_dir: str = "out"
    stages: List[str] = field(default_factory=lambda: PIPELINE_STAGES)


@dataclass
class StageStatus:
    """Status of a single pipeline stage."""
    name: str
    enabled: bool = False
    data_available: bool = False
    checkpoints: List[str] = field(default_factory=list)
    latest_run: Optional[str] = None
    latest_metrics: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class PipelineStatus:
    """Overall pipeline status."""
    timestamp: str
    stages: Dict[str, StageStatus] = field(default_factory=dict)
    overall_health: str = "unknown"
    next_stage: Optional[str] = None
    blockers: List[str] = field(default_factory=list)


# ============== Data Loading Stage ==============

def check_data_loading(config: TrackerConfig) -> StageStatus:
    """Check data loading stage status."""
    status = StageStatus(name="data_loading")
    
    # Check for Waymo episodes
    waymo_dir = os.path.join(config.workspace_dir, config.data_dir, "waymo")
    episode_index = os.path.join(config.workspace_dir, "training", "episodes", "episode_index.json")
    
    if os.path.exists(waymo_dir):
        # Count episodes
        episode_files = []
        for root, dirs, files in os.walk(waymo_dir):
            episode_files.extend([f for f in files if f.endswith('.json') or f.endswith('.pkl')])
        
        status.enabled = True
        status.data_available = len(episode_files) > 0
    
    # Check episode index
    if os.path.exists(episode_index):
        try:
            with open(episode_index, 'r') as f:
                index_data = json.load(f)
                if isinstance(index_data, dict) and 'episodes' in index_data:
                    status.data_available = len(index_data['episodes']) > 0
        except Exception:
            pass
    
    return status


# ============== SSL Pretrain Stage ==============

def check_ssl_pretrain(config: TrackerConfig) -> StageStatus:
    """Check SSL pretraining stage status."""
    status = StageStatus(name="ssl_pretrain")
    
    # Look for SSL checkpoints
    ssl_out_dir = os.path.join(config.workspace_dir, config.output_dir, "ssl")
    
    if os.path.exists(ssl_out_dir):
        status.enabled = True
        
        # Find latest run
        runs = [d for d in os.listdir(ssl_out_dir) if os.path.isdir(os.path.join(ssl_out_dir, d))]
        if runs:
            runs.sort(reverse=True)
            status.latest_run = runs[0]
            
            # Check for checkpoints
            latest_path = os.path.join(ssl_out_dir, runs[0])
            for fname in ["final.pt", "best.pt", "checkpoint.pt"]:
                ckpt_path = os.path.join(latest_path, fname)
                if os.path.exists(ckpt_path):
                    status.checkpoints.append(fname)
            
            # Load metrics if available
            metrics_path = os.path.join(latest_path, "metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        status.latest_metrics = json.load(f)
                except Exception:
                    pass
    
    return status


# ============== Waypoint BC Stage ==============

def check_waypoint_bc(config: TrackerConfig) -> StageStatus:
    """Check waypoint BC stage status."""
    status = StageStatus(name="waypoint_bc")
    
    # Look for BC checkpoints
    bc_out_dir = os.path.join(config.workspace_dir, config.output_dir, "bc")
    sft_out_dir = os.path.join(config.workspace_dir, config.output_dir, "sft")
    
    search_dirs = [bc_out_dir, sft_out_dir]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            status.enabled = True
            
            runs = [d for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d))]
            if runs:
                runs.sort(reverse=True)
                if not status.latest_run or runs[0] > status.latest_run:
                    status.latest_run = runs[0]
                
                # Check for checkpoints
                for run in runs[:3]:  # Check top 3 runs
                    latest_path = os.path.join(search_dir, run)
                    for fname in ["final.pt", "best.pt", "checkpoint.pt"]:
                        ckpt_path = os.path.join(latest_path, fname)
                        if os.path.exists(ckpt_path):
                            if fname not in status.checkpoints:
                                status.checkpoints.append(f"{run}/{fname}")
                
                # Load metrics if available
                latest_path = os.path.join(search_dir, runs[0])
                metrics_path = os.path.join(latest_path, "metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r') as f:
                            status.latest_metrics = json.load(f)
                    except Exception:
                        pass
    
    return status


# ============== RL Refinement Stage ==============

def check_rl_refinement(config: TrackerConfig) -> StageStatus:
    """Check RL refinement stage status."""
    status = StageStatus(name="rl_refinement")
    
    # Look for RL checkpoints
    rl_out_dir = os.path.join(config.workspace_dir, config.output_dir, "rl")
    
    if os.path.exists(rl_out_dir):
        status.enabled = True
        
        runs = [d for d in os.listdir(rl_out_dir) if os.path.isdir(os.path.join(rl_out_dir, d))]
        if runs:
            runs.sort(reverse=True)
            status.latest_run = runs[0]
            
            # Check for checkpoints
            latest_path = os.path.join(rl_out_dir, runs[0])
            for fname in ["final.pt", "best.pt", "best_reward.pt", "checkpoint.pt"]:
                ckpt_path = os.path.join(latest_path, fname)
                if os.path.exists(ckpt_path):
                    status.checkpoints.append(fname)
            
            # Load metrics if available
            metrics_path = os.path.join(latest_path, "metrics.json")
            train_metrics_path = os.path.join(latest_path, "train_metrics.json")
            
            for mp in [metrics_path, train_metrics_path]:
                if os.path.exists(mp):
                    try:
                        with open(mp, 'r') as f:
                            status.latest_metrics = json.load(f)
                        break
                    except Exception:
                        pass
    
    return status


# ============== CARLA Eval Stage ==============

def check_carla_eval(config: TrackerConfig) -> StageStatus:
    """Check CARLA evaluation stage status."""
    status = StageStatus(name="carla_eval")
    
    # Look for eval results
    eval_out_dir = os.path.join(config.workspace_dir, config.output_dir, "eval")
    
    if os.path.exists(eval_out_dir):
        status.enabled = True
        
        runs = [d for d in os.listdir(eval_out_dir) if os.path.isdir(os.path.join(eval_out_dir, d))]
        if runs:
            runs.sort(reverse=True)
            status.latest_run = runs[0]
            
            # Load metrics if available
            latest_path = os.path.join(eval_out_dir, runs[0])
            for fname in ["metrics.json", "results.json"]:
                metrics_path = os.path.join(latest_path, fname)
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r') as f:
                            status.latest_metrics = json.load(f)
                        break
                    except Exception:
                        pass
    
    return status


# ============== Pipeline Tracker ==============

class PipelineProgressTracker:
    """Tracks progress across all pipeline stages."""
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.stage_checkers = {
            "data_loading": check_data_loading,
            "ssl_pretrain": check_ssl_pretrain,
            "waypoint_bc": check_waypoint_bc,
            "rl_refinement": check_rl_refinement,
            "carla_eval": check_carla_eval
        }
    
    def get_status(self) -> PipelineStatus:
        """Get overall pipeline status."""
        status = PipelineStatus(timestamp=datetime.now().isoformat())
        
        # Check each stage
        for stage_name in self.config.stages:
            if stage_name in self.stage_checkers:
                stage_status = self.stage_checkers[stage_name](self.config)
                status.stages[stage_name] = stage_status
        
        # Determine overall health
        health = self._compute_health(status)
        status.overall_health = health
        
        # Determine next stage
        status.next_stage = self._get_next_stage(status)
        
        # Identify blockers
        status.blockers = self._get_blockers(status)
        
        return status
    
    def _compute_health(self, status: PipelineStatus) -> str:
        """Compute overall pipeline health."""
        if not status.stages:
            return "unknown"
        
        # Check if data is available
        data_stage = status.stages.get("data_loading")
        if not data_stage or not data_stage.data_available:
            return "no_data"
        
        # Check progression
        enabled_stages = [s for s in status.stages.values() if s.enabled]
        
        if not enabled_stages:
            return "no_training"
        
        # Check if RL has run (indicates full pipeline completion)
        rl_stage = status.stages.get("rl_refinement")
        if rl_stage and rl_stage.latest_metrics:
            return "healthy"
        
        # Check if BC has run
        bc_stage = status.stages.get("waypoint_bc")
        if bc_stage and bc_stage.latest_metrics:
            return "partial"
        
        return "initializing"
    
    def _get_next_stage(self, status: PipelineStatus) -> Optional[str]:
        """Determine the next stage that should run."""
        stage_order = ["data_loading", "ssl_pretrain", "waypoint_bc", "rl_refinement", "carla_eval"]
        
        for stage_name in stage_order:
            stage = status.stages.get(stage_name)
            if not stage or not stage.enabled:
                return stage_name
            if not stage.checkpoints:
                return stage_name
        
        return None  # Pipeline complete
    
    def _get_blockers(self, status: PipelineStatus) -> List[str]:
        """Identify pipeline blockers."""
        blockers = []
        
        # Check for data
        data_stage = status.stages.get("data_loading")
        if not data_stage or not data_stage.data_available:
            blockers.append("No waymo episode data available")
        
        return blockers
    
    def print_status(self, status: PipelineStatus):
        """Print human-readable status."""
        print(f"{'='*60}")
        print(f"Pipeline Progress Tracker")
        print(f"{'='*60}")
        print(f"Timestamp: {status.timestamp}")
        print(f"Overall Health: {status.overall_health}")
        print(f"Next Stage: {status.next_stage or 'complete'}")
        print()
        
        if status.blockers:
            print("⚠️  Blockers:")
            for blocker in status.blockers:
                print(f"  - {blocker}")
            print()
        
        print("Stage Status:")
        print("-" * 60)
        
        for stage_name in self.config.stages:
            stage = status.stages.get(stage_name)
            if not stage:
                continue
            
            # Status icon
            if stage.checkpoints:
                icon = "✅"
            elif stage.enabled:
                icon = "🔄"
            else:
                icon = "⏳"
            
            print(f"{icon} {stage_name}")
            
            if stage.latest_run:
                print(f"   Latest run: {stage.latest_run}")
            
            if stage.checkpoints:
                print(f"   Checkpoints: {', '.join(stage.checkpoints[:3])}")
            
            if stage.latest_metrics:
                # Show key metrics
                if 'loss' in stage.latest_metrics:
                    print(f"   Loss: {stage.latest_metrics['loss']:.4f}")
                if 'reward' in stage.latest_metrics:
                    print(f"   Reward: {stage.latest_metrics['reward']:.2f}")
                if 'ade' in stage.latest_metrics:
                    print(f"   ADE: {stage.latest_metrics['ade']:.2f}m")
                if 'fde' in stage.latest_metrics:
                    print(f"   FDE: {stage.latest_metrics['fde']:.2f}m")
            
            print()
        
        print("=" * 60)
    
    def save_status(self, status: PipelineStatus, output_path: str):
        """Save status to JSON file."""
        # Convert to dict
        status_dict = {
            "timestamp": status.timestamp,
            "overall_health": status.overall_health,
            "next_stage": status.next_stage,
            "blockers": status.blockers,
            "stages": {}
        }
        
        for stage_name, stage in status.stages.items():
            stage_dict = {
                "enabled": stage.enabled,
                "data_available": stage.data_available,
                "checkpoints": stage.checkpoints,
                "latest_run": stage.latest_run,
                "latest_metrics": stage.latest_metrics
            }
            status_dict["stages"][stage_name] = stage_dict
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(status_dict, f, indent=2)
        
        print(f"Status saved to: {output_path}")


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Track pipeline progress")
    parser.add_argument("--workspace", type=str, default="/data/.openclaw/workspace",
                        help="Workspace directory")
    parser.add_argument("--stages", nargs="+", default=PIPELINE_STAGES,
                        help="Stages to check")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (optional)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable")
    
    args = parser.parse_args()
    
    # Create config
    config = TrackerConfig(
        workspace_dir=args.workspace,
        stages=args.stages
    )
    
    # Get status
    tracker = PipelineProgressTracker(config)
    status = tracker.get_status()
    
    if args.json:
        # Print JSON output
        import json
        output = {
            "timestamp": status.timestamp,
            "overall_health": status.overall_health,
            "next_stage": status.next_stage,
            "blockers": status.blockers,
            "stages": {}
        }
        for stage_name, stage in status.stages.items():
            output["stages"][stage_name] = {
                "enabled": stage.enabled,
                "data_available": stage.data_available,
                "checkpoints": stage.checkpoints,
                "latest_run": stage.latest_run,
                "latest_metrics": stage.latest_metrics
            }
        print(json.dumps(output, indent=2))
    else:
        # Print human-readable status
        tracker.print_status(status)
        
        # Save if requested
        if args.output:
            tracker.save_status(status, args.output)


if __name__ == "__main__":
    main()
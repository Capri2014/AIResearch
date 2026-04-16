#!/usr/bin/env python3
"""
Pipeline Checkpoint Manager - Unified checkpoint management across all pipeline stages.

Manages, validates, and traces lineage of checkpoints from:
- SSL pretrain (CombinedSSLModel, JEPA, MIM)
- Waypoint BC (WaypointBCModel)
- RL Refinement (RefinementPolicy)

Supports checkpoint health validation, lineage tracing, and unified loading.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch
import numpy as np


# Pipeline stage names
STAGE_SSL = "ssl"
STAGE_WAYPOINT_BC = "waypoint_bc"
STAGE_RL = "rl"
STAGE_ALL = [STAGE_SSL, STAGE_WAYPOINT_BC, STAGE_RL]


@dataclass
class CheckpointHealth:
    """Health information for a checkpoint."""
    exists: bool = False
    size_mb: float = 0.0
    can_load: bool = False
    model_type: Optional[str] = None
    stage: Optional[str] = None
    run_id: Optional[str] = None
    epoch: Optional[int] = None
    metrics_available: bool = False
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class CheckpointLineage:
    """Lineage information for a pipeline checkpoint chain."""
    waymo_data: Optional[str] = None
    ssl_checkpoint: Optional[str] = None
    waypoint_bc_checkpoint: Optional[str] = None
    rl_checkpoint: Optional[str] = None
    total_stages: int = 0


class PipelineCheckpointManager:
    """Unified checkpoint manager for all pipeline stages."""
    
    def __init__(self, base_dir: str = "training/out"):
        self.base_dir = Path(base_dir)
        
    def find_checkpoints(self, stage: str) -> list[dict]:
        """Find all checkpoints for a given stage."""
        checkpoints = []
        
        if stage == STAGE_SSL:
            # Find SSL checkpoints
            ssl_dirs = list(self.base_dir.glob("ssl*/"))
            for ssl_dir in ssl_dirs:
                for pt_file in ["best.pt", "final.pt", "checkpoint.pt"]:
                    ckpt_path = ssl_dir / pt_file
                    if ckpt_path.exists():
                        checkpoints.append({
                            "path": str(ckpt_path),
                            "run_id": ssl_dir.name,
                            "epoch": self._extract_epoch(pt_file),
                            "stage": STAGE_SSL,
                            "type": "ssl"
                        })
                        
        elif stage == STAGE_WAYPOINT_BC:
            # Find waypoint BC checkpoints
            bc_dirs = list(self.base_dir.glob("bc*/")) + list(self.base_dir.glob("waypoint_bc*/"))
            for bc_dir in bc_dirs:
                for pt_file in ["best.pt", "final.pt", "checkpoint.pt"]:
                    ckpt_path = bc_dir / pt_file
                    if ckpt_path.exists():
                        checkpoints.append({
                            "path": str(ckpt_path),
                            "run_id": bc_dir.name,
                            "epoch": self._extract_epoch(pt_file),
                            "stage": STAGE_WAYPOINT_BC,
                            "type": "waypoint_bc"
                        })
                        
        elif stage == STAGE_RL:
            # Find RL checkpoints
            rl_dirs = list(self.base_dir.glob("rl*/")) + list(self.base_dir.glob("rl_refine*/"))
            for rl_dir in rl_dirs:
                for pt_file in ["best.pt", "final.pt", "best_reward.pt", "checkpoint.pt"]:
                    ckpt_path = rl_dir / pt_file
                    if ckpt_path.exists():
                        checkpoints.append({
                            "path": str(ckpt_path),
                            "run_id": rl_dir.name,
                            "epoch": self._extract_epoch(pt_file),
                            "stage": STAGE_RL,
                            "type": "rl"
                        })
                        
        return checkpoints
    
    def _extract_epoch(self, pt_file: str) -> Optional[int]:
        """Extract epoch number from checkpoint filename."""
        if "best" in pt_file or "final" in pt_file:
            return None
        if "epoch_" in pt_file:
            try:
                return int(pt_file.replace("checkpoint.pt", "").split("_")[-1])
            except:
                pass
        return None
    
    def check_health(self, checkpoint_path: str) -> CheckpointHealth:
        """Check the health of a checkpoint file."""
        health = CheckpointHealth()
        
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            health.exists = False
            health.error = f"Checkpoint not found: {checkpoint_path}"
            return health
            
        health.exists = True
        health.size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        
        # Try loading checkpoint
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            # Identify checkpoint type
            if "model_state" in state:
                if "encoder" in str(state.get("config", {})).lower():
                    health.model_type = "ssl"
                    health.stage = STAGE_SSL
                elif "waypoint" in str(state.get("config", {})).lower():
                    health.model_type = "waypoint_bc"
                    health.stage = STAGE_WAYPOINT_BC
                else:
                    health.model_type = "unknown"
            elif "policy_state" in state or "delta_head" in str(state):
                health.model_type = "rl"
                health.stage = STAGE_RL
            else:
                health.model_type = "unknown"
                
            health.run_id = state.get("run_id", ckpt_path.parent.name)
            health.epoch = state.get("epoch")
            health.can_load = True
            
            # Check for metrics
            metrics_path = ckpt_path.parent / "metrics.json"
            if metrics_path.exists():
                health.metrics_available = True
                with open(metrics_path) as f:
                    health.metrics = json.load(f)
                    
        except Exception as e:
            health.can_load = False
            health.error = str(e)
            
        return health
    
    def trace_lineage(self, waypoint_bc_path: Optional[str] = None, 
                   rl_path: Optional[str] = None) -> CheckpointLineage:
        """Trace the lineage of checkpoints from data to final model."""
        lineage = CheckpointLineage()
        
        if waypoint_bc_path:
            lineage.waypoint_bc_checkpoint = waypoint_bc_path
            lineage.total_stages = 2
            
        if rl_path:
            lineage.rl_checkpoint = rl_path
            lineage.total_stages = 3
            
        return lineage
    
    def compare_checkpoints(self, checkpoint_paths: list[str]) -> dict:
        """Compare multiple checkpoints."""
        results = []
        
        for path in checkpoint_paths:
            health = self.check_health(path)
            results.append({
                "path": path,
                "exists": health.exists,
                "size_mb": health.size_mb,
                "can_load": health.can_load,
                "model_type": health.model_type,
                "stage": health.stage,
                "run_id": health.run_id,
                "epoch": health.epoch,
                "metrics": health.metrics if health.metrics_available else None
            })
            
        return {"checkpoints": results, "count": len(results)}
    
    def validate_pipeline(self) -> dict:
        """Validate the entire pipeline has working checkpoints."""
        validation = {
            "stages": {},
            "ready": True,
            "gaps": []
        }
        
        # Check each stage
        for stage in STAGE_ALL:
            cks = self.find_checkpoints(stage)
            validation["stages"][stage] = {
                "count": len(cks),
                "checkpoints": [c["path"] for c in cks]
            }
            if len(cks) == 0:
                validation["gaps"].append(f"No {stage} checkpoints found")
                validation["ready"] = False
                
        return validation


def load_checkpoint_for_stage(checkpoint_path: str, stage: str):
    """Load a checkpoint for a specific pipeline stage."""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if stage == STAGE_SSL:
        # Try loading SSL model
        try:
            from training.pretrain.run_combined_ssl import CombinedSSLModel
            model = CombinedSSLModel()
            if "model_state" in state:
                model.load_state_dict(state["model_state"])
            return model
        except ImportError:
            return state
            
    elif stage == STAGE_WAYPOINT_BC:
        # Try loading waypoint BC model  
        try:
            from training.sft.train_waypoint_bc import WaypointBCModel
            model = WaypointBCModel()
            if "model_state" in state:
                model.load_state_dict(state["model_state"])
            return model
        except ImportError:
            return state
            
    elif stage == STAGE_RL:
        # Return RL state
        return state
        
    return state


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Checkpoint Manager - Unified checkpoint management"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List checkpoints")
    list_parser.add_argument("--stage", choices=STAGE_ALL, help="Filter by stage")
    list_parser.add_argument("--run-id", help="Filter by run ID")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check checkpoint health")
    health_parser.add_argument("checkpoint", help="Checkpoint path")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate pipeline")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare checkpoints")
    compare_parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load checkpoint")
    load_parser.add_argument("checkpoint", help="Checkpoint path")
    load_parser.add_argument("--stage", choices=STAGE_ALL, required=True, help="Pipeline stage")
    
    args = parser.parse_args()
    
    manager = PipelineCheckpointManager()
    
    if args.command == "list":
        stage = args.stage or STAGE_WAYPOINT_BC
        checkpoints = manager.find_checkpoints(stage)
        print(f"Found {len(checkpoints)} {stage} checkpoints:")
        for ck in checkpoints:
            print(f"  - {ck['path']} (run_id={ck['run_id']}, epoch={ck['epoch']})")
            
    elif args.command == "health":
        health = manager.check_health(args.checkpoint)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"  Exists: {health.exists}")
        print(f"  Size: {health.size_mb:.2f} MB")
        print(f"  Can load: {health.can_load}")
        print(f"  Model type: {health.model_type}")
        print(f"  Stage: {health.stage}")
        print(f"  Run ID: {health.run_id}")
        print(f"  Epoch: {health.epoch}")
        print(f"  Metrics: {health.metrics_available}")
        if health.error:
            print(f"  Error: {health.error}")
            
    elif args.command == "validate":
        validation = manager.validate_pipeline()
        print("Pipeline Validation:")
        for stage, info in validation["stages"].items():
            print(f"  {stage}: {info['count']} checkpoints")
        print(f"  Ready: {validation['ready']}")
        if validation["gaps"]:
            print("  Gaps:")
            for gap in validation["gaps"]:
                print(f"    - {gap}")
                
    elif args.command == "compare":
        results = manager.compare_checkpoints(args.checkpoints)
        print(f"Compared {results['count']} checkpoints:")
        for i, ck in enumerate(results['checkpoints']):
            print(f"\n{i+1}. {ck['path']}")
            print(f"   Exists: {ck['exists']}, Can load: {ck['can_load']}")
            print(f"   Type: {ck['model_type']}, Run ID: {ck['run_id']}")
            
    elif args.command == "load":
        model = load_checkpoint_for_stage(args.checkpoint, args.stage)
        print(f"Loaded {type(model).__name__} from {args.checkpoint}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
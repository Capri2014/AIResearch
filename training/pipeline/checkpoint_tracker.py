#!/usr/bin/env python3
"""
Pipeline Checkpoint Tracker

Tracks and validates checkpoint artifacts across pipeline stages.
Ensures proper dependencies between stages and provides versioning.

Usage:
    python checkpoint_tracker.py --list
    python checkpoint_tracker.py --latest ssl
    python checkpoint_tracker.py --validate ssl,bc,rl
    
    python -m training.pipeline.checkpoint_tracker --latest bc
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Base directory for all checkpoint artifacts
BASE_DIR = Path("out")
PIPELINE_DIR = BASE_DIR / "pipeline_benchmark"
ARTIFACTS = {
    "ssl": PIPELINE_DIR / "ssl" / "encoder_final.pt",
    "bc": PIPELINE_DIR / "bc" / "waypoint_final.pt", 
    "rl": PIPELINE_DIR / "rl" / "model_final.pt",
    "sft": BASE_DIR / "sft_waypoint" / "final_model.pt",
}

# Alternative paths for RL/SSL checkpoint storage
SSL_CKPT_PATHS = [
    PIPELINE_DIR / "ssl" / "encoder_final.pt",
    BASE_DIR / "ssl_pretrain" / "encoder_final.pt",
    BASE_DIR / "ssl_pretrain" / "checkpoint.pt",
    BASE_DIR / "checkpoints" / "ssl_latest.pt",
]
BC_CKPT_PATHS = [
    PIPELINE_DIR / "bc" / "waypoint_final.pt",
    BASE_DIR / "bc_waypoint" / "final_model.pt",
    BASE_DIR / "checkpoints" / "bc_latest.pt",
]
RL_CKPT_PATHS = [
    PIPELINE_DIR / "rl" / "model_final.pt",
    BASE_DIR / "rl_refine" / "model_final.pt",
    BASE_DIR / "rl_delta_waypoint" / "final_model.pt",
    BASE_DIR / "checkpoints" / "rl_latest.pt",
]
SFT_CKPT_PATHS = [
    BASE_DIR / "sft_waypoint" / "final_model.pt",
    BASE_DIR / "sft_waypoint_model.pt",
    BASE_DIR / "checkpoints" / "sft_latest.pt",
]


def get_checkpoint_path(stage: str) -> Optional[Path]:
    """Get the checkpoint path for a given stage, checking multiple possible locations."""
    path_list = {
        "ssl": SSL_CKPT_PATHS,
        "bc": BC_CKPT_PATHS,
        "rl": RL_CKPT_PATHS,
        "sft": SFT_CKPT_PATHS,
    }.get(stage.lower(), [])
    
    for path in path_list:
        if path.exists():
            return path
    
    # Try direct artifact path
    direct = ARTIFACTS.get(stage.lower())
    if direct and direct.exists():
        return direct
    
    return None


def list_checkpoints() -> dict:
    """List all available checkpoints with their status."""
    result = {}
    for stage in ["ssl", "bc", "rl", "sft"]:
        path = get_checkpoint_path(stage)
        if path:
            stat = path.stat()
            result[stage] = {
                "path": str(path),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "exists": True,
            }
        else:
            result[stage] = {"exists": False}
    
    return result


def get_latest_checkpoint(stage: str) -> Optional[str]:
    """Get the path to the latest checkpoint for a stage."""
    path = get_checkpoint_path(stage)
    return str(path) if path else None


def validate_dependencies(stages: list) -> dict:
    """Validate that checkpoints exist for the requested stages."""
    required = {
        "ssl": ["data"],
        "bc": ["ssl"],
        "rl": ["bc"],
        "carla": ["rl"],
    }
    
    results = {}
    for stage in stages:
        results[stage] = {"valid": True, "missing": [], "warnings": []}
        
        # Check if stage is valid
        if stage not in required and stage != "data":
            if stage != "data":
                results[stage]["warnings"].append(f"Unknown stage: {stage}")
        
        # Check dependencies
        if stage in required:
            for dep in required[stage]:
                dep_path = get_checkpoint_path(dep)
                if not dep_path:
                    results[stage]["missing"].append(dep)
                    results[stage]["valid"] = False
                else:
                    results[stage]["dependencies"] = {
                        dep: dep_path
                    }
        else:
            # For data stage or unknown, just note it
            pass
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Pipeline Checkpoint Tracker")
    parser.add_argument("--list", action="store_true", help="List all available checkpoints")
    parser.add_argument("--latest", type=str, help="Get latest checkpoint for stage")
    parser.add_argument("--validate", type=str, help="Comma-separated stages to validate")
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    if args.list:
        result = list_checkpoints()
        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        return
    
    if args.latest:
        path = get_latest_checkpoint(args.latest)
        if path:
            print(path)
        else:
            print(f"No checkpoint found for stage: {args.latest}", file=sys.stderr)
            sys.exit(1)
        return
    
    if args.validate:
        stages = [s.strip() for s in args.validate.split(",")]
        result = validate_dependencies(stages)
        print(json.dumps(result, indent=2))
        
        # Exit with error if any validation failed
        failed = [s for s, r in result.items() if not r["valid"]]
        if failed:
            print(f"Validation failed for: {failed}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Default: list all
    result = list_checkpoints()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
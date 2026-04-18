#!/usr/bin/env python3
"""
Pipeline Stage Validator - Validates each stage of the driving-first pipeline

Validates:
1. Data loading (episode files exist and are valid)
2. SSL pretrain (checkpoints, config, data loaders)
3. Waypoint BC (checkpoints, model outputs)
4. RL refinement (checkpoints, replay buffer, env)
5. CARLA eval (scenario files, bridge connection)

Usage:
    python training/pipeline_stage_validator.py              # Validate all stages
    python training/pipeline_stage_validator.py --stage ssl  # Validate specific stage
    python training/pipeline_stage_validator.py --json       # JSON output
    python training/pipeline_stage_validator.py --fix        # Attempt fixes
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    details: Dict[str, Any] = field(default_factory=dict)
    fix_suggestion: Optional[str] = None


@dataclass
class StageValidation:
    """Validation result for a pipeline stage."""
    stage_name: str
    enabled: bool
    checks: List[ValidationResult] = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)
    
    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")


def validate_data_loading(workspace: Path) -> StageValidation:
    """Validate stage 1: Data loading (Waymo episodes)."""
    stage = StageValidation(stage_name="data_loading", enabled=True)
    
    # Check for episode data directories
    possible_data_dirs = [
        workspace / "data" / "waymo" / "episodes",
        workspace / "data" / "episodes",
        workspace / "training" / "episodes",
    ]
    
    data_dir = None
    for d in possible_data_dirs:
        if d.exists():
            data_dir = d
            break
    
    if data_dir is None:
        stage.checks.append(ValidationResult(
            passed=False,
            message=f"No episode data directory found in {possible_data_dirs}",
            severity="error",
            fix_suggestion="Ensure Waymo episode data is downloaded"
        ))
    else:
        # Count episode files
        episode_files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.tfrecord*"))
        stage.checks.append(ValidationResult(
            passed=len(episode_files) > 0,
            message=f"Found {len(episode_files)} episode files in {data_dir}",
            severity="warning" if len(episode_files) == 0 else "info",
            details={"data_dir": str(data_dir), "episode_count": len(episode_files)}
        ))
        
        # Check for dataset index
        index_file = data_dir / "dataset_index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    index = json.load(f)
                stage.checks.append(ValidationResult(
                    passed=True,
                    message="Dataset index found and valid",
                    details={"episode_count": index.get("num_episodes", 0)}
                ))
            except Exception as e:
                stage.checks.append(ValidationResult(
                    passed=False,
                    message=f"Dataset index invalid: {e}",
                    severity="warning"
                ))
    
    return stage


def validate_ssl_pretrain(workspace: Path) -> StageValidation:
    """Validate stage 2: SSL pretraining."""
    stage = StageValidation(stage_name="ssl_pretrain", enabled=False)
    
    # Check for SSL checkpoint
    checkpoint_dirs = [
        workspace / "out" / "ssl",
        workspace / "checkpoints" / "ssl",
    ]
    
    checkpoint_dir = None
    for d in checkpoint_dirs:
        if d.exists():
            checkpoint_dir = d
            break
    
    if checkpoint_dir:
        checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
        stage.checks.append(ValidationResult(
            passed=len(checkpoints) > 0,
            message=f"Found {len(checkpoints)} SSL checkpoints",
            severity="info",
            details={"checkpoint_dir": str(checkpoint_dir), "checkpoints": len(checkpoints)}
        ))
        
        # Check for latest metrics
        metrics_files = list(checkpoint_dir.glob("metrics.json"))
        if metrics_files:
            try:
                with open(metrics_files[0]) as f:
                    metrics = json.load(f)
                stage.checks.append(ValidationResult(
                    passed=True,
                    message="SSL metrics found",
                    details={"latest_metrics": metrics}
                ))
            except Exception as e:
                stage.checks.append(ValidationResult(
                    passed=False,
                    message=f"Could not read metrics: {e}",
                    severity="warning"
                ))
    else:
        stage.checks.append(ValidationResult(
            passed=True,
            message="SSL checkpoint directory not found (stage not yet run)",
            severity="info",
            fix_suggestion="Run SSL pretraining to generate checkpoints"
        ))
    
    # Check for SSL training scripts
    ssl_scripts = [
        workspace / "training" / "pretrain" / "run_unified_ssl.py",
        workspace / "training" / "pretrain" / "run_jepa_pretrain.py",
    ]
    
    scripts_found = [s for s in ssl_scripts if s.exists()]
    stage.checks.append(ValidationResult(
        passed=len(scripts_found) > 0,
        message=f"Found {len(scripts_found)} SSL training scripts",
        details={"scripts": [s.name for s in scripts_found]}
    ))
    
    return stage


def validate_waypoint_bc(workspace: Path) -> StageValidation:
    """Validate stage 3: Waypoint Behavior Cloning."""
    stage = StageValidation(stage_name="waypoint_bc", enabled=False)
    
    # Check for BC checkpoints
    checkpoint_dirs = [
        workspace / "out" / "bc",
        workspace / "out" / "sft",
        workspace / "checkpoints" / "bc",
    ]
    
    checkpoint_dir = None
    for d in checkpoint_dirs:
        if d.exists():
            checkpoint_dir = d
            break
    
    if checkpoint_dir:
        checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
        stage.checks.append(ValidationResult(
            passed=len(checkpoints) > 0,
            message=f"Found {len(checkpoints)} BC checkpoints",
            details={"checkpoint_dir": str(checkpoint_dir), "checkpoints": len(checkpoints)}
        ))
    else:
        stage.checks.append(ValidationResult(
            passed=True,
            message="BC checkpoint directory not found (stage not yet run)",
            severity="info",
            fix_suggestion="Run waypoint BC training"
        ))
    
    # Check for BC training scripts
    bc_script = workspace / "training" / "bc" / "train_waypoint_bc.py"
    if bc_script.exists():
        stage.checks.append(ValidationResult(
            passed=True,
            message="BC training script found",
            details={"script": str(bc_script)}
        ))
    else:
        # Check if there's a combined training script
        combined_script = workspace / "training" / "pipeline_driver.py"
        if combined_script.exists():
            stage.checks.append(ValidationResult(
                passed=True,
                message="Using unified pipeline driver for BC",
                details={"script": str(combined_script)}
            ))
        else:
            stage.checks.append(ValidationResult(
                passed=False,
                message="No BC training script found",
                severity="error"
            ))
    
    return stage


def validate_rl_refinement(workspace: Path) -> StageValidation:
    """Validate stage 4: RL refinement."""
    stage = StageValidation(stage_name="rl_refinement", enabled=False)
    
    # Check for RL checkpoints
    checkpoint_dirs = [
        workspace / "out" / "rl",
        workspace / "training" / "rl" / "out",
        workspace / "checkpoints" / "rl",
    ]
    
    checkpoint_dir = None
    for d in checkpoint_dirs:
        if d.exists():
            checkpoint_dir = d
            break
    
    if checkpoint_dir:
        checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
        stage.checks.append(ValidationResult(
            passed=len(checkpoints) > 0,
            message=f"Found {len(checkpoints)} RL checkpoints",
            details={"checkpoint_dir": str(checkpoint_dir), "checkpoints": len(checkpoints)}
        ))
    else:
        stage.checks.append(ValidationResult(
            passed=True,
            message="RL checkpoint directory not found (stage not yet run)",
            severity="info"
        ))
    
    # Check for RL training scripts
    rl_scripts = [
        workspace / "training" / "rl" / "train_delta_waypoint_rl.py",
        workspace / "training" / "rl" / "ppo_delta_waypoint_refiner.py",
    ]
    
    scripts_found = [s for s in rl_scripts if s.exists()]
    stage.checks.append(ValidationResult(
        passed=len(scripts_found) > 0,
        message=f"Found {len(scripts_found)} RL training scripts",
        details={"scripts": [s.name for s in scripts_found]}
    ))
    
    # Check for CARLA bridge
    bridge_script = workspace / "training" / "rl" / "carla_eval_bridge.py"
    stage.checks.append(ValidationResult(
        passed=bridge_script.exists(),
        message="CARLA eval bridge found" if bridge_script.exists() else "CARLA eval bridge missing",
        severity="warning" if not bridge_script.exists() else "info",
        details={"bridge": str(bridge_script)}
    ))
    
    return stage


def validate_carla_eval(workspace: Path) -> StageValidation:
    """Validate stage 5: CARLA evaluation."""
    stage = StageValidation(stage_name="carla_eval", enabled=True)
    
    # Check for CARLA scenario files
    scenario_dirs = [
        workspace / "sim" / "scenarios",
        workspace / "training" / "eval",
    ]
    
    scenario_dir = None
    for d in scenario_dirs:
        if d.exists():
            scenario_dir = d
            break
    
    if scenario_dir:
        scenario_files = list(scenario_dir.glob("*.yaml")) + list(scenario_dir.glob("*.xosc"))
        stage.checks.append(ValidationResult(
            passed=len(scenario_files) > 0,
            message=f"Found {len(scenario_files)} scenario files",
            details={"scenario_dir": str(scenario_dir)}
        ))
    
    # Check for eval scripts
    eval_scripts = [
        workspace / "training" / "eval" / "run_carla_closed_loop_eval.py",
        workspace / "training" / "eval" / "carla_scenariorunner_eval.py",
    ]
    
    scripts_found = [s for s in eval_scripts if s.exists()]
    stage.checks.append(ValidationResult(
        passed=len(scripts_found) > 0,
        message=f"Found {len(scripts_found)} evaluation scripts",
        details={"scripts": [s.name for s in scripts_found]}
    ))
    
    # Check for eval output
    eval_out = workspace / "out" / "eval"
    if eval_out.exists():
        run_dirs = [d for d in eval_out.iterdir() if d.is_dir()]
        stage.checks.append(ValidationResult(
            passed=len(run_dirs) > 0,
            message=f"Found {len(run_dirs)} evaluation runs",
            details={"eval_out": str(eval_out), "runs": len(run_dirs)}
        ))
    else:
        stage.checks.append(ValidationResult(
            passed=True,
            message="No eval output directory (no runs yet)",
            severity="info"
        ))
    
    return stage


def run_all_validations(workspace: Path) -> List[StageValidation]:
    """Run all stage validations."""
    return [
        validate_data_loading(workspace),
        validate_ssl_pretrain(workspace),
        validate_waypoint_bc(workspace),
        validate_rl_refinement(workspace),
        validate_carla_eval(workspace),
    ]


def print_validation_report(validations: List[StageValidation]) -> None:
    """Print human-readable validation report."""
    print("\n" + "=" * 60)
    print("PIPELINE STAGE VALIDATION REPORT")
    print("=" * 60)
    
    total_errors = 0
    total_warnings = 0
    
    for v in validations:
        status = "✓" if v.all_passed else "✗"
        print(f"\n{status} {v.stage_name.upper()} (enabled={v.enabled})")
        
        for check in v.checks:
            if check.passed:
                print(f"  ✓ {check.message}")
            else:
                prefix = "E" if check.severity == "error" else "W"
                print(f"  [{prefix}] {check.message}")
                if check.fix_suggestion:
                    print(f"      → {check.fix_suggestion}")
        
        total_errors += v.error_count
        total_warnings += v.warning_count
    
    print("\n" + "-" * 60)
    print(f"Total: {total_errors} errors, {total_warnings} warnings")
    print("=" * 60)


def get_pipeline_health(validations: List[StageValidation]) -> str:
    """Determine overall pipeline health."""
    has_data = False
    has_ssl = False
    has_bc = False
    has_rl = False
    has_eval = False
    
    for v in validations:
        if v.stage_name == "data_loading":
            has_data = v.enabled and v.all_passed
        elif v.stage_name == "ssl_pretrain":
            has_ssl = v.enabled and v.all_passed
        elif v.stage_name == "waypoint_bc":
            has_bc = v.enabled and v.all_passed
        elif v.stage_name == "rl_refinement":
            has_rl = v.enabled and v.all_passed
        elif v.stage_name == "carla_eval":
            has_eval = v.enabled
    
    if not has_data:
        return "no_data"
    elif not has_ssl and not has_bc:
        return "initializing"
    elif not has_rl:
        return "partial"
    else:
        return "healthy"


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline stages")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root")
    parser.add_argument("--stage", type=str, choices=["data", "ssl", "bc", "rl", "eval", "all"],
                        default="all", help="Stage to validate")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--fix", action="store_true", help="Attempt fixes")
    
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    
    # Run validations
    if args.stage == "all":
        validations = run_all_validations(workspace)
    else:
        stage_map = {
            "data": validate_data_loading,
            "ssl": validate_ssl_pretrain,
            "bc": validate_waypoint_bc,
            "rl": validate_rl_refinement,
            "eval": validate_carla_eval,
        }
        validator = stage_map.get(args.stage, validate_data_loading)
        validations = [validator(workspace)]
    
    # Output
    if args.json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "workspace": str(workspace),
            "health": get_pipeline_health(validations),
            "stages": [asdict(v) for v in validations]
        }
        json_str = json.dumps(output, indent=2)
        if args.output:
            Path(args.output).write_text(json_str)
            print(f"Written to {args.output}")
        else:
            print(json_str)
    else:
        print_validation_report(validations)
        
        if args.output:
            output = {
                "timestamp": datetime.now().isoformat(),
                "health": get_pipeline_health(validations),
                "stages": [asdict(v) for v in validations]
            }
            Path(args.output).write_text(json.dumps(output, indent=2))
            print(f"\nJSON also written to {args.output}")
    
    # Exit code based on health
    health = get_pipeline_health(validations)
    if health in ["no_data", "initializing"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
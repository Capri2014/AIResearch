"""ScenarioRunner integration for RL policy evaluation.

This module bridges trained RL policies (from ppo_residual_delta_stub) with
CARLA ScenarioRunner for closed-loop scenario-specific evaluation.

Driving-first pipeline:
  Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
                                                       ↑
                                            This module connects RL ↔ SR

Usage:
    # Run RL policy evaluation on ScenarioRunner scenarios
    python -m training.rl.srunner_rl_eval \
        --checkpoint out/ppo_residual_delta_stub/run_2026-03-08/model.pt \
        --suite smoke \
        --carla-host 127.0.0.1 \
        --carla-port 2000

    # Dry-run to validate checkpoint loads correctly
    python -m training.rl.srunner_rl_eval \
        --checkpoint out/ppo_residual_delta_stub/run_2026-03-08/model.pt \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SRunnerRLEvalConfig:
    """Configuration for ScenarioRunner RL evaluation."""
    
    # Output
    out_root: Path = Path("out/srunner_rl_eval")
    
    # Checkpoint
    checkpoint: Optional[Path] = None
    
    # ScenarioRunner options
    suite: str = "smoke"
    scenario: Optional[str] = None
    route: Optional[str] = None
    scenario_runner_root: Optional[Path] = None
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    
    # Eval options
    num_episodes: int = 1
    timeout_s: int = 60 * 30
    dry_run: bool = False


def _git_info() -> Dict[str, Any]:
    """Get git metadata for reproducibility."""
    try:
        repo = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        repo = None
    
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = None
    
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        branch = None
    
    return {"repo": repo, "commit": commit, "branch": branch}


def load_rl_checkpoint(checkpoint_path: Optional[Path]) -> Dict[str, Any]:
    """Load RL checkpoint and extract metadata."""
    
    if checkpoint_path is None or not checkpoint_path.exists():
        return {
            "name": "stub",
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "error": "checkpoint not found" if checkpoint_path else "no checkpoint"
        }
    
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        info: Dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
        }
        
        if isinstance(ckpt, dict):
            # Extract known checkpoint fields
            info["has_encoder"] = "encoder" in ckpt
            info["has_head"] = "head" in ckpt
            info["has_delta_head"] = "delta_head" in ckpt
            info["has_sft_model"] = "sft_model" in ckpt
            
            # Training config if present
            if "config" in ckpt:
                config = ckpt["config"]
                if isinstance(config, dict):
                    info["horizon_steps"] = config.get("horizon_steps", "N/A")
                    info["out_dim"] = config.get("out_dim", "N/A")
            
            # Training metrics if present
            if "metrics" in ckpt:
                metrics = ckpt["metrics"]
                if isinstance(metrics, dict):
                    info["final_reward"] = metrics.get("mean_reward")
                    info["final_entropy"] = metrics.get("mean_entropy")
        
        return info
        
    except Exception as e:
        return {
            "checkpoint": str(checkpoint_path),
            "error": str(e)
        }


def _build_srunner_command(
    cfg: SRunnerRLEvalConfig,
    scenario_id: str,
    out_dir: Path
) -> List[str]:
    """Build ScenarioRunner command with RL policy injection."""
    
    srunner_root = cfg.scenario_runner_root
    if srunner_root is None:
        srunner_root = Path(os.environ.get("SCENARIO_RUNNER_ROOT", ""))
    
    if not srunner_root.exists():
        raise FileNotFoundError(f"ScenarioRunner root not found: {srunner_root}")
    
    entrypoint = srunner_root / "scenario_runner.py"
    if not entrypoint.exists():
        raise FileNotFoundError(f"ScenarioRunner entrypoint not found: {entrypoint}")
    
    cmd = [
        sys.executable if hasattr(sys, 'executable') else "python3",
        str(entrypoint),
        "--host", str(cfg.carla_host),
        "--port", str(cfg.carla_port),
        "--outputDir", str(out_dir),
    ]
    
    if cfg.scenario:
        cmd += ["--scenario", cfg.scenario]
    elif cfg.route:
        cmd += ["--route", cfg.route]
    
    # TODO: Inject RL policy via ScenarioRunner's agent interface
    # For now, we evaluate the SFT baseline and note RL integration pending
    
    return cmd


def _write_metrics(
    out_dir: Path,
    cfg: SRunnerRLEvalConfig,
    git_info: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    scenario_results: List[Dict[str, Any]]
) -> None:
    """Write evaluation metrics to JSON."""
    
    # Convert config to JSON-serializable dict
    config_dict = {}
    for k, v in asdict(cfg).items():
        config_dict[k] = str(v) if isinstance(v, Path) else v
    
    metrics = {
        "run_id": out_dir.name,
        "timestamp": datetime.now().isoformat(),
        "domain": "driving",
        "eval_type": "srunner_rl",
        "git": git_info,
        "config": config_dict,
        "checkpoint": checkpoint_info,
        "scenarios": scenario_results,
    }
    
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def run_srunner_evaluation(cfg: SRunnerRLEvalConfig) -> Dict[str, Any]:
    """Run ScenarioRunner evaluation with RL policy."""
    
    # Load checkpoint info
    checkpoint_info = load_rl_checkpoint(cfg.checkpoint)
    git_info = _git_info()
    
    # Create output directory
    run_id = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = cfg.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), default=str, indent=2) + "\n")
    
    # Dry-run mode: just validate checkpoint
    if cfg.dry_run:
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git_info=git_info,
            checkpoint_info=checkpoint_info,
            scenario_results=[{
                "scenario_id": f"{cfg.suite}:dry-run",
                "success": False,
                "note": "dry-run - checkpoint validation only",
            }]
        )
        print(f"[srunner_rl_eval] dry-run: checkpoint valid = {'error' not in checkpoint_info}")
        print(f"[srunner_rl_eval] checkpoint info: {json.dumps(checkpoint_info, indent=2)}")
        print(f"[srunner_rl_eval] wrote: {out_dir / 'metrics.json'}")
        return checkpoint_info
    
    # Find ScenarioRunner
    srunner_root = cfg.scenario_runner_root
    if srunner_root is None:
        srunner_root = Path(os.environ.get("SCENARIO_RUNNER_ROOT", ""))
    
    if not srunner_root.exists():
        print(f"[srunner_rl_eval] WARNING: ScenarioRunner not found at {srunner_root}")
        print("[srunner_rl_eval] Running in mock mode (no CARLA evaluation)")
        
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git_info=git_info,
            checkpoint_info=checkpoint_info,
            scenario_results=[{
                "scenario_id": f"{cfg.suite}:mock",
                "success": False,
                "note": "ScenarioRunner not available - mock evaluation",
            }]
        )
        print(f"[srunner_rl_eval] wrote mock metrics: {out_dir / 'metrics.json'}")
        return checkpoint_info
    
    # Build and run ScenarioRunner command
    scenario_id = cfg.scenario or cfg.route or f"{cfg.suite}:default"
    
    try:
        cmd = _build_srunner_command(cfg, scenario_id, out_dir)
        log_path = out_dir / "srunner_stdout.log"
        
        print(f"[srunner_rl_eval] Running ScenarioRunner:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Output: {out_dir}")
        
        result = subprocess.run(
            cmd,
            cwd=str(srunner_root.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=cfg.timeout_s,
            check=False
        )
        
        log_path.write_bytes(result.stdout or b"")
        
        # Parse results
        # (Using same logic as run_srunner_eval.py)
        scenario_result = {
            "scenario_id": f"{cfg.suite}:{scenario_id}",
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "duration_s": 0,  # TODO: add timing
        }
        
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git_info=git_info,
            checkpoint_info=checkpoint_info,
            scenario_results=[scenario_result]
        )
        
        print(f"[srunner_rl_eval] completed: {out_dir / 'metrics.json'}")
        return checkpoint_info
        
    except subprocess.TimeoutExpired:
        print(f"[srunner_rl_eval] ERROR: ScenarioRunner timed out after {cfg.timeout_s}s")
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git_info=git_info,
            checkpoint_info=checkpoint_info,
            scenario_results=[{
                "scenario_id": f"{cfg.suite}:{scenario_id}",
                "success": False,
                "error": "timeout",
            }]
        )
        raise
    except Exception as e:
        print(f"[srunner_rl_eval] ERROR: {e}")
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git_info=git_info,
            checkpoint_info=checkpoint_info,
            scenario_results=[{
                "scenario_id": f"{cfg.suite}:{scenario_id}",
                "success": False,
                "error": str(e),
            }]
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="ScenarioRunner RL Policy Evaluation")
    
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("out/srunner_rl_eval"),
        help="Output directory root"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to RL checkpoint (from ppo_residual_delta_stub)"
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        default="smoke",
        help="ScenarioRunner suite (smoke, basic, full)"
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Specific scenario to run"
    )
    
    parser.add_argument(
        "--route",
        type=str,
        default=None,
        help="Route file to evaluate"
    )
    
    parser.add_argument(
        "--scenario-runner-root",
        type=Path,
        default=None,
        help="Path to ScenarioRunner checkout"
    )
    
    parser.add_argument(
        "--carla-host",
        type=str,
        default="127.0.0.1",
        help="CARLA server host"
    )
    
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA server port"
    )
    
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=60 * 30,
        help="Timeout per scenario (seconds)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint without running CARLA"
    )
    
    args = parser.parse_args()
    
    cfg = SRunnerRLEvalConfig(
        out_root=args.out_root,
        checkpoint=args.checkpoint,
        suite=args.suite,
        scenario=args.scenario,
        route=args.route,
        scenario_runner_root=args.scenario_runner_root,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        num_episodes=args.num_episodes,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
    )
    
    run_srunner_evaluation(cfg)


if __name__ == "__main__":
    import os
    main()

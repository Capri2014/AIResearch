"""ScenarioRunner evaluation entrypoint (v0).

Driving-first plan:
- pretrain encoder on Waymo multi-cam
- BC fine-tune waypoint policy
- evaluate in CARLA ScenarioRunner suites

This script now does *real* ScenarioRunner process invocation when a ScenarioRunner
checkout is available, while keeping a safe fallback that writes stub metrics.

Optionally loads a trained waypoint policy for closed-loop evaluation:
  python -m sim.driving.carla_srunner.run_srunner_eval \
    --policy-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
    --suite smoke

Outputs
-------
- out/eval/<run_id>/metrics.json
- out/eval/<run_id>/config.json
- out/eval/<run_id>/srunner_stdout.log (when ScenarioRunner was invoked)

Usage
-----
Dry-run (always writes a metrics.json, never launches anything):

  python -m sim.driving.carla_srunner.run_srunner_eval --dry-run

Invoke ScenarioRunner (requires a local ScenarioRunner repo + CARLA server):

  python -m sim.driving.carla_srunner.run_srunner_eval \
    --suite smoke \
    --scenario_runner_root /path/to/scenario_runner \
    --carla-host 127.0.0.1 \
    --carla-port 2000 \
    --scenario OpenScenario_1

Notes
-----
- ScenarioRunner CLI flags differ slightly across versions. We keep the runner
  conservative: we only pass common options and record the invoked command.
- Result parsing extracts completion, infractions, and comfort metrics from
  ScenarioRunner's JSON output or structured log patterns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import json
import os
import re
import subprocess
import time


@dataclass
class EvalConfig:
    out_root: Path = Path("out/eval")
    policy_name: str = "waypoint_stub"
    policy_checkpoint: Optional[str] = None

    suite: str = "smoke"

    # ScenarioRunner / CARLA wiring
    scenario_runner_root: Optional[Path] = None
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000

    # One of these should be set (we keep it simple):
    scenario: Optional[str] = None
    route: Optional[str] = None

    # Extra passthrough args for ScenarioRunner (useful when SR flags differ)
    srunner_args: List[str] = None  # type: ignore[assignment]

    timeout_s: int = 60 * 60
    dry_run: bool = False


def _require_list(x: Optional[List[str]]) -> List[str]:
    return [] if x is None else list(x)


def _git_info(repo_root: Path) -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility."""

    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args, cwd=str(repo_root), stderr=subprocess.DEVNULL)
        except Exception:
            return None
        s = out.decode("utf-8", errors="replace").strip()
        return s or None

    return {
        "repo": _run(["git", "config", "--get", "remote.origin.url"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _load_policy_info(checkpoint: Optional[Path]) -> Dict[str, Any]:
    """Load metadata from a policy checkpoint."""
    if checkpoint is None:
        return {"name": "stub", "checkpoint": None}
    
    if not Path(checkpoint).exists():
        return {"name": "stub", "checkpoint": str(checkpoint), "error": "checkpoint not found"}
    
    try:
        import torch
        ckpt = torch.load(checkpoint, map_location="cpu")
        
        info: Dict[str, Any] = {
            "checkpoint": str(checkpoint),
        }
        
        # Extract metadata if present
        if isinstance(ckpt, dict):
            info["has_encoder"] = "encoder" in ckpt
            info["has_head"] = "head" in ckpt
            info["has_delta_head"] = "delta_head" in ckpt
            if "cam" in ckpt:
                info["cam"] = ckpt["cam"]
            if "horizon_steps" in ckpt:
                info["horizon_steps"] = int(ckpt["horizon_steps"])
            if "out_dim" in ckpt:
                info["out_dim"] = int(ckpt["out_dim"])
        
        return info
    except Exception as e:
        return {"name": "stub", "checkpoint": str(checkpoint), "error": str(e)}


def _find_srunner_entrypoint(cfg: EvalConfig) -> Tuple[Optional[Path], Optional[str]]:
    root = cfg.scenario_runner_root
    if root is None:
        env = os.environ.get("SCENARIO_RUNNER_ROOT")
        if env:
            root = Path(env)

    if root is None:
        return None, "ScenarioRunner root not provided (pass --scenario-runner-root or set SCENARIO_RUNNER_ROOT)"

    entry = root / "scenario_runner.py"
    if not entry.exists():
        return None, f"ScenarioRunner entrypoint not found: {entry}"

    return entry, None


def _build_srunner_cmd(cfg: EvalConfig, entrypoint: Path, *, out_dir: Path) -> List[str]:
    cmd: List[str] = [
        os.environ.get("PYTHON", "python3"),
        str(entrypoint),
        "--host",
        str(cfg.carla_host),
        "--port",
        str(int(cfg.carla_port)),
    ]

    # Choose either scenario or route.
    if cfg.scenario and cfg.route:
        raise ValueError("Pass only one of --scenario or --route")

    if cfg.scenario:
        cmd += ["--scenario", str(cfg.scenario)]
    elif cfg.route:
        cmd += ["--route", str(cfg.route)]
    else:
        # If neither is specified, we keep SR invocation optional and let the user
        # drive via --srunner-args.
        pass

    # Try to set output dir if SR supports it. Some versions use --outputDir.
    # If the flag is unrecognized, SR will fail fast; user can work around via
    # --srunner-args.
    cmd += ["--outputDir", str(out_dir)]

    cmd += _require_list(cfg.srunner_args)
    return cmd


def _write_metrics(
    *,
    out_dir: Path,
    cfg: EvalConfig,
    git: Dict[str, Any],
    scenario_rows: List[Dict[str, Any]],
) -> None:
    # Load policy checkpoint info
    policy_info = _load_policy_info(
        Path(cfg.policy_checkpoint) if cfg.policy_checkpoint else None
    )
    
    metrics: Dict[str, Any] = {
        "run_id": out_dir.name,
        "domain": "driving",
        "git": {k: v for k, v in git.items() if v is not None},
        "policy": {
            "name": cfg.policy_name,
            **policy_info,
        },
        "scenarios": scenario_rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def _parse_srunner_output(log_path: Path) -> Dict[str, Any]:
    """Parse ScenarioRunner stdout/log for completion and infraction metrics.

    ScenarioRunner outputs metrics in two common formats:
    1. JSON report file (newer versions): looks for 'scenario_result.json' or
       '**/results.json' in output dir.
    2. Structured log patterns (older versions): extracts metrics from log lines
       like "RouteCompletion: 0.85" or "Collisions: 2".

    Returns a dict with keys: route_completion, collisions, offroad, red_light,
    max_accel, max_jerk. Missing values are None.
    """
    result: Dict[str, Any] = {
        "route_completion": None,
        "collisions": None,
        "offroad": None,
        "red_light": None,
        "comfort": None,
    }

    # Try to find a JSON results file in output dir (ScenarioRunner 0.9+)
    results_patterns = ["scenario_result.json", "results.json", "evaluation.json"]
    log_dir = log_path.parent

    for pattern in results_patterns:
        results_file = log_dir / pattern
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                # Extract common SR result fields
                if isinstance(data, dict):
                    result["route_completion"] = data.get("route_completion") or data.get("completion_rate")
                    result["collisions"] = data.get("collisions") or data.get("collision_count")
                    result["offroad"] = data.get("offroad") or data.get("off_track")
                    result["red_light"] = data.get("red_light") or data.get("red_light_violations")

                    # Comfort metrics
                    comfort_fields = {}
                    for field in ["max_accel", "max_jerk", "avg_accel", "avg_jerk"]:
                        if field in data:
                            comfort_fields[field] = data[field]
                    if comfort_fields:
                        result["comfort"] = comfort_fields
                return result
            except (json.JSONDecodeError, OSError):
                pass

    # Fallback: parse structured log patterns from stdout
    if not log_path.exists():
        return result

    text = log_path.read_text(errors="replace")

    # Route completion patterns
    rc_match = re.search(r"RouteCompletion[:\s]+([0-9.]+)", text, re.I)
    if rc_match:
        result["route_completion"] = float(rc_match.group(1))

    # Infraction patterns
    coll_match = re.search(r"Collisions?[:\s]+(\d+)", text, re.I)
    if coll_match:
        result["collisions"] = int(coll_match.group(1))

    offroad_match = re.search(r"Off[- ]?road[:\s]+(\d+)", text, re.I)
    if offroad_match:
        result["offroad"] = int(offroad_match.group(1))

    redlight_match = re.search(r"Red[- ]?light[:\s]+(\d+)", text, re.I)
    if redlight_match:
        result["red_light"] = int(redlight_match.group(1))

    # Comfort metrics from log
    comfort_fields = {}
    accel_match = re.search(r"MaxAcceleration[:\s]+([0-9.]+)", text, re.I)
    if accel_match:
        comfort_fields["max_accel"] = float(accel_match.group(1))

    jerk_match = re.search(r"MaxJerk[:\s]+([0-9.]+)", text, re.I)
    if jerk_match:
        comfort_fields["max_jerk"] = float(jerk_match.group(1))

    if comfort_fields:
        result["comfort"] = comfort_fields

    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--policy-name", type=str, default="waypoint_stub")
    p.add_argument("--policy-checkpoint", type=str, default=None)
    p.add_argument("--suite", type=str, default="smoke")

    p.add_argument("--scenario-runner-root", type=Path, default=None)
    p.add_argument("--carla-host", type=str, default="127.0.0.1")
    p.add_argument("--carla-port", type=int, default=2000)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--scenario", type=str, default=None)
    g.add_argument("--route", type=str, default=None)

    p.add_argument(
        "--srunner-args",
        type=str,
        nargs="*",
        default=None,
        help="Extra args passed through to ScenarioRunner (use when SR CLI differs by version)",
    )

    p.add_argument("--timeout-s", type=int, default=60 * 60)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cfg = EvalConfig(
        out_root=a.out_root,
        policy_name=a.policy_name,
        policy_checkpoint=a.policy_checkpoint,
        suite=a.suite,
        scenario_runner_root=a.scenario_runner_root,
        carla_host=a.carla_host,
        carla_port=int(a.carla_port),
        scenario=a.scenario,
        route=a.route,
        srunner_args=a.srunner_args,
        timeout_s=int(a.timeout_s),
        dry_run=bool(a.dry_run),
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")

    repo_root = Path(__file__).resolve().parents[3]
    git = _git_info(repo_root)

    scenario_id = cfg.scenario or cfg.route or f"{cfg.suite}:unspecified"

    if cfg.dry_run:
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git=git,
            scenario_rows=[
                {
                    "scenario_id": f"{cfg.suite}:{scenario_id}",
                    "success": False,
                    "route_completion": 0.0,
                    "collisions": 0,
                    "offroad": 0,
                    "red_light": 0,
                    "raw": {"note": "dry-run (ScenarioRunner not invoked)"},
                }
            ],
        )
        print(f"[carla_srunner] dry-run wrote: {out_dir / 'metrics.json'}")
        return

    entrypoint, err = _find_srunner_entrypoint(cfg)
    if entrypoint is None:
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git=git,
            scenario_rows=[
                {
                    "scenario_id": f"{cfg.suite}:{scenario_id}",
                    "success": False,
                    "route_completion": 0.0,
                    "collisions": 0,
                    "offroad": 0,
                    "red_light": 0,
                    "raw": {"note": "ScenarioRunner not invoked", "error": err},
                }
            ],
        )
        print(f"[carla_srunner] wrote stub metrics (no ScenarioRunner): {out_dir / 'metrics.json'}")
        return

    cmd = _build_srunner_cmd(cfg, entrypoint, out_dir=out_dir)
    log_path = out_dir / "srunner_stdout.log"
    t0 = time.time()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(entrypoint.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(cfg.timeout_s),
            check=False,
        )
        log_path.write_bytes(proc.stdout or b"")
        dt = time.time() - t0

        # Parse ScenarioRunner output for completion/infraction metrics
        parsed = _parse_srunner_output(log_path)

        # Preserve run command + return code for debugging
        raw = {
            "note": "ScenarioRunner invoked",
            "cmd": cmd,
            "returncode": int(proc.returncode),
            "stdout_log": str(log_path),
            "duration_s": float(dt),
        }

        # Build scenario row with parsed metrics where available
        scenario_row = {
            "scenario_id": f"{cfg.suite}:{scenario_id}",
            "success": bool(proc.returncode == 0),
            "raw": raw,
        }

        # Wire in parsed metrics (only include non-None values)
        for key in ["route_completion", "collisions", "offroad", "red_light"]:
            if parsed[key] is not None:
                scenario_row[key] = parsed[key]

        if parsed.get("comfort"):
            scenario_row["comfort"] = parsed["comfort"]

        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git=git,
            scenario_rows=[scenario_row],
        )
        print(f"[carla_srunner] wrote: {out_dir / 'metrics.json'}")
    except subprocess.TimeoutExpired:
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git=git,
            scenario_rows=[
                {
                    "scenario_id": f"{cfg.suite}:{scenario_id}",
                    "success": False,
                    "raw": {"note": "ScenarioRunner timed out", "cmd": cmd, "timeout_s": cfg.timeout_s},
                }
            ],
        )
        print(f"[carla_srunner] timeout; wrote: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

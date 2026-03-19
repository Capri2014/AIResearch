#!/usr/bin/env python3
"""
Universal Metrics Loader for RL after SFT pipeline.

Loads and validates evaluation metrics, then prints a structured report.
Handles multiple formats: scenario-based eval, combined SFT+RL eval, and RL training metrics.

Usage
-----
# Load and print metrics from an eval run
python -m training.rl.eval_metrics_loader out/eval/<run_id>

# Load and validate
python -m training.rl.eval_metrics_loader out/eval/<run_id> --validate

# Compare two eval runs (SFT vs RL)
python -m training.rl.eval_metrics_loader out/eval/<sft_run> out/eval/<rl_run> --compare

# Compare with custom names
python -m training.rl.eval_metrics_loader out/eval/<sft_run> out/eval/<rl_run> \
    --compare --baseline-name SFT --candidate-name RL

# Auto-detect latest eval run
python -m training.rl.eval_metrics_loader --latest

# Load RL training metrics
python -m training.rl.eval_metrics_loader out/rl_refinement_daily_2026_03_13/train_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Resolve repo root for schema
# File: <repo>/training/rl/eval_metrics_loader.py → repo is parents[2]
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]
_SCHEMA_PATH = _REPO_ROOT / "data" / "schema" / "metrics.json"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _nan_to_none(obj: Any) -> Any:
    """Recursively replace NaN/Inf float values with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(item) for item in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _to_native(obj: Any) -> Any:
    """Convert numpy/Python special types to native JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Schema validation (lightweight)
# ---------------------------------------------------------------------------

def _load_schema() -> Dict:
    """Load the metrics JSON schema."""
    if not _SCHEMA_PATH.exists():
        return {}
    return json.loads(_SCHEMA_PATH.read_text())


def _validate(metrics: Dict, schema: Dict) -> tuple[bool, list[str]]:
    """Lightweight schema validation. Returns (is_valid, errors)."""
    errors = []

    if not schema:
        return True, []  # No schema available

    # Required top-level fields
    required = schema.get("required", [])
    for field in required:
        if field not in metrics:
            errors.append(f"Missing required field: {field}")

    # Domain field
    domain = metrics.get("domain", "")
    allowed_domains = schema.get("properties", {}).get("domain", {}).get("enum", [])
    if allowed_domains and domain not in allowed_domains:
        errors.append(f"Invalid domain '{domain}'; expected one of {allowed_domains}")

    # Scenarios must be a list
    scenarios = metrics.get("scenarios")
    if scenarios is not None and not isinstance(scenarios, list):
        errors.append(f"'scenarios' must be a list, got {type(scenarios).__name__}")
    elif isinstance(scenarios, list) and scenarios:
        # Check first scenario for required fields
        first = scenarios[0]
        scenario_props = schema.get("properties", {}).get("scenarios", {}).get("items", {}).get("properties", {})
        req = []
        for k, v in scenario_props.items():
            if v.get("required"):
                req.append(k)
        # Schema uses "required" array at scenario level — check success field
        if "success" not in first:
            errors.append("Scenario missing required field: success")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(metrics: Dict) -> str:
    """Detect the metrics format type."""
    scenarios = metrics.get("scenarios", [])

    # Check for combined SFT+RL format
    has_sft = any("sft" in str(s.get("scenario_id", "")).lower() for s in scenarios)
    has_rl = any("rl" in str(s.get("scenario_id", "")).lower() for s in scenarios)
    has_policy_type = any("policy_type" in s for s in scenarios[:3])

    if has_sft and has_rl:
        return "combined_sft_rl"
    if has_policy_type:
        return "policy_typed"

    # RL training metrics formats
    if "updates" in metrics or "rl_metrics" in metrics or "final_metrics" in metrics:
        return "rl_training"
    if "rewards" in metrics or "lengths" in metrics:
        return "rl_training"

    # Standard eval format
    if scenarios:
        return "scenario_eval"

    return "unknown"


# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------

def load_metrics(path: Path) -> Dict:
    """Load a metrics JSON file or find one in a directory."""
    if path.is_dir():
        candidates = ["metrics.json", "train_metrics.json"]
        for c in candidates:
            candidate = path / c
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No metrics.json found in {path}")

    raw = json.loads(path.read_text())
    return _nan_to_none(raw)


def get_latest_eval_dir(out_root: Path = None) -> Optional[Path]:
    """Return the most recently modified eval directory."""
    if out_root is None:
        out_root = _REPO_ROOT / "out" / "eval"
    if not out_root.exists():
        return None
    dirs = [d for d in out_root.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    if not dirs:
        return None
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0]


# ---------------------------------------------------------------------------
# Per-format summary extraction
# ---------------------------------------------------------------------------

def _extract_summary_scenarios(metrics: Dict) -> Dict[str, Any]:
    """Extract summary from scenario-based eval."""
    scenarios = metrics.get("scenarios", [])
    if not scenarios:
        return {}

    ades = [s.get("ade") or 0 for s in scenarios if s.get("ade") is not None]
    fdes = [s.get("fde") or 0 for s in scenarios if s.get("fde") is not None]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    returns = [s.get("return") or 0 for s in scenarios if s.get("return") is not None]

    def _safe_mean(vals):
        return float(np.mean(vals)) if vals else None

    def _safe_std(vals):
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    return {
        "ade_mean": _safe_mean(ades),
        "ade_std": _safe_std(ades),
        "fde_mean": _safe_mean(fdes),
        "fde_std": _safe_std(fdes),
        "success_rate": _safe_mean(successes),
        "return_mean": _safe_mean(returns),
        "num_episodes": len(scenarios),
    }


def _extract_summary_combined(metrics: Dict) -> Dict[str, Any]:
    """Extract separate SFT and RL summaries from combined eval."""
    scenarios = metrics.get("scenarios", [])
    if not scenarios:
        return {}

    sft_scenarios = [s for s in scenarios if "sft" in str(s.get("scenario_id", "")).lower()]
    rl_scenarios = [s for s in scenarios if "rl" in str(s.get("scenario_id", "")).lower()]

    sft_sum = _extract_summary_scenarios({"scenarios": sft_scenarios}) if sft_scenarios else {}
    rl_sum = _extract_summary_scenarios({"scenarios": rl_scenarios}) if rl_scenarios else {}

    result = {"sft": sft_sum, "rl": rl_sum}

    # Compute delta if both present
    if sft_sum.get("ade_mean") and rl_sum.get("ade_mean"):
        delta_ade = sft_sum["ade_mean"] - rl_sum["ade_mean"]
        delta_fde = sft_sum["fde_mean"] - rl_sum["fde_mean"]
        result["delta"] = {
            "ade_pct": delta_ade / sft_sum["ade_mean"] * 100 if sft_sum["ade_mean"] else 0,
            "fde_pct": delta_fde / sft_sum["fde_mean"] * 100 if sft_sum["fde_mean"] else 0,
            "success_diff": rl_sum["success_rate"] - sft_sum["success_rate"],
        }

    return result


def _extract_summary_training(metrics: Dict) -> Dict[str, Any]:
    """Extract summary from RL training metrics."""
    summary = {"type": "rl_training"}

    # Try rl_metrics format (newer)
    rl = metrics.get("rl_metrics", {})
    if rl:
        summary["final_avg_reward"] = rl.get("final_avg_reward")
        summary["best_avg_reward"] = rl.get("best_avg_reward")
        summary["total_updates"] = rl.get("total_updates")

    # Try final_metrics format (older)
    fm = metrics.get("final_metrics", {})
    if fm and not summary.get("final_avg_reward"):
        summary["final_avg_reward"] = fm.get("avg_reward")
        summary["final_episodes"] = fm.get("num_episodes")

    # Try rewards/lengths arrays
    rewards = metrics.get("rewards", [])
    lengths = metrics.get("lengths", [])
    if rewards and not summary.get("final_avg_reward"):
        summary["final_avg_reward"] = float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
    if lengths and "num_episodes" not in summary:
        summary["num_episodes"] = len(lengths)

    # Try updates format
    updates = metrics.get("updates", [])
    if updates and not summary.get("total_updates"):
        last = updates[-1]
        summary["steps"] = last.get("steps")
        summary["num_episodes"] = last.get("num_episodes", len(updates))

    return summary


def extract_summary(metrics: Dict) -> Dict[str, Any]:
    """Extract the best available summary from metrics."""
    fmt = _detect_format(metrics)

    if fmt in ("combined_sft_rl", "policy_typed"):
        return _extract_summary_combined(metrics)
    if fmt == "rl_training":
        return _extract_summary_training(metrics)
    return _extract_summary_scenarios(metrics)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt(x: Any, decimals: int = 3) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "N/A"
        return f"{x:.{decimals}f}"
    return str(x)


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def print_report(metrics: Dict, validate: bool = True, indent: str = "") -> None:
    """Print a formatted report for a single metrics file."""
    run_id = metrics.get("run_id", "unknown")
    domain = metrics.get("domain", "unknown")
    fmt = _detect_format(metrics)

    print(f"\n{indent}{'=' * 58}")
    print(f"{indent}METRICS REPORT: {run_id}")
    print(f"{indent}{'=' * 58}")

    # Header info
    print(f"{indent}  Domain:   {domain}")
    print(f"{indent}  Format:   {fmt}")
    policy_name = metrics.get("policy", {}).get("name", "unknown")
    print(f"{indent}  Policy:   {policy_name}")
    if timestamp := metrics.get("timestamp"):
        print(f"{indent}  Time:     {timestamp}")

    # Git info
    git = metrics.get("git", {})
    if git:
        commit = git.get("commit", "")
        branch = git.get("branch", "")
        if commit:
            print(f"{indent}  Commit:   {commit[:8]}")
        if branch:
            print(f"{indent}  Branch:   {branch}")

    # Schema validation
    if validate:
        schema = _load_schema()
        is_valid, errors = _validate(metrics, schema)
        if is_valid:
            print(f"{indent}  Schema:   ✅ Valid")
        else:
            print(f"{indent}  Schema:   ❌ Invalid ({len(errors)} error(s))")
            for e in errors[:3]:
                print(f"{indent}    - {e}")

    # Per-format summary
    summary = extract_summary(metrics)
    num_scenarios = len(metrics.get("scenarios", []))

    print(f"{indent}{'-' * 58}")
    if fmt == "rl_training":
        print(f"{indent}SUMMARY (RL Training)")
    else:
        print(f"{indent}SUMMARY ({num_scenarios} scenario(s))")
    print(f"{indent}{'-' * 58}")

    if fmt == "combined_sft_rl" and "sft" in summary:
        sft = summary.get("sft", {})
        rl = summary.get("rl", {})
        delta = summary.get("delta", {})

        print(f"{indent}SFT:")
        print(f"{indent}  ADE: {_fmt(sft.get('ade_mean'))}m ± {_fmt(sft.get('ade_std'))}m")
        print(f"{indent}  FDE: {_fmt(sft.get('fde_mean'))}m ± {_fmt(sft.get('fde_std'))}m")
        print(f"{indent}  Success Rate: {_fmt_pct(sft.get('success_rate'))}")

        print(f"{indent}RL:")
        print(f"{indent}  ADE: {_fmt(rl.get('ade_mean'))}m ± {_fmt(rl.get('ade_std'))}m")
        print(f"{indent}  FDE: {_fmt(rl.get('fde_mean'))}m ± {_fmt(rl.get('fde_std'))}m")
        print(f"{indent}  Success Rate: {_fmt_pct(rl.get('success_rate'))}")

        if delta:
            print(f"{indent}Delta (RL - SFT):")
            print(f"{indent}  ADE: {_fmt(delta.get('ade_pct'))}%")
            print(f"{indent}  FDE: {_fmt(delta.get('fde_pct'))}%")
            print(f"{indent}  Success: {_fmt(delta.get('success_diff'))}")

    elif fmt == "rl_training":
        print(f"{indent}  Type: RL Training")
        for k, v in summary.items():
            if k == "type":
                continue
            if v is not None:
                print(f"{indent}  {k}: {_fmt(v)}")

    else:
        print(f"{indent}  ADE: {_fmt(summary.get('ade_mean'))}m ± {_fmt(summary.get('ade_std'))}m")
        print(f"{indent}  FDE: {_fmt(summary.get('fde_mean'))}m ± {_fmt(summary.get('fde_std'))}m")
        print(f"{indent}  Success Rate: {_fmt_pct(summary.get('success_rate'))}")
        if summary.get("return_mean") is not None:
            print(f"{indent}  Avg Return: {_fmt(summary.get('return_mean'))}")

    print(f"{indent}{'=' * 58}")


def print_3line_summary(metrics: Dict) -> None:
    """Print minimal 3-line comparison suitable for PR reports."""
    fmt = _detect_format(metrics)
    summary = extract_summary(metrics)

    if fmt == "combined_sft_rl":
        sft = summary.get("sft", {})
        rl = summary.get("rl", {})
        delta = summary.get("delta", {})
        print(f"SFT:  ADE={_fmt(sft.get('ade_mean'))}m, FDE={_fmt(sft.get('fde_mean'))}m, Success={_fmt_pct(sft.get('success_rate'))}")
        print(f"RL:   ADE={_fmt(rl.get('ade_mean'))}m, FDE={_fmt(rl.get('fde_mean'))}m, Success={_fmt_pct(rl.get('success_rate'))}")
        print(f"Delta: ADE {_fmt(delta.get('ade_pct'))}%, FDE {_fmt(delta.get('fde_pct'))}%, Success {_fmt(delta.get('success_diff'))}")
    elif fmt == "rl_training":
        # Compact one-liner for RL training
        vals = []
        for k in ("final_avg_reward", "best_avg_reward", "num_episodes"):
            v = summary.get(k)
            if v is not None:
                vals.append(f"{k}={_fmt(v)}")
        print("Training: " + ", ".join(vals) if vals else "Training: N/A")
    else:
        print(f"ADE={_fmt(summary.get('ade_mean'))}m, FDE={_fmt(summary.get('fde_mean'))}m, Success={_fmt_pct(summary.get('success_rate'))}")


def print_comparison(
    base_metrics: Dict,
    cand_metrics: Dict,
    base_name: str = "baseline",
    cand_name: str = "candidate",
) -> None:
    """Print side-by-side comparison of two eval runs."""
    base_sum = extract_summary(base_metrics)
    cand_sum = extract_summary(cand_metrics)

    print(f"\n{'=' * 60}")
    print(f"COMPARISON: {base_name} vs {cand_name}")
    print(f"{'=' * 60}")

    # Extract scalar values safely
    def _ade(m):
        if isinstance(m, dict):
            if "sft" in m:
                return m["rl"].get("ade_mean"), m["sft"].get("ade_mean")
            return m.get("ade_mean")
        return None

    def _fde(m):
        if isinstance(m, dict):
            if "sft" in m:
                return m["rl"].get("fde_mean"), m["sft"].get("fde_mean")
            return m.get("fde_mean")
        return None

    def _succ(m):
        if isinstance(m, dict):
            if "sft" in m:
                return m["rl"].get("success_rate"), m["sft"].get("success_rate")
            return m.get("success_rate")
        return None

    b_ade, _ = _ade(base_sum) if isinstance(_ade(base_sum), tuple) else (_ade(base_sum), None)
    c_ade, _ = _ade(cand_sum) if isinstance(_ade(cand_sum), tuple) else (_ade(cand_sum), None)
    b_fde, _ = _fde(base_sum) if isinstance(_fde(base_sum), tuple) else (_fde(base_sum), None)
    c_fde, _ = _fde(cand_sum) if isinstance(_fde(cand_sum), tuple) else (_fde(cand_sum), None)
    b_succ, _ = _succ(base_sum) if isinstance(_succ(base_sum), tuple) else (_succ(base_sum), None)
    c_succ, _ = _succ(cand_sum) if isinstance(_succ(cand_sum), tuple) else (_succ(cand_sum), None)

    # Compute delta
    ade_imp = None
    fde_imp = None
    if b_ade and c_ade:
        ade_imp = (b_ade - c_ade) / b_ade * 100 if b_ade else None
    if b_fde and c_fde:
        fde_imp = (b_fde - c_fde) / b_fde * 100 if b_fde else None
    succ_imp = (c_succ - b_succ) if (b_succ is not None and c_succ is not None) else None

    print(f"\n{base_name.upper()} ({base_metrics.get('run_id', '?')}):")
    print(f"  ADE: {_fmt(b_ade)}m | FDE: {_fmt(b_fde)}m | Success: {_fmt_pct(b_succ)}")

    print(f"\n{cand_name.upper()} ({cand_metrics.get('run_id', '?')}):")
    print(f"  ADE: {_fmt(c_ade)}m | FDE: {_fmt(c_fde)}m | Success: {_fmt_pct(c_succ)}")

    print(f"\n{'-' * 60}")
    print("3-LINE SUMMARY:")
    print(f"  ADE: {_fmt(b_ade)}m ({base_name}) → {_fmt(c_ade)}m ({cand_name}) [{_fmt(ade_imp)}%]")
    print(f"  FDE: {_fmt(b_fde)}m ({base_name}) → {_fmt(c_fde)}m ({cand_name}) [{_fmt(fde_imp)}%]")
    print(f"  Success: {_fmt_pct(b_succ)} ({base_name}) → {_fmt_pct(c_succ)} ({cand_name}) [{_fmt_pct(succ_imp)}]")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and print evaluation metrics for RL after SFT pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="Path(s) to metrics.json or eval directories. "
             "With two paths and --compare, shows comparison. "
             "With zero paths and --latest, auto-detects.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare two eval runs (requires exactly 2 paths)",
    )
    parser.add_argument(
        "--baseline-name", default="baseline",
        help="Name for the first/baseline run in comparison (default: baseline)",
    )
    parser.add_argument(
        "--candidate-name", default="candidate",
        help="Name for the second/candidate run in comparison (default: candidate)",
    )
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Validate against metrics schema (default: True)",
    )
    parser.add_argument(
        "--no-validate", action="store_false", dest="validate",
        help="Skip schema validation",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Auto-detect and load the latest eval directory",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print the 3-line summary",
    )
    args = parser.parse_args()

    # Auto-detect latest
    if args.latest:
        latest = get_latest_eval_dir()
        if not latest:
            print("No eval directories found in out/eval/", file=sys.stderr)
            return 1
        args.paths = [latest]
        print(f"Using latest eval: {latest.name}")

    if not args.paths:
        parser.print_help()
        return 0

    # Load paths
    try:
        loaded = [load_metrics(p) for p in args.paths]
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Compare mode
    if args.compare:
        if len(loaded) != 2:
            print("Error: --compare requires exactly 2 paths", file=sys.stderr)
            return 1
        print_comparison(loaded[0], loaded[1], args.baseline_name, args.candidate_name)
        return 0

    # Print each
    for i, (path, metrics) in enumerate(zip(args.paths, loaded)):
        if not args.quiet:
            print(f"\n[{i + 1}/{len(loaded)}] {path}")
        if args.quiet:
            print_3line_summary(metrics)
        else:
            print_report(metrics, validate=args.validate)
            if len(args.paths) == 1:
                # Also print compact 3-line summary
                print("\nCompact summary:")
                print_3line_summary(metrics)

    return 0


if __name__ == "__main__":
    sys.exit(_main())

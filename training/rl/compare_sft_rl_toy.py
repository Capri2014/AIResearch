#!/usr/bin/env python3
"""Compare SFT-only vs RL-refined policy on toy waypoint RL env.

Loads two evaluation runs (SFT and RL) and prints a 3-line comparison report:
- ADE: SFT vs RL (meters and % improvement)
- FDE: SFT vs RL (meters and % improvement)  
- Success Rate: SFT vs RL (% points)

Usage:
    python -m training.rl.compare_sft_rl_toy --sft-run-id <id> --rl-run-id <id>
    python -m training.rl.compare_sft_rl_toy --latest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def load_metrics(run_id: str) -> Optional[dict]:
    """Load metrics from an evaluation run."""
    metrics_path = Path(__file__).parents[2] / "out" / "eval" / run_id / "metrics.json"
    if not metrics_path.exists():
        # Try alternate location
        metrics_path = Path(__file__).parents[0] / "out" / "eval" / run_id / "metrics.json"
    if not metrics_path.exists():
        print(f"[error] metrics not found: {metrics_path}")
        return None
    return json.loads(metrics_path.read_text())


def find_latest_run(policy_type: str) -> Optional[str]:
    """Find the latest run for a given policy type."""
    eval_dir = Path(__file__).parents[2] / "out" / "eval"
    if not eval_dir.exists():
        eval_dir = Path(__file__).parents[0] / "out" / "eval"
    if not eval_dir.exists():
        return None
    
    runs = sorted(eval_dir.iterdir(), reverse=True)
    for run in runs:
        if run.is_dir() and (run / "metrics.json").exists():
            return run.name
    return None


def compute_improvement(sft_val: float, rl_val: float) -> tuple[float, str]:
    """Compute improvement from SFT to RL. Positive improvement means RL is better."""
    if sft_val == 0:
        return 0.0, "N/A"
    delta = sft_val - rl_val  # Lower is better for errors
    pct = (delta / sft_val) * 100
    direction = "+" if delta > 0 else "-"
    return abs(pct), direction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-run-id", type=str, default=None)
    p.add_argument("--rl-run-id", type=str, default=None)
    p.add_argument("--latest", action="store_true")
    a = p.parse_args()

    # Find runs
    if a.latest:
        # Find runs from today
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        eval_dir = Path(__file__).parents[2] / "out" / "eval"
        
        sft_run = None
        rl_run = None
        for run in sorted(eval_dir.iterdir(), reverse=True):
            if not run.is_dir():
                continue
            if today in run.name:
                # Check if already compared both
                if sft_run is None:
                    sft_run = run.name
                elif rl_run is None:
                    rl_run = run.name
                    break
        
        if sft_run and rl_run:
            print(f"[info] Using runs: sft={sft_run}, rl={rl_run}")
    else:
        sft_run = a.sft_run_id or "20260502-213312"
        rl_run = a.rl_run_id or "20260502-213316"

    # Load metrics
    sft_metrics = load_metrics(sft_run)
    rl_metrics = load_metrics(rl_run)
    
    if not sft_metrics or not rl_metrics:
        print("[error] Failed to load metrics for one or both runs")
        return

    sft_summary = sft_metrics.get("summary", {})
    rl_summary = rl_metrics.get("summary", {})

    sft_ade = sft_summary.get("ade_mean", 0)
    rl_ade = rl_summary.get("ade_mean", 0)
    sft_fde = sft_summary.get("fde_mean", 0)
    rl_fde = rl_summary.get("fde_mean", 0)
    sft_success = sft_summary.get("success_rate", 0)
    rl_success = rl_summary.get("success_rate", 0)

    # Compute improvements
    ade_pct, ade_dir = compute_improvement(sft_ade, rl_ade)
    fde_pct, fde_dir = compute_improvement(sft_fde, rl_fde)
    success_pts = (rl_success - sft_success) * 100

    # Print 3-line report
    print(f"\n{'='*50}")
    print("SFT vs RL Comparison (Toy Waypoint RL Env)")
    print(f"{'='*50}")
    print(f"ADE:   SFT={sft_ade:.4f}m  RL={rl_ade:.4f}m  → {ade_dir}{ade_pct:.1f}% (lower is better)")
    print(f"FDE:   SFT={sft_fde:.4f}m  RL={rl_fde:.4f}m  → {fde_dir}{fde_pct:.1f}% (lower is better)")
    print(f"Succ:  SFT={sft_success:.1%}  RL={rl_success:.1%}  → {'+' if success_pts > 0 else ''}{success_pts:.1f}pp")
    print(f"{'='*50}")

    # Determine winner
    if ade_pct > 0:
        print("\n→ RL policy shows improvement over SFT-only baseline")
    elif ade_pct < 0:
        print("\n→ SFT-only baseline performs better (this is unexpected)")
    else:
        print("\n→ No meaningful difference between policies")

    # Write comparison JSON
    comparison = {
        "run_id": f"comparison_{sft_run}_vs_{rl_run}",
        "timestamp": sft_metrics.get("timestamp", ""),
        "sft_run": sft_run,
        "rl_run": rl_run,
        "sft": sft_summary,
        "rl": rl_summary,
        "improvement": {
            "ade_delta": sft_ade - rl_ade,
            "ade_delta_pct": ade_pct,
            "fde_delta": sft_fde - rl_fde,
            "fde_delta_pct": fde_pct,
            "success_delta_pp": success_pts,
        },
    }
    
    out_dir = Path(__file__).parents[2] / "out" / "eval" / f"comparison_{sft_run}_vs_{rl_run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"\n[wrote] {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
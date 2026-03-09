#!/usr/bin/env python3
"""
Deterministic evaluation runner for waypoint RL policy.

Runs deterministic evaluation on toy waypoint environment and:
1. Writes metrics.json to out/eval/<run_id>/
2. Validates against schema
3. Prints 3-line comparison report

Usage
-----
# Run 20 episode evaluation with default settings
python -m training.rl.run_det_eval

# Quick smoke with 5 episodes
python -m training.rl.run_det_eval --episodes 5

# Custom seed base
python -m training.rl.run_det_eval --seed-base 100 --episodes 20

# Skip validation (faster)
python -m training.rl.run_det_eval --no-validate

# Output to custom directory
python -m training.rl.run_det_eval --out-root out/eval_custom
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic evaluation runner for waypoint RL"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes per policy (default: 20)"
    )
    parser.add_argument(
        "--seed-base", type=int, default=42,
        help="Base seed for deterministic evaluation (default: 42)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Max steps per episode (default: 50)"
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("out/eval"),
        help="Output directory root (default: out/eval)"
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip schema validation"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress non-essential output"
    )
    
    args = parser.parse_args()
    
    # Build command for compare_sft_vs_rl
    compare_cmd = [
        sys.executable, "-m", "training.rl.compare_sft_vs_rl",
        "--episodes", str(args.episodes),
        "--seed-base", str(args.seed_base),
        "--max-steps", str(args.max_steps),
        "--out-root", str(args.out_root),
    ]
    
    if not args.quiet:
        print(f"[run_det_eval] Running deterministic evaluation...")
        print(f"[run_det_eval]   episodes: {args.episodes}")
        print(f"[run_det_eval]   seed-base: {args.seed_base}")
        print(f"[run_det_eval]   max-steps: {args.max_steps}")
        print()
    
    # Run comparison
    result = subprocess.run(
        compare_cmd,
        cwd=str(repo_root),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"[run_det_eval] Error: compare_sft_vs_rl failed with code {result.returncode}")
        sys.exit(result.returncode)
    
    # Find the output directories (they have timestamps)
    out_root = Path(args.out_root)
    if not out_root.exists():
        print(f"[run_det_eval] Error: output directory not found: {out_root}")
        sys.exit(1)
    
    # Get most recent run
    run_dirs = sorted(out_root.glob("*_sft"))
    if not run_dirs:
        print(f"[run_det_eval] Error: no SFT output directories found")
        sys.exit(1)
    
    latest_sft = run_dirs[-1]
    run_id = latest_sft.name.replace("_sft", "")
    latest_rl = out_root / f"{run_id}_rl"
    
    if not args.quiet:
        print(f"\n[run_det_eval] Output directories:")
        print(f"  SFT:  {latest_sft}")
        print(f"  RL:   {latest_rl}")
    
    # Validate against schema
    if not args.no_validate:
        if not args.quiet:
            print(f"\n[run_det_eval] Validating metrics against schema...")
        
        for metrics_file in [latest_sft / "metrics.json", latest_rl / "metrics.json"]:
            validate_cmd = [
                sys.executable, "-m", "training.rl.validate_metrics",
                str(metrics_file)
            ]
            
            result = subprocess.run(
                validate_cmd,
                cwd=str(repo_root),
                capture_output=True
            )
            
            if result.returncode != 0:
                print(f"[run_det_eval] Warning: {metrics_file.name} validation failed")
            elif not args.quiet:
                print(f"  {metrics_file.name}: ✅ VALID")
    
    if not args.quiet:
        print(f"\n[run_det_eval] Done!")
        print(f"\n[run_det_eval] To compare with prior runs:")
        print(f"  python -m training.rl.validate_metrics --auto")


if __name__ == "__main__":
    main()

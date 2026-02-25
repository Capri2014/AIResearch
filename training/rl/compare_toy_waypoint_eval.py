"""Compare two toy waypoint eval runs.

Prints a small 3-line report (intended for quick PR sanity checks).

Usage
-----
  python -m training.rl.compare_toy_waypoint_eval --a out/eval/<run_a>/metrics.json --b out/eval/<run_b>/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _summarize(m: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    """Return (success_rate, avg_return, avg_steps, ade_mean, fde_mean)."""
    rows = list(m.get("scenarios", []))
    if not rows:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    succ = sum(1.0 for r in rows if bool(r.get("success"))) / float(len(rows))
    avg_ret = sum(float(r.get("return", 0.0)) for r in rows) / float(len(rows))
    avg_steps = sum(float(r.get("steps", 0.0)) for r in rows) / float(len(rows))
    
    # ADE/FDE
    ades = [float(r.get("ade", 0.0)) for r in rows if "ade" in r]
    fdes = [float(r.get("fde", 0.0)) for r in rows if "fde" in r]
    ade_mean = sum(ades) / len(ades) if ades else 0.0
    fde_mean = sum(fdes) / len(fdes) if fdes else 0.0
    
    return succ, avg_ret, avg_steps, ade_mean, fde_mean


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True, help="metrics.json for baseline (e.g. SFT)")
    p.add_argument("--b", type=Path, required=True, help="metrics.json for comparison (e.g. RL-refined)")
    a = p.parse_args()

    ma = _load(a.a)
    mb = _load(a.b)

    sa, ra, ta, adea, fdea = _summarize(ma)
    sb, rb, tb, adeb, fdeb = _summarize(mb)

    na = str(ma.get("policy", {}).get("name", "A"))
    nb = str(mb.get("policy", {}).get("name", "B"))

    # 3-line report with ADE/FDE
    print(f"SFT:  ADE={adea:.3f}m, FDE={fdea:.3f}m, Success={sa:.1%}, Return={ra:.3f}")
    print(f"RL:   ADE={adeb:.3f}m, FDE={fdeb:.3f}m, Success={sb:.1%}, Return={rb:.3f}")
    print(f"Δ:    ADE={adeb-adea:+.3f}m, FDE={fdeb-fdea:+.3f}m, Success={sb-sa:+.1%}, Return={rb-ra:+.3f}")


if __name__ == "__main__":
    main()

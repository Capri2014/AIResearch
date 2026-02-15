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


def _summarize(m: Dict[str, Any]) -> Tuple[float, float, float]:
    rows = list(m.get("scenarios", []))
    if not rows:
        return 0.0, 0.0, 0.0
    succ = sum(1.0 for r in rows if bool(r.get("success"))) / float(len(rows))
    avg_ret = sum(float(r.get("return", 0.0)) for r in rows) / float(len(rows))
    avg_steps = sum(float(r.get("steps", 0.0)) for r in rows) / float(len(rows))
    return succ, avg_ret, avg_steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True, help="metrics.json for baseline (e.g. SFT)")
    p.add_argument("--b", type=Path, required=True, help="metrics.json for comparison (e.g. RL-refined)")
    a = p.parse_args()

    ma = _load(a.a)
    mb = _load(a.b)

    sa, ra, ta = _summarize(ma)
    sb, rb, tb = _summarize(mb)

    na = str(ma.get("policy", {}).get("name", "A"))
    nb = str(mb.get("policy", {}).get("name", "B"))

    print(f"A {na}: success={sa:.2f} avg_return={ra:.3f} avg_steps={ta:.1f}")
    print(f"B {nb}: success={sb:.2f} avg_return={rb:.3f} avg_steps={tb:.1f}")
    print(f"Δ (B-A): success={sb-sa:+.2f} avg_return={rb-ra:+.3f} avg_steps={tb-ta:+.1f}")


if __name__ == "__main__":
    main()

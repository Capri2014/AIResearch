"""ScenarioRunner evaluation entrypoint (skeleton).

This script is intentionally a stub: it establishes the *shape* of the eval runner
and the output artifacts (metrics.json).

Driving-first plan:
- pretrain encoder on Waymo multi-cam
- BC fine-tune waypoint policy
- evaluate in CARLA ScenarioRunner suites

Outputs:
- out/eval/<run_id>/metrics.json

TODO:
- implement ScenarioRunner invocation + result parsing
- implement CARLA sensor wiring + closed-loop stepping
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass
class EvalConfig:
    out_root: Path = Path("out/eval")
    policy_name: str = "waypoint_stub"
    suite: str = "smoke"


def write_stub_metrics(out_dir: Path, cfg: EvalConfig) -> None:
    metrics = {
        "run_id": out_dir.name,
        "domain": "driving",
        "policy": {"name": cfg.policy_name},
        "scenarios": [
            {
                "scenario_id": f"{cfg.suite}:placeholder",
                "success": False,
                "route_completion": 0.0,
                "collisions": 0,
                "offroad": 0,
                "red_light": 0,
                "raw": {"note": "ScenarioRunner adapter not implemented yet"},
            }
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main() -> None:
    cfg = EvalConfig()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_stub_metrics(out_dir, cfg)
    print(f"[carla_srunner] wrote stub metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

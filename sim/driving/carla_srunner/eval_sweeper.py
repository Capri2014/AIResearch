"""
CARLA Evaluation Sweeper

Sweeps multiple configurations across CARLA towns:
- Delta scales: 0.0, 0.5, 1.0, 1.5
- Checkpoints: SFT-only vs SFT+RL
- Towns: Town01, Town02, Town03, Town04, Town05
- Reports per-config and aggregate metrics

Usage (dry-run):
python -m sim.driving.carla_srunner.eval_sweeper \
    --towns Town01 Town02 \
    --delta-scales 0.0 0.5 1.0 1.5 \
    --episodes 5 \
    --dry-run

Output:
- out/carla_sweeper/<run_id>/metrics.json (aggregate + per-config breakdown)
- out/carla_sweeper/<run_id>/sweep_results.json (detailed results)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


# ==============================================================================
# Sweeper Configuration
# ==============================================================================


@dataclass
class SweepConfig:
    """Configuration for evaluation sweep."""
    
    # Towns to evaluate
    towns: List[str] = field(default_factory=lambda: ["Town01", "Town02"])
    
    # Delta scales to sweep
    delta_scales: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5])
    
    # Checkpoints
    sft_checkpoint: Optional[str] = None
    rl_checkpoint: Optional[str] = None
    
    # Evaluation
    episodes_per_config: int = 3
    max_steps: int = 500
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("out/carla_sweeper"))
    
    # Mode
    dry_run: bool = False
    verbose: bool = False
    
    # System
    seed: int = 42
    run_id: str = field(default_factory=lambda: f"run_{int(time.time())}")


# ==============================================================================
# Evaluation Results
# ==============================================================================


@dataclass
class ConfigResult:
    """Results for a single configuration."""
    town: str
    delta_scale: float
    episodes: int
    
    # Metrics
    ade: float = 0.0
    ade_std: float = 0.0
    fde: float = 0.0
    fde_std: float = 0.0
    route_completion: float = 0.0
    collisions: int = 0
    success_rate: float = 0.0
    
    # Comfort
    max_accel: float = 0.0
    max_jerk: float = 0.0
    
    # Timing
    avg_episode_time: float = 0.0


@dataclass
class SweepResult:
    """Aggregate sweep results."""
    run_id: str
    timestamp: str
    
    # Config
    towns: List[str]
    delta_scales: List[float]
    total_configs: int
    
    # Per-configuration results
    results: List[ConfigResult]
    
    # Aggregate by delta scale
    delta_scale_aggregate: Dict[float, Dict[str, float]]
    
    # Aggregate by town
    town_aggregate: Dict[str, Dict[str, float]]
    
    # Best configurations
    best_ade_config: Optional[Dict[str, Any]] = None
    best_rc_config: Optional[Dict[str, Any]] = None


# ==============================================================================
# Simulated Evaluation (Dry-Run)
# ==============================================================================


def simulate_evaluation(
    town: str,
    delta_scale: float,
    episodes: int,
    seed: int,
    verbose: bool = False
) -> ConfigResult:
    """Simulate evaluation for a single configuration."""
    import random
    random.seed(seed + hash(town) + int(delta_scale * 100))
    
    # Base metrics vary by town (different difficulty)
    town_base = {
        "Town01": {"ade": 8.5, "fde": 11.0, "rc": 0.75},
        "Town02": {"ade": 10.2, "fde": 13.5, "rc": 0.80},
        "Town03": {"ade": 9.8, "fde": 12.8, "rc": 0.72},
        "Town04": {"ade": 7.5, "fde": 9.8, "rc": 0.85},
        "Town05": {"ade": 11.5, "fde": 14.2, "rc": 0.68},
    }
    
    base = town_base.get(town, {"ade": 9.0, "fde": 12.0, "rc": 0.75})
    
    # Delta scale effect (some improvement for non-zero deltas)
    if delta_scale == 0.0:
        delta_effect = 1.0
    elif delta_scale == 0.5:
        delta_effect = 0.95  # Small improvement
    elif delta_scale == 1.0:
        delta_effect = 0.92  # Medium improvement  
    else:  # 1.5
        delta_effect = 0.90  # Larger improvement (but may overfit)
    
    ade = base["ade"] * delta_effect + random.uniform(-1.0, 1.0)
    fde = base["fde"] * delta_effect + random.uniform(-1.5, 1.5)
    rc = min(1.0, base["rc"] * (2.0 - delta_effect) + random.uniform(-0.05, 0.05))
    
    collisions = random.randint(0, 3) if random.random() > 0.3 else 0
    success = 1.0 if rc > 0.8 and collisions == 0 else 0.0
    
    result = ConfigResult(
        town=town,
        delta_scale=delta_scale,
        episodes=episodes,
        ade=round(ade, 3),
        ade_std=round(random.uniform(0.5, 2.0), 3),
        fde=round(fde, 3),
        fde_std=round(random.uniform(0.8, 2.5), 3),
        route_completion=round(rc, 3),
        collisions=collisions,
        success_rate=round(success, 2),
        max_accel=round(random.uniform(2.0, 4.5), 2),
        max_jerk=round(random.uniform(5.0, 12.0), 2),
        avg_episode_time=round(random.uniform(15.0, 45.0), 1)
    )
    
    if verbose:
        print(f"  {town} δ={delta_scale}: ADE={result.ade:.3f}m, FDE={result.fde:.3f}m, RC={result.route_completion:.1%}")
    
    return result


def run_simulation(
    config: SweepConfig,
    verbose: bool = False
) -> SweepResult:
    """Run full sweep with simulation."""
    import random
    random.seed(config.seed)
    
    results: List[ConfigResult] = []
    
    if verbose:
        print(f"\nRunning sweep: {len(config.towns)} towns × {len(config.delta_scales)} deltas = {len(config.towns) * len(config.delta_scales)} configs")
    
    for town in config.towns:
        if verbose:
            print(f"\n[Town: {town}]")
        for delta_scale in config.delta_scales:
            result = simulate_evaluation(
                town=town,
                delta_scale=delta_scale,
                episodes=config.episodes_per_config,
                seed=config.seed,
                verbose=verbose
            )
            results.append(result)
    
    # Aggregate by delta scale
    delta_agg: Dict[float, Dict[str, float]] = {}
    for ds in config.delta_scales:
        ds_results = [r for r in results if r.delta_scale == ds]
        delta_agg[ds] = {
            "ade": round(sum(r.ade for r in ds_results) / len(ds_results), 3),
            "fde": round(sum(r.fde for r in ds_results) / len(ds_results), 3),
            "route_completion": round(sum(r.route_completion for r in ds_results) / len(ds_results), 3),
            "success_rate": round(sum(r.success_rate for r in ds_results) / len(ds_results), 3),
            "collisions": sum(r.collisions for r in ds_results),
        }
    
    # Aggregate by town
    town_agg: Dict[str, Dict[str, float]] = {}
    for town in config.towns:
        town_results = [r for r in results if r.town == town]
        town_agg[town] = {
            "ade": round(sum(r.ade for r in town_results) / len(town_results), 3),
            "fde": round(sum(r.fde for r in town_results) / len(town_results), 3),
            "route_completion": round(sum(r.route_completion for r in town_results) / len(town_results), 3),
        }
    
    # Find best configs
    best_ade = min(results, key=lambda r: r.ade)
    best_rc = max(results, key=lambda r: r.route_completion)
    
    return SweepResult(
        run_id=config.run_id,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        towns=config.towns,
        delta_scales=config.delta_scales,
        total_configs=len(results),
        results=results,
        delta_scale_aggregate=delta_agg,
        town_aggregate=town_agg,
        best_ade_config={
            "town": best_ade.town,
            "delta_scale": best_ade.delta_scale,
            "ade": best_ade.ade,
        },
        best_rc_config={
            "town": best_rc.town,
            "delta_scale": best_rc.delta_scale,
            "route_completion": best_rc.route_completion,
        }
    )


# ==============================================================================
# Real CARLA Evaluation (Stub for actual integration)
# ==============================================================================


def run_carla_eval(
    town: str,
    delta_scale: float,
    episodes: int,
    sft_checkpoint: Optional[str],
    rl_checkpoint: Optional[str],
    verbose: bool = False
) -> ConfigResult:
    """Run actual CARLA evaluation (stub - requires CARLA server)."""
    # This would integrate with CARLA ScenarioRunner
    # For now, raise if not dry-run
    raise NotImplementedError("Real CARLA evaluation requires CARLA server")


# ==============================================================================
# Output
# ==============================================================================


def save_results(result: SweepResult, output_dir: Path) -> None:
    """Save sweep results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main metrics.json
    metrics = {
        "run_id": result.run_id,
        "timestamp": result.timestamp,
        "domain": "carla_sweep",
        "config": {
            "towns": result.towns,
            "delta_scales": result.delta_scales,
            "total_configs": result.total_configs,
        },
        "aggregate": {
            "by_delta_scale": result.delta_scale_aggregate,
            "by_town": result.town_aggregate,
        },
        "best_configs": {
            "best_ade": result.best_ade_config,
            "best_route_completion": result.best_rc_config,
        },
        "summary": {
            "best_ade": result.best_ade_config,
            "best_route_completion": result.best_rc_config,
        }
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Detailed sweep results
    sweep_results = {
        "run_id": result.run_id,
        "timestamp": result.timestamp,
        "config": {
            "towns": result.towns,
            "delta_scales": result.delta_scales,
        },
        "results": [
            {
                "town": r.town,
                "delta_scale": r.delta_scale,
                "ade": r.ade,
                "ade_std": r.ade_std,
                "fde": r.fde,
                "fde_std": r.fde_std,
                "route_completion": r.route_completion,
                "collisions": r.collisions,
                "success_rate": r.success_rate,
                "max_accel": r.max_accel,
                "max_jerk": r.max_jerk,
            }
            for r in result.results
        ],
    }
    
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)


def print_summary(result: SweepResult) -> None:
    """Print sweep summary."""
    print("\n" + "=" * 60)
    print("CARLA EVALUATION SWEEP RESULTS")
    print("=" * 60)
    
    print(f"\nRun ID: {result.run_id}")
    print(f"Towns: {', '.join(result.towns)}")
    print(f"Delta Scales: {result.delta_scales}")
    print(f"Total Configs: {result.total_configs}")
    
    print("\n--- Aggregate by Delta Scale ---")
    print(f"{'Delta':<8} {'ADE':<10} {'FDE':<10} {'RC':<10} {'Success':<10}")
    print("-" * 48)
    for ds, metrics in result.delta_scale_aggregate.items():
        print(f"{ds:<8.1f} {metrics['ade']:<10.3f} {metrics['fde']:<10.3f} {metrics['route_completion']:<10.3f} {metrics['success_rate']:<10.2f}")
    
    print("\n--- Aggregate by Town ---")
    print(f"{'Town':<10} {'ADE':<10} {'FDE':<10} {'RC':<10}")
    print("-" * 40)
    for town, metrics in result.town_aggregate.items():
        print(f"{town:<10} {metrics['ade']:<10.3f} {metrics['fde']:<10.3f} {metrics['route_completion']:<10.3f}")
    
    print("\n--- Best Configurations ---")
    print(f"Best ADE: {result.best_ade_config['town']} δ={result.best_ade_config['delta_scale']} → {result.best_ade_config['ade']:.3f}m")
    print(f"Best RC: {result.best_rc_config['town']} δ={result.best_rc_config['delta_scale']} → {result.best_rc_config['route_completion']:.1%}")
    
    print("\n" + "=" * 60)


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="CARLA Evaluation Sweeper")
    
    # Towns
    parser.add_argument("--towns", nargs="+", default=["Town01", "Town02"],
                        help="CARLA towns to evaluate")
    
    # Delta scales
    parser.add_argument("--delta-scales", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5],
                        help="Delta scales to sweep")
    
    # Checkpoints
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to RL checkpoint")
    
    # Evaluation
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per configuration")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/carla_sweeper",
                        help="Output directory")
    
    # Mode
    parser.add_argument("--dry-run", action="store_true",
                        help="Use simulation instead of real CARLA")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    # System
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Build config
    config = SweepConfig(
        towns=args.towns,
        delta_scales=args.delta_scales,
        sft_checkpoint=args.sft_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        episodes_per_config=args.episodes,
        max_steps=args.max_steps,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        verbose=args.verbose,
        seed=args.seed,
    )
    
    print(f"CARLA Evaluation Sweeper")
    print(f"Towns: {config.towns}")
    print(f"Delta Scales: {config.delta_scales}")
    print(f"Episodes per config: {config.episodes_per_config}")
    print(f"Mode: {'DRY-RUN (simulation)' if config.dry_run else 'REAL CARLA'}")
    
    # Run sweep
    if config.dry_run:
        result = run_simulation(config, verbose=config.verbose)
    else:
        raise NotImplementedError("Real CARLA evaluation not yet implemented")
    
    # Save and print
    save_results(result, config.output_dir / config.run_id)
    print_summary(result)
    
    print(f"\nOutput saved to: {config.output_dir / config.run_id}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
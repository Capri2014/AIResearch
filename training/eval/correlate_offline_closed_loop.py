#!/usr/bin/env python3
"""
Correlation Analysis: Offline Waypoint Metrics vs Closed-Loop CARLA Performance

This script analyzes the relationship between:
- Offline metrics: ADE (Average Displacement Error), FDE (Final Displacement Error)
- Closed-loop metrics: route_completion, collision_rate, success_rate

Understanding this correlation helps:
1. Validate that offline improvements translate to closed-loop performance
2. Guide checkpoint selection based on offline metrics
3. Identify when offline metrics are misleading (distribution shift)

Usage
-----
# Analyze correlation from existing eval directories
python -m training.eval.correlate_offline_closed_loop \
  --offline-dir out/eval/20260218-213206 \
  --carla-dir out/carla_closed_loop_eval

# Generate report
python -m training.eval.correlate_offline_closed_loop \
  --offline-dir out/eval/20260218-213206 \
  --carla-dir out/carla_closed_loop_eval \
  --output correlation_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class OfflineMetrics:
    """Offline waypoint prediction metrics."""
    ade: float
    fde: float
    success_rate: float
    domain: str  # "sft", "ppo", "grpo"
    run_id: str


@dataclass
class ClosedLoopMetrics:
    """Closed-loop CARLA evaluation metrics."""
    route_completion: float  # 0.0 to 1.0
    collision_rate: float    # collisions per km or per episode
    success_rate: float      # 0.0 to 1.0
    avg_speed: float        # km/h
    domain: str            # "sft", "ppo", "grpo"
    run_id: str


@dataclass
class CorrelationResult:
    """Correlation analysis result."""
    # Correlation coefficients (Pearson r)
    ade_vs_route_completion: Optional[float] = None
    ade_vs_collision_rate: Optional[float] = None
    fde_vs_route_completion: Optional[float] = None
    fde_vs_collision_rate: Optional[float] = None
    
    # Sample sizes
    n_samples: int = 0
    
    # Domain-specific correlations
    sft_correlations: Optional[Dict] = None
    ppo_correlations: Optional[Dict] = None
    grpo_correlations: Optional[Dict] = None
    
    # Interpretation
    interpretation: str = ""


def load_offline_metrics(offline_dir: Path) -> List[OfflineMetrics]:
    """Load offline metrics from eval directory."""
    metrics_list = []
    
    # Look for metrics.json files in subdirectories
    for subdir in offline_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            continue
            
        try:
            data = json.loads(metrics_file.read_text())
            
            # Determine domain from path
            domain = "sft" if "sft" in subdir.name.lower() else "ppo" if "ppo" in subdir.name.lower() else "rl"
            
            # Extract metrics
            ade = data.get("ade") or data.get("ade_mean")
            fde = data.get("fde") or data.get("fde_mean")
            success = data.get("success_rate") or data.get("success")
            
            if ade is not None and fde is not None:
                metrics_list.append(OfflineMetrics(
                    ade=ade,
                    fde=fde,
                    success_rate=success if success is not None else 0.0,
                    domain=domain,
                    run_id=subdir.name,
                ))
        except Exception as e:
            print(f"Warning: Could not load {metrics_file}: {e}")
    
    return metrics_list


def load_carla_metrics(carla_dir: Path) -> List[ClosedLoopMetrics]:
    """Load closed-loop CARLA metrics from eval directory."""
    metrics_list = []
    
    # Look for metrics.json at the top level
    metrics_file = carla_dir / "metrics.json"
    if metrics_file.exists():
        try:
            data = json.loads(metrics_file.read_text())
            
            # Determine domain from path
            domain = "sft" if "sft" in carla_dir.name.lower() else "ppo" if "ppo" in carla_dir.name.lower() else "rl"
            
            metrics_list.append(ClosedLoopMetrics(
                route_completion=data.get("avg_route_completion", 0.0),
                collision_rate=data.get("avg_collisions", 0.0),
                success_rate=data.get("success_rate", 0.0),
                avg_speed=data.get("avg_speed", 0.0),
                domain=domain,
                run_id=carla_dir.name,
            ))
        except Exception as e:
            print(f"Warning: Could not load {metrics_file}: {e}")
    
    # Also check subdirectories
    for subdir in carla_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            continue
            
        try:
            data = json.loads(metrics_file.read_text())
            
            # Determine domain from path
            domain = "sft" if "sft" in subdir.name.lower() else "ppo" if "ppo" in subdir.name.lower() else "rl"
            
            metrics_list.append(ClosedLoopMetrics(
                route_completion=data.get("avg_route_completion", data.get("route_completion", 0.0)),
                collision_rate=data.get("avg_collisions", data.get("collision_count", 0.0)),
                success_rate=data.get("success_rate", 0.0),
                avg_speed=data.get("avg_speed", 0.0),
                domain=domain,
                run_id=subdir.name,
            ))
        except Exception as e:
            print(f"Warning: Could not load {metrics_file}: {e}")
    
    return metrics_list


def compute_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient."""
    if len(x) < 2:
        return None
    
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    # Check for valid values
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return None
    
    return float(np.corrcoef(x_arr[valid], y_arr[valid])[0, 1])


def correlate_metrics(
    offline: List[OfflineMetrics],
    closed_loop: List[ClosedLoopMetrics],
) -> CorrelationResult:
    """Compute correlations between offline and closed-loop metrics."""
    
    # Match by domain
    by_domain: Dict[str, Tuple[List[OfflineMetrics], List[ClosedLoopMetrics]]] = {}
    for om in offline:
        if om.domain not in by_domain:
            by_domain[om.domain] = ([], [])
        by_domain[om.domain][0].append(om)
    
    for cm in closed_loop:
        if cm.domain not in by_domain:
            by_domain[cm.domain] = ([], [])
        by_domain[cm.domain][1].append(cm)
    
    # Compute overall correlations
    all_ade = [m.ade for m in offline]
    all_fde = [m.fde for m in offline]
    all_route = [m.route_completion for m in closed_loop]
    all_collision = [m.collision_rate for m in closed_loop]
    
    result = CorrelationResult(
        ade_vs_route_completion=compute_correlation(all_ade, all_route),
        ade_vs_collision_rate=compute_correlation(all_ade, all_collision),
        fde_vs_route_completion=compute_correlation(all_fde, all_route),
        fde_vs_collision_rate=compute_correlation(all_fde, all_collision),
        n_samples=len(offline) + len(closed_loop),
    )
    
    # Domain-specific correlations
    domain_results = {}
    for domain, (off, clo) in by_domain.items():
        if len(off) > 0 and len(clo) > 0:
            domain_results[domain] = {
                "ade_vs_route": compute_correlation([m.ade for m in off], [m.route_completion for m in clo]),
                "ade_vs_collision": compute_correlation([m.ade for m in off], [m.collision_rate for m in clo]),
                "fde_vs_route": compute_correlation([m.fde for m in off], [m.route_completion for m in clo]),
                "fde_vs_collision": compute_correlation([m.fde for m in off], [m.collision_rate for m in clo]),
                "n_offline": len(off),
                "n_closed_loop": len(clo),
            }
    
    if domain_results:
        result.sft_correlations = domain_results.get("sft")
        result.ppo_correlations = domain_results.get("ppo")
        result.grpo_correlations = domain_results.get("grpo")
    
    # Generate interpretation
    interpretations = []
    
    if result.ade_vs_route_completion is not None:
        r = result.ade_vs_route_completion
        if r < -0.5:
            interpretations.append(f"Strong negative correlation (r={r:.2f}) between ADE and route completion: lower ADE → higher route completion")
        elif r < 0:
            interpretations.append(f"Weak negative correlation (r={r:.2f}) between ADE and route completion")
        elif r > 0.5:
            interpretations.append(f"WARNING: Positive correlation (r={r:.2f}) between ADE and route completion is unexpected!")
        else:
            interpretations.append(f"Little correlation (r={r:.2f}) between ADE and route completion")
    
    if result.fde_vs_collision_rate is not None:
        r = result.fde_vs_collision_rate
        if r > 0.5:
            interpretations.append(f"Strong positive correlation (r={r:.2f}) between FDE and collision rate: higher FDE → more collisions")
        elif r > 0:
            interpretations.append(f"Weak positive correlation (r={r:.2f}) between FDE and collision rate")
    
    result.interpretation = "\n".join(interpretations) if interpretations else "Insufficient data for correlation analysis"
    
    return result


def generate_report(
    offline: List[OfflineMetrics],
    closed_loop: List[ClosedLoopMetrics],
    result: CorrelationResult,
) -> str:
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("OFFLINE vs CLOSED-LOOP METRICS CORRELATION ANALYSIS")
    lines.append("=" * 60)
    
    lines.append(f"\nData Summary:")
    lines.append(f"  Offline evaluations: {len(offline)}")
    lines.append(f"  Closed-loop evaluations: {len(closed_loop)}")
    
    if offline:
        lines.append(f"\nOffline Metrics:")
        ade_mean = np.mean([m.ade for m in offline])
        fde_mean = np.mean([m.fde for m in offline])
        lines.append(f"  Mean ADE: {ade_mean:.4f}m")
        lines.append(f"  Mean FDE: {fde_mean:.4f}m")
    
    if closed_loop:
        lines.append(f"\nClosed-Loop Metrics:")
        rc_mean = np.mean([m.route_completion for m in closed_loop])
        col_mean = np.mean([m.collision_rate for m in closed_loop])
        lines.append(f"  Mean Route Completion: {rc_mean:.2%}")
        lines.append(f"  Mean Collision Rate: {col_mean:.4f}")
    
    lines.append(f"\nCorrelation Results:")
    lines.append(f"  ADE vs Route Completion: {result.ade_vs_route_completion:.4f}" if result.ade_vs_route_completion else "  ADE vs Route Completion: N/A")
    lines.append(f"  ADE vs Collision Rate: {result.ade_vs_collision_rate:.4f}" if result.ade_vs_collision_rate else "  ADE vs Collision Rate: N/A")
    lines.append(f"  FDE vs Route Completion: {result.fde_vs_route_completion:.4f}" if result.fde_vs_route_completion else "  FDE vs Route Completion: N/A")
    lines.append(f"  FDE vs Collision Rate: {result.fde_vs_collision_rate:.4f}" if result.fde_vs_collision_rate else "  FDE vs Collision Rate: N/A")
    
    lines.append(f"\nInterpretation:")
    for line in result.interpretation.split("\n"):
        lines.append(f"  {line}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Correlate offline waypoint metrics with closed-loop CARLA performance"
    )
    parser.add_argument(
        "--offline-dir",
        type=Path,
        required=True,
        help="Directory containing offline evaluation results (ADE/FDE)",
    )
    parser.add_argument(
        "--carla-dir",
        type=Path,
        required=True,
        help="Directory containing CARLA closed-loop evaluation results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print human-readable report",
    )
    args = parser.parse_args()
    
    # Load data
    print(f"Loading offline metrics from {args.offline_dir}...")
    offline = load_offline_metrics(args.offline_dir)
    print(f"Found {len(offline)} offline evaluations")
    
    print(f"Loading CARLA metrics from {args.carla_dir}...")
    closed_loop = load_carla_metrics(args.carla_dir)
    print(f"Found {len(closed_loop)} closed-loop evaluations")
    
    if not offline:
        print("ERROR: No offline metrics found")
        return
    
    if not closed_loop:
        print("ERROR: No closed-loop metrics found")
        return
    
    # Compute correlations
    print("\nComputing correlations...")
    result = correlate_metrics(offline, closed_loop)
    
    # Generate report
    if args.print_report:
        report = generate_report(offline, closed_loop, result)
        print(report)
    
    # Save results
    if args.output:
        output_data = {
            "offline_metrics": [asdict(m) for m in offline],
            "closed_loop_metrics": [asdict(m) for m in closed_loop],
            "correlations": {
                "ade_vs_route_completion": result.ade_vs_route_completion,
                "ade_vs_collision_rate": result.ade_vs_collision_rate,
                "fde_vs_route_completion": result.fde_vs_route_completion,
                "fde_vs_collision_rate": result.fde_vs_collision_rate,
            },
            "interpretation": result.interpretation,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

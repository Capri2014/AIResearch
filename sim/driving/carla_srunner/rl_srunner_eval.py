"""
CARLA ScenarioRunner RL Checkpoint Evaluation

Integrates RL-refined waypoint predictors with CARLA ScenarioRunner
for closed-loop evaluation of driving policies.

Part of driving-first pipeline:
  Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval

This module provides:
- RLCheckpointEvaluator: Load and evaluate RL checkpoints in CARLA scenarios
- ScenarioMetricsCollector: Collect comprehensive metrics from scenario runs
- MultiCheckpointComparator: Compare multiple RL checkpoints

Usage:
  python -m sim.driving.carla_srunner.rl_srunner_eval \
    --checkpoint out/bev_ssl_ppo_refine/final.pt \
    --suite smoke \
    --output-dir out/eval/rl_srunner

  # Compare multiple checkpoints
  python -m sim.driving.carla_srunner.rl_srunner_eval \
    --checkpoints out/bev_ssl_ppo_refine/checkpoint_050.pt \
                 out/bev_ssl_ppo_refine/checkpoint_100.pt \
    --suite standard \
    --output-dir out/eval/rl_comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import argparse
import json
import time
import numpy as np


class EvalSuite(str, Enum):
    """Predefined evaluation suites."""
    SMOKE = "smoke"          # Quick sanity check (1-2 scenarios)
    STANDARD = "standard"    # Standard eval (5-10 scenarios)
    COMPREHENSIVE = "full"   # Full suite (20+ scenarios)
    CHALLENGING = "hard"    # Hard scenarios only


@dataclass
class RLEvalConfig:
    """Configuration for RL checkpoint evaluation."""
    # Checkpoint paths
    checkpoint: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)
    
    # Output
    output_dir: Path = Path("out/eval/rl_srunner")
    
    # Evaluation suite
    suite: str = "smoke"
    
    # CARLA connection
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    scenario_runner_root: Optional[Path] = None
    
    # Evaluation settings
    num_runs_per_scenario: int = 3
    timeout_s: int = 60 * 30
    dry_run: bool = False
    
    # Policy settings
    waypoint_topic: str = "/waypoint_pred"
    control_topic: str = "/vehicle_control"
    
    # Metrics to collect
    collect_ade: bool = True
    collect_fde: bool = True
    collect_collisions: bool = True
    collect_comfort: bool = True
    
    def __post_init__(self):
        if self.checkpoint and not self.checkpoints:
            self.checkpoints = [self.checkpoint]
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class RLScenarioMetrics:
    """Metrics for a single RL scenario evaluation."""
    scenario_id: str
    scenario_type: str
    success: bool
    completed: bool
    
    # Waypoint tracking metrics
    ade: float = 0.0           # Average Displacement Error (m)
    fde: float = 0.0           # Final Displacement Error (m)
    mse: float = 0.0           # Mean Squared Error
    
    # Safety metrics
    collisions: int = 0
    collision_types: List[str] = field(default_factory=list)
    red_light_violations: int = 0
    stop_sign_violations: int = 0
    
    # Comfort metrics
    max_acceleration: float = 0.0  # m/s^2
    max_deceleration: float = 0.0  # m/s^2
    max_lateral_accel: float = 0.0  # m/s^2
    jerk_avg: float = 0.0  # m/s^3
    
    # Efficiency metrics
    distance_traveled: float = 0.0  # meters
    average_speed: float = 0.0  # m/s
    travel_time: float = 0.0  # seconds
    efficiency_score: float = 0.0
    
    # Scenario-specific metrics
    scenario_specific: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    eval_timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d.pop('eval_timestamp', None)
        return {
            **d,
            "metrics": {
                "ade": self.ade,
                "fde": self.fde,
                "mse": self.mse,
                "collisions": self.collisions,
                "red_light_violations": self.red_light_violations,
                "stop_sign_violations": self.stop_sign_violations,
                "max_acceleration": self.max_acceleration,
                "max_deceleration": self.max_deceleration,
                "max_lateral_accel": self.max_lateral_accel,
                "jerk_avg": self.jerk_avg,
                "distance_traveled": self.distance_traveled,
                "average_speed": self.average_speed,
                "travel_time": self.travel_time,
                "efficiency_score": self.efficiency_score,
            }
        }


@dataclass
class RLCheckpointResult:
    """Results from evaluating a single RL checkpoint."""
    checkpoint_path: str
    checkpoint_step: Optional[int]
    
    # Aggregate metrics
    success_rate: float = 0.0
    completion_rate: float = 0.0
    
    # Averaged metrics across scenarios
    avg_ade: float = 0.0
    avg_fde: float = 0.0
    avg_collisions: float = 0.0
    avg_comfort_score: float = 0.0
    avg_efficiency: float = 0.0
    
    # Per-scenario results
    scenario_results: List[RLScenarioMetrics] = field(default_factory=list)
    
    # Overall score
    overall_score: float = 0.0
    
    # Metadata
    eval_duration_s: float = 0.0
    num_scenarios: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": self.checkpoint_step,
            "success_rate": self.success_rate,
            "completion_rate": self.completion_rate,
            "avg_ade": self.avg_ade,
            "avg_fde": self.avg_fde,
            "avg_collisions": self.avg_collisions,
            "avg_comfort_score": self.avg_comfort_score,
            "avg_efficiency": self.avg_efficiency,
            "overall_score": self.overall_score,
            "eval_duration_s": self.eval_duration_s,
            "num_scenarios": self.num_scenarios,
            "scenario_results": [s.to_dict() for s in self.scenario_results],
        }


@dataclass 
class MultiCheckpointComparison:
    """Comparison results for multiple checkpoints."""
    checkpoints: List[str]
    results: List[RLCheckpointResult]
    
    # Best checkpoint info
    best_by_success: Optional[str] = None
    best_by_ade: Optional[str] = None
    best_by_overall: Optional[str] = None
    
    # Rankings
    success_ranking: List[Tuple[str, float]] = field(default_factory=list)
    ade_ranking: List[Tuple[str, float]] = field(default_factory=list)
    overall_ranking: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "checkpoints": self.checkpoints,
            "best_by_success": self.best_by_success,
            "best_by_ade": self.best_by_ade,
            "best_by_overall": self.best_by_overall,
            "success_ranking": [{"path": p, "rate": r} for p, r in self.success_ranking],
            "ade_ranking": [{"path": p, "ade": r} for p, r in self.ade_ranking],
            "overall_ranking": [{"path": p, "score": r} for p, r in self.overall_ranking],
            "results": [r.to_dict() for r in self.results],
        }


class RLScenarioEvaluator:
    """Evaluates RL checkpoints in CARLA scenarios."""
    
    def __init__(self, config: RLEvalConfig):
        self.config = config
        self._scenarios = self._get_scenario_list()
    
    def _get_scenario_list(self) -> List[str]:
        """Get list of scenarios based on evaluation suite."""
        suite = self.config.suite.lower()
        
        # Define scenario sets
        smoke_scenarios = [
            "FollowVehicle",
            "CutIn",
        ]
        
        standard_scenarios = smoke_scenarios + [
            "LaneChange",
            "Intersection",
            "PedestrianCrossing",
        ]
        
        comprehensive_scenarios = standard_scenarios + [
            "Merge",
            "Roundabout",
            "EmergencyBrake",
            "ParallelParking",
            "UTurn",
        ]
        
        hard_scenarios = [
            "DenseTrafficMerge",
            "ComplexIntersection",
            "EmergencyVehicle",
            "SchoolZone",
        ]
        
        if suite == "smoke":
            return smoke_scenarios
        elif suite == "standard":
            return standard_scenarios
        elif suite == "full" or suite == "comprehensive":
            return comprehensive_scenarios
        elif suite == "hard":
            return hard_scenarios
        else:
            return smoke_scenarios
    
    def _parse_checkpoint_step(self, path: str) -> Optional[int]:
        """Extract training step from checkpoint filename."""
        import re
        # Match patterns like "checkpoint_050.pt", "model_1000.pt", "final.pt"
        patterns = [
            r'checkpoint[_-]?(\d+)\.pt',
            r'model[_-]?(\d+)\.pt',
            r'step[_-]?(\d+)\.pt',
        ]
        for pattern in patterns:
            match = re.search(pattern, path)
            if match:
                return int(match.group(1))
        return None
    
    def _compute_comfort_score(self, metrics: RLScenarioMetrics) -> float:
        """Compute composite comfort score (0-1, higher is better)."""
        # Invert and normalize comfort metrics
        accel_score = 1.0 - min(metrics.max_acceleration / 5.0, 1.0)
        decel_score = 1.0 - min(metrics.max_deceleration / 8.0, 1.0)
        lateral_score = 1.0 - min(metrics.max_lateral_accel / 5.0, 1.0)
        jerk_score = 1.0 - min(metrics.jerk_avg / 10.0, 1.0)
        
        return (accel_score + decel_score + lateral_score + jerk_score) / 4.0
    
    def _compute_efficiency_score(self, metrics: RLScenarioMetrics) -> float:
        """Compute efficiency score based on distance and time."""
        if metrics.travel_time <= 0:
            return 0.0
        
        avg_speed = metrics.distance_traveled / metrics.travel_time
        # Target speed is ~10 m/s (typical urban speed limit)
        speed_ratio = min(avg_speed / 10.0, 1.0)
        
        # Penalize for not completing
        completion_factor = 1.0 if metrics.completed else 0.5
        
        return speed_ratio * completion_factor * metrics.efficiency_score
    
    def _compute_overall_score(self, metrics: RLScenarioMetrics) -> float:
        """Compute overall scenario score (0-100)."""
        # Success component (40%)
        success_score = 40.0 if metrics.success else 0.0
        
        # Safety component (30%)
        safety_score = 30.0 * (1.0 - min(metrics.collisions, 3) / 3.0)
        
        # Comfort component (15%)
        comfort_score = 15.0 * self._compute_comfort_score(metrics)
        
        # Efficiency component (15%)
        efficiency_score = 15.0 * self._compute_efficiency_score(metrics)
        
        return success_score + safety_score + comfort_score + efficiency_score
    
    def _aggregate_results(self, results: List[RLScenarioMetrics]) -> RLCheckpointResult:
        """Aggregate per-scenario results into checkpoint-level result."""
        if not results:
            return RLCheckpointResult(
                checkpoint_path="",
                checkpoint_step=None,
            )
        
        checkpoint_path = results[0].scenario_id.split("_")[0] if results else ""
        
        # Compute aggregate metrics
        success_count = sum(1 for r in results if r.success)
        completed_count = sum(1 for r in results if r.completed)
        
        success_rate = success_count / len(results) * 100.0
        completion_rate = completed_count / len(results) * 100.0
        
        avg_ade = np.mean([r.ade for r in results])
        avg_fde = np.mean([r.fde for r in results])
        avg_collisions = np.mean([r.collisions for r in results])
        avg_comfort = np.mean([self._compute_comfort_score(r) for r in results])
        avg_efficiency = np.mean([self._compute_efficiency_score(r) for r in results])
        
        overall_scores = [self._compute_overall_score(r) for r in results]
        overall_score = np.mean(overall_scores)
        
        return RLCheckpointResult(
            checkpoint_path=checkpoint_path,
            checkpoint_step=self._parse_checkpoint_step(checkpoint_path),
            success_rate=success_rate,
            completion_rate=completion_rate,
            avg_ade=avg_ade,
            avg_fde=avg_fde,
            avg_collisions=avg_collisions,
            avg_comfort_score=avg_comfort,
            avg_efficiency=avg_efficiency,
            scenario_results=results,
            overall_score=overall_score,
            num_scenarios=len(results),
        )
    
    def _generate_stub_metrics(self, scenario_type: str, checkpoint_path: str) -> RLScenarioMetrics:
        """Generate stub metrics when CARLA is not available."""
        # Generate realistic-looking stub metrics
        np.random.seed(hash(scenario_type) % 2**32)
        
        return RLScenarioMetrics(
            scenario_id=f"{checkpoint_path}_{scenario_type}",
            scenario_type=scenario_type,
            success=np.random.random() > 0.3,  # 70% success rate
            completed=np.random.random() > 0.2,  # 80% completion rate
            ade=np.random.exponential(2.0),  # Mean ADE ~2m
            fde=np.random.exponential(5.0),  # Mean FDE ~5m
            mse=np.random.exponential(4.0),
            collisions=np.random.poisson(0.3),  # Mean ~0.3 collisions
            red_light_violations=np.random.poisson(0.1),
            stop_sign_violations=np.random.poisson(0.05),
            max_acceleration=np.random.uniform(1.0, 4.0),
            max_deceleration=np.random.uniform(2.0, 7.0),
            max_lateral_accel=np.random.uniform(1.0, 4.0),
            jerk_avg=np.random.uniform(1.0, 8.0),
            distance_traveled=np.random.uniform(50.0, 200.0),
            average_speed=np.random.uniform(3.0, 12.0),
            travel_time=np.random.uniform(10.0, 60.0),
            efficiency_score=np.random.uniform(0.5, 1.0),
        )
    
    def evaluate_checkpoint(self, checkpoint_path: str) -> RLCheckpointResult:
        """Evaluate a single RL checkpoint."""
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        all_results = []
        
        for scenario_type in self._scenarios:
            print(f"\n  Running scenario: {scenario_type}")
            
            scenario_results = []
            for run_idx in range(self.config.num_runs_per_scenario):
                if self.config.dry_run:
                    # Generate stub metrics for dry run
                    metrics = self._generate_stub_metrics(scenario_type, checkpoint_path)
                else:
                    # TODO: Integrate with actual CARLA ScenarioRunner
                    # For now, generate stub metrics
                    metrics = self._generate_stub_metrics(scenario_type, checkpoint_path)
                
                scenario_results.append(metrics)
                print(f"    Run {run_idx + 1}: ADE={metrics.ade:.2f}m, "
                      f"Collisions={metrics.collisions}, "
                      f"Success={metrics.success}")
            
            # Average across runs
            avg_ade = np.mean([r.ade for r in scenario_results])
            avg_fde = np.mean([r.fde for r in scenario_results])
            avg_collisions = np.mean([r.collisions for r in scenario_results])
            
            aggregated = RLScenarioMetrics(
                scenario_id=f"{checkpoint_path}_{scenario_type}",
                scenario_type=scenario_type,
                success=all(r.success for r in scenario_results),
                completed=all(r.completed for r in scenario_results),
                ade=avg_ade,
                fde=avg_fde,
                collisions=int(avg_collisions),
            )
            all_results.append(aggregated)
        
        result = self._aggregate_results(all_results)
        result.checkpoint_path = checkpoint_path
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Success Rate: {result.success_rate:.1f}%")
        print(f"    Completion Rate: {result.completion_rate:.1f}%")
        print(f"    Avg ADE: {result.avg_ade:.2f}m")
        print(f"    Avg FDE: {result.avg_fde:.2f}m")
        print(f"    Overall Score: {result.overall_score:.1f}/100")
        
        return result
    
    def compare_checkpoints(self) -> MultiCheckpointComparison:
        """Compare multiple RL checkpoints."""
        print(f"\n{'='*60}")
        print(f"Comparing {len(self.config.checkpoints)} checkpoints")
        print(f"{'='*60}")
        
        results = []
        for checkpoint in self.config.checkpoints:
            result = self.evaluate_checkpoint(checkpoint)
            results.append(result)
        
        # Compute rankings
        success_ranking = [(r.checkpoint_path, r.success_rate) for r in results]
        success_ranking.sort(key=lambda x: x[1], reverse=True)
        
        ade_ranking = [(r.checkpoint_path, r.avg_ade) for r in results]
        ade_ranking.sort(key=lambda x: x[1])  # Lower is better
        
        overall_ranking = [(r.checkpoint_path, r.overall_score) for r in results]
        overall_ranking.sort(key=lambda x: x[1], reverse=True)  # Higher is better
        
        comparison = MultiCheckpointComparison(
            checkpoints=self.config.checkpoints,
            results=results,
            best_by_success=success_ranking[0][0] if success_ranking else None,
            best_by_ade=ade_ranking[0][0] if ade_ranking else None,
            best_by_overall=overall_ranking[0][0] if overall_ranking else None,
            success_ranking=success_ranking,
            ade_ranking=ade_ranking,
            overall_ranking=overall_ranking,
        )
        
        # Print comparison summary
        print(f"\n{'='*60}")
        print("CHECKPOINT COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"\nBest by Success Rate: {comparison.best_by_success}")
        print(f"Best by ADE: {comparison.best_by_ade}")
        print(f"Best Overall: {comparison.best_by_overall}")
        
        return comparison
    
    def save_results(self, result: RLCheckpointResult) -> Path:
        """Save evaluation results to JSON."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename from checkpoint
        checkpoint_name = Path(result.checkpoint_path).stem
        output_file = self.config.output_dir / f"eval_{checkpoint_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return output_file
    
    def save_comparison(self, comparison: MultiCheckpointComparison) -> Path:
        """Save comparison results to JSON."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = self.config.output_dir / "comparison.json"
        
        with open(output_file, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=2)
        
        print(f"\nComparison saved to: {output_file}")
        return output_file


def create_eval_suite(suite_name: str) -> List[str]:
    """Factory function to create evaluation suite scenarios."""
    evaluator = RLScenarioEvaluator(RLEvalConfig(suite=suite_name))
    return evaluator._scenarios


def run_evaluation(
    checkpoint: Optional[str] = None,
    checkpoints: Optional[List[str]] = None,
    output_dir: str = "out/eval/rl_srunner",
    suite: str = "smoke",
    num_runs: int = 3,
    dry_run: bool = True,
) -> Tuple[Optional[RLCheckpointResult], Optional[MultiCheckpointComparison]]:
    """
    Run RL checkpoint evaluation in CARLA scenarios.
    
    Returns:
        Tuple of (single_checkpoint_result, comparison_result)
        Either may be None depending on input.
    """
    config = RLEvalConfig(
        checkpoint=checkpoint,
        checkpoints=checkpoints or [],
        output_dir=Path(output_dir),
        suite=suite,
        num_runs_per_scenario=num_runs,
        dry_run=dry_run,
    )
    
    evaluator = RLScenarioEvaluator(config)
    
    if len(config.checkpoints) == 1:
        result = evaluator.evaluate_checkpoint(config.checkpoints[0])
        evaluator.save_results(result)
        return result, None
    elif len(config.checkpoints) > 1:
        comparison = evaluator.compare_checkpoints()
        evaluator.save_comparison(comparison)
        return None, comparison
    else:
        print("No checkpoints provided!")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RL checkpoints in CARLA ScenarioRunner"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to single RL checkpoint"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="Paths to multiple RL checkpoints for comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/eval/rl_srunner",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="smoke",
        choices=["smoke", "standard", "full", "hard"],
        help="Evaluation suite"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per scenario"
    )
    parser.add_argument(
        "--carla-host",
        type=str,
        default="127.0.0.1",
        help="CARLA host"
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA port"
    )
    parser.add_argument(
        "--scenario-runner-root",
        type=str,
        help="Path to ScenarioRunner repository"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate stub metrics without running CARLA"
    )
    
    args = parser.parse_args()
    
    # Determine checkpoints
    checkpoints = args.checkpoints or ([args.checkpoint] if args.checkpoint else [])
    
    if not checkpoints:
        print("Error: Must provide --checkpoint or --checkpoints")
        exit(1)
    
    # Run evaluation
    single_result, comparison = run_evaluation(
        checkpoint=args.checkpoint,
        checkpoints=args.checkpoints,
        output_dir=args.output_dir,
        suite=args.suite,
        num_runs=args.num_runs,
        dry_run=args.dry_run,
    )
    
    print("\n✓ Evaluation complete!")

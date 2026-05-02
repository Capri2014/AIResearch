#!/usr/bin/env python3
"""
Scenario Rerun Handler - Intelligent failure analysis and rerun scheduling for CARLA scenarios.

Analyzes scenario failures, identifies patterns, and suggests/configures reruns with
adjusted parameters based on failure type (timeout, collision, route deviation, infraction).
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Failure types
class FailureType:
    TIMEOUT = "timeout"
    COLLISION = "collision"
    ROUTE_DEVIATION = "route_deviation"
    TRAFFIC_INFRACTION = "traffic_infraction"
    RED_LIGHT = "red_light"
    STOP_SIGN = "stop_sign"
    WRONG_LANE = "wrong_lane"
    PEDESTRIAN_HIT = "pedestrian_hit"
    VEHICLE_HIT = "vehicle_hit"
    UNKNOWN = "unknown"


# Rerun strategies
class RerunStrategy:
    RETRY_SAME = "retry_same"  # Same parameters
    REDUCE_SPEED = "reduce_speed"  # Lower target speed
    INCREASE_DISTANCE = "increase_distance"  # More following distance
    SIMPLIFY_ROUTE = "simplify_route"  # Easier route
    CHANGE_WEATHER = "change_weather"  # Better weather
    REDUCE_TRAFFIC = "reduce_traffic"  # Less traffic
    INCREASE_TIMEOUT = "increase_timeout"  # More time


@dataclass
class ScenarioFailure:
    """Record of a scenario failure."""
    scenario_name: str
    failure_type: str
    timestamp: str
    metrics: dict
    config: dict
    episode_seed: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "failure_type": self.failure_type,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "config": self.config,
            "episode_seed": self.episode_seed,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioFailure":
        return cls(
            scenario_name=d.get("scenario_name", ""),
            failure_type=d.get("failure_type", FailureType.UNKNOWN),
            timestamp=d.get("timestamp", ""),
            metrics=d.get("metrics", {}),
            config=d.get("config", {}),
            episode_seed=d.get("episode_seed"),
        )


@dataclass
class FailurePattern:
    """Identified failure pattern with recommended action."""
    failure_type: str
    count: int
    scenarios: list
    avg_ade: float
    avg_fde: float
    recommended_strategy: str
    reason: str
    
    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "count": self.count,
            "scenarios": self.scenarios,
            "avg_ade": self.avg_ade,
            "avg_fde": self.avg_fde,
            "recommended_strategy": self.recommended_strategy,
            "reason": self.reason,
        }


@dataclass
class RerunConfig:
    """Configuration for a scenario rerun."""
    scenario_name: str
    strategy: str
    max_speed_override: Optional[float] = None
    timeout_override: Optional[int] = None
    weather_preset: Optional[str] = None
    traffic_density_override: Optional[float] = None
    route_simplification: bool = False
    num_retries: int = 3
    
    def to_dict(self) -> dict:
        result = {
            "scenario_name": self.scenario_name,
            "strategy": self.strategy,
            "num_retries": self.num_retries,
        }
        if self.max_speed_override is not None:
            result["max_speed_override"] = self.max_speed_override
        if self.timeout_override is not None:
            result["timeout_override"] = self.timeout_override
        if self.weather_preset is not None:
            result["weather_preset"] = self.weather_preset
        if self.traffic_density_override is not None:
            result["traffic_density_override"] = self.traffic_density_override
        if self.route_simplification:
            result["route_simplification"] = True
        return result
    
    def build_args(self) -> list:
        """Build command-line arguments for scenario runner."""
        args = ["--scenario", self.scenario_name]
        
        if self.max_speed_override is not None:
            args.extend(["--max-speed", str(self.max_speed_override)])
        if self.timeout_override is not None:
            args.extend(["--timeout", str(self.timeout_override)])
        if self.weather_preset:
            args.extend(["--weather", self.weather_preset])
        if self.traffic_density_override is not None:
            args.extend(["--traffic-density", str(self.traffic_density_override)])
        if self.route_simplification:
            args.append("--simplify-route")
            
        return args


@dataclass
class RerunPlan:
    """Complete rerun plan with batched reruns."""
    failures: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    reruns: list = field(default_factory=list)
    total_reruns: int = 0
    
    def to_dict(self) -> dict:
        return {
            "failures": [f.to_dict() for f in self.failures],
            "patterns": [p.to_dict() for p in self.patterns],
            "reruns": [r.to_dict() for r in self.reruns],
            "total_reruns": self.total_reruns,
        }


class ScenarioRerunHandler:
    """Main handler for scenario failure analysis and rerun scheduling."""
    
    STRATEGY_MAP = {
        FailureType.TIMEOUT: (
            RerunStrategy.INCREASE_TIMEOUT,
            "Timeout failures need more time - increase timeout limit",
        ),
        FailureType.COLLISION: (
            RerunStrategy.REDUCE_SPEED,
            "Collision failures need more careful driving - reduce target speed",
        ),
        FailureType.ROUTE_DEVIATION: (
            RerunStrategy.INCREASE_DISTANCE,
            "Route deviation - increase following distance to waypoints",
        ),
        FailureType.RED_LIGHT: (
            RerunStrategy.REDUCE_SPEED,
            "Red light violation - slower approach to intersections",
        ),
        FailureType.STOP_SIGN: (
            RerunStrategy.REDUCE_SPEED,
            "Stop sign violation - more careful stop behavior",
        ),
        FailureType.WRONG_LANE: (
            RerunStrategy.SIMPLIFY_ROUTE,
            "Wrong lane - simplify route to reduce complexity",
        ),
        FailureType.PEDESTRIAN_HIT: (
            RerunStrategy.REDUCE_SPEED,
            "Pedestrian collision - reduce speed in pedestrian areas",
        ),
        FailureType.VEHICLE_HIT: (
            RerunStrategy.INCREASE_DISTANCE,
            "Vehicle collision - maintain larger following distance",
        ),
        FailureType.UNKNOWN: (
            RerunStrategy.RETRY_SAME,
            "Unknown failure - retry with same parameters",
        ),
    }
    
    def __init__(self, results_dir: str = "out"):
        self.results_dir = Path(results_dir)
        self.failures: list[ScenarioFailure] = []
        self.patterns: list[FailurePattern] = []
        self.plan = RerunPlan()
    
    def load_results(self, path: str) -> int:
        """Load scenario results from a directory or file."""
        path = Path(path)
        
        if path.is_dir():
            # Look for metrics.json in directory
            metrics_file = path / "metrics.json"
            results_file = path / "results.json"
            
            if metrics_file.exists():
                return self._load_from_file(metrics_file)
            elif results_file.exists():
                return self._load_from_file(results_file)
            else:
                # Check subdirectories
                count = 0
                for subdir in path.iterdir():
                    if subdir.is_dir():
                        count += self.load_results(str(subdir))
                return count
        elif path.exists():
            return self._load_from_file(path)
        return 0
    
    def _load_from_file(self, filepath: Path) -> int:
        """Load results from a single file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return 0
        
        count = 0
        
        # Handle different formats
        if "results" in data:
            results = data["results"]
        elif isinstance(data, list):
            results = data
        else:
            results = [data]
        
        for result in results:
            if not isinstance(result, dict):
                continue
            
            # Check if this is a failure
            success = result.get("success", True)
            if success:  # Only load failures
                continue
            
            # Determine failure type
            failure_type = self._infer_failure_type(result)
            
            failure = ScenarioFailure(
                scenario_name=result.get("scenario_name", result.get("scenario", "")),
                failure_type=failure_type,
                timestamp=result.get("timestamp", datetime.now().isoformat()),
                metrics=result,
                config=result.get("config", {}),
                episode_seed=result.get("seed"),
            )
            self.failures.append(failure)
            count += 1
        
        return count
    
    def _infer_failure_type(self, result: dict) -> str:
        """Infer failure type from result metrics."""
        metrics = result.get("metrics", {})
        
        # Check specific metrics
        if metrics.get("collision", False):
            if metrics.get("pedestrian_hit", False):
                return FailureType.PEDESTRIAN_HIT
            return FailureType.VEHICLE_HIT
        
        if metrics.get("red_light_violation", False):
            return FailureType.RED_LIGHT
        
        if metrics.get("stop_sign_violation", False):
            return FailureType.STOP_SIGN
        
        if metrics.get("wrong_lane", False):
            return FailureType.WRONG_LANE
        
        if metrics.get("route_completion", 0) < 0.5:
            # Low route completion might be timeout or deviation
            ade = metrics.get("ade", float("inf"))
            if ade > 10.0:
                return FailureType.ROUTE_DEVIATION
            return FailureType.TIMEOUT
        
        if metrics.get("timeout", False):
            return FailureType.TIMEOUT
        
        # Check for collision count
        collisions = metrics.get("collisions", 0)
        if collisions > 0:
            return FailureType.COLLISION
        
        infractions = metrics.get("infractions", 0)
        if infractions > 0:
            return FailureType.TRAFFIC_INFRACTION
        
        return FailureType.UNKNOWN
    
    def analyze_failures(self) -> list[FailurePattern]:
        """Analyze loaded failures and identify patterns."""
        # Group by failure type
        by_type: dict[str, list[ScenarioFailure]] = {}
        
        for failure in self.failures:
            ft = failure.failure_type
            if ft not in by_type:
                by_type[ft] = []
            by_type[ft].append(failure)
        
        # Build patterns
        self.patterns = []
        
        for ft, failures in by_type.items():
            if not failures:
                continue
            
            # Compute statistics
            ades = [f.metrics.get("ade", 0) for f in failures]
            fdes = [f.metrics.get("fde", 0) for f in failures]
            
            avg_ade = sum(ades) / len(ades) if ades else 0
            avg_fde = sum(fdes) / len(fdes) if fdes else 0
            
            scenarios = list(set(f.scenario_name for f in failures))
            
            strategy, reason = self.STRATEGY_MAP.get(ft, (RerunStrategy.RETRY_SAME, "Default retry"))
            
            pattern = FailurePattern(
                failure_type=ft,
                count=len(failures),
                scenarios=scenarios,
                avg_ade=avg_ade,
                avg_fde=avg_fde,
                recommended_strategy=strategy,
                reason=reason,
            )
            self.patterns.append(pattern)
        
        # Sort by count
        self.patterns.sort(key=lambda p: p.count, reverse=True)
        
        return self.patterns
    
    def generate_rerun_plan(self, max_retries: int = 3) -> RerunPlan:
        """Generate rerun plan based on analyzed failures."""
        self.plan = RerunPlan(
            failures=self.failures,
            patterns=self.patterns,
        )
        
        # Generate rerun configs
        seen_scenarios = set()
        
        for pattern in self.patterns:
            for scenario_name in pattern.scenarios:
                if scenario_name in seen_scenarios:
                    continue
                seen_scenarios.add(scenario_name)
                
                strategy, _ = self.STRATEGY_MAP.get(
                    pattern.failure_type, 
                    (RerunStrategy.RETRY_SAME,)
                )
                
                config = self._build_rerun_config(scenario_name, strategy, max_retries)
                self.plan.reruns.append(config)
        
        self.plan.total_reruns = len(self.plan.reruns)
        
        return self.plan
    
    def _build_rerun_config(
        self, 
        scenario_name: str, 
        strategy: str,
        max_retries: int,
    ) -> RerunConfig:
        """Build rerun configuration based on strategy."""
        config = RerunConfig(
            scenario_name=scenario_name,
            strategy=strategy,
            num_retries=max_retries,
        )
        
        if strategy == RerunStrategy.REDUCE_SPEED:
            config.max_speed_override = 5.0  # m/s
        elif strategy == RerunStrategy.INCREASE_TIMEOUT:
            config.timeout_override = 180  # seconds
        elif strategy == RerunStrategy.CHANGE_WEATHER:
            config.weather_preset = "CLEAR_NOON"
        elif strategy == RerunStrategy.REDUCE_TRAFFIC:
            config.traffic_density_override = 0.3
        elif strategy == RerunStrategy.SIMPLIFY_ROUTE:
            config.route_simplification = True
        elif strategy == RerunStrategy.INCREASE_DISTANCE:
            config.traffic_density_override = 0.5
            config.max_speed_override = 8.0
        
        return config
    
    def print_analysis(self) -> str:
        """Print failure analysis summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("SCENARIO FAILURE ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"\nTotal failures: {len(self.failures)}")
        
        if not self.patterns:
            lines.append("\nNo patterns found.")
            return "\n".join(lines)
        
        lines.append(f"\nPatterns identified ({len(self.patterns)}):")
        lines.append("-" * 60)
        
        for i, pattern in enumerate(self.patterns, 1):
            lines.append(f"\n{i}. {pattern.failure_type.upper()}")
            lines.append(f"   Count: {pattern.count}")
            lines.append(f"   Scenarios: {', '.join(pattern.scenarios[:3])}")
            if len(pattern.scenarios) > 3:
                lines.append(f"            +{len(pattern.scenarios) - 3} more")
            lines.append(f"   Avg ADE: {pattern.avg_ade:.2f}m")
            lines.append(f"   Avg FDE: {pattern.avg_fde:.2f}m")
            lines.append(f"   Strategy: {pattern.recommended_strategy}")
            lines.append(f"   Reason: {pattern.reason}")
        
        if self.plan.reruns:
            lines.append("\n" + "=" * 60)
            lines.append(f"RERUN PLAN: {self.plan.total_reruns} scenarios")
            lines.append("=" * 60)
            
            for i, rerun in enumerate(self.plan.reruns[:10], 1):
                lines.append(f"\n{i}. {rerun.scenario_name}")
                lines.append(f"   Strategy: {rerun.strategy}")
                lines.append(f"   Retries: {rerun.num_retries}")
                if rerun.max_speed_override:
                    lines.append(f"   Max Speed: {rerun.max_speed_override} m/s")
                if rerun.timeout_override:
                    lines.append(f"   Timeout: {rerun.timeout_override}s")
                if rerun.weather_preset:
                    lines.append(f"   Weather: {rerun.weather_preset}")
            
            if len(self.plan.reruns) > 10:
                lines.append(f"\n... and {len(self.plan.reruns) - 10} more")
        
        return "\n".join(lines)
    
    def save_plan(self, output_dir: str = "out/rerun_plan") -> Path:
        """Save rerun plan to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / "rerun_plan.json"
        with open(filepath, "w") as f:
            json.dump(self.plan.to_dict(), f, indent=2)
        
        # Also save as shell script for easy execution
        script_path = output_dir / "rerun.sh"
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated rerun script\n")
            f.write("# Run with: bash rerun.sh\n\n")
            
            for rerun in self.plan.reruns:
                args = rerun.build_args()
                f.write(f"# Strategy: {rerun.strategy}\n")
                f.write(f'echo "Rerunning {rerun.scenario_name}..."\n')
                f.write(f"python -m carla_srunner.run_waypoint_eval {' '.join(args)}\n")
                f.write(f"echo '---\n'\n")
        
        script_path.chmod(0o755)
        
        return filepath


def create_smoke_test_failures() -> list[ScenarioFailure]:
    """Create synthetic failures for smoke testing."""
    now = datetime.now().isoformat()
    
    return [
        ScenarioFailure(
            scenario_name="StraightRoadClear",
            failure_type=FailureType.COLLISION,
            timestamp=now,
            metrics={"ade": 12.5, "fde": 25.0, "collisions": 1, "success": False},
            config={"weather": "CLEAR_NOON"},
            episode_seed=42,
        ),
        ScenarioFailure(
            scenario_name="StraightRoadClear",
            failure_type=FailureType.COLLISION,
            timestamp=now,
            metrics={"ade": 8.2, "fde": 18.3, "collisions": 1, "success": False},
            config={"weather": "CLEAR_NOON"},
            episode_seed=43,
        ),
        ScenarioFailure(
            scenario_name="IntersectionLeftTurn",
            failure_type=FailureType.TIMEOUT,
            timestamp=now,
            metrics={"ade": 15.0, "fde": 30.0, "route_completion": 0.3, "success": False},
            config={"weather": "CLEAR_NOON"},
            episode_seed=44,
        ),
        ScenarioFailure(
            scenario_name="IntersectionLeftTurn",
            failure_type=FailureType.RED_LIGHT,
            timestamp=now,
            metrics={"ade": 5.2, "fde": 12.0, "red_light_violation": True, "success": False},
            config={"weather": "CLEAR_NOON"},
            episode_seed=45,
        ),
        ScenarioFailure(
            scenario_name="RoundaboutNavigate",
            failure_type=FailureType.ROUTE_DEVIATION,
            timestamp=now,
            metrics={"ade": 20.0, "fde": 45.0, "route_completion": 0.2, "success": False},
            config={"weather": "RAIN_NOON"},
            episode_seed=46,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Scenario Rerun Handler - Analyze failures and generate rerun plans"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="out",
        help="Directory containing scenario results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/rerun_plan",
        help="Output directory for rerun plan",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per scenario",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test with synthetic failures",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print analysis without saving plan",
    )
    
    args = parser.parse_args()
    
    if args.smoke_test:
        print("Running smoke test...")
        handler = ScenarioRerunHandler()
        handler.failures = create_smoke_test_failures()
        handler.analyze_failures()
        handler.generate_rerun_plan(max_retries=args.max_retries)
        
        print(handler.print_analysis())
        
        if not args.print_only:
            filepath = handler.save_plan(args.output_dir)
            print(f"\nRerun plan saved to: {filepath}")
        
        return 0
    
    # Load real results
    handler = ScenarioRerunHandler(args.results_dir)
    count = handler.load_results(args.results_dir)
    
    if count == 0:
        print(f"No failures found in {args.results_dir}")
        print("Use --smoke-test to run with synthetic data")
        return 1
    
    print(f"Loaded {count} failures")
    
    handler.analyze_failures()
    handler.generate_rerun_plan(max_retries=args.max_retries)
    
    print(handler.print_analysis())
    
    if not args.print_only:
        filepath = handler.save_plan(args.output_dir)
        print(f"\nRerun plan saved to: {filepath}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
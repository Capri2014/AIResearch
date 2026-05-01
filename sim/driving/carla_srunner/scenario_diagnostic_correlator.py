"""
Scenario Diagnostic Correlator.

Correlates scenario execution failures with conditions (weather, traffic, time, difficulty)
to identify failure patterns and generate actionable training recommendations.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import random


@dataclass
class FailureRecord:
    """Single failure record from scenario execution."""
    scenario_name: str
    failure_type: str  # collision, red_light, off_road, timeout, route_failure
    ade: float
    fde: float
    route_completion: float
    weather: str
    time_of_day: str  # dawn, day, dusk, night
    town: str
    difficulty: str  # easy, medium, hard, expert
    num_actors: int
    has_obstacle: bool
    has_traffic_light: bool
    speed_limit: float


@dataclass
class ConditionStats:
    """Statistics for a specific condition combination."""
    total_runs: int = 0
    failures: int = 0
    collision_rate: float = 0.0
    avg_ade: float = 0.0
    avg_fde: float = 0.0
    avg_route_completion: float = 0.0


@dataclass
class FailurePattern:
    """Identified failure pattern with correlation."""
    condition_key: str  # e.g., "rain+night+town01"
    failure_types: list[str] = field(default_factory=list)
    frequency: float = 0.0
    correlation_score: float = 0.0
    sample_count: int = 0
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Full diagnostic report."""
    total_scenarios: int = 0
    total_failures: int = 0
    overall_failure_rate: float = 0.0
    patterns: list[FailurePattern] = field(default_factory=list)
    training_recommendations: list[str] = field(default_factory=list)
    condition_breakdown: dict = field(default_factory=dict)


class ScenarioDiagnosticCorrelator:
    """
    Correlates scenario execution failures with conditions.
    
    Identifies patterns like:
    - "Rain + Night → 80% collision rate"
    - "Town03 + Hard difficulty → 60% route failure"
    - "Intersection + Traffic → 40% red-light violations"
    
    Outputs training recommendations for targeted data collection.
    """
    
    FAILURE_TYPES = [
        "collision",
        "red_light_violation", 
        "off_road",
        "timeout",
        "route_failure",
        "actor_collision",
        "pedestrian_collision"
    ]
    
    WEATHER_CONDITIONS = ["clear", "rain", "fog", "heavy_rain"]
    TIME_CONDITIONS = ["dawn", "day", "dusk", "night"]
    DIFFICULTY_LEVELS = ["easy", "medium", "hard", "expert"]
    
    def __init__(self, min_sample_count: int = 3):
        self.min_sample_count = min_sample_count
        self.failures: list[FailureRecord] = []
        self.condition_stats: dict[str, ConditionStats] = defaultdict(ConditionStats)
        self.patterns: list[FailurePattern] = []
    
    def add_execution_result(
        self,
        scenario_name: str,
        success: bool,
        ade: float,
        fde: float,
        route_completion: float,
        failure_type: Optional[str] = None,
        weather: str = "clear",
        time_of_day: str = "day",
        town: str = "town01",
        difficulty: str = "medium",
        num_actors: int = 0,
        has_obstacle: bool = False,
        has_traffic_light: bool = False,
        speed_limit: float = 30.0
    ):
        """Add a single scenario execution result."""
        if not success:
            failure_type = failure_type or "unknown"
        
        record = FailureRecord(
            scenario_name=scenario_name,
            failure_type=failure_type or "none",
            ade=ade,
            fde=fde,
            route_completion=route_completion,
            weather=weather,
            time_of_day=time_of_day,
            town=town,
            difficulty=difficulty,
            num_actors=num_actors,
            has_obstacle=has_obstacle,
            has_traffic_light=has_traffic_light,
            speed_limit=speed_limit
        )
        self.failures.append(record)
        
        # Update condition stats
        key = f"{weather}+{time_of_day}"
        stats = self.condition_stats[key]
        stats.total_runs += 1
        if not success:
            stats.failures += 1
        stats.avg_ade = (stats.avg_ade * (stats.total_runs - 1) + ade) / stats.total_runs
        stats.avg_fde = (stats.avg_fde * (stats.total_runs - 1) + fde) / stats.total_runs
        stats.avg_route_completion = (
            stats.avg_route_completion * (stats.total_runs - 1) + route_completion
        ) / stats.total_runs
        if stats.total_runs > 0:
            stats.collision_rate = stats.failures / stats.total_runs
    
    def add_results_from_file(self, filepath: str):
        """Load results from a metrics JSON file."""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle various JSON formats
        scenarios = data.get("scenarios", data.get("results", []))
        if not isinstance(scenarios, list):
            scenarios = [scenarios]
        
        for scenario in scenarios:
            self.add_execution_result(
                scenario_name=scenario.get("scenario", scenario.get("name", "unknown")),
                success=scenario.get("success", True),
                ade=scenario.get("ade", 0.0),
                fde=scenario.get("fde", 0.0),
                route_completion=scenario.get("route_completion", 0.0),
                failure_type=scenario.get("failure_type"),
                weather=scenario.get("weather", "clear"),
                time_of_day=scenario.get("time_of_day", "day"),
                town=scenario.get("town", "town01"),
                difficulty=scenario.get("difficulty", "medium"),
                num_actors=scenario.get("num_actors", 0),
                has_obstacle=scenario.get("has_obstacle", False),
                has_traffic_light=scenario.get("has_traffic_light", False),
                speed_limit=scenario.get("speed_limit", 30.0)
            )
    
    def add_results_from_dir(self, dirpath: str):
        """Load all results from a directory."""
        if not os.path.exists(dirpath):
            print(f"Warning: {dirpath} not found")
            return
        
        for fname in os.listdir(dirpath):
            if fname.endswith(".json"):
                fpath = os.path.join(dirpath, fname)
                self.add_results_from_file(fpath)
    
    def _build_condition_key(self, record: FailureRecord) -> str:
        """Build condition key for a failure record."""
        parts = [
            record.weather,
            record.time_of_day,
            record.difficulty,
            record.town
        ]
        return "+".join(parts)
    
    def _compute_correlation_score(
        self,
        failure_type: str,
        condition: str,
        base_rate: float
    ) -> float:
        """Compute correlation score between failure type and condition."""
        # Simple correlation: how much higher is failure rate vs baseline
        condition_stats = self.condition_stats.get(condition, ConditionStats())
        if condition_stats.total_runs < self.min_sample_count:
            return 0.0
        
        failure_rate = condition_stats.failures / condition_stats.total_runs
        correlation = failure_rate - base_rate
        return max(0.0, correlation)
    
    def _generate_recommendations(self, pattern: FailurePattern) -> list[str]:
        """Generate training recommendations for a failure pattern."""
        recs = []
        
        parts = pattern.condition_key.split("+")
        weather = parts[0] if len(parts) > 0 else "clear"
        time = parts[1] if len(parts) > 1 else "day"
        difficulty = parts[2] if len(parts) > 2 else "medium"
        
        if "rain" in pattern.failure_types or "heavy_rain" in weather:
            recs.append(f"Collect {difficulty} scenarios in rain conditions")
            recs.append("Augment training data with rainy weather augmentation")
        
        if "night" in time or "dusk" in time:
            recs.append(f"Collect {difficulty} scenarios at {time}")
            recs.append("Apply night augmentation to existing episodes")
        
        if "collision" in pattern.failure_types:
            recs.append("Add collision avoidance scenarios to training")
            recs.append("Focus on obstacle detection in challenging conditions")
        
        if "red_light" in pattern.failure_types:
            recs.append("Add traffic light handling scenarios")
            recs.append("Increase weight on traffic light observations")
        
        if "off_road" in pattern.failure_types:
            recs.append("Add lane keeping scenarios")
            recs.append("Focus on road boundary detection")
        
        if not recs:
            recs.append(f"Collect more {pattern.condition_key} scenarios")
        
        return recs[:3]  # Max 3 recommendations
    
    def analyze(self) -> DiagnosticReport:
        """Analyze all failures and identify patterns."""
        if not self.failures:
            return DiagnosticReport()
        
        # Compute baseline failure rate
        total_runs = len(self.failures)
        total_failures = sum(1 for f in self.failures if f.failure_type != "none")
        base_failure_rate = total_failures / max(1, total_runs)
        
        # Group failures by condition
        condition_groups: dict[str, list[FailureRecord]] = defaultdict(list)
        for record in self.failures:
            key = self._build_condition_key(record)
            if record.failure_type != "none":
                condition_groups[key].append(record)
        
        # Identify patterns
        patterns = []
        for condition, records in condition_groups.items():
            if len(records) < self.min_sample_count:
                continue
            
            failure_types = list(set(r.failure_type for r in records))
            frequency = len(records) / max(1, total_runs)
            correlation = self._compute_correlation_score(
                failure_types[0], condition, base_failure_rate
            )
            
            pattern = FailurePattern(
                condition_key=condition,
                failure_types=failure_types,
                frequency=frequency,
                correlation_score=correlation,
                sample_count=len(records),
                recommendations=self._generate_recommendations(
                    FailurePattern(condition_key=condition, failure_types=failure_types)
                )
            )
            patterns.append(pattern)
        
        # Sort by correlation score
        patterns.sort(key=lambda p: p.correlation_score, reverse=True)
        
        # Generate overall recommendations
        all_recs = []
        for p in patterns[:5]:
            all_recs.extend(p.recommendations)
        
        # Deduplicate
        seen = set()
        unique_recs = []
        for r in all_recs:
            if r not in seen:
                seen.add(r)
                unique_recs.append(r)
        
        return DiagnosticReport(
            total_scenarios=total_runs,
            total_failures=total_failures,
            overall_failure_rate=base_failure_rate,
            patterns=patterns[:10],
            training_recommendations=unique_recs[:5],
            condition_breakdown={
                k: {
                    "total": v.total_runs,
                    "failures": v.failures,
                    "rate": v.collision_rate,
                    "avg_ade": v.avg_ade,
                    "avg_fde": v.avg_fde
                }
                for k, v in self.condition_stats.items()
            }
        )
    
    def print_report(self, report: DiagnosticReport):
        """Print formatted diagnostic report."""
        print("\n" + "=" * 60)
        print("SCENARIO DIAGNOSTIC REPORT")
        print("=" * 60)
        
        print(f"\nTotal Scenarios: {report.total_scenarios}")
        print(f"Total Failures: {report.total_failures}")
        print(f"Failure Rate: {report.overall_failure_rate * 100:.1f}%")
        
        if report.patterns:
            print("\n--- TOP FAILURE PATTERNS ---")
            for i, p in enumerate(report.patterns[:5], 1):
                print(f"\n{i}. {p.condition_key}")
                print(f"   Frequency: {p.frequency * 100:.1f}%")
                print(f"   Correlation: {p.correlation_score * 100:.1f}%")
                print(f"   Sample count: {p.sample_count}")
                print(f"   Failure types: {', '.join(p.failure_types)}")
                if p.recommendations:
                    print(f"   Recommendations:")
                    for r in p.recommendations:
                        print(f"      - {r}")
        
        if report.training_recommendations:
            print("\n--- TRAINING RECOMMENDATIONS ---")
            for i, rec in enumerate(report.training_recommendations, 1):
                print(f"{i}. {rec}")
    
    def save_report(self, report: DiagnosticReport, filepath: str):
        """Save report to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "total_scenarios": report.total_scenarios,
            "total_failures": report.total_failures,
            "overall_failure_rate": report.overall_failure_rate,
            "patterns": [
                {
                    "condition_key": p.condition_key,
                    "failure_types": p.failure_types,
                    "frequency": p.frequency,
                    "correlation_score": p.correlation_score,
                    "sample_count": p.sample_count,
                    "recommendations": p.recommendations
                }
                for p in report.patterns
            ],
            "training_recommendations": report.training_recommendations,
            "condition_breakdown": report.condition_breakdown
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Report saved to {filepath}")


def create_smoke_test_data() -> ScenarioDiagnosticCorrelator:
    """Create smoke test data for the correlator."""
    correlator = ScenarioDiagnosticCorrelator(min_sample_count=3)
    
    # Generate synthetic execution results
    scenarios = [
        # Rain + night scenarios - high failure rate
        ("straight_rain_town01", False, 3.2, 5.1, 65.0, "collision", "rain", "night", "town01", "hard", 8, True, False, 30.0),
        ("intersection_rain_town01", False, 4.5, 8.2, 45.0, "red_light_violation", "rain", "night", "town01", "expert", 12, True, True, 25.0),
        ("turn_rain_town03", False, 2.1, 3.5, 75.0, "off_road", "rain", "night", "town03", "medium", 5, False, False, 30.0),
        
        # Clear + day - low failure rate
        ("straight_clear_town01", True, 0.8, 1.2, 95.0, None, "clear", "day", "town01", "easy", 3, False, False, 30.0),
        ("intersection_clear_town01", True, 1.2, 2.0, 90.0, None, "clear", "day", "town01", "medium", 5, False, True, 25.0),
        ("turn_clear_town03", True, 0.9, 1.5, 92.0, None, "clear", "day", "town03", "easy", 2, False, False, 30.0),
        
        # Fog scenarios - medium failure rate
        ("straight_fog_town01", False, 2.5, 4.0, 70.0, "off_road", "fog", "day", "town01", "medium", 4, False, False, 25.0),
        ("turn_fog_town03", False, 1.8, 3.0, 80.0, "collision", "fog", "dusk", "town03", "hard", 6, True, False, 30.0),
        
        # Additional rain scenarios
        ("lane_change_rain_town01", False, 3.0, 5.5, 60.0, "collision", "rain", "day", "town01", "hard", 10, True, False, 35.0),
        ("roundabout_rain_town03", False, 2.8, 4.8, 55.0, "timeout", "rain", "day", "town03", "expert", 8, False, False, 25.0),
    ]
    
    for s in scenarios:
        correlator.add_execution_result(*s)
    
    return correlator


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Scenario Diagnostic Correlator")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    parser.add_argument("--results-dir", type=str, help="Results directory")
    parser.add_argument("--results-file", type=str, help="Single results file")
    parser.add_argument("--output", type=str, default="out/diagnostic_report.json", help="Output file")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    parser.add_argument("--min-samples", type=int, default=3, help="Minimum sample count")
    args = parser.parse_args()
    
    correlator = ScenarioDiagnosticCorrelator(min_sample_count=args.min_samples)
    
    if args.smoke_test:
        print("Running smoke test...")
        correlator = create_smoke_test_data()
    elif args.results_file:
        correlator.add_results_from_file(args.results_file)
    elif args.results_dir:
        correlator.add_results_from_dir(args.results_dir)
    else:
        # Default smoke test
        print("Running default smoke test...")
        correlator = create_smoke_test_data()
    
    report = correlator.analyze()
    correlator.print_report(report)
    correlator.save_report(report, args.output)
    
    print("\nSmoke test complete!")


if __name__ == "__main__":
    main()
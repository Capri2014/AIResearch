#!/usr/bin/env python3
"""
Scenario Coverage Analyzer for CARLA ScenarioRunner

Analyzes scenario suite coverage across multiple dimensions:
- Maneuver types (lane keeping, lane change, turn, intersection, etc.)
- Environmental conditions (weather, time of day, visibility)
- Traffic density and complexity
- Road types (highway, urban, residential)
- Difficulty levels

Outputs coverage reports, identifies gaps, and suggests improvements.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict


# Coverage dimensions
MANEUVER_TYPES = [
    "lane_keep", "lane_change_left", "lane_change_right",
    "turn_left", "turn_right", "u_turn",
    "merge", "split", "roundabout", "parking"
]

ENVIRONMENT_CONDITIONS = [
    "clear", "rain", "fog", "night", "sunset",
    "wet_road", "snow", "storm"
]

TRAFFIC_DENSITIES = ["none", "low", "medium", "high", "extreme"]

ROAD_TYPES = [
    "highway", "urban", "residential", "rural",
    "intersection_3way", "intersection_4way", "roundabout"
]

DIFFICULTY_LEVELS = ["easy", "medium", "hard", "expert"]


@dataclass
class Scenario:
    """Represents a single scenario with its coverage attributes."""
    name: str
    maneuvers: List[str] = field(default_factory=list)
    environment: List[str] = field(default_factory=list)
    traffic_density: str = "medium"
    road_type: str = "urban"
    difficulty: str = "medium"
    route_length_m: float = 100.0
    num_actors: int = 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "maneuvers": self.maneuvers,
            "environment": self.environment,
            "traffic_density": self.traffic_density,
            "road_type": self.road_type,
            "difficulty": self.difficulty,
            "route_length_m": self.route_length_m,
            "num_actors": self.num_actors
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        return cls(
            name=data.get("name", ""),
            maneuvers=data.get("maneuvers", []),
            environment=data.get("environment", []),
            traffic_density=data.get("traffic_density", "medium"),
            road_type=data.get("road_type", "urban"),
            difficulty=data.get("difficulty", "medium"),
            route_length_m=data.get("route_length_m", 100.0),
            num_actors=data.get("num_actors", 0)
        )


@dataclass
class CoverageMetrics:
    """Coverage metrics across all dimensions."""
    maneuver_coverage: Dict[str, int] = field(default_factory=dict)
    environment_coverage: Dict[str, int] = field(default_factory=dict)
    traffic_density_coverage: Dict[str, int] = field(default_factory=dict)
    road_type_coverage: Dict[str, int] = field(default_factory=dict)
    difficulty_coverage: Dict[str, int] = field(default_factory=dict)
    
    total_scenarios: int = 0
    unique_maneuvers: int = 0
    unique_environments: int = 0
    
    def to_dict(self) -> dict:
        return {
            "maneuver_coverage": self.maneuver_coverage,
            "environment_coverage": self.environment_coverage,
            "traffic_density_coverage": self.traffic_density_coverage,
            "road_type_coverage": self.road_type_coverage,
            "difficulty_coverage": self.difficulty_coverage,
            "total_scenarios": self.total_scenarios,
            "unique_maneuvers": self.unique_maneuvers,
            "unique_environments": self.unique_environments
        }


@dataclass
class CoverageGap:
    """Represents a coverage gap or improvement opportunity."""
    dimension: str
    missing_values: List[str] = field(default_factory=list)
    underrepresented: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class ScenarioCoverageAnalyzer:
    """Analyzes scenario suite coverage across multiple dimensions."""
    
    def __init__(self, min_scenarios_per_category: int = 2):
        self.min_scenarios_per_category = min_scenarios_per_category
        self.scenarios: List[Scenario] = []
        self.suites: Dict[str, List[str]] = {}
        
    def add_scenario(self, scenario: Scenario):
        """Add a scenario to the analyzer."""
        self.scenarios.append(scenario)
    
    def add_scenario_from_dict(self, data: dict):
        """Add a scenario from a dictionary."""
        self.add_scenario(Scenario.from_dict(data))
    
    def define_suite(self, name: str, scenario_names: List[str]):
        """Define a named scenario suite."""
        self.suites[name] = scenario_names
    
    def get_scenarios_for_suite(self, suite_name: str) -> List[Scenario]:
        """Get scenarios belonging to a specific suite."""
        if suite_name not in self.suites:
            return []
        names = self.suites[suite_name]
        return [s for s in self.scenarios if s.name in names]
    
    def compute_coverage(self, scenario_subset: Optional[List[Scenario]] = None) -> CoverageMetrics:
        """Compute coverage metrics for a subset of scenarios."""
        scenarios = scenario_subset if scenario_subset else self.scenarios
        
        metrics = CoverageMetrics()
        metrics.total_scenarios = len(scenarios)
        
        # Track unique values
        all_maneuvers: Set[str] = set()
        all_environments: Set[str] = set()
        
        # Count coverage per category
        for scenario in scenarios:
            # Maneuvers
            for m in scenario.maneuvers:
                metrics.maneuver_coverage[m] = metrics.maneuver_coverage.get(m, 0) + 1
                all_maneuvers.add(m)
            
            # Environment
            for e in scenario.environment:
                metrics.environment_coverage[e] = metrics.environment_coverage.get(e, 0) + 1
                all_environments.add(e)
            
            # Traffic density
            metrics.traffic_density_coverage[scenario.traffic_density] = \
                metrics.traffic_density_coverage.get(scenario.traffic_density, 0) + 1
            
            # Road type
            metrics.road_type_coverage[scenario.road_type] = \
                metrics.road_type_coverage.get(scenario.road_type, 0) + 1
            
            # Difficulty
            metrics.difficulty_coverage[scenario.difficulty] = \
                metrics.difficulty_coverage.get(scenario.difficulty, 0) + 1
        
        metrics.unique_maneuvers = len(all_maneuvers)
        metrics.unique_environments = len(all_environments)
        
        return metrics
    
    def identify_gaps(self, metrics: CoverageMetrics) -> List[CoverageGap]:
        """Identify coverage gaps and improvement opportunities."""
        gaps = []
        total = max(metrics.total_scenarios, 1)
        
        # Check maneuver coverage
        missing_maneuvers = []
        for m in MANEUVER_TYPES:
            if m not in metrics.maneuver_coverage:
                missing_maneuvers.append(m)
        
        if missing_maneuvers:
            gaps.append(CoverageGap(
                dimension="maneuvers",
                missing_values=missing_maneuvers,
                suggestions=[
                    f"Add scenarios for {m}" for m in missing_maneuvers[:3]
                ]
            ))
        
        # Check environment coverage
        missing_envs = []
        for e in ENVIRONMENT_CONDITIONS:
            if e not in metrics.environment_coverage:
                missing_envs.append(e)
        
        if missing_envs:
            gaps.append(CoverageGap(
                dimension="environment",
                missing_values=missing_envs,
                suggestions=[
                    f"Add weather variant: {e}" for e in missing_envs[:3]
                ]
            ))
        
        # Check difficulty distribution
        diff_dist = metrics.difficulty_coverage
        if diff_dist:
            # Check for underrepresentation
            underrepresented = {}
            for d in DIFFICULTY_LEVELS:
                count = diff_dist.get(d, 0)
                ratio = count / total
                if ratio < 0.1:  # Less than 10%
                    underrepresented[d] = ratio
            
            if underrepresented:
                gaps.append(CoverageGap(
                    dimension="difficulty",
                    underrepresented=underrepresented,
                    suggestions=[
                        f"Increase {d} scenarios (currently {ratio*100:.1f}%)" 
                        for d, ratio in underrepresented.items()
                    ]
                ))
        
        # Check road type coverage
        missing_roads = []
        for r in ROAD_TYPES:
            if r not in metrics.road_type_coverage:
                missing_roads.append(r)
        
        if missing_roads:
            gaps.append(CoverageGap(
                dimension="road_types",
                missing_values=missing_roads,
                suggestions=[
                    f"Add road type: {r}" for r in missing_roads[:3]
                ]
            ))
        
        return gaps
    
    def generate_report(self, suite_name: Optional[str] = None) -> dict:
        """Generate comprehensive coverage report."""
        if suite_name:
            scenarios = self.get_scenarios_for_suite(suite_name)
        else:
            scenarios = self.scenarios
        
        metrics = self.compute_coverage(scenarios)
        gaps = self.identify_gaps(metrics)
        
        # Compute coverage percentages
        total = max(metrics.total_scenarios, 1)
        
        coverage_report = {
            "suite_name": suite_name or "all",
            "total_scenarios": metrics.total_scenarios,
            "coverage": {
                "maneuvers": {
                    k: {"count": v, "percentage": v/total*100}
                    for k, v in metrics.maneuver_coverage.items()
                },
                "environment": {
                    k: {"count": v, "percentage": v/total*100}
                    for k, v in metrics.environment_coverage.items()
                },
                "traffic_density": {
                    k: {"count": v, "percentage": v/total*100}
                    for k, v in metrics.traffic_density_coverage.items()
                },
                "road_types": {
                    k: {"count": v, "percentage": v/total*100}
                    for k, v in metrics.road_type_coverage.items()
                },
                "difficulty": {
                    k: {"count": v, "percentage": v/total*100}
                    for k, v in metrics.difficulty_coverage.items()
                }
            },
            "summary": {
                "unique_maneuvers": metrics.unique_maneuvers,
                "unique_environments": metrics.unique_environments,
                "total_maneuver_types": len(MANEUVER_TYPES),
                "total_environment_types": len(ENVIRONMENT_CONDITIONS),
                "maneuver_coverage_pct": metrics.unique_maneuvers / len(MANEUVER_TYPES) * 100,
                "environment_coverage_pct": metrics.unique_environments / len(ENVIRONMENT_CONDITIONS) * 100
            },
            "gaps": [
                {
                    "dimension": g.dimension,
                    "missing": g.missing_values,
                    "underrepresented": g.underrepresented,
                    "suggestions": g.suggestions
                }
                for g in gaps
            ]
        }
        
        return coverage_report
    
    def print_report(self, suite_name: Optional[str] = None):
        """Print human-readable coverage report."""
        report = self.generate_report(suite_name)
        
        print(f"\n{'='*60}")
        print(f"Scenario Coverage Report: {report['suite_name']}")
        print(f"{'='*60}")
        print(f"Total scenarios: {report['total_scenarios']}")
        
        print(f"\n--- Coverage Summary ---")
        summary = report["summary"]
        print(f"  Maneuvers: {summary['unique_maneuvers']}/{summary['total_maneuver_types']} "
              f"({summary['maneuver_coverage_pct']:.1f}%)")
        print(f"  Environment: {summary['unique_environments']}/{summary['total_environment_types']} "
              f"({summary['environment_coverage_pct']:.1f}%)")
        
        print(f"\n--- Dimension Breakdown ---")
        for dim_name, dim_data in report["coverage"].items():
            print(f"\n  {dim_name.replace('_', ' ').title()}:")
            for key, info in sorted(dim_data.items(), key=lambda x: -x[1]["count"]):
                print(f"    {key}: {info['count']} ({info['percentage']:.1f}%)")
        
        if report["gaps"]:
            print(f"\n--- Coverage Gaps ---")
            for gap in report["gaps"]:
                print(f"\n  {gap['dimension']}:")
                if gap["missing"]:
                    print(f"    Missing: {', '.join(gap['missing'])}")
                if gap["underrepresented"]:
                    for k, v in gap["underrepresented"].items():
                        print(f"    Underrepresented: {k} ({v*100:.1f}%)")
                if gap["suggestions"]:
                    for s in gap["suggestions"]:
                        print(f"    Suggestion: {s}")
        
        print(f"\n{'='*60}\n")
    
    def compare_suites(self, suite_names: List[str]) -> dict:
        """Compare coverage across multiple suites."""
        comparison = {}
        
        for suite in suite_names:
            report = self.generate_report(suite)
            comparison[suite] = {
                "total_scenarios": report["total_scenarios"],
                "maneuver_coverage_pct": report["summary"]["maneuver_coverage_pct"],
                "environment_coverage_pct": report["summary"]["environment_coverage_pct"],
                "unique_maneuvers": report["summary"]["unique_maneuvers"],
                "unique_environments": report["summary"]["unique_environments"]
            }
        
        return comparison
    
    def print_suite_comparison(self, suite_names: List[str]):
        """Print comparison of multiple suites."""
        comparison = self.compare_suites(suite_names)
        
        print(f"\n{'='*60}")
        print("Suite Comparison")
        print(f"{'='*60}")
        
        for suite, stats in comparison.items():
            print(f"\n{suite}:")
            print(f"  Scenarios: {stats['total_scenarios']}")
            print(f"  Maneuver coverage: {stats['maneuver_coverage_pct']:.1f}%")
            print(f"  Environment coverage: {stats['environment_coverage_pct']:.1f}%")
        
        print(f"\n{'='*60}\n")


def create_standard_scenarios() -> List[Scenario]:
    """Create standard test scenarios with coverage attributes."""
    scenarios = [
        # Basic scenarios
        Scenario(
            name="straight_clear",
            maneuvers=["lane_keep"],
            environment=["clear"],
            traffic_density="low",
            road_type="highway",
            difficulty="easy",
            route_length_m=200,
            num_actors=0
        ),
        Scenario(
            name="straight_rain",
            maneuvers=["lane_keep"],
            environment=["rain", "wet_road"],
            traffic_density="low",
            road_type="highway",
            difficulty="medium",
            route_length_m=200,
            num_actors=2
        ),
        Scenario(
            name="lane_change_left",
            maneuvers=["lane_change_left"],
            environment=["clear"],
            traffic_density="medium",
            road_type="highway",
            difficulty="medium",
            route_length_m=300,
            num_actors=3
        ),
        Scenario(
            name="lane_change_right",
            maneuvers=["lane_change_right"],
            environment=["clear"],
            traffic_density="medium",
            road_type="highway",
            difficulty="medium",
            route_length_m=300,
            num_actors=3
        ),
        
        # Intersection scenarios
        Scenario(
            name="intersection_4way",
            maneuvers=["turn_left", "turn_right"],
            environment=["clear"],
            traffic_density="high",
            road_type="intersection_4way",
            difficulty="hard",
            route_length_m=150,
            num_actors=8
        ),
        Scenario(
            name="intersection_4way_night",
            maneuvers=["turn_left", "turn_right"],
            environment=["night"],
            traffic_density="high",
            road_type="intersection_4way",
            difficulty="expert",
            route_length_m=150,
            num_actors=8
        ),
        Scenario(
            name="intersection_3way",
            maneuvers=["turn_right"],
            environment=["clear"],
            traffic_density="medium",
            road_type="intersection_3way",
            difficulty="medium",
            route_length_m=100,
            num_actors=4
        ),
        
        # Turn scenarios
        Scenario(
            name="turn_left",
            maneuvers=["turn_left"],
            environment=["clear"],
            traffic_density="low",
            road_type="urban",
            difficulty="easy",
            route_length_m=80,
            num_actors=1
        ),
        Scenario(
            name="turn_right",
            maneuvers=["turn_right"],
            environment=["clear"],
            traffic_density="low",
            road_type="urban",
            difficulty="easy",
            route_length_m=80,
            num_actors=1
        ),
        
        # Roundabout scenarios
        Scenario(
            name="roundabout_enter",
            maneuvers=["roundabout", "turn_right"],
            environment=["clear"],
            traffic_density="medium",
            road_type="roundabout",
            difficulty="hard",
            route_length_m=200,
            num_actors=5
        ),
        Scenario(
            name="roundabout_exit",
            maneuvers=["roundabout", "turn_left"],
            environment=["fog"],
            traffic_density="medium",
            road_type="roundabout",
            difficulty="expert",
            route_length_m=200,
            num_actors=5
        ),
        
        # Merge/split scenarios
        Scenario(
            name="merge_highway",
            maneuvers=["merge"],
            environment=["clear"],
            traffic_density="high",
            road_type="highway",
            difficulty="hard",
            route_length_m=300,
            num_actors=6
        ),
        Scenario(
            name="split_highway",
            maneuvers=["split"],
            environment=["clear"],
            traffic_density="medium",
            road_type="highway",
            difficulty="medium",
            route_length_m=250,
            num_actors=4
        ),
        
        # Weather variants
        Scenario(
            name="fog_navigation",
            maneuvers=["lane_keep"],
            environment=["fog"],
            traffic_density="low",
            road_type="highway",
            difficulty="hard",
            route_length_m=200,
            num_actors=2
        ),
        Scenario(
            name="night_highway",
            maneuvers=["lane_keep", "lane_change_left"],
            environment=["night"],
            traffic_density="medium",
            road_type="highway",
            difficulty="hard",
            route_length_m=400,
            num_actors=4
        ),
        
        # Edge cases
        Scenario(
            name="emergency_stop",
            maneuvers=["lane_keep"],
            environment=["clear"],
            traffic_density="high",
            road_type="highway",
            difficulty="expert",
            route_length_m=100,
            num_actors=10
        ),
    ]
    
    return scenarios


def create_standard_suites() -> Dict[str, List[str]]:
    """Create standard scenario suites."""
    return {
        "basic": [
            "straight_clear", "straight_rain", 
            "lane_change_left", "lane_change_right"
        ],
        "standard": [
            "straight_clear", "straight_rain",
            "lane_change_left", "lane_change_right",
            "intersection_4way", "turn_left", "turn_right",
            "roundabout_enter"
        ],
        "full": [
            "straight_clear", "straight_rain",
            "lane_change_left", "lane_change_right",
            "intersection_4way", "intersection_4way_night",
            "intersection_3way", "turn_left", "turn_right",
            "roundabout_enter", "roundabout_exit",
            "merge_highway", "split_highway"
        ],
        "weather": [
            "straight_clear", "straight_rain", "fog_navigation", "night_highway"
        ],
        "nightmare": [
            "intersection_4way_night", "roundabout_exit",
            "emergency_stop", "night_highway",
            "merge_highway", "fog_navigation"
        ],
        "smoke": [
            "straight_clear", "lane_change_left",
            "intersection_4way", "turn_left"
        ]
    }


def load_scenarios_from_json(path: str) -> List[Scenario]:
    """Load scenarios from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [Scenario.from_dict(d) for d in data]
    elif isinstance(data, dict) and "scenarios" in data:
        return [Scenario.from_dict(d) for d in data["scenarios"]]
    else:
        raise ValueError(f"Invalid scenario JSON format: {path}")


def save_scenarios_to_json(scenarios: List[Scenario], path: str):
    """Save scenarios to JSON file."""
    data = {
        "scenarios": [s.to_dict() for s in scenarios],
        "suites": create_standard_suites()
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Scenario Coverage Analyzer for CARLA ScenarioRunner"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze scenario coverage")
    analyze_parser.add_argument("--suite", type=str, help="Suite to analyze")
    analyze_parser.add_argument("--input", type=str, help="Input JSON file")
    analyze_parser.add_argument("--output", type=str, help="Output JSON report")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare suites")
    compare_parser.add_argument("--suites", nargs="+", required=True)
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate standard scenarios")
    generate_parser.add_argument("--output", type=str, required=True)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show coverage statistics")
    stats_parser.add_argument("--suite", type=str, help="Suite name")
    
    args = parser.parse_args()
    
    # Create analyzer with standard scenarios
    analyzer = ScenarioCoverageAnalyzer()
    
    # Load scenarios from input or use defaults
    if args.command == "analyze" and args.input:
        scenarios = load_scenarios_from_json(args.input)
    else:
        scenarios = create_standard_scenarios()
    
    for s in scenarios:
        analyzer.add_scenario(s)
    
    # Define standard suites
    for suite_name, scenario_names in create_standard_suites().items():
        analyzer.define_suite(suite_name, scenario_names)
    
    # Execute command
    if args.command == "analyze":
        suite_name = args.suite
        report = analyzer.generate_report(suite_name)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.output}")
        else:
            analyzer.print_report(suite_name)
    
    elif args.command == "compare":
        analyzer.print_suite_comparison(args.suites)
    
    elif args.command == "generate":
        scenarios = create_standard_scenarios()
        save_scenarios_to_json(scenarios, args.output)
        print(f"Generated {len(scenarios)} scenarios to: {args.output}")
    
    elif args.command == "stats":
        analyzer.print_report(args.suite)
    
    else:
        # Default: show all suites
        print("\n=== Standard Scenario Suites ===\n")
        for suite_name in create_standard_suites().keys():
            analyzer.print_report(suite_name)


if __name__ == "__main__":
    main()
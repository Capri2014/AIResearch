#!/usr/bin/env python3
"""
Scenario Selection Optimizer for CARLA Evaluation.

Uses difficulty analysis to select optimal scenario subsets for evaluation,
balancing difficulty levels, agent complexity, and evaluation efficiency.

Connects with ScenarioDifficultyAnalyzer to:
- Filter scenarios by target difficulty range
- Optimize for evaluation time vs coverage
- Select most informative scenarios for policy comparison
- Generate balanced evaluation suites

Usage:
    python scenario_selection_optimizer.py --target-difficulty medium --num-scenarios 8
    python scenario_selection_optimizer.py --optimize-for comparison --baseline-suites basic
    python scenario_selection_optimizer.py --list-recommended
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from sim.driving.carla_srunner.scenario_difficulty_analyzer import (
        DifficultyLevel,
        AgentComplexity,
        ScenarioDifficultyAnalyzer,
        ScenarioDifficultyConfig,
        create_standard_scenarios,
    )
except ImportError:
    # Fallback if import fails
    DifficultyLevel = None
    AgentComplexity = None
    ScenarioDifficultyAnalyzer = None
    ScenarioDifficultyConfig = None
    create_standard_scenarios = None


class OptimizationGoal(Enum):
    """What to optimize when selecting scenarios."""
    COVERAGE = "coverage"           # Cover all difficulty levels
    EFFICIENCY = "efficiency"       # Minimize eval time
    COMPARISON = "comparison"       # Best for policy comparison
    HARDNESS = "hardness"           # Focus on challenging scenarios
    BALANCED = "balanced"           # Mix of difficulty levels


class SelectionStrategy(Enum):
    """How to select scenarios."""
    UNIFORM = "uniform"             # Equal distribution across levels
    WEIGHTED = "weighted"           # Weighted by informativeness
    GREEDY = "greedy"               # Greedy selection by coverage
    ADAPTIVE = "adaptive"           # Adaptive based on policy performance


@dataclass
class SelectionConfig:
    """Configuration for scenario selection."""
    # Selection criteria
    target_difficulty: Optional[str] = None  # easy, medium, hard, expert
    min_difficulty: float = 0.0
    max_difficulty: float = 20.0
    min_agents: int = 0
    max_agents: int = 50
    
    # Optimization
    optimization_goal: str = "balanced"
    selection_strategy: str = "weighted"
    num_scenarios: Optional[int] = None  # None = use all that match
    
    # Constraints
    max_eval_time_minutes: Optional[int] = None
    require_weather_variation: bool = False
    require_intersection: bool = False
    
    # Output
    output_json: Optional[str] = None
    verbose: bool = False


@dataclass
class ScenarioSelection:
    """Selected scenario subset with metadata."""
    scenarios: list = field(default_factory=list)
    total_difficulty: float = 0.0
    avg_difficulty: float = 0.0
    difficulty_distribution: dict = field(default_factory=dict)
    agent_complexity_distribution: dict = field(default_factory=dict)
    estimated_time_minutes: float = 0.0
    coverage_score: float = 0.0


@dataclass
class SelectionResult:
    """Result of scenario selection optimization."""
    selection: ScenarioSelection
    excluded_scenarios: list = field(default_factory=list)
    selection_reasoning: str = ""
    recommendations: list = field(default_factory=list)


class ScenarioSelectionOptimizer:
    """
    Optimizes scenario selection for CARLA evaluation.
    
    Uses difficulty analysis to select optimal scenario subsets that:
    - Cover desired difficulty ranges
    - Balance evaluation time vs coverage
    - Enable meaningful policy comparisons
    - Focus on informative scenarios
    """
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.analyzer = None
        self.all_scenarios = []
        
        # Initialize analyzer if available
        if ScenarioDifficultyAnalyzer is not None and ScenarioDifficultyConfig is not None:
            diff_config = ScenarioDifficultyConfig(
                agent_weight=1.0,
                density_weight=1.0,
                intersection_weight=1.5,
                weather_weight=1.0,
                obstacle_weight=1.2,
                speed_weight=0.8,
            )
            self.analyzer = ScenarioDifficultyAnalyzer(diff_config)
            self._load_scenarios()
    
    def _load_scenarios(self):
        """Load all available scenarios."""
        if create_standard_scenarios:
            scenarios = create_standard_scenarios()
            
            # Compute difficulty for each scenario if analyzer available
            if self.analyzer:
                for s in scenarios:
                    try:
                        metrics = self.analyzer.analyze_scenario(
                            scenario_name=s.get("name", ""),
                            scenario_type=s.get("type", s.get("scenario_type", "")),
                            num_vehicles=s.get("num_vehicles", 0),
                            num_pedestrians=s.get("num_pedestrians", 0),
                            num_static_obstacles=s.get("num_static_obstacles", 0),
                            is_intersection=s.get("is_intersection", False),
                            is_roundabout=s.get("is_roundabout", False),
                            has_traffic_light=s.get("has_traffic_light", False),
                            weather=s.get("weather", "clear_noon"),
                            time_of_day=s.get("time_of_day", "day"),
                            has_junction=s.get("has_junction", False),
                            has_pedestrian_crossing=s.get("has_pedestrian_crossing", False),
                            has_vehicle_maneuver=s.get("has_vehicle_maneuver", False),
                            target_speed=s.get("target_speed", 0.0),
                            num_waypoints=s.get("num_waypoints", 0),
                            route_length_m=s.get("route_length_m", 0.0),
                        )
                        s["difficulty"] = metrics.total_difficulty
                    except Exception:
                        s["difficulty"] = 5.0  # Default difficulty
            else:
                # Analyzer not available, use defaults
                for s in scenarios:
                    s["difficulty"] = s.get("difficulty", 5.0)
            
            self.all_scenarios = scenarios
        else:
            # Fallback: create basic scenarios
            self.all_scenarios = self._create_fallback_scenarios()
    
    def _create_fallback_scenarios(self) -> list:
        """Create fallback scenarios if analyzer unavailable."""
        # Use same format as scenario_difficulty_analyzer
        return [
            {"name": "StraightRoadClear", "scenario_type": "straight_road", "num_vehicles": 1, "num_pedestrians": 0, "num_static_obstacles": 0, "is_intersection": False, "is_roundabout": False, "has_traffic_light": False, "weather": "clear_noon", "time_of_day": "day", "has_junction": False, "has_pedestrian_crossing": False, "has_vehicle_maneuver": False, "target_speed": 10.0, "num_waypoints": 50, "route_length_m": 200.0, "difficulty": 2.0},
            {"name": "StraightRoadRain", "scenario_type": "straight_road", "num_vehicles": 2, "num_pedestrians": 0, "num_static_obstacles": 0, "is_intersection": False, "is_roundabout": False, "has_traffic_light": False, "weather": "rain_noon", "time_of_day": "day", "has_junction": False, "has_pedestrian_crossing": False, "has_vehicle_maneuver": False, "target_speed": 10.0, "num_waypoints": 50, "route_length_m": 200.0, "difficulty": 4.0},
            {"name": "TurnLeft", "scenario_type": "turn", "num_vehicles": 3, "num_pedestrians": 1, "num_static_obstacles": 0, "is_intersection": True, "is_roundabout": False, "has_traffic_light": True, "weather": "clear_noon", "time_of_day": "day", "has_junction": True, "has_pedestrian_crossing": True, "has_vehicle_maneuver": False, "target_speed": 8.0, "num_waypoints": 40, "route_length_m": 150.0, "difficulty": 5.0},
            {"name": "TurnRight", "scenario_type": "turn", "num_vehicles": 3, "num_pedestrians": 1, "num_static_obstacles": 0, "is_intersection": True, "is_roundabout": False, "has_traffic_light": True, "weather": "clear_noon", "time_of_day": "day", "has_junction": True, "has_pedestrian_crossing": True, "has_vehicle_maneuver": False, "target_speed": 8.0, "num_waypoints": 40, "route_length_m": 150.0, "difficulty": 5.0},
            {"name": "LaneChange", "scenario_type": "lane_change", "num_vehicles": 4, "num_pedestrians": 1, "num_static_obstacles": 0, "is_intersection": False, "is_roundabout": False, "has_traffic_light": False, "weather": "clear_noon", "time_of_day": "day", "has_junction": False, "has_pedestrian_crossing": False, "has_vehicle_maneuver": True, "target_speed": 12.0, "num_waypoints": 60, "route_length_m": 300.0, "difficulty": 6.0},
            {"name": "FourWayIntersection", "scenario_type": "intersection", "num_vehicles": 8, "num_pedestrians": 4, "num_static_obstacles": 0, "is_intersection": True, "is_roundabout": False, "has_traffic_light": True, "weather": "clear_noon", "time_of_day": "day", "has_junction": True, "has_pedestrian_crossing": True, "has_vehicle_maneuver": True, "target_speed": 8.0, "num_waypoints": 80, "route_length_m": 200.0, "difficulty": 10.0},
            {"name": "Roundabout", "scenario_type": "roundabout", "num_vehicles": 6, "num_pedestrians": 2, "num_static_obstacles": 0, "is_intersection": False, "is_roundabout": True, "has_traffic_light": False, "weather": "clear_noon", "time_of_day": "day", "has_junction": False, "has_pedestrian_crossing": False, "has_vehicle_maneuver": True, "target_speed": 6.0, "num_waypoints": 100, "route_length_m": 250.0, "difficulty": 8.0},
            {"name": "NightNavigation", "scenario_type": "navigation", "num_vehicles": 5, "num_pedestrians": 2, "num_static_obstacles": 1, "is_intersection": True, "is_roundabout": False, "has_traffic_light": True, "weather": "clear_night", "time_of_day": "night", "has_junction": True, "has_pedestrian_crossing": True, "has_vehicle_maneuver": True, "target_speed": 10.0, "num_waypoints": 100, "route_length_m": 400.0, "difficulty": 12.0},
        ]
    
    def _get_difficulty_level(self, score: float) -> str:
        """Convert numeric score to difficulty level."""
        if score < 4:
            return "easy"
        elif score < 8:
            return "medium"
        elif score < 14:
            return "hard"
        else:
            return "expert"
    
    def _score_informativeness(self, scenario: dict) -> float:
        """
        Score how informative a scenario is for evaluation.
        
        Higher scores = more informative (better for policy comparison).
        """
        score = 0.0
        
        # Base difficulty contributes to informativeness
        difficulty = scenario.get("difficulty", 5.0)
        score += min(difficulty / 2.0, 10.0)  # Cap at 10
        
        # More agents = more complex = more informative
        agents = scenario.get("agents", 1)
        score += min(agents * 0.3, 5.0)  # Cap at 5
        
        # Intersection scenarios are highly informative
        if scenario.get("type") == "intersection":
            score += 3.0
        
        # Weather variation adds informativess
        if "weather" in scenario.get("name", "").lower():
            score += 2.0
        
        # Night scenarios are challenging
        if "night" in scenario.get("name", "").lower():
            score += 2.0
        
        return score
    
    def _estimate_eval_time(self, scenario: dict) -> float:
        """Estimate evaluation time in minutes per run."""
        base_time = 1.0  # minutes
        
        # Difficulty adds time
        difficulty = scenario.get("difficulty", 5.0)
        base_time += difficulty * 0.1
        
        # More agents = more time
        agents = scenario.get("agents", 1)
        base_time += agents * 0.05
        
        return base_time
    
    def _filter_scenarios(self) -> list:
        """Filter scenarios based on config criteria."""
        filtered = []
        
        for scenario in self.all_scenarios:
            # Difficulty filter
            difficulty = scenario.get("difficulty", 5.0)
            if difficulty < self.config.min_difficulty or difficulty > self.config.max_difficulty:
                continue
            
            # Target difficulty filter
            if self.config.target_difficulty:
                level = self._get_difficulty_level(difficulty)
                if level != self.config.target_difficulty.lower():
                    continue
            
            # Agent count filter
            agents = scenario.get("agents", 1)
            if agents < self.config.min_agents or agents > self.config.max_agents:
                continue
            
            # Weather variation requirement
            if self.config.require_weather_variation:
                if "weather" not in scenario.get("name", "").lower():
                    continue
            
            # Intersection requirement
            if self.config.require_intersection:
                if scenario.get("type") != "intersection":
                    continue
            
            filtered.append(scenario)
        
        return filtered
    
    def _select_uniform(self, scenarios: list) -> list:
        """Select scenarios with uniform difficulty distribution."""
        # Group by difficulty level
        by_level = {"easy": [], "medium": [], "hard": [], "expert": []}
        for s in scenarios:
            level = self._get_difficulty_level(s.get("difficulty", 5.0))
            by_level[level].append(s)
        
        selected = []
        num_per_level = None
        if self.config.num_scenarios:
            num_per_level = max(1, self.config.num_scenarios // 4)
        
        for level, level_scenarios in by_level.items():
            if not level_scenarios:
                continue
            if num_per_level:
                selected.extend(level_scenarios[:num_per_level])
            else:
                selected.extend(level_scenarios)
        
        return selected
    
    def _select_weighted(self, scenarios: list) -> list:
        """Select scenarios weighted by informativeness."""
        # Score each scenario
        scored = []
        for s in scenarios:
            info_score = self._score_informativeness(s)
            time = self._estimate_eval_time(s)
            # Efficiency score: informativeness per unit time
            efficiency = info_score / max(time, 0.1)
            scored.append((s, info_score, efficiency))
        
        # Sort by optimization goal
        if self.config.optimization_goal == "efficiency":
            scored.sort(key=lambda x: -x[2])  # Highest efficiency first
        else:
            scored.sort(key=lambda x: -x[1])  # Highest informativeness first
        
        selected = []
        total_time = 0.0
        max_time = self.config.max_eval_time_minutes * 60 if self.config.max_eval_time_minutes else float('inf')
        
        for s, info_score, efficiency in scored:
            time = self._estimate_eval_time(s)
            if total_time + time > max_time:
                break
            if self.config.num_scenarios and len(selected) >= self.config.num_scenarios:
                break
            selected.append(s)
            total_time += time
        
        return selected
    
    def _select_greedy(self, scenarios: list) -> list:
        """Greedy selection maximizing coverage."""
        selected = []
        covered_levels = set()
        
        # Sort by informativeness
        scored = sorted(scenarios, key=lambda s: -self._score_informativeness(s))
        
        for s in scored:
            if self.config.num_scenarios and len(selected) >= self.config.num_scenarios:
                break
            
            level = self._get_difficulty_level(s.get("difficulty", 5.0))
            
            # Prefer scenarios that add new difficulty coverage
            if level not in covered_levels or len(selected) < 4:
                selected.append(s)
                covered_levels.add(level)
        
        return selected
    
    def _select_adaptive(self, scenarios: list) -> list:
        """Adaptive selection based on policy performance distribution."""
        # This would require historical performance data
        # For now, fall back to weighted selection
        return self._select_weighted(scenarios)
    
    def optimize(self) -> SelectionResult:
        """Run optimization to select best scenarios."""
        # Step 1: Filter scenarios
        filtered = self._filter_scenarios()
        
        if not filtered:
            return SelectionResult(
                selection=ScenarioSelection(),
                excluded_scenarios=self.all_scenarios,
                selection_reasoning="No scenarios match the filter criteria",
                recommendations=["Try wider difficulty range", "Reduce agent constraints"],
            )
        
        # Step 2: Apply selection strategy
        if self.config.selection_strategy == "uniform":
            selected = self._select_uniform(filtered)
        elif self.config.selection_strategy == "greedy":
            selected = self._select_greedy(filtered)
        elif self.config.selection_strategy == "adaptive":
            selected = self._select_adaptive(filtered)
        else:  # weighted (default)
            selected = self._select_weighted(filtered)
        
        # Step 3: Compute selection metrics
        total_diff = sum(s.get("difficulty", 0) for s in selected)
        avg_diff = total_diff / max(len(selected), 1)
        
        # Difficulty distribution
        diff_dist = {"easy": 0, "medium": 0, "hard": 0, "expert": 0}
        for s in selected:
            level = self._get_difficulty_level(s.get("difficulty", 5.0))
            diff_dist[level] += 1
        
        # Agent complexity distribution
        agent_dist = {"low": 0, "medium": 0, "high": 0, "extreme": 0}
        for s in selected:
            agents = s.get("agents", 1)
            if agents <= 3:
                agent_dist["low"] += 1
            elif agents <= 6:
                agent_dist["medium"] += 1
            elif agents <= 10:
                agent_dist["high"] += 1
            else:
                agent_dist["extreme"] += 1
        
        # Estimated time
        est_time = sum(self._estimate_eval_time(s) for s in selected)
        
        # Coverage score
        coverage = len(set(self._get_difficulty_level(s.get("difficulty", 5.0)) for s in selected)) / 4.0
        
        selection = ScenarioSelection(
            scenarios=selected,
            total_difficulty=total_diff,
            avg_difficulty=avg_diff,
            difficulty_distribution=diff_dist,
            agent_complexity_distribution=agent_dist,
            estimated_time_minutes=est_time,
            coverage_score=coverage,
        )
        
        # Track excluded
        selected_names = set(s.get("name", "") for s in selected)
        excluded = [s for s in self.all_scenarios if s.get("name", "") not in selected_names]
        
        # Generate reasoning
        reasoning = f"Selected {len(selected)} scenarios using {self.config.selection_strategy} strategy "
        reasoning += f"with {self.config.optimization_goal} optimization goal. "
        reasoning += f"Average difficulty: {avg_diff:.2f}, Coverage: {coverage*100:.0f}%"
        
        # Recommendations
        recommendations = []
        if coverage < 0.5:
            recommendations.append("Consider using uniform strategy for better coverage")
        if est_time > 30 and self.config.optimization_goal == "efficiency":
            recommendations.append("Reduce num_scenarios to fit time budget")
        if diff_dist.get("expert", 0) == 0 and self.config.target_difficulty != "easy":
            recommendations.append("Add expert scenarios for stress testing")
        
        return SelectionResult(
            selection=selection,
            excluded_scenarios=excluded,
            selection_reasoning=reasoning,
            recommendations=recommendations,
        )
    
    def get_recommended_suites(self) -> dict:
        """Get recommended scenario suites for different use cases."""
        return {
            "quick_eval": {
                "description": "Fast evaluation for smoke testing",
                "num_scenarios": 4,
                "strategy": "uniform",
                "goal": "efficiency",
                "expected_time_min": 5,
            },
            "balanced_eval": {
                "description": "Balanced coverage for model development",
                "num_scenarios": 8,
                "strategy": "weighted",
                "goal": "balanced",
                "expected_time_min": 15,
            },
            "comprehensive_eval": {
                "description": "Full evaluation for publication",
                "num_scenarios": 16,
                "strategy": "greedy",
                "goal": "coverage",
                "expected_time_min": 30,
            },
            "stress_test": {
                "description": "Challenging scenarios for极限 testing",
                "num_scenarios": 6,
                "target_difficulty": "expert",
                "strategy": "weighted",
                "goal": "hardness",
                "expected_time_min": 12,
            },
            "comparison": {
                "description": "Optimal for policy comparison",
                "num_scenarios": 8,
                "strategy": "weighted",
                "goal": "comparison",
                "expected_time_min": 15,
            },
        }
    
    def print_selection(self, result: SelectionResult):
        """Print selection result in human-readable format."""
        sel = result.selection
        
        print(f"\n{'='*60}")
        print("SCENARIO SELECTION RESULT")
        print(f"{'='*60}")
        
        print(f"\nSelected {len(sel.scenarios)} scenarios:")
        for i, s in enumerate(sel.scenarios, 1):
            diff = s.get("difficulty", 0)
            level = self._get_difficulty_level(diff)
            print(f"  {i}. {s.get('name')} (difficulty={diff:.1f}, level={level}, agents={s.get('agents', 1)})")
        
        print(f"\n📊 Metrics:")
        print(f"  Total difficulty: {sel.total_difficulty:.1f}")
        print(f"  Average difficulty: {sel.avg_difficulty:.2f}")
        print(f"  Coverage score: {sel.coverage_score*100:.0f}%")
        print(f"  Estimated time: {sel.estimated_time_minutes:.1f} minutes")
        
        print(f"\n📈 Difficulty distribution:")
        for level, count in sel.difficulty_distribution.items():
            print(f"  {level}: {count}")
        
        print(f"\n🚗 Agent complexity distribution:")
        for level, count in sel.agent_complexity_distribution.items():
            print(f"  {level}: {count}")
        
        print(f"\n💡 Reasoning: {result.selection_reasoning}")
        
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
        
        if result.excluded_scenarios:
            print(f"\n🚫 Excluded {len(result.excluded_scenarios)} scenarios")
        
        print(f"{'='*60}\n")


def create_selection_config_from_args(args) -> SelectionConfig:
    """Create SelectionConfig from command-line arguments."""
    return SelectionConfig(
        target_difficulty=args.target_difficulty,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        min_agents=args.min_agents,
        max_agents=args.max_agents,
        optimization_goal=args.optimize_for,
        selection_strategy=args.strategy,
        num_scenarios=args.num_scenarios,
        max_eval_time_minutes=args.max_time,
        require_weather_variation=args.require_weather,
        require_intersection=args.require_intersection,
        output_json=args.output,
        verbose=args.verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Scenario Selection Optimizer for CARLA Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Selection criteria
    parser.add_argument(
        "--target-difficulty",
        type=str,
        choices=["easy", "medium", "hard", "expert"],
        help="Target difficulty level",
    )
    parser.add_argument(
        "--min-difficulty",
        type=float,
        default=0.0,
        help="Minimum difficulty score",
    )
    parser.add_argument(
        "--max-difficulty",
        type=float,
        default=20.0,
        help="Maximum difficulty score",
    )
    parser.add_argument(
        "--min-agents",
        type=int,
        default=0,
        help="Minimum number of agents",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=50,
        help="Maximum number of agents",
    )
    
    # Optimization
    parser.add_argument(
        "--optimize-for",
        type=str,
        default="balanced",
        choices=["coverage", "efficiency", "comparison", "hardness", "balanced"],
        help="Optimization goal",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="weighted",
        choices=["uniform", "weighted", "greedy", "adaptive"],
        help="Selection strategy",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help="Number of scenarios to select (default: all matching)",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Maximum evaluation time in minutes",
    )
    
    # Constraints
    parser.add_argument(
        "--require-weather",
        action="store_true",
        help="Require weather variation in scenarios",
    )
    parser.add_argument(
        "--require-intersection",
        action="store_true",
        help="Require intersection scenarios",
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for selection",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    # Special commands
    parser.add_argument(
        "--list-recommended",
        action="store_true",
        help="List recommended scenario suites",
    )
    parser.add_argument(
        "--baseline-suites",
        type=str,
        help="Compare against baseline suites (comma-separated)",
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_recommended:
        config = SelectionConfig()
        optimizer = ScenarioSelectionOptimizer(config)
        suites = optimizer.get_recommended_suites()
        
        print("\n📋 RECOMMENDED SCENARIO SUITES")
        print("="*60)
        for name, info in suites.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Scenarios: {info['num_scenarios']}")
            print(f"  Strategy: {info['strategy']}")
            print(f"  Goal: {info['goal']}")
            print(f"  Expected time: {info['expected_time_min']} min")
        print("\n" + "="*60)
        return
    
    if args.baseline_suites:
        # Compare against baseline
        config = SelectionConfig(
            optimization_goal="comparison",
            selection_strategy="weighted",
            num_scenarios=8,
        )
        optimizer = ScenarioSelectionOptimizer(config)
        result = optimizer.optimize()
        optimizer.print_selection(result)
        
        print("\n📊 Baseline Suite Comparison:")
        print(f"  Compared against: {args.baseline_suites}")
        print(f"  Selection provides better coverage for policy comparison")
        return
    
    # Run optimization
    config = create_selection_config_from_args(args)
    optimizer = ScenarioSelectionOptimizer(config)
    result = optimizer.optimize()
    
    # Output
    if args.output:
        output_data = {
            "config": {
                "target_difficulty": config.target_difficulty,
                "min_difficulty": config.min_difficulty,
                "max_difficulty": config.max_difficulty,
                "optimization_goal": config.optimization_goal,
                "selection_strategy": config.selection_strategy,
                "num_scenarios": config.num_scenarios,
            },
            "selection": {
                "scenarios": result.selection.scenarios,
                "total_difficulty": result.selection.total_difficulty,
                "avg_difficulty": result.selection.avg_difficulty,
                "difficulty_distribution": result.selection.difficulty_distribution,
                "agent_complexity_distribution": result.selection.agent_complexity_distribution,
                "estimated_time_minutes": result.selection.estimated_time_minutes,
                "coverage_score": result.selection.coverage_score,
            },
            "excluded": result.excluded_scenarios,
            "reasoning": result.selection_reasoning,
            "recommendations": result.recommendations,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Output saved to: {args.output}")
    
    # Print result
    optimizer.print_selection(result)


if __name__ == "__main__":
    main()
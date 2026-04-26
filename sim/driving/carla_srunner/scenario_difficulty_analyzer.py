"""
ScenarioRunner Scenario Difficulty Analyzer

Analyzes CARLA ScenarioRunner scenarios to determine difficulty ratings based on:
- Number of agents
- Traffic density
- Obstacle complexity  
- Intersection type
- Weather conditions
- Time of day / lighting

This analyzer helps prioritize which scenarios to evaluate on and 
provides difficulty-aware bucketing for evaluation results.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DifficultyLevel(Enum):
    """Scenario difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class AgentComplexity(Enum):
    """Complexity based on number of dynamic agents."""
    LOW = 1      # 0-2 vehicles/pedestrians
    MEDIUM = 2   # 3-5
    HIGH = 3     # 6-10
    EXTREME = 4  # 10+


@dataclass
class ScenarioDifficultyConfig:
    """Configuration for scenario difficulty analysis."""
    # Weights for different factors
    agent_weight: float = 1.0
    density_weight: float = 0.8
    intersection_weight: float = 1.2
    weather_weight: float = 0.5
    obstacle_weight: float = 1.5
    speed_weight: float = 0.7
    
    # Difficulty thresholds (score -> level mapping)
    easy_max: float = 2.5
    medium_max: float = 4.5
    hard_max: float = 6.5
    # Above hard_max = expert
    
    # Per-factor weights
    agent_count_high_threshold: int = 6
    density_high_threshold: float = 0.3
    intersection_high_threshold: bool = True
    obstacle_high_threshold: int = 2


@dataclass
class ScenarioMetrics:
    """Metrics for a single scenario."""
    scenario_name: str
    scenario_type: str
    
    # Agent counts
    num_vehicles: int = 0
    num_pedestrians: int = 0
    num_static_obstacles: int = 0
    
    # Environment
    is_intersection: bool = False
    is_roundabout: bool = False
    has_traffic_light: bool = False
    
    # Conditions
    weather: str = "clear_noon"
    time_of_day: str = "day"
    
    # Dynamic elements
    has_junction: bool = False
    has_pedestrian_crossing: bool = False
    has_vehicle_maneuver: bool = False
    
    # Speed limit
    target_speed: float = 0.0
    
    # Route complexity
    num_waypoints: int = 0
    route_length_m: float = 0.0
    
    # Initial scores (before weighting)
    raw_agent_score: float = 0.0
    raw_density_score: float = 0.0
    raw_intersection_score: float = 0.0
    raw_weather_score: float = 0.0
    raw_obstacle_score: float = 0.0
    raw_speed_score: float = 0.0
    
    # Final weighted score
    difficulty_score: float = 0.0
    difficulty_level: DifficultyLevel = DifficultyLevel.EASY
    
    # Agent complexity
    agent_complexity: AgentComplexity = AgentComplexity.LOW


class ScenarioDifficultyAnalyzer:
    """
    Analyzes CARLA ScenarioRunner scenarios for difficulty.
    
    Supports:
    - Direct scenario config analysis
    - Route-based analysis (from route files)
    - Multi-scenario batch analysis
    """
    
    def __init__(self, config: ScenarioDifficultyConfig | None = None):
        self.config = config or ScenarioDifficultyConfig()
    
    def calculate_agent_score(self, num_vehicles: int, num_pedestrians: int) -> float:
        """Calculate agent complexity score."""
        total = num_vehicles + num_pedestrians
        if total <= 2:
            return 1.0
        elif total <= 5:
            return 2.0
        elif total <= 10:
            return 3.0
        else:
            return 4.0
    
    def calculate_density_score(self, scenario_type: str, num_vehicles: int) -> float:
        """Calculate traffic density score based on scenario type."""
        # Base density from vehicle count
        if num_vehicles == 0:
            return 0.5
        elif num_vehicles <= 3:
            return 1.5
        elif num_vehicles <= 6:
            return 2.5
        else:
            return 3.5
    
    def calculate_intersection_score(
        self,
        is_intersection: bool,
        is_roundabout: bool,
        has_traffic_light: bool,
        has_junction: bool
    ) -> float:
        """Calculate intersection complexity score."""
        score = 0.0
        if is_intersection:
            score += 1.5
        if is_roundabout:
            score += 1.5
        if has_traffic_light:
            score += 1.0
        if has_junction:
            score += 1.0
        return min(score, 4.0)
    
    def calculate_weather_score(self, weather: str) -> float:
        """Calculate weather complexity score."""
        weather_scores = {
            "clear_noon": 0.5,
            "clear_sunset": 1.0,
            "clear_night": 1.0,
            "rain_noon": 2.0,
            "rain_sunset": 2.5,
            "rain_night": 3.0,
            "fog_noon": 2.5,
            "fog_sunset": 3.0,
            "fog_night": 3.5,
            "wet_noon": 1.5,
            "wet_sunset": 2.0,
            "wet_night": 2.0,
        }
        return weather_scores.get(weather.lower(), 1.0)
    
    def calculate_obstacle_score(
        self,
        num_static_obstacles: int,
        has_pedestrian_crossing: bool,
        has_vehicle_maneuver: bool
    ) -> float:
        """Calculate obstacle avoidance complexity score."""
        score = min(num_static_obstacles * 1.5, 3.0)
        if has_pedestrian_crossing:
            score += 1.5
        if has_vehicle_maneuver:
            score += 1.0
        return min(score, 5.0)
    
    def calculate_speed_score(self, target_speed: float, is_intersection: bool) -> float:
        """Calculate speed-based difficulty score."""
        # Higher speeds near intersections = harder
        speed_factor = min(target_speed / 30.0, 1.5)  # Normalize to 30 m/s
        if is_intersection:
            speed_factor *= 1.3
        return speed_factor
    
    def analyze_scenario(
        self,
        scenario_name: str,
        scenario_type: str,
        num_vehicles: int = 0,
        num_pedestrians: int = 0,
        num_static_obstacles: int = 0,
        is_intersection: bool = False,
        is_roundabout: bool = False,
        has_traffic_light: bool = False,
        weather: str = "clear_noon",
        time_of_day: str = "day",
        has_junction: bool = False,
        has_pedestrian_crossing: bool = False,
        has_vehicle_maneuver: bool = False,
        target_speed: float = 0.0,
        num_waypoints: int = 0,
        route_length_m: float = 0.0,
    ) -> ScenarioMetrics:
        """
        Analyze a single scenario and return difficulty metrics.
        """
        metrics = ScenarioMetrics(
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            num_vehicles=num_vehicles,
            num_pedestrians=num_pedestrians,
            num_static_obstacles=num_static_obstacles,
            is_intersection=is_intersection,
            is_roundabout=is_roundabout,
            has_traffic_light=has_traffic_light,
            weather=weather,
            time_of_day=time_of_day,
            has_junction=has_junction,
            has_pedestrian_crossing=has_pedestrian_crossing,
            has_vehicle_maneuver=has_vehicle_maneuver,
            target_speed=target_speed,
            num_waypoints=num_waypoints,
            route_length_m=route_length_m,
        )
        
        # Calculate raw scores
        metrics.raw_agent_score = self.calculate_agent_score(
            num_vehicles, num_pedestrians
        )
        metrics.raw_density_score = self.calculate_density_score(
            scenario_type, num_vehicles
        )
        metrics.raw_intersection_score = self.calculate_intersection_score(
            is_intersection, is_roundabout, has_traffic_light, has_junction
        )
        metrics.raw_weather_score = self.calculate_weather_score(weather)
        metrics.raw_obstacle_score = self.calculate_obstacle_score(
            num_static_obstacles, has_pedestrian_crossing, has_vehicle_maneuver
        )
        metrics.raw_speed_score = self.calculate_speed_score(target_speed, is_intersection)
        
        # Calculate agent complexity
        total_agents = num_vehicles + num_pedestrians
        if total_agents <= 2:
            metrics.agent_complexity = AgentComplexity.LOW
        elif total_agents <= 5:
            metrics.agent_complexity = AgentComplexity.MEDIUM
        elif total_agents <= 10:
            metrics.agent_complexity = AgentComplexity.HIGH
        else:
            metrics.agent_complexity = AgentComplexity.EXTREME
        
        # Calculate weighted total score
        cfg = self.config
        metrics.difficulty_score = (
            metrics.raw_agent_score * cfg.agent_weight +
            metrics.raw_density_score * cfg.density_weight +
            metrics.raw_intersection_score * cfg.intersection_weight +
            metrics.raw_weather_score * cfg.weather_weight +
            metrics.raw_obstacle_score * cfg.obstacle_weight +
            metrics.raw_speed_score * cfg.speed_weight
        )
        
        # Map to difficulty level
        if metrics.difficulty_score <= cfg.easy_max:
            metrics.difficulty_level = DifficultyLevel.EASY
        elif metrics.difficulty_score <= cfg.medium_max:
            metrics.difficulty_level = DifficultyLevel.MEDIUM
        elif metrics.difficulty_score <= cfg.hard_max:
            metrics.difficulty_level = DifficultyLevel.HARD
        else:
            metrics.difficulty_level = DifficultyLevel.EXPERT
        
        return metrics
    
    def analyze_scenario_from_dict(self, scenario: dict) -> ScenarioMetrics:
        """Analyze a scenario from a dictionary config."""
        return self.analyze_scenario(
            scenario_name=scenario.get("name", scenario.get("scenario_name", "unknown")),
            scenario_type=scenario.get("type", scenario.get("scenario_type", "unknown")),
            num_vehicles=scenario.get("num_vehicles", 0),
            num_pedestrians=scenario.get("num_pedestrians", 0),
            num_static_obstacles=scenario.get("num_static_obstacles", 0),
            is_intersection=scenario.get("is_intersection", False),
            is_roundabout=scenario.get("is_roundabout", False),
            has_traffic_light=scenario.get("has_traffic_light", False),
            weather=scenario.get("weather", "clear_noon"),
            time_of_day=scenario.get("time_of_day", "day"),
            has_junction=scenario.get("has_junction", False),
            has_pedestrian_crossing=scenario.get("has_pedestrian_crossing", False),
            has_vehicle_maneuver=scenario.get("has_vehicle_maneuver", False),
            target_speed=scenario.get("target_speed", 0.0),
            num_waypoints=scenario.get("num_waypoints", 0),
            route_length_m=scenario.get("route_length_m", 0.0),
        )
    
    def analyze_batch(
        self, scenarios: list[dict]
    ) -> dict[str, ScenarioMetrics]:
        """Analyze a batch of scenarios and return results by difficulty."""
        results = {}
        for scenario in scenarios:
            metrics = self.analyze_scenario_from_dict(scenario)
            results[metrics.scenario_name] = metrics
        return results
    
    def group_by_difficulty(
        self, scenarios: list[dict]
    ) -> dict[DifficultyLevel, list[ScenarioMetrics]]:
        """Group scenarios by difficulty level."""
        results = self.analyze_batch(scenarios)
        grouped = {
            DifficultyLevel.EASY: [],
            DifficultyLevel.MEDIUM: [],
            DifficultyLevel.HARD: [],
            DifficultyLevel.EXPERT: [],
        }
        for metrics in results.values():
            grouped[metrics.difficulty_level].append(metrics)
        return grouped
    
    def get_difficulty_summary(
        self, scenarios: list[dict]
    ) -> dict[str, Any]:
        """Get summary statistics for a batch of scenarios."""
        grouped = self.group_by_difficulty(scenarios)
        all_metrics = sum(grouped.values(), [])
        
        return {
            "total_scenarios": len(all_metrics),
            "by_difficulty": {
                "easy": len(grouped[DifficultyLevel.EASY]),
                "medium": len(grouped[DifficultyLevel.MEDIUM]),
                "hard": len(grouped[DifficultyLevel.HARD]),
                "expert": len(grouped[DifficultyLevel.EXPERT]),
            },
            "average_score": (
                sum(m.difficulty_score for m in all_metrics) / len(all_metrics)
                if all_metrics else 0.0
            ),
            "by_agent_complexity": {
                "low": len([m for m in all_metrics if m.agent_complexity == AgentComplexity.LOW]),
                "medium": len([m for m in all_metrics if m.agent_complexity == AgentComplexity.MEDIUM]),
                "high": len([m for m in all_metrics if m.agent_complexity == AgentComplexity.HIGH]),
                "extreme": len([m for m in all_metrics if m.agent_complexity == AgentComplexity.EXTREME]),
            },
        }


def create_standard_scenarios() -> list[dict]:
    """Create standard CARLA ScenarioRunner scenarios with difficulty attributes."""
    return [
        {
            "name": "StraightRoadYield",
            "type": "straight_road_yield",
            "num_vehicles": 1,
            "num_pedestrians": 0,
            "num_static_obstacles": 0,
            "is_intersection": False,
            "is_roundabout": False,
            "has_traffic_light": False,
            "weather": "clear_noon",
            "time_of_day": "day",
            "has_junction": False,
            "has_pedestrian_crossing": False,
            "has_vehicle_maneuver": False,
            "target_speed": 10.0,
            "num_waypoints": 50,
            "route_length_m": 200.0,
        },
        {
            "name": "IntersectionLeftTurn",
            "type": "intersection_left_turn",
            "num_vehicles": 3,
            "num_pedestrians": 2,
            "num_static_obstacles": 0,
            "is_intersection": True,
            "is_roundabout": False,
            "has_traffic_light": True,
            "weather": "clear_noon",
            "time_of_day": "day",
            "has_junction": True,
            "has_pedestrian_crossing": True,
            "has_vehicle_maneuver": True,
            "target_speed": 8.0,
            "num_waypoints": 100,
            "route_length_m": 350.0,
        },
        {
            "name": "RoundaboutMerge",
            "type": "roundabout_merge",
            "num_vehicles": 5,
            "num_pedestrians": 0,
            "num_static_obstacles": 0,
            "is_intersection": False,
            "is_roundabout": True,
            "has_traffic_light": False,
            "weather": "clear_noon",
            "time_of_day": "day",
            "has_junction": True,
            "has_pedestrian_crossing": False,
            "has_vehicle_maneuver": True,
            "target_speed": 12.0,
            "num_waypoints": 150,
            "route_length_m": 400.0,
        },
        {
            "name": "UrbanJunctionPedestrianCrossing",
            "type": "urban_junction_pedestrian",
            "num_vehicles": 4,
            "num_pedestrians": 4,
            "num_static_obstacles": 1,
            "is_intersection": True,
            "is_roundabout": False,
            "has_traffic_light": True,
            "weather": "clear_noon",
            "time_of_day": "day",
            "has_junction": True,
            "has_pedestrian_crossing": True,
            "has_vehicle_maneuver": False,
            "target_speed": 8.0,
            "num_waypoints": 120,
            "route_length_m": 300.0,
        },
        {
            "name": "HighwayMerge",
            "type": "highway_merge",
            "num_vehicles": 8,
            "num_pedestrians": 0,
            "num_static_obstacles": 0,
            "is_intersection": False,
            "is_roundabout": False,
            "has_traffic_light": False,
            "weather": "clear_noon",
            "time_of_day": "day",
            "has_junction": True,
            "has_pedestrian_crossing": False,
            "has_vehicle_maneuver": True,
            "target_speed": 25.0,
            "num_waypoints": 200,
            "route_length_m": 800.0,
        },
        {
            "name": "NightRainIntersection",
            "type": "intersection_right_turn",
            "num_vehicles": 3,
            "num_pedestrians": 2,
            "num_static_obstacles": 0,
            "is_intersection": True,
            "is_roundabout": False,
            "has_traffic_light": True,
            "weather": "rain_night",
            "time_of_day": "night",
            "has_junction": True,
            "has_pedestrian_crossing": True,
            "has_vehicle_maneuver": True,
            "target_speed": 6.0,
            "num_waypoints": 80,
            "route_length_m": 250.0,
        },
        {
            "name": "FoggyPedestrianAvoidance",
            "type": "pedestrian_avoidance",
            "num_vehicles": 1,
            "num_pedestrians": 3,
            "num_static_obstacles": 2,
            "is_intersection": False,
            "is_roundabout": False,
            "has_traffic_light": False,
            "weather": "fog_noon",
            "time_of_day": "day",
            "has_junction": False,
            "has_pedestrian_crossing": True,
            "has_vehicle_maneuver": False,
            "target_speed": 8.0,
            "num_waypoints": 60,
            "route_length_m": 180.0,
        },
        {
            "name": "ComplexUrbanRoute",
            "type": "complex_urban",
            "num_vehicles": 12,
            "num_pedestrians": 6,
            "num_static_obstacles": 2,
            "is_intersection": True,
            "is_roundabout": True,
            "has_traffic_light": True,
            "weather": "clear_sunset",
            "time_of_day": "sunset",
            "has_junction": True,
            "has_pedestrian_crossing": True,
            "has_vehicle_maneuver": True,
            "target_speed": 10.0,
            "num_waypoints": 300,
            "route_length_m": 1000.0,
        },
    ]


def main():
    """Main entry point for scenario difficulty analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze CARLA ScenarioRunner scenario difficulty"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="Path to JSON file containing scenario configurations",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for difficulty analysis results",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose analysis",
    )
    args = parser.parse_args()
    
    # Load or create scenarios
    if args.scenarios and os.path.exists(args.scenarios):
        with open(args.scenarios) as f:
            scenarios = json.load(f)
    else:
        scenarios = create_standard_scenarios()
    
    # Analyze
    analyzer = ScenarioDifficultyAnalyzer()
    results = analyzer.analyze_batch(scenarios)
    summary = analyzer.get_difficulty_summary(scenarios)
    
    # Output
    if args.json_output or args.output:
        output_data = {
            "summary": summary,
            "scenarios": [
                {
                    "name": m.scenario_name,
                    "type": m.scenario_type,
                    "score": m.difficulty_score,
                    "level": m.difficulty_level.value,
                    "complexity": m.agent_complexity.name,
                }
                for m in results.values()
            ],
        }
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))
    else:
        # Human-readable output
        print("=" * 60)
        print("SCENARIO DIFFICULTY ANALYSIS")
        print("=" * 60)
        
        print(f"\nTotal scenarios: {summary['total_scenarios']}")
        print(f"Average difficulty score: {summary['average_score']:.2f}")
        
        print("\nBy Difficulty Level:")
        for level, count in summary["by_difficulty"].items():
            print(f"  {level.capitalize()}: {count}")
        
        print("\nBy Agent Complexity:")
        for level, count in summary["by_agent_complexity"].items():
            print(f"  {level.capitalize()}: {count}")
        
        if args.verbose:
            print("\n" + "-" * 60)
            print("DETAILED SCENARIO ANALYSIS:")
            print("-" * 60)
            for name, metrics in sorted(results.items()):
                print(f"\n{name} ({metrics.scenario_type})")
                print(f"  Level: {metrics.difficulty_level.value}")
                print(f"  Score: {metrics.difficulty_score:.2f}")
                print(f"  Agents: {metrics.num_vehicles}v + {metrics.num_pedestrians}p")
                print(f"  Weather: {metrics.weather}")
                print(f"  Raw scores: A={metrics.raw_agent_score:.1f}, "
                      f"D={metrics.raw_density_score:.1f}, "
                      f"I={metrics.raw_intersection_score:.1f}, "
                      f"W={metrics.raw_weather_score:.1f}, "
                      f"O={metrics.raw_obstacle_score:.1f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Scenario Performance Analyzer

Analyzes policy performance across scenario difficulties, identifies patterns,
and provides actionable insights for improving driving models.

This analyzer correlates scenario difficulty (from ScenarioDifficultyAnalyzer)
with actual evaluation metrics to identify:
- Performance bottlenecks by difficulty level
- Scenario categories where the policy struggles most
- Recommendations for targeted training data generation
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try to import scenario difficulty analyzer
try:
    from sim.driving.carla_srunner.scenario_difficulty_analyzer import (
        ScenarioDifficultyAnalyzer,
        DifficultyLevel,
    )
    DIFFICULTY_AVAILABLE = True
except ImportError:
    DIFFICULTY_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single scenario run."""
    scenario_name: str
    ade: float  # Average Displacement Error (meters)
    fde: float  # Final Displacement Error (meters)
    success_rate: float  # 0-1
    route_completion: float  # 0-1
    collisions: int
    infractions: int
    duration: float  # seconds
    

@dataclass
class DifficultyBreakdown:
    """Performance breakdown by difficulty level."""
    difficulty: str
    num_scenarios: int
    mean_ade: float = 0.0
    mean_fde: float = 0.0
    mean_success_rate: float = 0.0
    mean_route_completion: float = 0.0
    total_collisions: int = 0
    total_infractions: int = 0
    
    
@dataclass
class CategoryBreakdown:
    """Performance breakdown by scenario category."""
    category: str
    num_scenarios: int
    mean_ade: float = 0.0
    mean_fde: float = 0.0
    mean_success_rate: float = 0.0
    
    
@dataclass
class PerformanceInsight:
    """Insight about policy performance."""
    category: str  # difficulty, category, or pattern
    description: str
    severity: str  # critical, warning, info
    metric: str
    value: float
    recommendation: str
    
    
@dataclass
class PerformanceReport:
    """Complete performance analysis report."""
    timestamp: str
    total_scenarios: int
    overall_metrics: PerformanceMetrics
    difficulty_breakdown: list[DifficultyBreakdown]
    category_breakdown: list[CategoryBreakdown]
    insights: list[PerformanceInsight]
    bottlenecks: list[str]
    recommendations: list[str]
    

class ScenarioPerformanceAnalyzer:
    """
    Analyzes scenario evaluation performance and correlates with difficulty.
    
    Usage:
        analyzer = ScenarioPerformanceAnalyzer()
        analyzer.add_result("straight_clear", ade=2.1, fde=3.2, success_rate=0.95)
        analyzer.add_result("intersection_left", ade=5.8, fde=12.3, success_rate=0.65)
        report = analyzer.analyze()
        print(report)
    """
    
    SCENARIO_CATEGORIES = {
        "straight": ["straight_clear", "straight_traffic", "straight_pedestrian"],
        "turn": ["turn_left", "turn_right", "intersection_left", "intersection_right"],
        "lane_change": ["lane_change_left", "lane_change_right", "merge"],
        "roundabout": ["roundabout_enter", "roundabout_exit", "roundabout_navigate"],
        "intersection": ["intersection_4way", "intersection_t", "stop_sign"],
        "weather": ["night_clear", "rain_clear", "fog_clear"],
    }
    
    def __init__(self, difficulty_analyzer: Optional[object] = None):
        """Initialize the performance analyzer."""
        self.results: list[PerformanceMetrics] = []
        self.difficulty_analyzer = difficulty_analyzer
        
    def add_result(
        self,
        scenario_name: str,
        ade: float = 0.0,
        fde: float = 0.0,
        success_rate: float = 0.0,
        route_completion: float = 0.0,
        collisions: int = 0,
        infractions: int = 0,
        duration: float = 0.0,
    ):
        """Add a scenario evaluation result."""
        self.results.append(PerformanceMetrics(
            scenario_name=scenario_name,
            ade=ade,
            fde=fde,
            success_rate=success_rate,
            route_completion=route_completion,
            collisions=collisions,
            infractions=infractions,
            duration=duration,
        ))
        
    def add_results_from_file(self, metrics_path: str) -> int:
        """Load results from a metrics.json file."""
        path = Path(metrics_path)
        if not path.exists():
            print(f"Warning: Metrics file not found: {metrics_path}")
            return 0
            
        with open(path) as f:
            data = json.load(f)
            
        # Handle different metric formats
        if "results" in data:
            results = data["results"]
        elif "scenarios" in data:
            results = data["scenarios"]
        else:
            # Single scenario or aggregate format
            results = [data]
            
        count = 0
        for item in results:
            # Handle various field names
            name = item.get("scenario", item.get("name", f"scenario_{count}"))
            ade = item.get("ade", item.get("ADE", 0.0))
            fde = item.get("fde", item.get("FDE", 0.0))
            success = item.get("success_rate", item.get("success", item.get("Success", 0.0)))
            rc = item.get("route_completion", item.get("routeCompletion", item.get("rc", 0.0)))
            collisions = item.get("collisions", item.get("Collisions", 0))
            infractions = item.get("infractions", item.get("Infractions", 0))
            duration = item.get("duration", item.get("Duration", 0.0))
            
            self.add_result(
                scenario_name=name,
                ade=ade,
                fde=fde,
                success_rate=success,
                route_completion=rc,
                collisions=collisions,
                infractions=infractions,
                duration=duration,
            )
            count += 1
            
        return count
    
    def add_results_from_dir(self, results_dir: str) -> int:
        """Load all metrics.json files from a directory."""
        from pathlib import Path
        
        dir_path = Path(results_dir)
        if not dir_path.exists():
            print(f"Warning: Results directory not found: {results_dir}")
            return 0
            
        count = 0
        for metrics_file in dir_path.rglob("metrics.json"):
            try:
                loaded = self.add_results_from_file(str(metrics_file))
                count += loaded
            except Exception as e:
                print(f"Warning: Failed to load {metrics_file}: {e}")
                
        return count
        
    def _get_scenario_difficulty(self, scenario_name: str) -> str:
        """Get difficulty level for a scenario."""
        if self.difficulty_analyzer is not None:
            try:
                return self.difficulty_analyzer.get_difficulty_level(scenario_name)
            except Exception:
                pass
                
        # Fallback: infer from scenario name patterns
        name_lower = scenario_name.lower()
        
        # Check for difficulty indicators in name
        if "hard" in name_lower or "expert" in name_lower:
            return "expert"
        elif "night" in name_lower or "rain" in name_lower or "fog" in name_lower:
            return "hard"
        elif "left" in name_lower or "right" in name_lower or "turn" in name_lower:
            return "medium"
        else:
            return "easy"
            
    def _get_scenario_category(self, scenario_name: str) -> str:
        """Get category for a scenario."""
        name_lower = scenario_name.lower()
        
        for category, keywords in self.SCENARIO_CATEGORIES.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return category
                    
        # Default categories based on common patterns
        if "intersection" in name_lower or "stop" in name_lower:
            return "intersection"
        elif "straight" in name_lower:
            return "straight"
        elif "turn" in name_lower:
            return "turn"
        elif "lane" in name_lower or "change" in name_lower or "merge" in name_lower:
            return "lane_change"
        elif "roundabout" in name_lower:
            return "roundabout"
        elif "night" in name_lower or "rain" in name_lower or "fog" in name_lower:
            return "weather"
        else:
            return "other"
            
    def _compute_difficulty_breakdown(self) -> list[DifficultyBreakdown]:
        """Compute performance breakdown by difficulty level."""
        by_difficulty: dict[str, list[PerformanceMetrics]] = {}
        
        for result in self.results:
            difficulty = self._get_scenario_difficulty(result.scenario_name)
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = []
            by_difficulty[difficulty].append(result)
            
        breakdown = []
        for difficulty, results in sorted(by_difficulty.items()):
            if not results:
                continue
                
            breakdown.append(DifficultyBreakdown(
                difficulty=difficulty,
                num_scenarios=len(results),
                mean_ade=sum(r.ade for r in results) / len(results),
                mean_fde=sum(r.fde for r in results) / len(results),
                mean_success_rate=sum(r.success_rate for r in results) / len(results),
                mean_route_completion=sum(r.route_completion for r in results) / len(results),
                total_collisions=sum(r.collisions for r in results),
                total_infractions=sum(r.infractions for r in results),
            ))
            
        return breakdown
        
    def _compute_category_breakdown(self) -> list[CategoryBreakdown]:
        """Compute performance breakdown by scenario category."""
        by_category: dict[str, list[PerformanceMetrics]] = {}
        
        for result in self.results:
            category = self._get_scenario_category(result.scenario_name)
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
            
        breakdown = []
        for category, results in sorted(by_category.items()):
            if not results:
                continue
                
            breakdown.append(CategoryBreakdown(
                category=category,
                num_scenarios=len(results),
                mean_ade=sum(r.ade for r in results) / len(results),
                mean_fde=sum(r.fde for r in results) / len(results),
                mean_success_rate=sum(r.success_rate for r in results) / len(results),
            ))
            
        return breakdown
        
    def _generate_insights(self) -> list[PerformanceInsight]:
        """Generate performance insights based on analysis."""
        insights = []
        
        # Get breakdowns for reference
        diff_breakdown = {b.difficulty: b for b in self._compute_difficulty_breakdown()}
        cat_breakdown = {b.category: b for b in self._compute_category_breakdown()}
        
        # Insight: High ADE on hard scenarios
        if "expert" in diff_breakdown:
            expert = diff_breakdown["expert"]
            if expert.mean_ade > 5.0:
                insights.append(PerformanceInsight(
                    category="difficulty",
                    description=f"Expert-level scenarios have high ADE ({expert.mean_ade:.2f}m)",
                    severity="critical",
                    metric="ade",
                    value=expert.mean_ade,
                    recommendation="Generate more expert-level training scenarios or increase RL exploration",
                ))
                
        # Insight: Poor intersection performance
        if "intersection" in cat_breakdown:
            inter = cat_breakdown["intersection"]
            if inter.mean_success_rate < 0.7:
                insights.append(PerformanceInsight(
                    category="category",
                    description=f"Intersection scenarios have low success rate ({inter.mean_success_rate:.1%})",
                    severity="critical",
                    metric="success_rate",
                    value=inter.mean_success_rate,
                    recommendation="Add more intersection-specific training data and improve traffic actor handling",
                ))
                
        # Insight: Weather impact
        if "weather" in cat_breakdown:
            weather = cat_breakdown["weather"]
            if weather.mean_success_rate < 0.6:
                insights.append(PerformanceInsight(
                    category="category",
                    description=f"Weather scenarios have poor performance ({weather.mean_success_rate:.1%} success)",
                    severity="warning",
                    metric="success_rate",
                    value=weather.mean_success_rate,
                    recommendation="Augment training data with weather variations (night, rain, fog)",
                ))
                
        # Insight: High collision rate
        total_collisions = sum(r.collisions for r in self.results)
        if total_collisions > len(self.results) * 0.3:
            insights.append(PerformanceInsight(
                category="pattern",
                description=f"High collision rate: {total_collisions} collisions in {len(self.results)} scenarios",
                severity="critical",
                metric="collisions",
                value=total_collisions,
                recommendation="Improve collision avoidance in policy or add collision-focused RL training",
            ))
            
        # Insight: Performance degradation with complexity
        if "easy" in diff_breakdown and "expert" in diff_breakdown:
            easy_ade = diff_breakdown["easy"].mean_ade
            expert_ade = diff_breakdown["expert"].mean_ade
            degradation = (expert_ade - easy_ade) / max(easy_ade, 0.1)
            if degradation > 1.0:  # More than 100% worse
                insights.append(PerformanceInsight(
                    category="difficulty",
                    description=f"Performance degrades significantly with difficulty (ADE +{degradation:.0%})",
                    severity="warning",
                    metric="ade",
                    value=degradation,
                    recommendation="Curriculum learning: start with easy scenarios, progressively increase difficulty",
                ))
                
        return insights
        
    def _identify_bottlenecks(self) -> list[str]:
        """Identify primary performance bottlenecks."""
        bottlenecks = []
        
        diff_breakdown = {b.difficulty: b for b in self._compute_difficulty_breakdown()}
        cat_breakdown = {b.category: b for b in self._compute_category_breakdown()}
        
        # Find worst performing category
        if cat_breakdown:
            worst_cat = min(cat_breakdown.values(), key=lambda x: x.mean_success_rate)
            if worst_cat.mean_success_rate < 0.5:
                bottlenecks.append(f"Category '{worst_cat.category}' has {worst_cat.mean_success_rate:.1%} success rate")
                
        # Find worst difficulty level
        if diff_breakdown:
            worst_diff = max(diff_breakdown.values(), key=lambda x: x.mean_ade)
            if worst_diff.mean_ade > 8.0:
                bottlenecks.append(f"Difficulty '{worst_diff.difficulty}' has {worst_diff.mean_ade:.1f}m mean ADE")
                
        # Check for collision issues
        total_collisions = sum(r.collisions for r in self.results)
        if total_collisions > len(self.results) * 0.5:
            bottlenecks.append(f"Excessive collisions: {total_collisions} total")
            
        return bottlenecks
        
    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        diff_breakdown = {b.difficulty: b for b in self._compute_difficulty_breakdown()}
        cat_breakdown = {b.category: b for b in self._compute_category_breakdown()}
        
        # Recommendation: Targeted data generation
        if cat_breakdown:
            worst_cats = sorted(cat_breakdown.values(), key=lambda x: x.mean_success_rate)[:2]
            for cat in worst_cats:
                if cat.mean_success_rate < 0.7:
                    recommendations.append(f"Generate more {cat.category} training scenarios")
                    
        # Recommendation: Curriculum learning
        if "expert" in diff_breakdown and diff_breakdown["expert"].mean_success_rate < 0.5:
            recommendations.append("Implement curriculum learning: start with easy scenarios")
            
        # Recommendation: Weather augmentation
        if "weather" in cat_breakdown and cat_breakdown["weather"].mean_success_rate < 0.6:
            recommendations.append("Add weather augmentation to training pipeline")
            
        # Recommendation: Collision avoidance
        total_collisions = sum(r.collisions for r in self.results)
        if total_collisions > len(self.results) * 0.3:
            recommendations.append("Add collision-focused RL training objective")
            
        # Recommendation: Intersection handling
        if "intersection" in cat_breakdown:
            inter = cat_breakdown["intersection"]
            if inter.mean_ade > 5.0:
                recommendations.append("Improve intersection handling: add yield/priority modeling")
                
        return recommendations
        
    def analyze(self) -> PerformanceReport:
        """Generate complete performance analysis report."""
        if not self.results:
            return PerformanceReport(
                timestamp="",
                total_scenarios=0,
                overall_metrics=PerformanceMetrics(
                    scenario_name="", ade=0, fde=0, success_rate=0,
                    route_completion=0, collisions=0, infractions=0, duration=0
                ),
                difficulty_breakdown=[],
                category_breakdown=[],
                insights=[],
                bottlenecks=[],
                recommendations=[],
            )
            
        # Compute overall metrics
        overall = PerformanceMetrics(
            scenario_name="overall",
            ade=sum(r.ade for r in self.results) / len(self.results),
            fde=sum(r.fde for r in self.results) / len(self.results),
            success_rate=sum(r.success_rate for r in self.results) / len(self.results),
            route_completion=sum(r.route_completion for r in self.results) / len(self.results),
            collisions=sum(r.collisions for r in self.results),
            infractions=sum(r.infractions for r in self.results),
            duration=sum(r.duration for r in self.results),
        )
        
        from datetime import datetime
        
        return PerformanceReport(
            timestamp=datetime.now().isoformat(),
            total_scenarios=len(self.results),
            overall_metrics=overall,
            difficulty_breakdown=self._compute_difficulty_breakdown(),
            category_breakdown=self._compute_category_breakdown(),
            insights=self._generate_insights(),
            bottlenecks=self._identify_bottlenecks(),
            recommendations=self._generate_recommendations(),
        )
        
    def print_report(self, report: Optional[PerformanceReport] = None):
        """Print formatted performance report."""
        if report is None:
            report = self.analyze()
            
        print("=" * 60)
        print("SCENARIO PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(f"\nTimestamp: {report.timestamp}")
        print(f"Total Scenarios: {report.total_scenarios}")
        
        print("\n--- Overall Metrics ---")
        om = report.overall_metrics
        print(f"  ADE:            {om.ade:.3f}m")
        print(f"  FDE:            {om.fde:.3f}m")
        print(f"  Success Rate:   {om.success_rate:.1%}")
        print(f"  Route Complete: {om.route_completion:.1%}")
        print(f"  Collisions:     {om.collisions}")
        print(f"  Infractions:    {om.infractions}")
        
        if report.difficulty_breakdown:
            print("\n--- Performance by Difficulty ---")
            for db in report.difficulty_breakdown:
                print(f"  {db.difficulty.upper():8s}: {db.num_scenarios:2d} scenarios | "
                      f"ADE={db.mean_ade:.2f}m | Success={db.mean_success_rate:.1%}")
                      
        if report.category_breakdown:
            print("\n--- Performance by Category ---")
            for cb in report.category_breakdown:
                print(f"  {cb.category:12s}: {cb.num_scenarios:2d} scenarios | "
                      f"ADE={cb.mean_ade:.2f}m | Success={cb.mean_success_rate:.1%}")
                      
        if report.insights:
            print("\n--- Insights ---")
            for ins in report.insights:
                severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(ins.severity, "⚪")
                print(f"  {severity_emoji} [{ins.severity.upper()}] {ins.description}")
                
        if report.bottlenecks:
            print("\n--- Bottlenecks ---")
            for bn in report.bottlenecks:
                print(f"  • {bn}")
                
        if report.recommendations:
            print("\n--- Recommendations ---")
            for rec in report.recommendations:
                print(f"  → {rec}")
                
        print("\n" + "=" * 60)
        
    def save_report(self, output_path: str):
        """Save report to JSON file."""
        report = self.analyze()
        
        # Convert to serializable format
        data = {
            "timestamp": report.timestamp,
            "total_scenarios": report.total_scenarios,
            "overall_metrics": {
                "ade": report.overall_metrics.ade,
                "fde": report.overall_metrics.fde,
                "success_rate": report.overall_metrics.success_rate,
                "route_completion": report.overall_metrics.route_completion,
                "collisions": report.overall_metrics.collisions,
                "infractions": report.overall_metrics.infractions,
            },
            "difficulty_breakdown": [
                {
                    "difficulty": b.difficulty,
                    "num_scenarios": b.num_scenarios,
                    "mean_ade": b.mean_ade,
                    "mean_fde": b.mean_fde,
                    "mean_success_rate": b.mean_success_rate,
                    "mean_route_completion": b.mean_route_completion,
                    "total_collisions": b.total_collisions,
                    "total_infractions": b.total_infractions,
                }
                for b in report.difficulty_breakdown
            ],
            "category_breakdown": [
                {
                    "category": b.category,
                    "num_scenarios": b.num_scenarios,
                    "mean_ade": b.mean_ade,
                    "mean_fde": b.mean_fde,
                    "mean_success_rate": b.mean_success_rate,
                }
                for b in report.category_breakdown
            ],
            "insights": [
                {
                    "category": i.category,
                    "description": i.description,
                    "severity": i.severity,
                    "metric": i.metric,
                    "value": i.value,
                    "recommendation": i.recommendation,
                }
                for i in report.insights
            ],
            "bottlenecks": report.bottlenecks,
            "recommendations": report.recommendations,
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Report saved to: {output_path}")


def create_smoke_test_results() -> ScenarioPerformanceAnalyzer:
    """Create analyzer with smoke test results."""
    analyzer = ScenarioPerformanceAnalyzer()
    
    # Add test scenarios with varying performance
    # Easy scenarios - good performance
    analyzer.add_result("straight_clear", ade=1.2, fde=2.1, success_rate=0.95, route_completion=0.98, collisions=0)
    analyzer.add_result("straight_traffic", ade=2.1, fde=3.5, success_rate=0.88, route_completion=0.92, collisions=1)
    
    # Medium scenarios - moderate performance
    analyzer.add_result("turn_left", ade=3.8, fde=7.2, success_rate=0.75, route_completion=0.82, collisions=2)
    analyzer.add_result("intersection_t", ade=4.2, fde=8.5, success_rate=0.70, route_completion=0.78, collisions=1)
    analyzer.add_result("lane_change_right", ade=3.1, fde=5.8, success_rate=0.80, route_completion=0.85, collisions=0)
    
    # Hard scenarios - poor performance
    analyzer.add_result("intersection_4way", ade=6.5, fde=12.3, success_rate=0.55, route_completion=0.62, collisions=3)
    analyzer.add_result("roundabout_navigate", ade=5.8, fde=11.2, success_rate=0.60, route_completion=0.68, collisions=2)
    
    # Expert scenarios - very poor performance
    analyzer.add_result("night_rain_intersection", ade=8.2, fde=15.8, success_rate=0.40, route_completion=0.48, collisions=4)
    analyzer.add_result("fog_roundabout_exit", ade=9.5, fde=18.2, success_rate=0.35, route_completion=0.42, collisions=5)
    
    return analyzer


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scenario Performance Analyzer - Correlate difficulty with policy performance"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze scenario performance")
    analyze_parser.add_argument("--input", "-i", help="Input metrics file or directory")
    analyze_parser.add_argument("--output", "-o", help="Output report JSON path")
    analyze_parser.add_argument("--difficulty", action="store_true", help="Use difficulty analyzer")
    
    # Add result command
    add_parser = subparsers.add_parser("add", help="Add a single scenario result")
    add_parser.add_argument("--scenario", required=True, help="Scenario name")
    add_parser.add_argument("--ade", type=float, default=0.0, help="ADE in meters")
    add_parser.add_argument("--fde", type=float, default=0.0, help="FDE in meters")
    add_parser.add_argument("--success", type=float, default=0.0, help="Success rate (0-1)")
    add_parser.add_argument("--rc", type=float, default=0.0, help="Route completion (0-1)")
    add_parser.add_argument("--collisions", type=int, default=0, help="Number of collisions")
    add_parser.add_argument("--infractions", type=int, default=0, help="Number of infractions")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show quick statistics")
    
    # Smoke test command
    smoke_parser = subparsers.add_parser("smoke", help="Run smoke test with synthetic data")
    
    args = parser.parse_args()
    
    if args.command == "smoke":
        print("Running smoke test...")
        analyzer = create_smoke_test_results()
        report = analyzer.analyze()
        analyzer.print_report(report)
        analyzer.save_report("out/performance_analysis_smoke/report.json")
        print("\nSmoke test: ✅ PASSED")
        
    elif args.command == "analyze":
        analyzer = ScenarioPerformanceAnalyzer()
        
        if args.difficulty and DIFFICULTY_AVAILABLE:
            diff_analyzer = ScenarioDifficultyAnalyzer()
            analyzer = ScenarioPerformanceAnalyzer(difficulty_analyzer=diff_analyzer)
            
        if args.input:
            input_path = Path(args.input)
            if input_path.is_dir():
                count = analyzer.add_results_from_dir(args.input)
                print(f"Loaded {count} scenario results from directory")
            else:
                count = analyzer.add_results_from_file(args.input)
                print(f"Loaded {count} scenario results from file")
        else:
            # Use smoke test data if no input provided
            analyzer = create_smoke_test_results()
            
        report = analyzer.analyze()
        analyzer.print_report(report)
        
        if args.output:
            analyzer.save_report(args.output)
            
    elif args.command == "add":
        analyzer = ScenarioPerformanceAnalyzer()
        analyzer.add_result(
            scenario_name=args.scenario,
            ade=args.ade,
            fde=args.fde,
            success_rate=args.success,
            route_completion=args.rc,
            collisions=args.collisions,
      infractions=args.infractions,
        )
        print(f"Added result for scenario: {args.scenario}")
        
    elif args.command == "stats":
        analyzer = create_smoke_test_results()
        report = analyzer.analyze()
        print(f"Total scenarios: {report.total_scenarios}")
        print(f"Overall ADE: {report.overall_metrics.ade:.3f}m")
        print(f"Overall Success: {report.overall_metrics.success_rate:.1%}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
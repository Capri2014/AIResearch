#!/usr/bin/env python3
"""
Scenario Result Aggregator

Aggregates results from scenario batch runs, integrates with difficulty analysis
and selection optimizer, and generates comprehensive evaluation reports with
difficulty-aware metrics.

Usage:
    python scenario_result_aggregator.py aggregate \
        --results-dir out/scenario_results \
        --output report.json
    
    python scenario_result_aggregator.py analyze \
        --results-dir out/scenario_results \
        --difficulty-config scenario_difficulty_analyzer.py
    
    python scenario_result_aggregator.py compare \
        --baseline baseline_results.json \
        --current current_results.json \
        --output comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# =============================================================================
# Enums
# =============================================================================

class PerformanceLevel(Enum):
    """Performance level based on metrics."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class ImprovementDirection(Enum):
    """Direction of improvement between runs."""
    IMPROVED = "improved"
    DEGRADED = "degraded"
    STABLE = "stable"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ScenarioMetrics:
    """Individual scenario metrics."""
    scenario_name: str
    success: bool
    ade: float  # Average Displacement Error
    fde: float  # Final Displacement Error
    route_completion: float  # Percentage
    collisions: int
    red_light_violations: int
    stop_sign_violations: int
    route_deviation: float
    duration: float
    difficulty_score: Optional[float] = None
    difficulty_level: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "success": self.success,
            "ade": self.ade,
            "fde": self.fde,
            "route_completion": self.route_completion,
            "collisions": self.collisions,
            "red_light_violations": self.red_light_violations,
            "stop_sign_violations": self.stop_sign_violations,
            "route_deviation": self.route_deviation,
            "duration": self.duration,
            "difficulty_score": self.difficulty_score,
            "difficulty_level": self.difficulty_level,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioMetrics":
        return cls(
            scenario_name=data["scenario_name"],
            success=data["success"],
            ade=data["ade"],
            fde=data["fde"],
            route_completion=data["route_completion"],
            collisions=data["collisions"],
            red_light_violations=data["red_light_violations"],
            stop_sign_violations=data["stop_sign_violations"],
            route_deviation=data["route_deviation"],
            duration=data["duration"],
            difficulty_score=data.get("difficulty_score"),
            difficulty_level=data.get("difficulty_level"),
            error=data.get("error"),
        )


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all scenarios."""
    # Overall stats
    total_scenarios: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    
    # ADE/FDE
    mean_ade: float = 0.0
    median_ade: float = 0.0
    std_ade: float = 0.0
    min_ade: float = 0.0
    max_ade: float = 0.0
    
    mean_fde: float = 0.0
    median_fde: float = 0.0
    std_fde: float = 0.0
    min_fde: float = 0.0
    max_fde: float = 0.0
    
    # Route completion
    mean_route_completion: float = 0.0
    median_route_completion: float = 0.0
    
    # Safety
    total_collisions: int = 0
    total_red_light_violations: int = 0
    total_stop_sign_violations: int = 0
    
    # Duration
    mean_duration: float = 0.0
    total_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "mean_ade": self.mean_ade,
            "median_ade": self.median_ade,
            "std_ade": self.std_ade,
            "min_ade": self.min_ade,
            "max_ade": self.max_ade,
            "mean_fde": self.mean_fde,
            "median_fde": self.median_fde,
            "std_fde": self.std_fde,
            "min_fde": self.min_fde,
            "max_fde": self.max_fde,
            "mean_route_completion": self.mean_route_completion,
            "median_route_completion": self.median_route_completion,
            "total_collisions": self.total_collisions,
            "total_red_light_violations": self.total_red_light_violations,
            "total_stop_sign_violations": self.total_stop_sign_violations,
            "mean_duration": self.mean_duration,
            "total_duration": self.total_duration,
        }


@dataclass
class DifficultyBreakdown:
    """Metrics broken down by difficulty level."""
    easy: AggregatedMetrics = field(default_factory=AggregatedMetrics)
    medium: AggregatedMetrics = field(default_factory=AggregatedMetrics)
    hard: AggregatedMetrics = field(default_factory=AggregatedMetrics)
    expert: AggregatedMetrics = field(default_factory=AggregatedMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "easy": self.easy.to_dict(),
            "medium": self.medium.to_dict(),
            "hard": self.hard.to_dict(),
            "expert": self.expert.to_dict(),
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    report_id: str
    timestamp: str
    total_scenarios: int
    aggregated: AggregatedMetrics
    difficulty_breakdown: DifficultyBreakdown
    per_scenario: List[ScenarioMetrics]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "total_scenarios": self.total_scenarios,
            "aggregated": self.aggregated.to_dict(),
            "difficulty_breakdown": self.difficulty_breakdown.to_dict(),
            "per_scenario": [s.to_dict() for s in self.per_scenario],
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Comparison between two evaluation runs."""
    baseline_report: EvaluationReport
    current_report: EvaluationReport
    
    # Improvement metrics
    success_rate_delta: float = 0.0
    ade_delta: float = 0.0
    fde_delta: float = 0.0
    route_completion_delta: float = 0.0
    collisions_delta: int = 0
    
    direction: ImprovementDirection = ImprovementDirection.STABLE
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_report_id": self.baseline_report.report_id,
            "current_report_id": self.current_report.report_id,
            "success_rate_delta": self.success_rate_delta,
            "ade_delta": self.ade_delta,
            "fde_delta": self.fde_delta,
            "route_completion_delta": self.route_completion_delta,
            "collisions_delta": self.collisions_delta,
            "direction": self.direction.value,
            "summary": self.summary,
        }


# =============================================================================
# Aggregator Class
# =============================================================================

class ScenarioResultAggregator:
    """Aggregates and analyzes scenario evaluation results."""
    
    def __init__(self, difficulty_analyzer=None):
        """
        Initialize the aggregator.
        
        Args:
            difficulty_analyzer: Optional ScenarioDifficultyAnalyzer instance
                                for difficulty-aware aggregation
        """
        self.difficulty_analyzer = difficulty_analyzer
        self.results: List[ScenarioMetrics] = []
        
    def add_result(self, result: ScenarioMetrics):
        """Add a single scenario result."""
        self.results.append(result)
        
    def add_results_from_dir(self, results_dir: str, pattern: str = "*.json"):
        """
        Load results from a directory.
        
        Args:
            results_dir: Directory containing result JSON files
            pattern: Glob pattern for result files
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
            
        for result_file in results_path.glob(pattern):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            self.add_result(ScenarioMetrics.from_dict(item))
                    elif isinstance(data, dict):
                        if "scenarios" in data:
                            for item in data["scenarios"]:
                                self.add_result(ScenarioMetrics.from_dict(item))
                        else:
                            self.add_result(ScenarioMetrics.from_dict(data))
            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}")
                
    def add_results_from_file(self, filepath: str):
        """Load results from a single JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            for item in data:
                self.add_result(ScenarioMetrics.from_dict(item))
        elif isinstance(data, dict):
            if "results" in data:
                for item in data["results"]:
                    self.add_result(ScenarioMetrics.from_dict(item))
            elif "scenarios" in data:
                for item in data["scenarios"]:
                    self.add_result(ScenarioMetrics.from_dict(item))
            else:
                self.add_result(ScenarioMetrics.from_dict(data))
                
    def _compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute basic statistics for a list of values."""
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            
        arr = np.array(values) if NUMPY_AVAILABLE else values
        
        if NUMPY_AVAILABLE:
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        else:
            sorted_vals = sorted(arr)
            n = len(sorted_vals)
            mean_val = sum(sorted_vals) / n
            std_val = (sum((x - mean_val) ** 2 for x in sorted_vals) / n) ** 0.5
            return {
                "mean": mean_val,
                "median": sorted_vals[n // 2],
                "std": std_val,
                "min": sorted_vals[0],
                "max": sorted_vals[-1],
            }
            
    def _aggregate_results(self, results: List[ScenarioMetrics]) -> AggregatedMetrics:
        """Aggregate a list of scenario results."""
        if not results:
            return AggregatedMetrics()
            
        # Filter successful results for ADE/FDE stats (exclude errors)
        valid_results = [r for r in results if r.error is None]
        
        agg = AggregatedMetrics()
        agg.total_scenarios = len(results)
        agg.success_count = sum(1 for r in results if r.success)
        agg.failure_count = agg.total_scenarios - agg.success_count
        agg.success_rate = agg.success_count / agg.total_scenarios if agg.total_scenarios > 0 else 0.0
        
        # ADE statistics
        ade_values = [r.ade for r in valid_results]
        if ade_values:
            stats = self._compute_statistics(ade_values)
            agg.mean_ade = stats["mean"]
            agg.median_ade = stats["median"]
            agg.std_ade = stats["std"]
            agg.min_ade = stats["min"]
            agg.max_ade = stats["max"]
            
        # FDE statistics
        fde_values = [r.fde for r in valid_results]
        if fde_values:
            stats = self._compute_statistics(fde_values)
            agg.mean_fde = stats["mean"]
            agg.median_fde = stats["median"]
            agg.std_fde = stats["std"]
            agg.min_fde = stats["min"]
            agg.max_fde = stats["max"]
            
        # Route completion
        rc_values = [r.route_completion for r in valid_results]
        if rc_values:
            stats = self._compute_statistics(rc_values)
            agg.mean_route_completion = stats["mean"]
            agg.median_route_completion = stats["median"]
            
        # Safety metrics
        agg.total_collisions = sum(r.collisions for r in results)
        agg.total_red_light_violations = sum(r.red_light_violations for r in results)
        agg.total_stop_sign_violations = sum(r.stop_sign_violations for r in results)
        
        # Duration
        duration_values = [r.duration for r in results]
        if duration_values:
            stats = self._compute_statistics(duration_values)
            agg.mean_duration = stats["mean"]
            agg.total_duration = sum(duration_values)
            
        return agg
        
    def _get_difficulty_level(self, result: ScenarioMetrics) -> str:
        """Get difficulty level for a result."""
        if result.difficulty_level:
            return result.difficulty_level.lower()
            
        # If we have a difficulty analyzer, compute it
        if self.difficulty_analyzer and result.difficulty_score is not None:
            if result.difficulty_score < 4:
                return "easy"
            elif result.difficulty_score < 8:
                return "medium"
            elif result.difficulty_score < 12:
                return "hard"
            else:
                return "expert"
                
        # Default to expert if no difficulty info
        return "expert"
        
    def aggregate(self, include_difficulty_breakdown: bool = True) -> EvaluationReport:
        """
        Aggregate all loaded results into a comprehensive report.
        
        Args:
            include_difficulty_breakdown: Whether to compute difficulty-aware breakdown
            
        Returns:
            EvaluationReport with aggregated metrics
        """
        if not self.results:
            raise ValueError("No results to aggregate")
            
        # Compute overall aggregation
        aggregated = self._aggregate_results(self.results)
        
        # Compute difficulty breakdown
        difficulty_breakdown = DifficultyBreakdown()
        
        if include_difficulty_breakdown:
            # Group results by difficulty
            by_difficulty = {
                "easy": [],
                "medium": [],
                "hard": [],
                "expert": [],
            }
            
            for result in self.results:
                level = self._get_difficulty_level(result)
                if level in by_difficulty:
                    by_difficulty[level].append(result)
                    
            # Aggregate each difficulty level
            difficulty_breakdown.easy = self._aggregate_results(by_difficulty["easy"])
            difficulty_breakdown.medium = self._aggregate_results(by_difficulty["medium"])
            difficulty_breakdown.hard = self._aggregate_results(by_difficulty["hard"])
            difficulty_breakdown.expert = self._aggregate_results(by_difficulty["expert"])
            
        # Create report
        report = EvaluationReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            total_scenarios=len(self.results),
            aggregated=aggregated,
            difficulty_breakdown=difficulty_breakdown,
            per_scenario=self.results,
            metadata={
                "difficulty_analyzer": self.difficulty_analyzer is not None,
            }
        )
        
        return report
        
    def compare(self, baseline: EvaluationReport, current: EvaluationReport) -> ComparisonResult:
        """
        Compare two evaluation reports.
        
        Args:
            baseline: Baseline evaluation report
            current: Current evaluation report
            
        Returns:
            ComparisonResult with delta metrics
        """
        # Compute deltas
        base_agg = baseline.aggregated
        curr_agg = current.aggregated
        
        success_rate_delta = curr_agg.success_rate - base_agg.success_rate
        ade_delta = curr_agg.mean_ade - base_agg.mean_ade
        fde_delta = curr_agg.mean_fde - base_agg.mean_fde
        route_completion_delta = curr_agg.mean_route_completion - base_agg.mean_route_completion
        collisions_delta = curr_agg.total_collisions - base_agg.total_collisions
        
        # Determine direction
        # Improvement: higher success rate, lower ADE/FDE, higher route completion, fewer collisions
        improvement_score = (
            success_rate_delta * 10  # Weight success rate heavily
            - ade_delta * 0.1  # Lower ADE is better
            - fde_delta * 0.05
            + route_completion_delta * 0.1
            - collisions_delta * 0.5
        )
        
        if improvement_score > 0.5:
            direction = ImprovementDirection.IMPROVED
        elif improvement_score < -0.5:
            direction = ImprovementDirection.DEGRADED
        else:
            direction = ImprovementDirection.STABLE
            
        # Generate summary
        summary_parts = []
        if success_rate_delta > 0.01:
            summary_parts.append(f"success rate +{success_rate_delta:.1%}")
        elif success_rate_delta < -0.01:
            summary_parts.append(f"success rate {success_rate_delta:.1%}")
            
        if ade_delta < -0.1:
            summary_parts.append(f"ADE -{ade_delta:.2f}m")
        elif ade_delta > 0.1:
            summary_parts.append(f"ADE +{ade_delta:.2f}m")
            
        if collisions_delta < 0:
            summary_parts.append(f"collisions {collisions_delta}")
        elif collisions_delta > 0:
            summary_parts.append(f"collisions +{collisions_delta}")
            
        summary = ", ".join(summary_parts) if summary_parts else "No significant changes"
        
        return ComparisonResult(
            baseline_report=baseline,
            current_report=current,
            success_rate_delta=success_rate_delta,
            ade_delta=ade_delta,
            fde_delta=fde_delta,
            route_completion_delta=route_completion_delta,
            collisions_delta=collisions_delta,
            direction=direction,
            summary=summary,
        )


# =============================================================================
# CLI Functions
# =============================================================================

def aggregate_command(args):
    """Handle aggregate subcommand."""
    aggregator = ScenarioResultAggregator()
    
    # Load results
    if args.results_file:
        aggregator.add_results_from_file(args.results_file)
    elif args.results_dir:
        aggregator.add_results_from_dir(args.results_dir, args.pattern)
    else:
        print("Error: Must specify --results-file or --results-dir")
        return 1
        
    # Aggregate
    report = aggregator.aggregate(include_difficulty_breakdown=not args.no_difficulty_breakdown)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report.to_dict(), indent=2))
        
    return 0


def analyze_command(args):
    """Handle analyze subcommand."""
    # Try to import difficulty analyzer
    try:
        from sim.driving.carla_srunner.scenario_difficulty_analyzer import ScenarioDifficultyAnalyzer
        difficulty_analyzer = ScenarioDifficultyAnalyzer()
    except ImportError:
        print("Warning: Could not load ScenarioDifficultyAnalyzer, proceeding without difficulty analysis")
        difficulty_analyzer = None
        
    aggregator = ScenarioResultAggregator(difficulty_analyzer=difficulty_analyzer)
    
    # Load results
    if args.results_file:
        aggregator.add_results_from_file(args.results_file)
    elif args.results_dir:
        aggregator.add_results_from_dir(args.results_dir, args.pattern)
    else:
        print("Error: Must specify --results-file or --results-dir")
        return 1
        
    # Aggregate with difficulty breakdown
    report = aggregator.aggregate(include_difficulty_breakdown=True)
    
    # Print analysis
    print(f"\n=== Evaluation Report: {report.report_id} ===")
    print(f"Total scenarios: {report.total_scenarios}")
    print(f"\n--- Overall Metrics ---")
    print(f"Success Rate: {report.aggregated.success_rate:.1%}")
    print(f"ADE: {report.aggregated.mean_ade:.2f}m (std: {report.aggregated.std_ade:.2f})")
    print(f"FDE: {report.aggregated.mean_fde:.2f}m (std: {report.aggregated.std_fde:.2f})")
    print(f"Route Completion: {report.aggregated.mean_route_completion:.1f}%")
    print(f"Total Collisions: {report.aggregated.total_collisions}")
    
    print(f"\n--- By Difficulty Level ---")
    breakdown = report.difficulty_breakdown
    
    for level, metrics in [
        ("Easy", breakdown.easy),
        ("Medium", breakdown.medium),
        ("Hard", breakdown.hard),
        ("Expert", breakdown.expert),
    ]:
        if metrics.total_scenarios > 0:
            print(f"\n{level} ({metrics.total_scenarios} scenarios):")
            print(f"  Success Rate: {metrics.success_rate:.1%}")
            print(f"  ADE: {metrics.mean_ade:.2f}m, FDE: {metrics.mean_fde:.2f}m")
            
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nFull report saved to {args.output}")
        
    return 0


def compare_command(args):
    """Handle compare subcommand."""
    # Load reports
    with open(args.baseline, 'r') as f:
        baseline_data = json.load(f)
        
    with open(args.current, 'r') as f:
        current_data = json.load(f)
        
    # Reconstruct reports (simplified - in production would use proper deserialization)
    baseline = EvaluationReport(
        report_id=baseline_data.get("report_id", "baseline"),
        timestamp=baseline_data.get("timestamp", ""),
        total_scenarios=baseline_data.get("total_scenarios", 0),
        aggregated=AggregatedMetrics(),  # Would reconstruct from data
        difficulty_breakdown=DifficultyBreakdown(),
        per_scenario=[],
    )
    # Note: For full comparison, would need to properly deserialize the nested data
    
    # For now, compare top-level metrics
    base_agg = baseline_data.get("aggregated", {})
    curr_agg = current_data.get("aggregated", {})
    
    print(f"\n=== Comparison Report ===")
    print(f"Baseline: {args.baseline}")
    print(f"Current: {args.current}")
    
    # Success rate
    base_sr = base_agg.get("success_rate", 0)
    curr_sr = curr_agg.get("success_rate", 0)
    sr_delta = curr_sr - base_sr
    print(f"\nSuccess Rate: {base_sr:.1%} -> {curr_sr:.1%} ({sr_delta:+.1%})")
    
    # ADE
    base_ade = base_agg.get("mean_ade", 0)
    curr_ade = curr_agg.get("mean_ade", 0)
    ade_delta = curr_ade - base_ade
    print(f"ADE: {base_ade:.2f}m -> {curr_ade:.2f}m ({ade_delta:+.2f}m)")
    
    # FDE
    base_fde = base_agg.get("mean_fde", 0)
    curr_fde = curr_agg.get("mean_fde", 0)
    fde_delta = curr_fde - base_fde
    print(f"FDE: {base_fde:.2f}m -> {curr_fde:.2f}m ({fde_delta:+.2f}m)")
    
    # Collisions
    base_col = base_agg.get("total_collisions", 0)
    curr_col = curr_agg.get("total_collisions", 0)
    col_delta = curr_col - base_col
    print(f"Collisions: {base_col} -> {curr_col} ({col_delta:+)})")
    
    # Summary
    print(f"\n=== Summary ===")
    if sr_delta > 0.05 and ade_delta < 0:
        print("✓ Significant improvement")
    elif sr_delta < -0.05 or ade_delta > 1:
        print("✗ Significant regression")
    else:
        print("○ Minor changes")
        
    # Save comparison
    if args.output:
        comparison = {
            "baseline": args.baseline,
            "current": args.current,
            "success_rate_delta": sr_delta,
            "ade_delta": ade_delta,
            "fde_delta": fde_delta,
            "collisions_delta": col_delta,
        }
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {args.output}")
        
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scenario Result Aggregator - Analyze and compare scenario evaluation results"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Aggregate command
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate results into a report")
    agg_parser.add_argument("--results-file", type=str, help="Path to results JSON file")
    agg_parser.add_argument("--results-dir", type=str, help="Directory containing result files")
    agg_parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for result files")
    agg_parser.add_argument("--output", type=str, help="Output file for report")
    agg_parser.add_argument("--no-difficulty-breakdown", action="store_true", help="Skip difficulty breakdown")
    agg_parser.set_defaults(func=aggregate_command)
    
    # Analyze command
    anal_parser = subparsers.add_parser("analyze", help="Analyze results with difficulty breakdown")
    anal_parser.add_argument("--results-file", type=str, help="Path to results JSON file")
    anal_parser.add_argument("--results-dir", type=str, help="Directory containing result files")
    anal_parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for result files")
    anal_parser.add_argument("--output", type=str, help="Output file for report")
    anal_parser.set_defaults(func=analyze_command)
    
    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Compare two evaluation reports")
    comp_parser.add_argument("--baseline", type=str, required=True, help="Baseline report JSON")
    comp_parser.add_argument("--current", type=str, required=True, help="Current report JSON")
    comp_parser.add_argument("--output", type=str, help="Output file for comparison")
    comp_parser.set_defaults(func=compare_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
        
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
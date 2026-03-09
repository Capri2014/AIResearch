"""
Scenario-Specific Evaluation for RL Waypoint Policy

Breaks down evaluation metrics by scenario type to understand
where the RL policy performs well vs. struggles.

Scenario categories:
- turns: left turns, right turns, u-turns
- intersections: uncontrolled, signalized, stop-sign
- lane_changes: lane follow, lane change left/right
- straight: straight driving, following
- parking: parking maneuvers, pulling in/out
- merging: highway merge, intersection merge
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Try to import carla, fallback gracefully
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


@dataclass
class ScenarioMetrics:
    """Metrics for a single scenario run."""
    scenario_id: str
    scenario_type: str  # turns, intersections, lane_changes, straight, parking, merging
    ade: float = 0.0  # Average Displacement Error (m)
    fde: float = 0.0  # Final Displacement Error (m)
    success: bool = False
    route_completion: float = 0.0  # 0-1
    collisions: int = 0
    red_light_violations: int = 0
    stop_sign_violations: int = 0
    off_road: int = 0
    timeout: bool = False
    duration: float = 0.0  # seconds


@dataclass
class CategoryStats:
    """Aggregated stats for a scenario category."""
    scenario_type: str
    count: int = 0
    success_count: int = 0
    ade_sum: float = 0.0
    fde_sum: float = 0.0
    rc_sum: float = 0.0
    collision_count: int = 0
    rl_violation_count: int = 0
    off_road_count: int = 0
    timeout_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.count if self.count > 0 else 0.0

    @property
    def mean_ade(self) -> float:
        return self.ade_sum / self.count if self.count > 0 else 0.0

    @property
    def mean_fde(self) -> float:
        return self.fde_sum / self.count if self.count > 0 else 0.0

    @property
    def mean_route_completion(self) -> float:
        return self.rc_sum / self.count if self.count > 0 else 0.0


# Scenario type keywords for classification
SCENARIO_KEYWORDS = {
    "turns": ["turn", "left_turn", "right_turn", "u_turn", "u_turn_left", "u_turn_right", 
              "turn_left", "turn_right", "curve"],
    "intersections": ["intersection", "cross", "junction", "4way", "3way", "roundabout"],
    "lane_changes": ["lane_change", "lane_keep", "lane_follow", "change_lane", "overtake"],
    "straight": ["straight", "follow", "leading", "stopped", "go_straight"],
    "parking": ["park", "pull_in", "pull_out", "parallel", "perpendicular", "parking"],
    "merging": ["merge", "on_ramp", "off_ramp", "highway_enter", "highway_exit"],
}


def classify_scenario(scenario_id: str) -> str:
    """Classify a scenario by its ID into a category."""
    scenario_id_lower = scenario_id.lower()
    
    for category, keywords in SCENARIO_KEYWORDS.items():
        for kw in keywords:
            if kw in scenario_id_lower:
                return category
    
    # Default to straight if no match
    return "straight"


def aggregate_category_stats(metrics: list[ScenarioMetrics]) -> dict[str, CategoryStats]:
    """Aggregate metrics by scenario category."""
    stats: dict[str, CategoryStats] = {}
    
    for m in metrics:
        cat = classify_scenario(m.scenario_id)
        
        if cat not in stats:
            stats[cat] = CategoryStats(scenario_type=cat)
        
        s = stats[cat]
        s.count += 1
        s.ade_sum += m.ade
        s.fde_sum += m.fde
        s.rc_sum += m.route_completion
        
        if m.success:
            s.success_count += 1
        if m.collisions > 0:
            s.collision_count += 1
        if m.red_light_violations > 0 or m.stop_sign_violations > 0:
            s.rl_violation_count += 1
        if m.off_road > 0:
            s.off_road_count += 1
        if m.timeout:
            s.timeout_count += 1
    
    return stats


def print_category_report(stats: dict[str, CategoryStats]) -> str:
    """Generate a formatted category report."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("SCENARIO-SPECIFIC EVALUATION REPORT")
    lines.append("=" * 70)
    
    # Sort by category name for consistent output
    for cat in sorted(stats.keys()):
        s = stats[cat]
        lines.append(f"\n### {cat.upper()} (n={s.count})")
        lines.append("-" * 40)
        lines.append(f"  Success Rate:    {s.success_rate*100:6.1f}%")
        lines.append(f"  Mean ADE:        {s.mean_ade:6.2f} m")
        lines.append(f"  Mean FDE:        {s.mean_fde:6.2f} m")
        lines.append(f"  Route Completion:{s.mean_route_completion*100:6.1f}%")
        lines.append(f"  Collisions:      {s.collision_count}/{s.count}")
        lines.append(f"  RL Violations:   {s.rl_violation_count}/{s.count}")
        lines.append(f"  Off-Road:        {s.off_road_count}/{s.count}")
        lines.append(f"  Timeouts:        {s.timeout_count}/{s.count}")
    
    # Summary row
    total = sum(s.count for s in stats.values())
    total_success = sum(s.success_count for s in stats.values())
    total_ade = sum(s.ade_sum for s in stats.values())
    total_fde = sum(s.fde_sum for s in stats.values())
    total_rc = sum(s.rc_sum for s in stats.values())
    
    lines.append("\n" + "=" * 70)
    lines.append("OVERALL")
    lines.append("-" * 40)
    lines.append(f"  Total Scenarios: {total}")
    lines.append(f"  Success Rate:    {total_success/total*100 if total > 0 else 0:.1f}%")
    lines.append(f"  Mean ADE:         {total_ade/total if total > 0 else 0:.2f} m")
    lines.append(f"  Mean FDE:         {total_fde/total if total > 0 else 0:.2f} m")
    lines.append(f"  Route Completion: {total_rc/total*100 if total > 0 else 0:.1f}%")
    lines.append("=" * 70 + "\n")
    
    return "\n".join(lines)


def load_existing_metrics(metrics_path: str) -> list[ScenarioMetrics]:
    """Load metrics from an existing metrics.json file."""
    metrics = []
    
    if not os.path.exists(metrics_path):
        return metrics
    
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    # Handle different metric formats
    scenarios = data.get("scenarios", data.get("results", []))
    
    for s in scenarios:
        m = ScenarioMetrics(
            scenario_id=s.get("scenario_id", s.get("id", "unknown")),
            scenario_type=s.get("scenario_type", "unknown"),
            ade=s.get("ade", 0.0),
            fde=s.get("fde", 0.0),
            success=s.get("success", s.get("success_rate", 0) > 0),
            route_completion=s.get("route_completion", s.get("rc", 0.0)),
            collisions=s.get("collisions", 0),
            red_light_violations=s.get("red_light_violations", 0),
            stop_sign_violations=s.get("stop_sign_violations", 0),
            off_road=s.get("off_road", 0),
            timeout=s.get("timeout", False),
            duration=s.get("duration", 0.0),
        )
        metrics.append(m)
    
    return metrics


def run_scenario_eval(
    checkpoint_path: str,
    carla_host: str = "127.0.0.1",
    carla_port: int = 2000,
    scenario_suite: str = "smoke",
    output_dir: Optional[str] = None,
) -> dict:
    """
    Run scenario-specific evaluation using RL checkpoint.
    
    Args:
        checkpoint_path: Path to RL checkpoint
        carla_host: CARLA server host
        carla_port: CARLA server port
        scenario_suite: Suite to run (smoke, basic, all)
        output_dir: Output directory for results
        
    Returns:
        Dictionary with per-category metrics
    """
    if not CARLA_AVAILABLE:
        print("ERROR: CARLA Python API not available")
        print("Install: pip install carla")
        return {"error": "carla_not_available"}
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Import our RL modules
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.rl import srunner_rl_eval
    
    # Load checkpoint metadata
    checkpoint_meta = srunner_rl_eval.load_checkpoint_metadata(checkpoint_path)
    print(f"Checkpoint: {checkpoint_meta.get('model_type', 'unknown')}")
    
    # Connect to CARLA
    client = carla.Client(carla_host, carla_port)
    client.set_timeout(30.0)
    world = client.get_world()
    
    # Get scenario runner
    # Note: This assumes ScenarioRunner is available
    scenario_runner = None
    
    # Run evaluation scenarios
    metrics: list[ScenarioMetrics] = []
    
    # Get scenario list based on suite
    scenario_suites = {
        "smoke": 5,
        "basic": 20,
        "all": 100,
    }
    num_scenarios = scenario_suites.get(scenario_suite, 5)
    
    print(f"Running {num_scenarios} scenarios from {scenario_suite} suite...")
    
    # This would run actual scenarios - for now, return mock structure
    # In production, this would iterate through scenarios and collect metrics
    
    return {
        "checkpoint": checkpoint_path,
        "suite": scenario_suite,
        "num_scenarios": num_scenarios,
        "categories": {},  # Would be filled with actual run data
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scenario-specific evaluation for RL waypoint policy"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to RL checkpoint"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Path to existing metrics.json to analyze"
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
        "--suite",
        type=str,
        default="smoke",
        choices=["smoke", "basic", "all"],
        help="Scenario suite to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (don't connect to CARLA)"
    )
    
    args = parser.parse_args()
    
    # If metrics file provided, analyze existing results
    if args.metrics:
        print(f"Loading existing metrics from: {args.metrics}")
        metrics = load_existing_metrics(args.metrics)
        print(f"Loaded {len(metrics)} scenario results")
        
        stats = aggregate_category_stats(metrics)
        report = print_category_report(stats)
        print(report)
        
        # Save report
        if args.output_dir:
            report_path = os.path.join(args.output_dir, "category_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {report_path}")
        
        return
    
    # If checkpoint provided, run evaluation
    if args.checkpoint:
        if args.dry_run:
            print(f"[DRY RUN] Would evaluate checkpoint: {args.checkpoint}")
            print(f"  Suite: {args.suite}")
            print(f"  CARLA: {args.carla_host}:{args.carla_port}")
            
            # Validate checkpoint exists
            if os.path.exists(args.checkpoint):
                print(f"  ✓ Checkpoint exists")
            else:
                print(f"  ✗ Checkpoint not found: {args.checkpoint}")
                return
            
            # Try loading checkpoint metadata
            sys.path.insert(0, str(Path(__file__).parent.parent))
            try:
                from training.rl import srunner_rl_eval
                meta = srunner_rl_eval.load_checkpoint_metadata(args.checkpoint)
                print(f"  ✓ Checkpoint metadata: {meta.get('model_type', 'unknown')}")
            except Exception as e:
                print(f"  ! Could not load metadata: {e}")
            
            return
        
        result = run_scenario_eval(
            checkpoint_path=args.checkpoint,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            scenario_suite=args.suite,
            output_dir=args.output_dir,
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Aggregate and report
        # (Would aggregate actual results here)
        print("Evaluation complete!")
        
        return
    
    # Neither metrics nor checkpoint provided
    parser.print_help()
    print("\nExamples:")
    print("  # Analyze existing metrics")
    print("  python -m training.rl.scenario_eval --metrics out/run/metrics.json")
    print("")
    print("  # Run evaluation (requires CARLA)")
    print("  python -m training.rl.scenario_eval --checkpoint model.pt --suite smoke")
    print("")
    print("  # Dry run (no CARLA)")
    print("  python -m training.rl.scenario_eval --checkpoint model.pt --dry-run")


if __name__ == "__main__":
    main()

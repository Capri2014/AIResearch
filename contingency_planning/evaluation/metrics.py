"""
Evaluation Metrics for Contingency Planning

Defines metrics for comparing different planning approaches.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ContingencyMetrics:
    """Metrics for evaluating contingency planning approaches."""
    
    # ===== Safety Metrics =====
    collision_rate: float = 0.0           # % of episodes with collision
    mrc_trigger_rate: float = 0.0        # % of episodes requiring MRC
    off_road_rate: float = 0.0           # % of episodes going off-road
    safety_violations: int = 0           # Total safety constraint violations
    
    # ===== Efficiency Metrics =====
    avg_completion_time: float = 0.0     # Time to reach goal (s)
    avg_speed: float = 0.0              # Average speed (m/s)
    deviation_from_optimal: float = 0.0  # Deviation from nominal plan (m)
    success_rate: float = 0.0            # % of episodes completing successfully
    
    # ===== Computation Metrics =====
    avg_planning_time: float = 0.0      # Average planning time (ms)
    max_planning_time: float = 0.0      # Max planning time (ms)
    tree_depth_used: int = 0            # Average tree branching depth
    
    # ===== Contingency-Specific =====
    correct_hypothesis_rate: float = 0.0 # % of correct contingency predictions
    belief_accuracy: float = 0.0        # Calibration of uncertainty estimates
    false_alarm_rate: float = 0.0       # False positive rate for contingencies
    miss_rate: float = 0.0              # False negative rate for contingencies
    
    # ===== Raw Data =====
    episode_lengths: List[int] = field(default_factory=list)
    planning_times: List[float] = field(default_factory=list)
    
    def __repr__(self):
        return (
            f"ContingencyMetrics(\n"
            f"  Safety: collision={self.collision_rate:.1%}, "
            f"mrc={self.mrc_trigger_rate:.1%}, "
            f"off_road={self.off_road_rate:.1%}\n"
            f"  Efficiency: success={self.success_rate:.1%}, "
            f"time={self.avg_completion_time:.2f}s, "
            f"speed={self.avg_speed:.2f}m/s\n"
            f"  Computation: plan_time={self.avg_planning_time:.1f}ms, "
            f"tree_depth={self.tree_depth_used}\n"
            f")"
        )


@dataclass
class ComparisonResults:
    """Results comparing multiple approaches."""
    approaches: Dict[str, ContingencyMetrics]
    scenario_name: str
    n_episodes: int
    
    def summary_table(self) -> str:
        """Generate markdown summary table."""
        headers = ["Metric", *self.approaches.keys()]
        rows = [
            ["Collision Rate", *[f"{m.collision_rate:.1%}" for m in self.approaches.values()]],
            ["MRC Trigger Rate", *[f"{m.mrc_trigger_rate:.1%}" for m in self.approaches.values()]],
            ["Success Rate", *[f"{m.success_rate:.1%}" for m in self.approaches.values()]],
            ["Avg Completion Time", *[f"{m.avg_completion_time:.2f}s" for m in self.approaches.values()]],
            ["Avg Speed", *[f"{m.avg_speed:.2f}m/s" for m in self.approaches.values()]],
            ["Avg Planning Time", *[f"{m.avg_planning_time:.1f}ms" for m in self.approaches.values()]],
            ["Tree Depth", *[str(m.tree_depth_used) for m in self.approaches.values()]],
            ["Correct Hypothesis", *[f"{m.correct_hypothesis_rate:.1%}" for m in self.approaches.values()]],
        ]
        
        # Format as markdown table
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = ["| " + " | ".join(row) + " |" for row in rows]
        
        return "\n".join([header_row, separator] + data_rows)
    
    def best_approach(self, metric: str) -> str:
        """Get best approach for a given metric."""
        if metric == "collision_rate" or metric == "mrc_trigger_rate":
            return min(self.approaches.keys(), 
                      key=lambda k: getattr(self.approaches[k], metric))
        else:
            return max(self.approaches.keys(), 
                      key=lambda k: getattr(self.approaches[k], metric))


def compute_metrics(episode_data: List[Dict]) -> ContingencyMetrics:
    """Compute metrics from raw episode data."""
    n = len(episode_data)
    if n == 0:
        return ContingencyMetrics()
    
    collisions = sum(1 for e in episode_data if e.get("collision", False))
    mrc_triggers = sum(1 for e in episode_data if e.get("mrc_triggered", False))
    off_roads = sum(1 for e in episode_data if e.get("off_road", False))
    successes = sum(1 for e in episode_data if e.get("success", False))
    
    completion_times = [e.get("completion_time", 0) for e in episode_data if e.get("completion_time")]
    speeds = [e.get("avg_speed", 0) for e in episode_data if e.get("avg_speed")]
    planning_times = []
    for e in episode_data:
        planning_times.extend(e.get("planning_times", []))
    
    correct_hypotheses = sum(1 for e in episode_data if e.get("correct_hypothesis", False))
    
    metrics = ContingencyMetrics(
        collision_rate=collisions / n,
        mrc_trigger_rate=mrc_triggers / n,
        off_road_rate=off_roads / n,
        success_rate=successes / n,
        avg_completion_time=np.mean(completion_times) if completion_times else 0,
        avg_speed=np.mean(speeds) if speeds else 0,
        avg_planning_time=np.mean(planning_times) * 1000 if planning_times else 0,  # ms
        max_planning_time=np.max(planning_times) * 1000 if planning_times else 0,
        correct_hypothesis_rate=correct_hypotheses / n,
    )
    
    return metrics

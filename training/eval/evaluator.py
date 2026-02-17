"""
Evaluation Framework for Autonomous Driving

Comprehensive evaluation including:
- Trajectory metrics (ADE, FDE)
- Safety metrics (collision, off-road)
- Comfort metrics (jerk, acceleration)
- Success rate metrics

Usage:
    from training.eval.evaluator import DrivingEvaluator
    
    evaluator = DrivingEvaluator(config)
    metrics = evaluator.evaluate(
        predictions=predicted_trajectories,
        ground_truth=expert_trajectories,
        scenarios=test_scenarios,
    )
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


# ============================================================================
# Core Metrics
# ============================================================================

@dataclass
class TrajectoryMetrics:
    """Trajectory prediction metrics."""
    ade: float = 0.0           # Average Displacement Error
    fde: float = 0.0           # Final Displacement Error
    miss_rate: float = 0.0      # Miss rate (FDE > threshold)
    
    # Additional trajectory metrics
    max_ade: float = 0.0       # Max ADE across timesteps
    trajectory_length: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "ade": self.ade,
            "fde": self.fde,
            "miss_rate": self.miss_rate,
            "max_ade": self.max_ade,
            "trajectory_length": self.trajectory_length,
        }


@dataclass
class SafetyMetrics:
    """Safety-related metrics."""
    collision_rate: float = 0.0     # Rate of collisions
    off_road_rate: float = 0.0       # Rate of off-road driving
    lane_violation_rate: float = 0.0 # Rate of lane violations
    min_distance_to_obstacle: float = 0.0
    safety_violations: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "collision_rate": self.collision_rate,
            "off_road_rate": self.off_road_rate,
            "lane_violation_rate": self.lane_violation_rate,
            "min_distance_to_obstacle": self.min_distance_to_obstacle,
            "safety_violations": self.safety_violations,
        }


@dataclass
class ComfortMetrics:
    """Comfort-related metrics."""
    mean_jerk: float = 0.0           # Mean jerk (comfort)
    max_jerk: float = 0.0           # Max jerk
    mean_acceleration: float = 0.0    # Mean acceleration
    max_acceleration: float = 0.0    # Max acceleration
    mean_curvature: float = 0.0      # Mean curvature rate
    steering_activity: float = 0.0    # Total steering effort
    
    def to_dict(self) -> Dict:
        return {
            "mean_jerk": self.mean_jerk,
            "max_jerk": self.max_jerk,
            "mean_acceleration": self.mean_acceleration,
            "max_acceleration": self.max_acceleration,
            "mean_curvature": self.mean_curvature,
            "steering_activity": self.steering_activity,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    trajectory: TrajectoryMetrics = None
    safety: SafetyMetrics = None
    comfort: ComfortMetrics = None
    
    success_rate: float = 0.0         # Overall success rate
    completion_rate: float = 0.0     # Episode completion rate
    
    num_samples: int = 0
    num_episodes: int = 0
    
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = {
            "trajectory": self.trajectory.to_dict() if self.trajectory else {},
            "safety": self.safety.to_dict() if self.safety else {},
            "comfort": self.comfort.to_dict() if self.comfort else {},
            "success_rate": self.success_rate,
            "completion_rate": self.completion_rate,
            "num_samples": self.num_samples,
            "num_episodes": self.num_episodes,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        return result
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 40,
            "Evaluation Results",
            "=" * 40,
            f"Samples: {self.num_samples}",
            f"Episodes: {self.num_episodes}",
            f"Success Rate: {self.success_rate:.2%}",
            "",
            "Trajectory Metrics:",
            f"  ADE: {self.trajectory.ade:.4f}",
            f"  FDE: {self.trajectory.fde:.4f}",
            f"  Miss Rate: {self.trajectory.miss_rate:.2%}",
            "",
            "Safety Metrics:",
            f"  Collision Rate: {self.safety.collision_rate:.2%}",
            f"  Off-Road Rate: {self.safety.off_road_rate:.2%}",
            "",
            "Comfort Metrics:",
            f"  Mean Jerk: {self.comfort.mean_jerk:.4f}",
            f"  Mean Accel: {self.comfort.mean_acceleration:.4f}",
            "=" * 40,
        ]
        return "\n".join(lines)


# ============================================================================
# Core Evaluation Functions
# ============================================================================

class TrajectoryEvaluator:
    """Evaluate trajectory predictions."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ade_threshold = self.config.get("ade_threshold", 2.0)  # meters
        self.fde_threshold = self.config.get("fde_threshold", 4.0)      # meters
    
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        masks: Optional[np.ndarray] = None,
    ) -> TrajectoryMetrics:
        """
        Evaluate trajectory predictions.
        
        Args:
            predictions: [B, T, 3] or [T, 3] predicted waypoints
            ground_truth: [B, T, 3] or [T, 3] ground truth waypoints
            masks: [B, T] or [T] valid waypoint mask
            
        Returns:
            TrajectoryMetrics
        """
        # Handle single batch
        if predictions.ndim == 2:
            predictions = predictions[np.newaxis, :, :]
            ground_truth = ground_truth[np.newaxis, :, :]
        
        B, T, D = predictions.shape
        assert D == 3, "Trajectory must be (x, y, heading)"
        
        # Handle masks
        if masks is None:
            masks = np.ones((B, T), dtype=bool)
        elif masks.ndim == 1:
            masks = masks[np.newaxis, :]
        
        # Compute displacement errors
        displacements = predictions - ground_truth  # [B, T, 3]
        distances = np.linalg.norm(displacements, axis=2)  # [B, T]
        
        # ADE (Average Displacement Error)
        ade = np.zeros(B)
        for b in range(B):
            valid = masks[b]
            if valid.sum() > 0:
                ade[b] = distances[b][valid].mean()
        
        # FDE (Final Displacement Error)
        fde = np.zeros(B)
        for b in range(B):
            valid = masks[b]
            if valid.sum() > 0:
                last_valid = valid[::-1].argmax()
                fde[b] = distances[b][last_valid]
        
        # Max ADE
        max_ade = np.zeros(B)
        for b in range(B):
            valid = masks[b]
            if valid.sum() > 0:
                max_ade[b] = distances[b][valid].max()
        
        # Miss rate (FDE > threshold)
        miss_rate = (fde > self.fde_threshold).mean()
        
        # Trajectory length
        trajectory_length = np.zeros(B)
        for b in range(B):
            diffs = np.diff(ground_truth[b], axis=0)
            trajectory_length[b] = np.linalg.norm(diffs, axis=1).sum()
        
        return TrajectoryMetrics(
            ade=float(ade.mean()),
            fde=float(fde.mean()),
            miss_rate=float(miss_rate),
            max_ade=float(max_ade.mean()),
            trajectory_length=float(trajectory_length.mean()),
        )


class SafetyEvaluator:
    """Evaluate safety metrics."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.collision_threshold = self.config.get("collision_threshold", 0.5)  # meters
        self.off_road_threshold = self.config.get("off_road_threshold", 2.0)    # meters
    
    def evaluate(
        self,
        trajectories: np.ndarray,
        obstacles: List[List[Dict]] = None,
        road_boundaries: np.ndarray = None,
    ) -> SafetyMetrics:
        """
        Evaluate safety metrics.
        
        Args:
            trajectories: [B, T, 3] predicted trajectories
            obstacles: List of obstacle lists per batch element
            road_boundaries: [B, T, 2] road boundary constraints
            
        Returns:
            SafetyMetrics
        """
        B, T, _ = trajectories.shape
        
        collision_count = 0
        off_road_count = 0
        lane_violation_count = 0
        min_distances = []
        
        for b in range(B):
            traj = trajectories[b]
            
            # Check collisions
            if obstacles and b < len(obstacles):
                for obs in obstacles[b]:
                    dist = self._distance_to_obstacle(traj, obs)
                    min_distances.append(dist)
                    
                    if dist < self.collision_threshold:
                        collision_count += 1
            
            # Check off-road
            if road_boundaries is not None:
                if self._is_off_road(traj[b], road_boundaries[b]):
                    off_road_count += 1
            
            # Check lane violations
            if road_boundaries is not None:
                if self._crosses_lane_boundary(traj[b], road_boundaries[b]):
                    lane_violation_count += 1
        
        min_dist = min(min_distances) if min_distances else 0.0
        
        return SafetyMetrics(
            collision_rate=collision_count / (B * T) if B * T > 0 else 0.0,
            off_road_rate=off_road_count / (B * T) if B * T > 0 else 0.0,
            lane_violation_rate=lane_violation_count / (B * T) if B * T > 0 else 0.0,
            min_distance_to_obstacle=min_dist,
            safety_violations=collision_count + off_road_count + lane_violation_count,
        )
    
    def _distance_to_obstacle(self, trajectory: np.ndarray, obstacle: Dict) -> float:
        """Compute minimum distance from trajectory to obstacle."""
        obs_pos = np.array([obstacle.get("x", 0), obstacle.get("y", 0)])
        
        min_dist = float("inf")
        for wp in trajectory:
            dist = np.linalg.norm(wp[:2] - obs_pos)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _is_off_road(self, waypoint: np.ndarray, boundary: np.ndarray) -> bool:
        """Check if waypoint is off road."""
        # Simplified: check if within road boundaries
        return False  # Would implement actual check
    
    def _crosses_lane_boundary(self, waypoint: np.ndarray, boundary: np.ndarray) -> bool:
        """Check if waypoint crosses lane boundary."""
        return False  # Would implement


class ComfortEvaluator:
    """Evaluate comfort metrics."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.dt = self.config.get("dt", 0.1)  # Time step
    
    def evaluate(
        self,
        trajectories: np.ndarray,
        velocities: np.ndarray = None,
    ) -> ComfortMetrics:
        """
        Evaluate comfort metrics.
        
        Args:
            trajectories: [B, T, 3] trajectories (x, y, heading)
            velocities: [B, T, 2] or None (vx, vy)
            
        Returns:
            ComfortMetrics
        """
        B, T, _ = trajectories.shape
        
        # Compute accelerations
        accelerations = np.zeros(B)
        max_accel = np.zeros(B)
        
        for b in range(B):
            traj = trajectories[b]
            
            # Compute velocity (finite difference)
            if velocities is not None:
                vel = velocities[b]
            else:
                vel = np.diff(traj[:, :2], axis=0) / self.dt
            
            # Compute acceleration
            accel = np.diff(vel, axis=0) / self.dt
            
            accelerations[b] = np.linalg.norm(accel, axis=1).mean()
            max_accel[b] = np.linalg.norm(accel, axis=1).max()
        
        # Compute jerk (derivative of acceleration)
        jerks = np.zeros(B)
        max_jerk = np.zeros(B)
        
        for b in range(B):
            traj = trajectories[b]
            
            # Velocity
            vel = np.diff(traj[:, :2], axis=0) / self.dt
            
            # Acceleration
            accel = np.diff(vel, axis=0) / self.dt
            
            # Jerk
            jerk = np.diff(accel, axis=0) / self.dt
            
            jerks[b] = np.linalg.norm(jerk, axis=1).mean()
            max_jerk[b] = np.linalg.norm(jerk, axis=1).max()
        
        # Compute curvature
        curvatures = np.zeros(B)
        for b in range(B):
            traj = trajectories[b]
            
            # Compute curvature using cross product
            dx = np.diff(traj[:, 0])
            dy = np.diff(traj[:, 1])
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
            curvatures[b] = curvature.mean()
        
        # Compute steering activity
        steering_activity = np.zeros(B)
        for b in range(B):
            traj = trajectories[b]
            
            # Heading change rate
            dheading = np.diff(traj[:, 2])
            steering_activity[b] = np.abs(dheading).sum()
        
        return ComfortMetrics(
            mean_jerk=float(jerks.mean()),
            max_jerk=float(max_jerk.mean()),
            mean_acceleration=float(accelerations.mean()),
            max_acceleration=float(max_accel.mean()),
            mean_curvature=float(curvatures.mean()),
            steering_activity=float(steering_activity.mean()),
        )


# ============================================================================
# Complete Evaluator
# ============================================================================

class DrivingEvaluator:
    """
    Complete evaluation framework for autonomous driving.
    
    Combines trajectory, safety, and comfort metrics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize sub-evaluators
        self.trajectory_eval = TrajectoryEvaluator(config)
        self.safety_eval = SafetyEvaluator(config)
        self.comfort_eval = ComfortEvaluator(config)
        
        # Metrics weights for overall score
        self.weights = self.config.get("weights", {
            "trajectory": 0.4,
            "safety": 0.4,
            "comfort": 0.2,
        })
    
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        obstacles: List[List[Dict]] = None,
        road_boundaries: np.ndarray = None,
        masks: Optional[np.ndarray] = None,
        velocities: np.ndarray = None,
        success_mask: np.ndarray = None,
        completion_mask: np.ndarray = None,
        metadata: Optional[Dict] = None,
    ) -> EvaluationResult:
        """
        Complete evaluation.
        
        Args:
            predictions: [B, T, 3] predicted trajectories
            ground_truth: [B, T, 3] ground truth
            obstacles: Obstacle lists per batch
            road_boundaries: Road boundaries
            masks: Valid waypoint mask
            velocities: Velocity data for comfort
            success_mask: [B] success per episode
            completion_mask: [B] completion per episode
            
        Returns:
            EvaluationResult
        """
        B = predictions.shape[0]
        
        # Trajectory metrics
        trajectory = self.trajectory_eval.evaluate(
            predictions, ground_truth, masks
        )
        
        # Safety metrics
        safety = self.safety_eval.evaluate(
            predictions, obstacles, road_boundaries
        )
        
        # Comfort metrics
        comfort = self.comfort_eval.evaluate(
            predictions, velocities
        )
        
        # Success and completion rates
        success_rate = float(success_mask.mean()) if success_mask is not None else 1.0
        completion_rate = float(completion_mask.mean()) if completion_mask is not None else 1.0
        
        # Overall score
        overall_score = (
            self.weights["trajectory"] * (1.0 - min(trajectory.ade / 10.0, 1.0)) +
            self.weights["safety"] * (1.0 - safety.collision_rate) +
            self.weights["comfort"] * (1.0 - min(comfort.mean_jerk / 10.0, 1.0))
        
        return EvaluationResult(
            trajectory=trajectory,
            safety=safety,
            comfort=comfort,
            success_rate=success_rate,
            completion_rate=completion_rate,
            num_samples=B,
            num_episodes=B,
            timestamp=time.time(),
            metadata=metadata or {},
        )
    
    def evaluate_episode(
        self,
        predicted_trajectory: np.ndarray,
        expert_trajectory: np.ndarray,
        obstacles: List[Dict] = None,
        road_boundary: np.ndarray = None,
        is_success: bool = True,
        is_completed: bool = True,
    ) -> EvaluationResult:
        """Evaluate a single episode."""
        return self.evaluate(
            predictions=predicted_trajectory[np.newaxis, :, :],
            ground_truth=expert_trajectory[np.newaxis, :, :],
            obstacles=[obstacles] if obstacles else None,
            road_boundaries=road_boundary[np.newaxis, :, :] if road_boundary is not None else None,
            success_mask=np.array([is_success]),
            completion_mask=np.array([is_completed]),
        )
    
    def compare_models(
        self,
        model_results: Dict[str, EvaluationResult],
    ) -> pd.DataFrame:
        """
        Compare evaluation results across models.
        
        Args:
            model_results: Dict of model_name -> EvaluationResult
            
        Returns:
            DataFrame with comparison table
        """
        import pandas as pd
        
        rows = []
        for model_name, result in model_results.items():
            row = {
                "model": model_name,
                "ADE": result.trajectory.ade,
                "FDE": result.trajectory.fde,
                "Miss Rate": result.trajectory.miss_rate,
                "Collision Rate": result.safety.collision_rate,
                "Off-Road Rate": result.safety.off_road_rate,
                "Mean Jerk": result.comfort.mean_jerk,
                "Success Rate": result.success_rate,
                "Completion Rate": result.completion_rate,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by FDE
        df = df.sort_values("FDE")
        
        return df
    
    def generate_report(
        self,
        result: EvaluationResult,
        output_path: Path,
        format: str = "markdown",
    ):
        """
        Generate evaluation report.
        
        Args:
            result: EvaluationResult
            output_path: Path to save report
            format: "markdown" or "json"
        """
        if format == "json":
            output_path.write_text(json.dumps(result.to_dict(), indent=2))
        
        elif format == "markdown":
            report = result.summary()
            output_path.write_text(report)
        
        else:
            raise ValueError(f"Unknown format: {format}")


# ============================================================================
# Scenario-Based Evaluation
# ============================================================================

class ScenarioEvaluator:
    """
    Evaluate performance on specific scenarios.
    
    Scenarios:
    - Highway driving
    - Urban intersection
    - Pedestrian crossing
    - Lane change
    - Emergency braking
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Scenario definitions
        self.scenarios = {
            "highway": {
                "speed_range": (20, 35),  # m/s
                "lane_count": (3, 6),
                "obstacle_density": "low",
            },
            "urban": {
                "speed_range": (5, 15),
                "lane_count": (1, 3),
                "obstacle_density": "high",
            },
            "intersection": {
                "speed_range": (0, 10),
                "traffic_light": True,
                "has_pedestrians": True,
            },
            "pedestrian": {
                "speed_range": (0, 15),
                "pedestrian_present": True,
                "crossing_intent": True,
            },
            "lane_change": {
                "speed_range": (10, 25),
                "lane_change_required": True,
            },
            "emergency": {
                "speed_range": (15, 30),
                "emergency_vehicle": False,
                "obstacle_ahead": True,
            },
        }
    
    def classify_scenario(
        self,
        state: Dict,
        objects: List[Dict],
    ) -> str:
        """
        Classify the current driving scenario.
        
        Returns:
            Scenario name or "unknown"
        """
        speed = state.get("speed", 0)
        
        # Check each scenario
        for name, criteria in self.scenarios.items():
            speed_range = criteria.get("speed_range", (0, float("inf")))
            
            if not (speed_range[0] <= speed <= speed_range[1]):
                continue
            
            # Additional checks
            if name == "urban":
                if len(objects) > 5:
                    return "urban"
            
            elif name == "intersection":
                if state.get("is_intersection", False):
                    return "intersection"
            
            elif name == "pedestrian":
                peds = [o for o in objects if o.get("type") == "pedestrian"]
                if peds:
                    return "pedestrian"
            
            elif name == "highway":
                if speed > 20 and len(objects) < 5:
                    return "highway"
            
            elif name == "lane_change":
                if state.get("lane_change_required", False):
                    return "lane_change"
        
        return "unknown"
    
    def evaluate_by_scenario(
        self,
        trajectories: np.ndarray,
        states: List[Dict],
        objects_list: List[List[Dict]],
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate trajectories grouped by scenario.
        
        Returns:
            Dict of scenario_name -> EvaluationResult
        """
        results = {}
        
        # Classify each sample
        scenario_samples = {}
        for i, (state, objects) in enumerate(zip(states, objects_list)):
            scenario = self.classify_scenario(state, objects)
            
            if scenario not in scenario_samples:
                scenario_samples[scenario] = []
            
            scenario_samples[scenario].append(i)
        
        # Evaluate each scenario
        for scenario, indices in scenario_samples.items():
            if len(indices) < 2:
                continue  # Need at least 2 samples
            
            scenario_trajs = trajectories[indices]
            
            # Would need ground truth for full evaluation
            # Simplified: just count samples
            results[scenario] = EvaluationResult(
                num_samples=len(indices),
                num_episodes=len(indices),
                success_rate=1.0,
                completion_rate=1.0,
                metadata={"scenario": scenario},
            )
        
        return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create evaluator
    evaluator = DrivingEvaluator()
    
    # Simulate predictions and ground truth
    B, T = 100, 20
    
    predictions = np.random.randn(B, T, 3) * 2
    ground_truth = np.random.randn(B, T, 3) * 2
    
    # Add some structure
    predictions[:, 0] = ground_truth[:, 0]  # Same start
    predictions = np.cumsum(predictions, axis=1) * 0.1
    ground_truth = np.cumsum(ground_truth, axis=1) * 0.1
    
    # Evaluate
    result = evaluator.evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        success_mask=np.random.rand(B) > 0.1,
        completion_mask=np.random.rand(B) > 0.05,
    )
    
    # Print summary
    print(result.summary())
    
    # Save report
    report_path = Path("out/eval/results.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.generate_report(result, report_path)
    
    print(f"\nReport saved to: {report_path}")

#!/usr/bin/env python3
"""
Scenario Execution Monitor for CARLA ScenarioRunner.

Monitors real-time execution metrics during scenario runs:
- Episode progress and timing
- Actor states (position, velocity)
- Collision detection
- Traffic light states
- Route completion percentage
- Saves metrics to JSON for downstream analysis.

This bridges scenario execution with metrics collection for the evaluation pipeline.
"""

import argparse
import json
import time
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import random
import math


@dataclass
class ActorState:
    """State of a single actor at a timestep."""
    actor_id: str
    role_name: str
    transform: dict  # {"x": float, "y": float, "z": float, "pitch": float, "yaw": float, "roll": float}
    velocity: dict  # {"x": float, "y": float, "z": float}
    acceleration: dict  # {"x": float, "y": float, "z": float}


@dataclass
class CollisionEvent:
    """Collision event during scenario execution."""
    timestamp: float
    actor_id: str
    other_actor_id: str
    position: dict  # {"x": float, "y": float, "z": float}
    impulse: float


@dataclass
class TrafficLightEvent:
    """Traffic light state change event."""
    timestamp: float
    actor_id: str
    state: str  # "red", "yellow", "green", "off"


@dataclass
class RouteProgress:
    """Route completion progress."""
    current_route_index: int
    total_routes: int
    distance_traveled: float
    route_length: float
    completion_percent: float


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution frame/timestep."""
    timestamp: float
    sim_time: float  # Simulation time in seconds
    actor_states: list = field(default_factory=list)
    collisions: list = field(default_factory=list)
    traffic_light_events: list = field(default_factory=list)
    route_progress: Optional[RouteProgress] = None
    ego_vehicle_speed: float = 0.0
    ego_vehicle_acceleration: float = 0.0
    distance_to_goal: float = 0.0


@dataclass
class ScenarioExecutionSummary:
    """Summary of full scenario execution."""
    scenario_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    failure_reason: Optional[str] = None
    final_route_progress: Optional[RouteProgress] = None
    total_collisions: int = 0
    total_infractions: int = 0
    max_ego_speed: float = 0.0
    mean_ego_speed: float = 0.0
    execution_metrics: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "scenario_name": self.scenario_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "total_collisions": self.total_collisions,
            "total_infractions": self.total_infractions,
            "max_ego_speed": self.max_ego_speed,
            "mean_ego_speed": self.mean_ego_speed,
        }
        if self.final_route_progress:
            data["final_route_progress"] = asdict(self.final_route_progress)
        return data


class ScenarioExecutionMonitor:
    """Monitors scenario execution in real-time."""
    
    def __init__(self, scenario_name: str = "default", output_dir: Optional[str] = None):
        self.scenario_name = scenario_name
        self.output_dir = Path(output_dir) if output_dir else Path("out/scenario_monitor")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Execution state
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metrics_history: list = []
        self.collisions: list = []
        self.traffic_light_events: list = []
        self.max_ego_speed: float = 0.0
        self.ego_speeds: list = []
        
        # Route tracking
        self.route_length: float = 0.0
        self.distance_traveled: float = 0.0
        self.current_route_index: int = 0
        self.total_routes: int = 1
    
    def start(self, route_length: float = 100.0, total_routes: int = 1):
        """Start monitoring a scenario execution."""
        self.start_time = time.time()
        self.route_length = route_length
        self.total_routes = total_routes
        self.distance_traveled = 0.0
        self.current_route_index = 0
        self.collisions = []
        self.traffic_light_events = []
        self.metrics_history = []
        self.max_ego_speed = 0.0
        self.ego_speeds = []
    
    def _compute_distance(self, pos1: dict, pos2: dict) -> float:
        """Compute Euclidean distance between two positions."""
        dx = pos1.get("x", 0) - pos2.get("x", 0)
        dy = pos1.get("y", 0) - pos2.get("y", 0)
        dz = pos1.get("z", 0) - pos2.get("z", 0)
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _compute_speed(self, velocity: dict) -> float:
        """Compute speed from velocity vector."""
        vx = velocity.get("x", 0)
        vy = velocity.get("y", 0)
        vz = velocity.get("z", 0)
        return math.sqrt(vx*vx + vy*vy + vz*vz)
    
    def update(
        self,
        sim_time: float,
        ego_transform: Optional[dict] = None,
        ego_velocity: Optional[dict] = None,
        goal_position: Optional[dict] = None,
        actor_states: Optional[list] = None,
    ):
        """Update monitor with current execution state."""
        if self.start_time is None:
            self.start(route_length=100.0)
        
        # Compute ego vehicle metrics
        ego_speed = 0.0
        ego_acceleration = 0.0
        distance_to_goal = 0.0
        
        if ego_velocity:
            ego_speed = self._compute_speed(ego_velocity)
            self.max_ego_speed = max(self.max_ego_speed, ego_speed)
            self.ego_speeds.append(ego_speed)
        
        if ego_transform and goal_position:
            distance_to_goal = self._compute_distance(ego_transform, goal_position)
            # Simple distance traveled estimate
            self.distance_traveled += ego_speed * 0.1  # Assuming ~0.1s per update
        
        # Route progress
        completion = 0.0
        if self.route_length > 0:
            completion = min(100.0, (self.distance_traveled / self.route_length) * 100.0)
        route_progress = RouteProgress(
            current_route_index=self.current_route_index,
            total_routes=self.total_routes,
            distance_traveled=self.distance_traveled,
            route_length=self.route_length,
            completion_percent=completion,
        )
        
        # Create execution metrics frame
        metrics = ExecutionMetrics(
            timestamp=time.time(),
            sim_time=sim_time,
            actor_states=actor_states or [],
            collisions=list(self.collisions),
            traffic_light_events=list(self.traffic_light_events),
            route_progress=route_progress,
            ego_vehicle_speed=ego_speed,
            ego_vehicle_acceleration=ego_acceleration,
            distance_to_goal=distance_to_goal,
        )
        
        self.metrics_history.append(asdict(metrics))
        
        return metrics
    
    def add_collision(self, actor_id: str, other_actor_id: str, position: dict, impulse: float = 0.0):
        """Record a collision event."""
        event = CollisionEvent(
            timestamp=time.time() - (self.start_time or time.time()),
            actor_id=actor_id,
            other_actor_id=other_actor_id,
            position=position,
            impulse=impulse,
        )
        self.collisions.append(asdict(event))
    
    def add_traffic_light_event(self, actor_id: str, state: str):
        """Record a traffic light state change."""
        event = TrafficLightEvent(
            timestamp=time.time() - (self.start_time or time.time()),
            actor_id=actor_id,
            state=state,
        )
        self.traffic_light_events.append(asdict(event))
    
    def stop(self, success: bool = True, failure_reason: Optional[str] = None) -> ScenarioExecutionSummary:
        """Stop monitoring and get execution summary."""
        self.end_time = time.time()
        
        duration = 0.0
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        
        mean_ego_speed = 0.0
        if self.ego_speeds:
            mean_ego_speed = sum(self.ego_speeds) / len(self.ego_speeds)
        
        completion = 0.0
        if self.route_length > 0:
            completion = min(100.0, (self.distance_traveled / self.route_length) * 100.0)
        
        final_route_progress = RouteProgress(
            current_route_index=self.current_route_index,
            total_routes=self.total_routes,
            distance_traveled=self.distance_traveled,
            route_length=self.route_length,
            completion_percent=completion,
        )
        
        summary = ScenarioExecutionSummary(
            scenario_name=self.scenario_name,
            start_time=self.start_time or time.time(),
            end_time=self.end_time or time.time(),
            duration=duration,
            success=success,
            failure_reason=failure_reason,
            final_route_progress=final_route_progress,
            total_collisions=len(self.collisions),
            total_infractions=len(self.traffic_light_events),
            max_ego_speed=self.max_ego_speed,
            mean_ego_speed=mean_ego_speed,
            execution_metrics=self.metrics_history,
        )
        
        return summary
    
    def save_summary(self, summary: ScenarioExecutionSummary) -> str:
        """Save execution summary to JSON."""
        output_file = self.output_dir / f"{self.scenario_name}_summary.json"
        with open(output_file, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        return str(output_file)
    
    def save_metrics_history(self) -> str:
        """Save full metrics history to JSON."""
        output_file = self.output_dir / f"{self.scenario_name}_metrics.json"
        with open(output_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        return str(output_file)


def compute_distance_between_waypoints(waypoints: list) -> float:
    """Compute total distance between consecutive waypoints."""
    if len(waypoints) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(waypoints) - 1):
        total += compute_distance_simple(waypoints[i], waypoints[i + 1])
    return total


def compute_distance_simple(p1: dict, p2: dict) -> float:
    """Simple Euclidean distance."""
    dx = p2.get("x", 0) - p1.get("x", 0)
    dy = p2.get("y", 0) - p1.get("y", 0)
    return math.sqrt(dx*dx + dy*dy)


def generate_mock_execution(
    scenario_name: str = "straight_clear",
    duration: float = 10.0,
    route_length: float = 100.0,
) -> ScenarioExecutionSummary:
    """Generate mock scenario execution for testing."""
    monitor = ScenarioExecutionMonitor(scenario_name)
    monitor.start(route_length=route_length)
    
    # Simulate 10 seconds of execution
    sim_time = 0.0
    for step in range(int(duration * 10)):  # 10 Hz
        sim_time += 0.1
        
        # Mock ego vehicle moving forward
        ego_transform = {
            "x": sim_time * 3.0,  # Moving at ~3 m/s
            "y": 0.0,
            "z": 0.0,
        }
        ego_velocity = {
            "x": 3.0 + random.uniform(-0.1, 0.1),
            "y": 0.0,
            "z": 0.0,
        }
        
        # Goal at end of route
        goal_position = {"x": route_length, "y": 0.0, "z": 0.0}
        
        # Occasional collision at ~30% of execution
        if random.random() < 0.03 and sim_time > 2.0:
            monitor.add_collision(
                actor_id="hero",
                other_actor_id="vehicle_1",
                position=ego_transform,
                impulse=random.uniform(1.0, 5.0),
            )
        
        monitor.update(
            sim_time=sim_time,
            ego_transform=ego_transform,
            ego_velocity=ego_velocity,
            goal_position=goal_position,
        )
    
    summary = monitor.stop(success=True)
    
    # Save outputs
    monitor.save_summary(summary)
    monitor.save_metrics_history()
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Scenario Execution Monitor for CARLA ScenarioRunner"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # monitor subcommand
    monitor_parser = subparsers.add_parser("monitor", help="Monitor scenario execution")
    monitor_parser.add_argument("--scenario", type=str, default="default")
    monitor_parser.add_argument("--route-length", type=float, default=100.0)
    monitor_parser.add_argument("--duration", type=float, default=10.0)
    monitor_parser.add_argument("--output-dir", type=str, default=None)
    
    # generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate mock execution")
    gen_parser.add_argument("--scenario", type=str, default="straight_clear")
    gen_parser.add_argument("--duration", type=float, default=10.0)
    gen_parser.add_argument("--route-length", type=float, default=100.0)
    gen_parser.add_argument("--output-dir", type=str, default=None)
    
    # stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--metrics-dir", type=str, required=True)
    
    # smoke subcommand
    subparsers.add_parser("smoke", help="Run smoke test")
    
    args = parser.parse_args()
    
    if args.command == "monitor":
        monitor = ScenarioExecutionMonitor(
            scenario_name=args.scenario,
            output_dir=args.output_dir,
        )
        monitor.start(route_length=args.route_length)
        
        # Simulate some updates
        for i in range(10):
            monitor.update(
                sim_time=i * 0.1,
                ego_transform={"x": i * 0.3, "y": 0, "z": 0},
                ego_velocity={"x": 3.0, "y": 0, "z": 0},
                goal_position={"x": args.route_length, "y": 0, "z": 0},
            )
        
        summary = monitor.stop(success=True)
        output_file = monitor.save_summary(summary)
        print(f"Summary saved to: {output_file}")
        print(f"  Success: {summary.success}")
        print(f"  Duration: {summary.duration:.2f}s")
        print(f"  Collisions: {summary.total_collisions}")
        print(f"  Max speed: {summary.max_ego_speed:.2f} m/s")
    
    elif args.command == "generate":
        summary = generate_mock_execution(
            scenario_name=args.scenario,
            duration=args.duration,
            route_length=args.route_length,
        )
        print(f"Generated execution for: {args.scenario}")
        print(f"  Success: {summary.success}")
        print(f"  Duration: {summary.duration:.2f}s")
        print(f"  Collisions: {summary.total_collisions}")
        print(f"  Max speed: {summary.max_ego_speed:.2f} m/s")
        print(f"  Mean speed: {summary.mean_ego_speed:.2f} m/s")
    
    elif args.command == "stats":
        metrics_dir = Path(args.metrics_dir)
        summary_files = list(metrics_dir.glob("*_summary.json"))
        
        if not summary_files:
            print(f"No summary files found in {metrics_dir}")
            return
        
        total_duration = 0.0
        total_collisions = 0
        total_success = 0
        max_speeds = []
        
        for f in summary_files:
            with open(f) as fp:
                data = json.load(fp)
                total_duration += data.get("duration", 0)
                total_collisions += data.get("total_collisions", 0)
                if data.get("success"):
                    total_success += 1
                max_speeds.append(data.get("max_ego_speed", 0))
        
        n = len(summary_files)
        print(f"Statistics for {n} scenario(s):")
        print(f"  Mean duration: {total_duration / n:.2f}s")
        print(f"  Mean collisions: {total_collisions / n:.2f}")
        print(f"  Success rate: {total_success / n * 100:.1f}%")
        print(f"  Max speed (mean): {sum(max_speeds) / n:.2f} m/s")
    
    elif args.command == "smoke" or args.command is None:
        print("Running smoke test...")
        
        # Test 1: Basic monitor
        monitor = ScenarioExecutionMonitor("smoke_test")
        monitor.start(route_length=100.0)
        for i in range(10):
            monitor.update(
                sim_time=i * 0.1,
                ego_transform={"x": i * 0.3, "y": 0, "z": 0},
                ego_velocity={"x": 3.0, "y": 0, "z": 0},
                goal_position={"x": 100.0, "y": 0, "z": 0},
            )
        summary = monitor.stop(success=True)
        print(f"  [1] Basic monitor: OK (duration={summary.duration:.2f}s)")
        
        # Test 2: Generate mock execution
        summary2 = generate_mock_execution("straight_clear", duration=5.0, route_length=50.0)
        print(f"  [2] Mock execution: OK")
        print(f"      Success: {summary2.success}")
        print(f"      Collisions: {summary2.total_collisions}")
        
        # Test 3: Multiple updates with collisions
        monitor3 = ScenarioExecutionMonitor("collision_test")
        monitor3.start(route_length=100.0)
        for i in range(20):
            monitor3.update(
                sim_time=i * 0.1,
                ego_transform={"x": i * 0.3, "y": i * 0.01, "z": 0},
                ego_velocity={"x": 3.0 + random.uniform(-0.5, 0.5), "y": 0, "z": 0},
                goal_position={"x": 100.0, "y": 0, "z": 0},
            )
            if i == 10:
                monitor3.add_collision(
                    actor_id="hero",
                    other_actor_id="vehicle_1",
                    position={"x": 3.0, "y": 0, "z": 0},
                    impulse=2.5,
                )
        summary3 = monitor3.stop(success=False, failure_reason="collision")
        print(f"  [3] Collision tracking: OK")
        print(f"      Collisions: {summary3.total_collisions}")
        
        print("\nSmoke test: ALL PASSED")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
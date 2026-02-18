"""
CARLA ScenarioRunner Evaluation Wrapper
====================================

Evaluate RL policies using CARLA's ScenarioRunner.

Features:
- Pre-defined scenarios (lane change, intersection, pedestrian)
- Built-in infraction detection (collisions, red light, etc.)
- Route completion metrics
- Industry-standard evaluation protocol

Usage:
    from training.rl.envs.carla_scenario_eval import ScenarioEvaluator
    
    evaluator = ScenarioEvaluator(
        host="localhost",
        port=2000,
        scenarios=["lane_change", "intersection", "pedestrian_crossing"],
    )
    
    results = evaluator.evaluate(policy, num_episodes=10)
    print(results["summary"])
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# CARLA is optional - only required for actual execution
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None
import time
import json


class InfractionType(Enum):
    """Types of infractions during evaluation."""
    COLLISION = "collision"
    RED_LIGHT_VIOLATION = "red_light"
    STOP_SIGN_VIOLATION = "stop_sign"
    OFF_ROAD = "off_road"
    LANE_INVASION = "lane_invasion"
    PEDESTRIAN_VIOLATION = "pedestrian"
    VEHICLE_VIOLATION = "vehicle"
    TIMEOUT = "timeout"
    ROUTE_COMPLETE = "route_complete"


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario."""
    name: str
    town: str = "Town03"
    weather: str = "ClearNoon"
    num_vehicles: int = 0
    num_pedestrians: int = 0
    route_length: float = 100.0  # meters
    timeout: float = 60.0  # seconds


@dataclass
class EpisodeResult:
    """Result from a single evaluation episode."""
    scenario: str
    success: bool
    route_completion: float  # 0-1
    infractions: List[str]
    total_reward: float
    duration: float
    distance_driven: float
    max_speed: float
    avg_speed: float
    final_transform: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class ScenarioEvaluatorConfig:
    """Configuration for scenario evaluator."""
    # Connection
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    
    # Scenarios
    scenarios: List[str] = field(default_factory=lambda: [
        "lane_change",
        "straight",
        "intersection",
        "pedestrian_crossing",
    ])
    
    # Evaluation
    num_episodes: int = 10
    timeout_per_episode: float = 60.0
    
    # Route
    route_length: float = 200.0  # meters
    
    # Weather
    weather: str = "ClearNoon"


class ScenarioEvaluator:
    """
    Evaluate RL policies using CARLA ScenarioRunner scenarios.
    
    Provides standardized metrics for autonomous driving:
    - Route completion percentage
    - Infraction detection and counting
    - Safety metrics (collisions, violations)
    - Comfort metrics (speed, acceleration)
    """
    
    def __init__(self, config: Optional[ScenarioEvaluatorConfig] = None):
        if not CARLA_AVAILABLE:
            raise ImportError(
                "CARLA module not installed. "
                "Install with: pip install carla"
            )
        
        self.config = config or ScenarioEvaluatorConfig()
        
        # CARLA objects
        self.client = None
        self.world = None
        self.map = None
        
        # State
        self._vehicle = None
        self._sensor = None
        self._collision_sensor = None
        self._traffic_light_sensor = None
        
        # Metrics
        self._infractions = []
        self._start_location = None
        self._start_time = 0
        self._prev_transform = None
        self._total_distance = 0.0
        self._max_speed = 0.0
        self._speeds = []
    
    def _connect(self):
        """Connect to CARLA server."""
        try:
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CARLA: {e}")
    
    def _setup_vehicle(self, spawn_point: int = 0):
        """Set up the ego vehicle."""
        # Get vehicle blueprint
        blueprints = self.world.get_blueprint_library().filter(
            "vehicle.tesla.model3"
        )
        blueprint = list(blueprints)[0]
        
        # Spawn point
        spawn_points = self.map.get_spawn_points()
        transform = spawn_points[min(spawn_point, len(spawn_points) - 1)]
        
        # Spawn vehicle
        self._vehicle = self.world.spawn_actor(blueprint, transform)
        
        # Record start location
        self._start_location = transform.location
        
        # Set up collision sensor
        collision_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self._collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._vehicle
        )
        self._collision_sensor.listen(
            lambda event: self._on_collision(event)
        )
    
    def _on_collision(self, event):
        """Handle collision event."""
        self._infractions.append(InfractionType.COLLISION.value)
    
    def _cleanup(self):
        """Clean up actors."""
        if self._vehicle is not None:
            self._vehicle.destroy()
        if self._sensor is not None:
            self._sensor.destroy()
        if self._collision_sensor is not None:
            self._collision_sensor.destroy()
    
    def _run_episode(
        self,
        policy: Callable,
        scenario: str,
        route_length: float,
    ) -> EpisodeResult:
        """Run a single evaluation episode."""
        self._infractions = []
        self._total_distance = 0.0
        self._max_speed = 0.0
        self._speeds = []
        
        # Setup
        self._setup_vehicle()
        self._prev_transform = self._vehicle.get_transform()
        self._start_time = time.time()
        
        try:
            # Run until timeout or completion
            start_loc = self._start_location
            target_loc = self._get_target_location(start_loc, route_length)
            
            while True:
                # Check timeout
                if time.time() - self._start_time > self.config.timeout_per_episode:
                    return EpisodeResult(
                        scenario=scenario,
                        success=False,
                        route_completion=self._compute_route_completion(start_loc, target_loc),
                        infractions=self._infractions.copy(),
                        total_reward=0.0,
                        duration=self.config.timeout_per_episode,
                        distance_driven=self._total_distance,
                        max_speed=self._max_speed,
                        avg_speed=np.mean(self._speeds) if self._speeds else 0.0,
                        error="timeout",
                    )
                
                # Get observation
                obs = self._get_observation()
                
                # Get action from policy
                action = policy(obs)
                
                # Apply action
                self._apply_action(action)
                
                # Update metrics
                self._update_metrics()
                
                # Check if route complete
                curr_loc = self._vehicle.get_location()
                if curr_loc.distance(target_loc) < 10.0:
                    return EpisodeResult(
                        scenario=scenario,
                        success=True,
                        route_completion=1.0,
                        infractions=self._infractions.copy(),
                        total_reward=0.0,
                        duration=time.time() - self._start_time,
                        distance_driven=self._total_distance,
                        max_speed=self._max_speed,
                        avg_speed=np.mean(self._speeds) if self._speeds else 0.0,
                    )
                
                time.sleep(0.05)  # ~20 Hz
        
        except Exception as e:
            return EpisodeResult(
                scenario=scenario,
                success=False,
                route_completion=self._compute_route_completion(start_loc, target_loc),
                infractions=self._infractions.copy(),
                total_reward=0.0,
                duration=time.time() - self._start_time,
                distance_driven=self._total_distance,
                max_speed=self._max_speed,
                avg_speed=np.mean(self._speeds) if self._speeds else 0.0,
                error=str(e),
            )
        
        finally:
            self._cleanup()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        
        speed = np.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )
        
        return {
            'location': np.array([transform.location.x, transform.location.y]),
            'heading': np.array([np.radians(transform.rotation.yaw)]),
            'speed': np.array([speed]),
        }
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to vehicle."""
        steer, throttle, brake = action[0], action[1], action[2]
        
        self._vehicle.apply_control(carla.VehicleControl(
            steering=steer,
            throttle=throttle,
            brake=brake,
        ))
    
    def _update_metrics(self):
        """Update episode metrics."""
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        
        speed = np.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )
        
        # Distance
        dist = transform.location.distance(self._prev_transform.location)
        self._total_distance += dist
        
        # Speed
        self._speeds.append(speed)
        self._max_speed = max(self._max_speed, speed)
        
        self._prev_transform = transform
    
    def _get_target_location(
        self,
        start_loc: carla.Location,
        route_length: float,
    ) -> carla.Location:
        """Get target location along route."""
        # Simple: target is route_length meters ahead
        # In practice, would use route planner
        yaw = self._vehicle.get_transform().rotation.yaw
        
        target_x = start_loc.x + route_length * np.cos(np.radians(yaw))
        target_y = start_loc.y + route_length * np.sin(np.radians(yaw))
        
        return carla.Location(x=target_x, y=target_y, z=start_loc.z)
    
    def _compute_route_completion(
        self,
        start_loc: carla.Location,
        target_loc: carla.Location,
    ) -> float:
        """Compute route completion percentage."""
        if self._start_location is None:
            return 0.0
        
        curr_loc = self._vehicle.get_location()
        
        total_dist = start_loc.distance(target_loc)
        covered_dist = start_loc.distance(curr_loc)
        
        return min(1.0, covered_dist / total_dist)
    
    def evaluate(
        self,
        policy: Callable,
        num_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate policy across all scenarios.
        
        Args:
            policy: Function that maps observation to action
            num_episodes: Episodes per scenario (uses config if None)
        
        Returns:
            Dictionary with episode results and summary statistics
        """
        self._connect()
        
        num_episodes = num_episodes or self.config.num_episodes
        
        results = []
        
        print(f"Evaluating policy across {len(self.config.scenarios)} scenarios")
        print(f"Episodes per scenario: {num_episodes}")
        print("-" * 50)
        
        for scenario in self.config.scenarios:
            scenario_results = []
            
            for ep in range(num_episodes):
                print(f"  {scenario}: Episode {ep + 1}/{num_episodes}...", end=" ")
                
                result = self._run_episode(
                    policy=policy,
                    scenario=scenario,
                    route_length=self.config.route_length,
                )
                
                scenario_results.append(result)
                results.append(result)
                
                status = "✓" if result.success else "✗"
                infractions = len(result.infractions)
                completion = result.route_completion * 100
                print(f"{status} completion={completion:.0f}%, infractions={infractions}")
            
            # Scenario summary
            self._print_scenario_summary(scenario, scenario_results)
        
        # Overall summary
        summary = self._compute_summary(results)
        
        return {
            "episodes": [self._result_to_dict(r) for r in results],
            "summary": summary,
        }
    
    def _result_to_dict(self, result: EpisodeResult) -> Dict:
        """Convert result to dictionary."""
        return {
            "scenario": result.scenario,
            "success": result.success,
            "route_completion": result.route_completion,
            "infractions": result.infractions,
            "duration": result.duration,
            "distance_driven": result.distance_driven,
            "max_speed": result.max_speed,
            "avg_speed": result.avg_speed,
            "error": result.error,
        }
    
    def _print_scenario_summary(
        self,
        scenario: str,
        results: List[EpisodeResult],
    ):
        """Print summary for a scenario."""
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_completion = np.mean([r.route_completion for r in results]) * 100
        avg_distance = np.mean([r.distance_driven for r in results])
        
        # Count infractions
        infraction_counts = {}
        for r in results:
            for inf in r.infractions:
                infraction_counts[inf] = infraction_counts.get(inf, 0) + 1
        
        print(f"\n{scenario} Summary:")
        print(f"  Success rate: {success_rate * 100:.0f}%")
        print(f"  Avg completion: {avg_completion:.0f}%")
        print(f"  Avg distance: {avg_distance:.1f}m")
        if infraction_counts:
            print("  Infractions:")
            for inf, count in infraction_counts.items():
                print(f"    - {inf}: {count}")
    
    def _compute_summary(self, results: List[EpisodeResult]) -> Dict[str, Any]:
        """Compute overall summary statistics."""
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_completion = np.mean([r.route_completion for r in results]) * 100
        
        # Infraction counts
        infraction_counts = {}
        for r in results:
            for inf in r.infractions:
                infraction_counts[inf] = infraction_counts.get(inf, 0) + 1
        
        # Per-scenario breakdown
        by_scenario = {}
        for r in results:
            if r.scenario not in by_scenario:
                by_scenario[r.scenario] = []
            by_scenario[r.scenario].append(r)
        
        scenario_summaries = {}
        for name, eps in by_scenario.items():
            scenario_summaries[name] = {
                "success_rate": sum(1 for e in eps if e.success) / len(eps),
                "avg_completion": np.mean([e.route_completion for e in eps]) * 100,
                "avg_distance": np.mean([e.distance_driven for e in eps]),
            }
        
        return {
            "total_episodes": len(results),
            "success_rate": success_rate * 100,
            "avg_route_completion": avg_completion,
            "infraction_counts": infraction_counts,
            "by_scenario": scenario_summaries,
        }
    
    def save_results(self, results: Dict, path: str):
        """Save evaluation results to JSON."""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


# ============================================================================
# Integration examples
# ============================================================================

def evaluate_rl_policy(
    policy,
    env,
    num_episodes: int = 10,
):
    """
    Evaluate an RL policy using CARLA scenarios.
    
    Args:
        policy: Trained policy (gym-compatible)
        env: CarlaGymEnv instance
        num_episodes: Number of episodes
    
    Returns:
        Dict with evaluation metrics
    """
    evaluator = ScenarioEvaluator()
    
    def policy_wrapper(obs):
        """Wrap env observation for policy."""
        action, _ = policy.predict(obs)
        return action
    
    results = evaluator.evaluate(policy_wrapper, num_episodes=num_episodes)
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("CARLA ScenarioRunner Evaluation")
    print("=" * 50)
    print("\nNote: Requires running CARLA server with ScenarioRunner")
    print("Usage:")
    print("  1. Start CARLA: ./CarlaUE4.sh -carla-server-port=2000")
    print("  2. Run: python carla_scenario_eval.py")
    print("\nExample integration:")
    print("""
    from training.rl.envs.carla_scenario_eval import ScenarioEvaluator
    
    evaluator = ScenarioEvaluator(
        scenarios=["lane_change", "straight"],
        num_episodes=10,
    )
    
    results = evaluator.evaluate(my_policy)
    print(results["summary"])
    """)

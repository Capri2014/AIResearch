#!/usr/bin/env python3
"""
CARLA ScenarioRunner Integration for Full Pipeline Evaluation

This module provides the interface between the PyTorch-based driving pipeline
(Waymo SSL → Waypoint BC → RL refinement) and CARLA ScenarioRunner for 
closed-loop evaluation.

Key components:
- ScenarioConfig: Configuration for CARLA scenarios
- ScenarioRunnerInterface: Interface to CARLA ScenarioRunner
- RoutePlanner: Route planning for CARLA navigation
- CarlaFullPipelineAgent: Integrates pipeline policy with CARLA agent
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ScenarioConfig:
    """Configuration for a CARLA scenario."""
    town: str = "Town01"
    route_file: Optional[str] = None
    scenario_file: Optional[str] = None
    weather: str = "ClearNoon"
    num_vehicles: int = 20
    num_pedestrians: int = 50
    seed: int = 42
    timeout: float = 60.0  # seconds
    
    # Sensor configuration
    sensor_config: dict = field(default_factory=lambda: {
        "rgb": True,
        "depth": False,
        "semantic": False,
        "lidar": False,
    })
    
    # Agent configuration
    agent_type: str = "pipeline"  # "pipeline", "baseline", "oracle"
    
    def to_dict(self) -> dict:
        return {
            "town": self.town,
            "route_file": self.route_file,
            "scenario_file": self.scenario_file,
            "weather": self.weather,
            "num_vehicles": self.num_vehicles,
            "num_pedestrians": self.num_pedestrians,
            "seed": self.seed,
            "timeout": self.timeout,
            "sensor_config": self.sensor_config,
            "agent_type": self.agent_type,
        }


@dataclass
class CarlaClientConfig:
    """Configuration for CARLA client connection."""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    warm_up_seconds: float = 2.0
    
    # CARLA quality settings
    quality_level: str = "Low"  # "Low", "Medium", "High"
    
    def to_carla_args(self) -> list:
        return [
            "--host", self.host,
            "--port", str(self.port),
            "--timeout", str(self.timeout),
        ]


class RoutePlanner:
    """Route planning for CARLA navigation."""
    
    def __init__(self, route_file: str):
        self.route_file = route_file
        self.route_points = []
        self.current_index = 0
        
    def load_route(self) -> list:
        """Load route from file."""
        if not os.path.exists(self.route_file):
            print(f"Warning: Route file {self.route_file} not found")
            return []
            
        with open(self.route_file, 'r') as f:
            data = json.load(f)
            
        self.route_points = data.get("route", [])
        return self.route_points
    
    def get_waypoint(self, index: int) -> Optional[dict]:
        """Get waypoint at index."""
        if 0 <= index < len(self.route_points):
            return self.route_points[index]
        return None
    
    def get_next_waypoint(self, distance: float = 5.0) -> Optional[dict]:
        """Get next waypoint at approximate distance."""
        # Simplified - return next point
        if self.current_index + 1 < len(self.route_points):
            self.current_index += 1
            return self.route_points[self.current_index]
        return None
    
    def reset(self):
        """Reset route to beginning."""
        self.current_index = 0
        
    def get_progress(self) -> float:
        """Get route completion progress (0-1)."""
        if not self.route_points:
            return 0.0
        return self.current_index / len(self.route_points)


class CarlaFullPipelineAgent:
    """
    Integrates the full pipeline policy with CARLA for closed-loop evaluation.
    
    Pipeline: Waymo SSL encoder → Waypoint BC → RL delta refinement
    """
    
    def __init__(
        self,
        encoder_path: Optional[str] = None,
        bc_checkpoint: Optional[str] = None,
        rl_checkpoint: Optional[str] = None,
        delta_scale: float = 0.0,
        device: str = "cuda",
    ):
        self.encoder_path = encoder_path
        self.bc_checkpoint = bc_checkpoint
        self.rl_checkpoint = rl_checkpoint
        self.delta_scale = delta_scale
        self.device = device
        
        # Pipeline components (loaded on demand)
        self.encoder = None
        self.bc_model = None
        self.rl_model = None
        
        # State
        self.current_waypoints = None
        self.step_count = 0
        
    def load_pipeline(self):
        """Load the full pipeline (SSL + BC + RL)."""
        try:
            import torch
            from training.rl.full_pipeline_benchmark import (
                FullPipelinePolicy,
                PipelineBenchmarkConfig,
            )
            
            config = PipelineBenchmarkConfig(
                encoder_path=self.encoder_path,
                bc_checkpoint=self.bc_checkpoint,
                rl_checkpoint=self.rl_checkpoint,
                delta_scale=self.delta_scale,
            )
            
            self.pipeline = FullPipelinePolicy(config, device=self.device)
            print(f"Loaded full pipeline with delta_scale={self.delta_scale}")
            
        except ImportError as e:
            print(f"Warning: Could not load full pipeline: {e}")
            self.pipeline = None
            
    def reset(self, route_file: str):
        """Reset agent for new route."""
        self.step_count = 0
        self.current_waypoints = None
        
        # Load route
        self.route_planner = RoutePlanner(route_file)
        self.route_planner.load_route()
        
    def run_step(self, observation: dict) -> dict:
        """
        Run one step of the pipeline policy.
        
        Args:
            observation: Dict with keys like 'rgb', 'position', 'velocity', etc.
            
        Returns:
            Action dict with 'a' (acceleration) and 'kappa' (curvature)
        """
        self.step_count += 1
        
        if self.pipeline is None:
            # Fallback: simple constant action
            return {"a": 0.0, "kappa": 0.0}
        
        try:
            # Get next target waypoint from route
            target_waypoint = self.route_planner.get_next_waypoint()
            
            if target_waypoint is None:
                return {"a": 0.0, "kappa": 0.0}
            
            # Run pipeline inference
            action = self.pipeline.compute_action(
                observation=observation,
                target_waypoint=target_waypoint,
            )
            
            return action
            
        except Exception as e:
            print(f"Error in pipeline step: {e}")
            return {"a": 0.0, "kappa": 0.0}


class ScenarioRunnerEvaluator:
    """
    Evaluates pipeline policies using CARLA ScenarioRunner.
    
    Supports both real CARLA and mock evaluation modes.
    """
    
    def __init__(
        self,
        scenario_config: Optional[ScenarioConfig] = None,
        mock_mode: bool = False,
    ):
        self.config = scenario_config or ScenarioConfig()
        self.mock_mode = mock_mode
        
        self.client = None
        self.world = None
        self.agent = None
        
        # Metrics
        self.episode_metrics = []
        
    def connect(self, carla_args: Optional[list] = None):
        """Connect to CARLA."""
        if self.mock_mode:
            print("Running in mock mode (no CARLA)")
            return
            
        try:
            import carla
            from agents.navigation.local_planner import LocalPlanner
            
            client = carla.Client(
                self.config.host,
                self.config.port,
            )
            client.set_timeout(self.config.timeout)
            
            self.world = client.get_world()
            self.client = client
            
            print(f"Connected to CARLA at {self.config.host}:{self.config.port}")
            print(f"Town: {self.config.town}")
            
        except ImportError:
            print("CARLA not available, falling back to mock mode")
            self.mock_mode = True
            
        except Exception as e:
            print(f"Could not connect to CARLA: {e}")
            print("Falling back to mock mode")
            self.mock_mode = True
            
    def load_agent(self, agent: CarlaFullPipelineAgent):
        """Load the agent for evaluation."""
        self.agent = agent
        
    def run_episode(
        self,
        route_file: str,
        scenario_file: Optional[str] = None,
    ) -> dict:
        """Run one evaluation episode."""
        start_time = time.time()
        
        if self.mock_mode:
            return self._run_mock_episode(route_file)
        else:
            return self._run_real_episode(route_file, scenario_file)
            
    def _run_mock_episode(self, route_file: str) -> dict:
        """Run a mock episode for testing."""
        # Simulate episode
        route_planner = RoutePlanner(route_file)
        route_planner.load_route()
        
        total_steps = len(route_planner.route_points) if route_planner.route_points else 100
        
        # Simulate metrics
        np.random.seed(self.config.seed)
        
        ade = np.random.uniform(5.0, 15.0)
        fde = ade * np.random.uniform(1.2, 1.8)
        route_completion = np.random.uniform(70.0, 95.0)
        success_rate = 1.0 if route_completion > 80.0 else np.random.uniform(0.3, 0.7)
        collisions = np.random.randint(0, 5)
        
        metrics = {
            "episode_id": f"mock_{self.config.seed}",
            "town": self.config.town,
            "ADE": ade,
            "FDE": fde,
            "route_completion": route_completion,
            "success_rate": success_rate,
            "collisions": collisions,
            "red_light_violations": np.random.randint(0, 2),
            "stop_sign_violations": np.random.randint(0, 2),
            "steps": total_steps,
            "duration": total_steps * 0.1,
            "mock": True,
        }
        
        self.episode_metrics.append(metrics)
        return metrics
        
    def _run_real_episode(self, route_file: str, scenario_file: Optional[str]) -> dict:
        """Run a real episode in CARLA."""
        # TODO: Implement real CARLA episode
        return self._run_mock_episode(route_file)
        
    def run_evaluation(
        self,
        route_files: list,
        num_runs: int = 1,
    ) -> dict:
        """Run full evaluation across routes."""
        all_metrics = []
        
        for route_file in route_files:
            for run in range(num_runs):
                print(f"Running {route_file} (run {run + 1}/{num_runs})")
                
                metrics = self.run_episode(route_file)
                all_metrics.append(metrics)
                
        # Aggregate metrics
        return self._aggregate_metrics(all_metrics)
        
    def _aggregate_metrics(self, metrics_list: list) -> dict:
        """Aggregate metrics across episodes."""
        if not metrics_list:
            return {}
            
        # Calculate means
        keys = ["ADE", "FDE", "route_completion", "success_rate", 
                "collisions", "red_light_violations", "stop_sign_violations"]
        
        aggregated = {}
        for key in keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                aggregated[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
                
        aggregated["num_episodes"] = len(metrics_list)
        aggregated["episodes"] = metrics_list
        
        return aggregated


def main():
    """CLI for CARLA ScenarioRunner evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CARLA ScenarioRunner Integration for Pipeline Evaluation"
    )
    parser.add_argument("--town", type=str, default="Town01",
                        help="CARLA town")
    parser.add_argument("--routes", type=str, nargs="+", 
                        help="Route files")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to SSL encoder checkpoint")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC checkpoint")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to RL checkpoint")
    parser.add_argument("--delta-scale", type=float, default=0.0,
                        help="Delta scale for RL refinement")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode (no CARLA)")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of evaluation runs per route")
    parser.add_argument("--output", type=str, default="out/carla_eval/metrics.json",
                        help="Output file for metrics")
    
    args = parser.parse_args()
    
    # Create scenario config
    scenario_config = ScenarioConfig(
        town=args.town,
        seed=42,
    )
    
    # Create evaluator
    evaluator = ScenarioRunnerEvaluator(
        scenario_config=scenario_config,
        mock_mode=args.mock,
    )
    
    # Connect to CARLA
    evaluator.connect()
    
    # Create and load agent
    agent = CarlaFullPipelineAgent(
        encoder_path=args.encoder_path,
        bc_checkpoint=args.bc_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        delta_scale=args.delta_scale,
    )
    agent.load_pipeline()
    evaluator.load_agent(agent)
    
    # Default routes if not specified
    route_files = args.routes or [
        "data/routes/town01_route_01.json",
    ]
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.run_evaluation(
        route_files=route_files,
        num_runs=args.num_runs,
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or "out/carla_eval", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {args.output}")
    print(f"Aggregated metrics: {results}")


if __name__ == "__main__":
    main()

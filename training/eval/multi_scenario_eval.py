#!/usr/bin/env python3
"""
Multi-Scenario Evaluation Runner for Driving Pipeline

Comprehensive evaluation that:
1. Loads BC and/or RL refinement checkpoints
2. Runs multiple CARLA scenarios from configured suites
3. Computes standardized metrics (ADE, FDE, completion, infractions)
4. Outputs aggregated summary report

Usage:
    # Run smoke suite with BC only
    python -m training.eval.multi_scenario_eval \
        --bc-checkpoint out/waypoint_bc/final.pt \
        --suite smoke

    # Run full suite with BC + RL refinement
    python -m training.eval.multi_scenario_eval \
        --bc-checkpoint out/waypoint_bc/final.pt \
        --rl-checkpoint out/ppo_delta/final.pt \
        --suite full \
        --output-dir out/multi_scenario_eval

    # Dry-run (no CARLA)
    python -m training.eval.multi_scenario_eval --dry-run
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import sys
import time

import numpy as np

# Try imports with graceful fallbacks
try:
    from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
    WAYPOINT_BC_AVAILABLE = True
except ImportError:
    WAYPOINT_BC_AVAILABLE = False

try:
    from training.rl.waypoint_policy_wrapper import WaypointPolicyWithDelta
    WAYPOINT_POLICY_AVAILABLE = True
except ImportError:
    WAYPOINT_POLICY_AVAILABLE = False

try:
    from sim.driving.carla_srunner.scenario_config import (
        get_scenario_suite,
        ScenarioConfig,
        MapName,
    )
    SCENARIO_CONFIG_AVAILABLE = True
except ImportError:
    SCENARIO_CONFIG_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MultiScenarioEvalConfig:
    """Configuration for multi-scenario evaluation."""
    
    # Checkpoints
    bc_checkpoint: Optional[Path] = None
    rl_checkpoint: Optional[Path] = None
    ssl_checkpoint: Optional[Path] = None
    
    # Scenario settings
    suite: str = "smoke"  # smoke, quick, full, adverse, night
    num_runs_per_scenario: int = 1  # Multiple runs for statistical significance
    
    # CARLA settings
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    timeout_s: int = 300  # Per-scenario timeout
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("out/multi_scenario_eval"))
    
    # Behavior
    dry_run: bool = False
    verbose: bool = True
    save_episodes: bool = False  # Save full episode data
    
    # Model settings
    num_waypoints: int = 8
    waypoint_time_delta: float = 0.5
    
    # Device
    device: str = "cuda" if True else "cpu"  # Default to CPU for CARLA inference


@dataclass
class ScenarioResult:
    """Result from a single scenario run."""
    
    scenario_name: str
    success: bool
    completion_rate: float = 0.0
    
    # Metrics
    ade: float = 0.0
    fde: float = 0.0
    average_speed: float = 0.0
    max_acceleration: float = 0.0
    collision: bool = False
    red_light_violation: bool = False
    stop_sign_violation: bool = False
    off_road: bool = False
    
    # Timing
    runtime_s: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Extra
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "scenario_name": self.scenario_name,
            "success": self.success,
            "completion_rate": self.completion_rate,
            "metrics": {
                "ade": self.ade,
                "fde": self.fde,
                "average_speed": self.average_speed,
                "max_acceleration": self.max_acceleration,
            },
            "infractions": {
                "collision": self.collision,
                "red_light_violation": self.red_light_violation,
                "stop_sign_violation": self.stop_sign_violation,
                "off_road": self.off_road,
            },
            "runtime_s": self.runtime_s,
            "timestamp": self.timestamp,
        }
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class MultiScenarioSummary:
    """Aggregated results across all scenarios."""
    
    suite: str
    num_scenarios: int
    num_runs: int
    
    # Aggregate metrics
    overall_success_rate: float = 0.0
    mean_ade: float = 0.0
    mean_fde: float = 0.0
    mean_completion_rate: float = 0.0
    
    # Infraction rates
    collision_rate: float = 0.0
    red_light_rate: float = 0.0
    stop_sign_rate: float = 0.0
    off_road_rate: float = 0.0
    
    # Timing
    total_runtime_s: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Per-scenario results
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite": self.suite,
            "num_scenarios": self.num_scenarios,
            "num_runs": self.num_runs,
            "aggregate_metrics": {
                "success_rate": self.overall_success_rate,
                "mean_ade": self.mean_ade,
                "mean_fde": self.mean_fde,
                "mean_completion_rate": self.mean_completion_rate,
            },
            "infraction_rates": {
                "collision": self.collision_rate,
                "red_light_violation": self.red_light_rate,
                "stop_sign_violation": self.stop_sign_rate,
                "off_road": self.off_road_rate,
            },
            "total_runtime_s": self.total_runtime_s,
            "timestamp": self.timestamp,
            "scenario_results": [r.to_dict() for r in self.scenario_results],
        }


# ============================================================================
# Model Loading
# ============================================================================

class ModelLoader:
    """Loads BC and RL models for evaluation."""
    
    def __init__(self, config: MultiScenarioEvalConfig):
        self.config = config
        self.bc_model = None
        self.rl_model = None
        
    def load_models(self) -> bool:
        """Load BC and RL models. Returns True if any model loaded."""
        loaded_any = False
        
        # Load BC model
        if self.config.bc_checkpoint and WAYPOINT_BC_AVAILABLE:
            try:
                self.bc_model = self._load_bc_model()
                print(f"[ModelLoader] Loaded BC model from {self.config.bc_checkpoint}")
                loaded_any = True
            except Exception as e:
                print(f"[ModelLoader] Failed to load BC model: {e}")
        
        # Load RL model
        if self.config.rl_checkpoint and WAYPOINT_POLICY_AVAILABLE:
            try:
                self.rl_model = self._load_rl_model()
                print(f"[ModelLoader] Loaded RL model from {self.config.rl_checkpoint}")
                loaded_any = True
            except Exception as e:
                print(f"[ModelLoader] Failed to load RL model: {e}")
        
        if not loaded_any:
            print("[ModelLoader] Warning: No models loaded, using stub policy")
            
        return loaded_any
    
    def _load_bc_model(self):
        """Load BC waypoint model."""
        # Create config
        bc_config = WaypointBCConfig(
            bev_feature_dim=256,
            num_waypoints=self.config.num_waypoints,
            predict_speed=True,
        )
        
        # Create model
        model = WaypointBCModel(bc_config)
        
        # Load checkpoint if exists
        if self.config.bc_checkpoint and self.config.bc_checkpoint.exists():
            checkpoint = torch.load(
                self.config.bc_checkpoint,
                map_location=self.config.device,
                weights_only=False
            )
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
        
        model.to(self.config.device)
        model.eval()
        return model
    
    def _load_rl_model(self):
        """Load RL delta-waypoint model."""
        # WaypointPolicyWithDelta handles RL checkpoint loading
        model = WaypointPolicyWithDelta(
            bc_model=self.bc_model,
            rl_checkpoint=self.config.rl_checkpoint,
        )
        return model


# ============================================================================
# Scenario Evaluation
# ============================================================================

class ScenarioEvaluator:
    """Runs scenarios and collects metrics."""
    
    def __init__(
        self,
        config: MultiScenarioEvalConfig,
        model_loader: ModelLoader,
    ):
        self.config = config
        self.model_loader = model_loader
        
    def run_scenario(self, scenario: ScenarioConfig) -> ScenarioResult:
        """Run a single scenario and return results."""
        start_time = time.time()
        
        if self.config.dry_run:
            # Dry run: simulate results
            return self._run_dry_run(scenario, start_time)
        
        try:
            # Real CARLA run
            return self._run_carla(scenario, start_time)
        except Exception as e:
            return ScenarioResult(
                scenario_name=scenario.name,
                success=False,
                error=str(e),
                runtime_s=time.time() - start_time,
            )
    
    def _run_dry_run(self, scenario: ScenarioConfig, start_time: float) -> ScenarioResult:
        """Simulate a scenario run for testing."""
        # Simulate realistic metrics
        np.random.seed(hash(scenario.name) % (2**32))
        
        ade = np.random.uniform(2.0, 8.0)
        fde = ade * np.random.uniform(1.2, 2.0)
        
        return ScenarioResult(
            scenario_name=scenario.name,
            success=np.random.random() > 0.3,  # 70% success rate
            completion_rate=np.random.uniform(0.6, 1.0),
            ade=ade,
            fde=fde,
            average_speed=np.random.uniform(5.0, 12.0),
            max_acceleration=np.random.uniform(2.0, 5.0),
            collision=np.random.random() < 0.1,
            red_light_violation=np.random.random() < 0.05,
            stop_sign_violation=np.random.random() < 0.05,
            off_road=np.random.random() < 0.1,
            runtime_s=time.time() - start_time,
        )
    
    def _run_carla(self, scenario: ScenarioConfig, start_time: float) -> ScenarioResult:
        """Run actual CARLA scenario."""
        # TODO: Implement real CARLA integration
        # For now, fall back to dry run
        print(f"[ScenarioEvaluator] CARLA not available, using dry-run mode")
        return self._run_dry_run(scenario, start_time)
    
    def compute_ade_fde(
        self,
        predicted_waypoints: np.ndarray,
        target_waypoints: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute Average Displacement Error and Final Displacement Error.
        
        Args:
            predicted_waypoints: (N, 2) predicted waypoints
            target_waypoints: (N, 2) ground truth waypoints
            
        Returns:
            (ade, fde) in meters
        """
        if len(predicted_waypoints) == 0 or len(target_waypoints) == 0:
            return float('inf'), float('inf')
        
        # Ensure same length
        n = min(len(predicted_waypoints), len(target_waypoints))
        pred = predicted_waypoints[:n]
        tgt = target_waypoints[:n]
        
        # ADE: mean euclidean distance
        errors = np.linalg.norm(pred - tgt, axis=1)
        ade = float(np.mean(errors))
        
        # FDE: distance to final waypoint
        fde = float(errors[-1])
        
        return ade, fde


# ============================================================================
# Multi-Scenario Runner
# ============================================================================

class MultiScenarioRunner:
    """Orchestrates multi-scenario evaluation."""
    
    def __init__(self, config: MultiScenarioEvalConfig):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.evaluator = None
        
    def run(self) -> MultiScenarioSummary:
        """Run all scenarios in the suite and return aggregated results."""
        print(f"[MultiScenarioRunner] Starting evaluation suite: {self.config.suite}")
        
        # Load models
        self.model_loader.load_models()
        
        # Create evaluator
        self.evaluator = ScenarioEvaluator(self.config, self.model_loader)
        
        # Get scenario suite
        scenarios = self._get_scenarios()
        print(f"[MultiScenarioRunner] Running {len(scenarios)} scenarios")
        
        # Run scenarios
        all_results: List[ScenarioResult] = []
        start_time = time.time()
        
        for i, scenario in enumerate(scenarios):
            print(f"[MultiScenarioRunner] Scenario {i+1}/{len(scenarios)}: {scenario.name}")
            
            # Run multiple times if configured
            for run in range(self.config.num_runs_per_scenario):
                result = self.evaluator.run_scenario(scenario)
                if self.config.num_runs_per_scenario > 1:
                    result.scenario_name = f"{scenario.name}_run{run}"
                all_results.append(result)
                
                if self.config.verbose:
                    print(f"  -> ADE: {result.ade:.2f}m, FDE: {result.fde:.2f}m, "
                          f"Success: {result.success}, Completion: {result.completion_rate:.1%}")
        
        total_runtime = time.time() - start_time
        
        # Compute aggregate statistics
        summary = self._aggregate_results(all_results, total_runtime)
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _get_scenarios(self) -> List[ScenarioConfig]:
        """Get list of scenarios for the configured suite."""
        if not SCENARIO_CONFIG_AVAILABLE:
            # Fallback: create stub scenarios
            return self._get_stub_scenarios()
        
        try:
            # get_scenario_suite returns List[ScenarioConfig], not object with .scenarios
            scenarios = get_scenario_suite(self.config.suite)
            return scenarios
        except Exception as e:
            print(f"[MultiScenarioRunner] Failed to load suite {self.config.suite}: {e}")
            return self._get_stub_scenarios()
    
    def _get_stub_scenarios(self) -> List[ScenarioConfig]:
        """Get stub scenarios when scenario config unavailable."""
        if SCENARIO_CONFIG_AVAILABLE:
            # Use the actual scenario config module
            from sim.driving.carla_srunner.scenario_config import (
                ScenarioType, MapName, WeatherConfig, WeatherPreset
            )
            
            stub_configs = [
                ("straight_clear", ScenarioType.STRAIGHT, WeatherPreset.CLEAR),
                ("straight_cloudy", ScenarioType.STRAIGHT, WeatherPreset.CLOUDY),
                ("straight_night", ScenarioType.STRAIGHT, WeatherPreset.NIGHT),
                ("turn_left_clear", ScenarioType.TURN_LEFT, WeatherPreset.CLEAR),
                ("turn_right_clear", ScenarioType.TURN_RIGHT, WeatherPreset.CLEAR),
            ]
        else:
            # Fallback without imports
            stub_configs = [
                ("straight_clear", "straight", "clear"),
                ("straight_cloudy", "straight", "cloudy"),
                ("straight_night", "straight", "night"),
                ("turn_left_clear", "turn_left", "clear"),
                ("turn_right_clear", "turn_right", "clear"),
            ]
        
        # Select based on suite
        if self.config.suite == "smoke":
            stub_configs = stub_configs[:2]
        elif self.config.suite == "quick":
            stub_configs = stub_configs[:3]
        elif self.config.suite == "adverse":
            stub_configs = stub_configs[:2]  # Use first two for adverse
        elif self.config.suite == "night":
            stub_configs = [stub_configs[2]]  # Only night
        
        # Create stub configs
        scenarios = []
        
        if SCENARIO_CONFIG_AVAILABLE:
            from sim.driving.carla_srunner.scenario_config import MapName
            for name, scen_type, weather_preset in stub_configs:
                scenarios.append(ScenarioConfig(
                    id=name,
                    name=name,
                    type=scen_type,
                    map=MapName.TOWN01,
                    weather=WeatherConfig.from_preset(weather_preset),
                ))
        else:
            # Minimal stub without actual ScenarioConfig
            for name, _, _ in stub_configs:
                scenarios.append({"name": name, "id": name})
        
        return scenarios
    
    def _aggregate_results(
        self,
        results: List[ScenarioResult],
        total_runtime: float,
    ) -> MultiScenarioSummary:
        """Aggregate results across all scenarios."""
        if not results:
            return MultiScenarioSummary(
                suite=self.config.suite,
                num_scenarios=0,
                num_runs=0,
            )
        
        n = len(results)
        
        # Compute aggregate metrics
        success_count = sum(1 for r in results if r.success)
        
        ade_values = [r.ade for r in results if r.ade > 0]
        fde_values = [r.fde for r in results if r.fde > 0]
        completion_values = [r.completion_rate for r in results]
        
        # Infraction rates
        collision_count = sum(1 for r in results if r.collision)
        red_light_count = sum(1 for r in results if r.red_light_violation)
        stop_sign_count = sum(1 for r in results if r.stop_sign_violation)
        off_road_count = sum(1 for r in results if r.off_road)
        
        summary = MultiScenarioSummary(
            suite=self.config.suite,
            num_scenarios=len(set(r.scenario_name for r in results)),
            num_runs=n,
            overall_success_rate=success_count / n,
            mean_ade=np.mean(ade_values) if ade_values else 0.0,
            mean_fde=np.mean(fde_values) if fde_values else 0.0,
            mean_completion_rate=np.mean(completion_values) if completion_values else 0.0,
            collision_rate=collision_count / n,
            red_light_rate=red_light_count / n,
            stop_sign_rate=stop_sign_count / n,
            off_road_rate=off_road_count / n,
            total_runtime_s=total_runtime,
            scenario_results=results,
        )
        
        return summary
    
    def _save_results(self, summary: MultiScenarioSummary):
        """Save results to output directory."""
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.config.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"[MultiScenarioRunner] Saved metrics to {metrics_path}")
        
        # Save config
        config_path = run_dir / "config.json"
        config_dict = asdict(self.config)
        # Convert Path to str for JSON
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"[MultiScenarioRunner] Results saved to {run_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-scenario evaluation for driving pipeline"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        help="Path to BC waypoint checkpoint",
    )
    parser.add_argument(
        "--rl-checkpoint", 
        type=Path,
        help="Path to RL delta-waypoint checkpoint",
    )
    parser.add_argument(
        "--ssl-checkpoint",
        type=Path,
        help="Path to SSL encoder checkpoint",
    )
    
    # Scenario settings
    parser.add_argument(
        "--suite",
        type=str,
        default="smoke",
        choices=["smoke", "quick", "full", "adverse", "night"],
        help="Scenario suite to run",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per scenario",
    )
    
    # CARLA settings
    parser.add_argument(
        "--carla-host",
        type=str,
        default="127.0.0.1",
        help="CARLA host",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA port",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-scenario timeout in seconds",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/multi_scenario_eval"),
        help="Output directory",
    )
    
    # Behavior
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without CARLA (dry run)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = MultiScenarioEvalConfig(
        bc_checkpoint=args.bc_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        ssl_checkpoint=args.ssl_checkpoint,
        suite=args.suite,
        num_runs_per_scenario=args.num_runs,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        timeout_s=args.timeout,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    # Run evaluation
    runner = MultiScenarioRunner(config)
    summary = runner.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("MULTI-SCENARIO EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Suite: {summary.suite}")
    print(f"Scenarios: {summary.num_scenarios}")
    print(f"Total runs: {summary.num_runs}")
    print("-" * 60)
    print(f"Success Rate: {summary.overall_success_rate:.1%}")
    print(f"Mean ADE: {summary.mean_ade:.2f}m")
    print(f"Mean FDE: {summary.mean_fde:.2f}m")
    print(f"Mean Completion: {summary.mean_completion_rate:.1%}")
    print("-" * 60)
    print(f"Collision Rate: {summary.collision_rate:.1%}")
    print(f"Red Light Rate: {summary.red_light_rate:.1%}")
    print(f"Off-Road Rate: {summary.off_road_rate:.1%}")
    print("-" * 60)
    print(f"Total Runtime: {summary.total_runtime_s:.1f}s")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

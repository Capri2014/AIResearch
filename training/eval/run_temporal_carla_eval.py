"""
CARLA Closed-Loop Evaluation for Temporal Waypoint BC

Evaluates temporal waypoint BC models in CARLA closed-loop scenarios.
This script connects the temporal waypoint BC (training/sft/train_temporal_waypoint_bc.py)
with the CARLA evaluation pipeline.

Pipeline stage: CARLA evaluation of temporal waypoint BC policies

Usage:
    python -m training.eval.run_temporal_carla_eval \
        --checkpoint out/temporal_waypoint_bc/best_model.pt \
        --output-dir out/temporal_carla_eval

Outputs:
- out/temporal_carla_eval/metrics.json (aggregated metrics)
- out/temporal_carla_eval/scenario_*.json (per-scenario results)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import logging
from datetime import datetime

# Lazy carla import
_carla_imported = False
_carla = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_carla():
    """Lazy import of carla module."""
    global _carla_imported, _carla
    if not _carla_imported:
        import carla as _carla_module
        _carla = _carla_module
        _carla_imported = True
    return _carla


@dataclass
class TemporalScenarioConfig:
    """Configuration for temporal waypoint evaluation scenario."""
    name: str
    weather: Any  # carla.WeatherParameters
    map_name: str = "Town01"
    num_episodes: int = 3
    sequence_length: int = 4  # Frames in temporal context
    spawn_points: List[str] = field(default_factory=list)
    target_points: List[str] = field(default_factory=list)


def create_clear_weather():
    """Clear daytime weather."""
    carla = _get_carla()
    return carla.WeatherParameters(
        sun_altitude_angle=70.0,
        cloudiness=0.0,
        precipitation=0.0,
        fog_density=0.0,
        fog_distance=0.0,
        wetness=0.0,
    )


def create_cloudy_weather():
    """Overcast weather."""
    carla = _get_carla()
    return carla.WeatherParameters(
        sun_altitude_angle=30.0,
        cloudiness=80.0,
        precipitation=0.0,
        fog_density=10.0,
        fog_distance=50.0,
        wetness=20.0,
    )


def create_night_weather():
    """Night driving conditions."""
    carla = _get_carla()
    return carla.WeatherParameters(
        sun_altitude_angle=-90.0,
        cloudiness=20.0,
        precipitation=0.0,
        fog_density=5.0,
        fog_distance=30.0,
        wetness=0.0,
    )


def create_rain_weather():
    """Rainy conditions."""
    carla = _get_carla()
    return carla.WeatherParameters(
        sun_altitude_angle=45.0,
        cloudiness=60.0,
        precipitation=80.0,
        fog_density=15.0,
        fog_distance=40.0,
        wetness=80.0,
    )


@dataclass
class TemporalEvalMetrics:
    """Metrics for temporal waypoint BC evaluation."""
    # Scenario info
    scenario_name: str = ""
    map_name: str = ""
    weather: str = ""
    
    # Core metrics
    route_completion: float = 0.0  # Percentage of route completed
    average_displacement_error: float = 0.0  # ADE
    final_displacement_error: float = 0.0  # FDE
    success_rate: float = 0.0  # Percentage of successful episodes
    
    # Temporal-specific metrics
    temporal_consistency: float = 0.0  # How smooth are predictions over time
    
    # Safety metrics
    collisions: int = 0
    red_light_violations: int = 0
    stop_sign_violations: int = 0
    
    # Performance metrics
    avg_inference_time_ms: float = 0.0
    total_episodes: int = 0
    successful_episodes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "map_name": self.map_name,
            "weather": self.weather,
            "route_completion": self.route_completion,
            "average_displacement_error": self.average_displacement_error,
            "final_displacement_error": self.final_displacement_error,
            "success_rate": self.success_rate,
            "temporal_consistency": self.temporal_consistency,
            "collisions": self.collisions,
            "red_light_violations": self.red_light_violations,
            "stop_sign_violations": self.stop_sign_violations,
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
        }


class TemporalWaypointAgent:
    """Agent that uses temporal waypoint BC for driving in CARLA.
    
    Maintains a buffer of recent frames for temporal context.
    """
    
    def __init__(
        self,
        vehicle,
        sensor_interface,
        policy_wrapper,
        sequence_length: int = 4,
    ):
        self.vehicle = vehicle
        self.sensor_interface = sensor_interface
        self.policy = policy_wrapper
        self.sequence_length = sequence_length
        
        # Temporal frame buffer
        self._frame_buffer: List[np.ndarray] = []
        self._max_buffer = sequence_length
        
        # Inference timing
        self._inference_times: List[float] = []
        
        # Metrics
        self._waypoints_history: List[np.ndarray] = []
    
    def run_step(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Execute one step of the agent.
        
        Args:
            input_data: Dict of sensor data from CARLA
        
        Returns:
            control: Dict with throttle, steer, brake
        """
        import time
        
        # Get front camera image
        camera_data = input_data.get("front_camera")
        if camera_data is None:
            return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
        
        # Extract image from camera data
        image = self._extract_camera_image(camera_data)
        
        # Add to temporal buffer
        self._frame_buffer.append(image)
        if len(self._frame_buffer) > self._max_buffer:
            self._frame_buffer = self._frame_buffer[-self._max_buffer:]
        
        # Get current speed
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Time inference
        start_time = time.perf_counter()
        
        # Predict with temporal context
        waypoints = self.policy.predict_temporal(self._frame_buffer)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)
        
        # Store for temporal consistency metric
        self._waypoints_history.append(waypoints.copy())
        
        # Convert to control
        control_dict = self.policy.waypoints_to_control(waypoints, speed)
        
        # Convert to carla control
        carla = _get_carla()
        control = carla.VehicleControl(
            throttle=control_dict.get("throttle", 0.0),
            steer=control_dict.get("steer", 0.0),
            brake=control_dict.get("brake", 0.0),
        )
        
        return control
    
    def _extract_camera_image(self, camera_data) -> np.ndarray:
        """Extract image array from camera data."""
        # Handle different camera data formats
        if hasattr(camera_data, 'frame'):
            # Sprite/Image object
            array = np.frombuffer(camera_data.raw_data, dtype=np.uint8)
            array = array.reshape((camera_data.height, camera_data.width, 4))
            return array[:, :, :3]  # RGB
        elif isinstance(camera_data, np.ndarray):
            return camera_data
        else:
            raise ValueError(f"Unknown camera data format: {type(camera_data)}")
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time in ms."""
        if not self._inference_times:
            return 0.0
        return np.mean(self._inference_times)
    
    def compute_temporal_consistency(self) -> float:
        """Compute temporal consistency metric.
        
        Measures how smooth the waypoint predictions are over time.
        Lower values = smoother predictions.
        """
        if len(self._waypoints_history) < 2:
            return 1.0
        
        # Compute differences between consecutive predictions
        diffs = []
        for i in range(1, len(self._waypoints_history)):
            diff = np.linalg.norm(
                self._waypoints_history[i] - self._waypoints_history[i-1]
            )
            diffs.append(diff)
        
        # Return inverse of mean difference (normalized)
        mean_diff = np.mean(diffs) if diffs else 0.0
        return float(np.exp(-mean_diff))  # Higher = more consistent
    
    def reset(self):
        """Reset agent state for new episode."""
        self._frame_buffer.clear()
        self._inference_times.clear()
        self._waypoints_history.clear()


def compute_ade(gt_waypoints: np.ndarray, pred_waypoints: np.ndarray) -> float:
    """Compute Average Displacement Error."""
    if len(gt_waypoints) == 0 or len(pred_waypoints) == 0:
        return float('inf')
    
    min_len = min(len(gt_waypoints), len(pred_waypoints))
    errors = np.linalg.norm(
        gt_waypoints[:min_len] - pred_waypoints[:min_len],
        axis=1
    )
    return float(np.mean(errors))


def compute_fde(gt_waypoints: np.ndarray, pred_waypoints: np.ndarray) -> float:
    """Compute Final Displacement Error."""
    if len(gt_waypoints) == 0 or len(pred_waypoints) == 0:
        return float('inf')
    
    final_gt = gt_waypoints[-1]
    final_pred = pred_waypoints[-1] if len(pred_waypoints) > 0 else pred_waypoints[0]
    return float(np.linalg.norm(final_gt - final_pred))


def get_default_scenarios() -> List[TemporalScenarioConfig]:
    """Get default evaluation scenarios."""
    carla = _get_carla()
    
    return [
        TemporalScenarioConfig(
            name="straight_clear",
            weather=create_clear_weather(),
            map_name="Town01",
            sequence_length=4,
        ),
        TemporalScenarioConfig(
            name="straight_cloudy",
            weather=create_cloudy_weather(),
            map_name="Town01",
            sequence_length=4,
        ),
        TemporalScenarioConfig(
            name="straight_night",
            weather=create_night_weather(),
            map_name="Town01",
            sequence_length=4,
        ),
        TemporalScenarioConfig(
            name="straight_rain",
            weather=create_rain_weather(),
            map_name="Town01",
            sequence_length=4,
        ),
        TemporalScenarioConfig(
            name="turn_clear",
            weather=create_clear_weather(),
            map_name="Town01",
            sequence_length=4,
        ),
    ]


def run_temporal_evaluation(
    checkpoint: Path,
    output_dir: Path,
    sequence_length: int = 4,
    hidden_dim: int = 256,
    num_rnn_layers: int = 2,
    encoder_name: str = "resnet34",
    scenarios: Optional[List[TemporalScenarioConfig]] = None,
    carla_port: int = 2000,
) -> Dict[str, Any]:
    """
    Run temporal waypoint BC evaluation in CARLA.
    
    Args:
        checkpoint: Path to temporal waypoint BC checkpoint
        output_dir: Directory for output metrics
        sequence_length: Frames in temporal context
        hidden_dim: LSTM hidden dimension
        num_rnn_layers: Number of LSTM layers
        encoder_name: CNN encoder name
        scenarios: List of scenarios to evaluate
        carla_port: CARLA server port
    
    Returns:
        Dictionary with aggregated metrics
    """
    from sim.driving.carla_srunner.temporal_policy_wrapper import (
        TemporalWaypointPolicyWrapper,
        TemporalPolicyConfig,
    )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default scenarios
    if scenarios is None:
        scenarios = get_default_scenarios()
    
    # Initialize policy wrapper
    cfg = TemporalPolicyConfig(
        checkpoint=checkpoint,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        num_rnn_layers=num_rnn_layers,
        encoder_name=encoder_name,
    )
    policy = TemporalWaypointPolicyWrapper(cfg)
    
    if not policy.initialize():
        raise RuntimeError("Failed to initialize temporal policy")
    
    logger.info(f"Loaded temporal waypoint BC from {checkpoint}")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  RNN layers: {num_rnn_layers}")
    logger.info(f"  Encoder: {encoder_name}")
    
    # Run each scenario
    scenario_results: List[Dict[str, Any]] = []
    
    for scenario in scenarios:
        logger.info(f"\n=== Running scenario: {scenario.name} ===")
        
        # Note: Full CARLA evaluation would require a running CARLA server
        # This is a stub that logs what would happen
        logger.info(f"  Weather: {scenario.weather}")
        logger.info(f"  Map: {scenario.map_name}")
        logger.info(f"  Episodes: {scenario.num_episodes}")
        
        # Create placeholder results
        result = {
            "scenario_name": scenario.name,
            "map_name": scenario.map_name,
            "weather": str(scenario.weather),
            "status": "skipped_no_carla",
            "note": "Requires running CARLA server",
        }
        scenario_results.append(result)
    
    # Aggregate metrics
    metrics = TemporalEvalMetrics(
        scenario_name="aggregated",
        total_episodes=sum(s.num_episodes for s in scenarios),
    )
    
    # Save results
    metrics_json = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint),
        "model_config": {
            "sequence_length": sequence_length,
            "hidden_dim": hidden_dim,
            "num_rnn_layers": num_rnn_layers,
            "encoder_name": encoder_name,
        },
        "aggregated_metrics": metrics.to_dict(),
        "scenario_results": scenario_results,
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {metrics_path}")
    
    return metrics_json


def main():
    """CLI for temporal waypoint BC CARLA evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CARLA Closed-Loop Evaluation for Temporal Waypoint BC"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to temporal waypoint BC checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4,
        help="Frames in temporal context (default: 4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="LSTM hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--num-rnn-layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        choices=["resnet18", "resnet34", "efficientnet_b0"],
        help="CNN encoder (default: resnet34)",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000)",
    )
    
    args = parser.parse_args()
    
    run_temporal_evaluation(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        num_rnn_layers=args.num_rnn_layers,
        encoder_name=args.encoder,
        carla_port=args.carla_port,
    )


if __name__ == "__main__":
    main()

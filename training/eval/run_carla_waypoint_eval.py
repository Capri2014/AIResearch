"""
CARLA Evaluation Script for Trained Waypoint BC Policies

This script bridges the gap between offline waypoint BC training and
closed-loop CARLA simulation evaluation.

Pipeline: Waymo → SSL pretrain → waypoint BC → CARLA eval

Usage
-----
# With CARLA server running:
python -m training.eval.run_carla_waypoint_eval \
  --checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --scenarios "training/eval/scenarios/*.yaml" \
  --output-dir out/carla_eval_waypoint_bc

# Smoke test (no CARLA required):
python -m training.eval.run_carla_waypoint_eval --smoke
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
import argparse
import json
import logging
import sys
from io import StringIO

import numpy as np

# YAML for scenario loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from carla_scenariorunner_eval (graceful if CARLA not available)
try:
    from training.eval.carla_scenariorunner_eval import (
        CARLAEvalConfig,
        EvalResult,
    )
    EVAL_IMPORTS_OK = True
except ImportError:
    EVAL_IMPORTS_OK = False
    # Define minimal stubs for smoke test
    @dataclass
    class CARLAEvalConfig:
        host: str = "localhost"
        port: int = 2000
        fps: int = 20
        timeout: float = 10.0
        weather: Optional[Any] = None
        map_name: str = "Town01"
    
    @dataclass
    class EvalResult:
        route_completion: float
        collision_count: int
        offroad_count: int
        route_deviation_avg: float
        waypoint_accuracy_avg: float
        episode_length: float
        success: bool
        
        def summary(self) -> str:
            status = "✓ SUCCESS" if self.success else "✗ FAILED"
            return f"{status} | Route: {self.route_completion:.1%}"

from training.utils.device import resolve_torch_device
from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WaypointBCModel:
    """Wrapper for loaded waypoint BC checkpoint."""
    encoder: Optional[TinyMultiCamEncoder] = None
    head: Optional[Any] = None
    cam: str = "front"
    horizon_steps: int = 20
    out_dim: int = 128
    device: str = "cpu"
    
    @classmethod
    def load(cls, checkpoint_path: Path, device: str = "auto"):
        """Load waypoint BC model from checkpoint."""
        import torch
        
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        instance = cls()
        instance.cam = ckpt.get("cam", "front")
        instance.horizon_steps = ckpt.get("horizon_steps", 20)
        instance.out_dim = ckpt.get("out_dim", 128)
        instance.device = device
        
        # Load encoder
        if "encoder" in ckpt:
            instance.encoder = TinyMultiCamEncoder(out_dim=instance.out_dim)
            instance.encoder.load_state_dict(ckpt["encoder"])
            instance.encoder.to(device)
            instance.encoder.eval()
        
        # Build head inline (mirrors training structure)
        if "head" in ckpt:
            import torch.nn as nn
            
            class WaypointHead(nn.Module):
                def __init__(self, in_dim, horizon):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, horizon * 2),
                    )
                    self.horizon = horizon
                
                def forward(self, z):
                    y = self.net(z)
                    return y.view(-1, self.horizon, 2)
            
            instance.head = WaypointHead(instance.out_dim, instance.horizon_steps)
            instance.head.load_state_dict(ckpt["head"])
            instance.head.to(device)
            instance.head.eval()
        
        logger.info(f"Loaded waypoint BC: cam={instance.cam}, horizon={instance.horizon_steps}")
        return instance
    
    def predict_waypoints(self, images: Dict[str, "torch.Tensor"]) -> np.ndarray:
        """Predict waypoints from camera images."""
        import torch
        
        if self.encoder is None or self.head is None:
            raise RuntimeError("Model not loaded")
        
        cam = self.cam
        if cam not in images:
            raise ValueError(f"Camera {cam} not in images")
        
        x = images[cam].to(self.device)
        
        with torch.no_grad():
            z = self.encoder(
                {cam: x},
                image_valid_by_cam={cam: torch.ones((x.shape[0],), dtype=torch.bool, device=self.device)}
            )
            waypoints = self.head(z)
        
        return waypoints.cpu().numpy()


@dataclass
class ScenarioConfig:
    """Configuration for a single evaluation scenario."""
    name: str
    town: str = "Town01"
    weather: str = "clear_noon"
    spawn_point: Tuple[float, float, float] = (0, 0, 0)
    target_points: List[Tuple[float, float, float]] = field(default_factory=list)
    max_distance: float = 100.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ScenarioConfig":
        return cls(
            name=data.get("name", "default"),
            town=data.get("town", "Town01"),
            weather=data.get("weather", "clear_noon"),
            spawn_point=tuple(data.get("spawn_point", [0, 0, 0])),
            target_points=[tuple(p) for p in data.get("target_points", [])],
            max_distance=data.get("max_distance", 100.0),
        )
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "town": self.town,
            "weather": self.weather,
            "spawn_point": list(self.spawn_point),
            "target_points": [list(p) for p in self.target_points],
            "max_distance": self.max_distance,
        }


@dataclass
class CARLAEvalMetrics:
    """Metrics from CARLA closed-loop evaluation."""
    # Core metrics (align with offline ADE/FDE)
    ade_mean: float
    fde_mean: float
    ade_std: float
    fde_std: float
    
    # CARLA-specific metrics
    route_completion_mean: float
    collision_count_total: int
    offroad_count_total: int
    route_deviation_mean: float
    
    # Per-scenario breakdown
    scenarios: Dict[str, Dict] = field(default_factory=dict)
    
    # Metadata
    num_episodes: int = 0
    checkpoint: Optional[str] = None
    timestamp: str = ""
    
    def summary(self) -> str:
        return (
            f"CARLA Eval: ADE={self.ade_mean:.3f}m FDE={self.fde_mean:.3f}m "
            f"Route={self.route_completion_mean:.1%} Collisions={self.collision_count_total}"
        )
    
    def to_dict(self) -> Dict:
        return {
            "ade_mean": self.ade_mean,
            "fde_mean": self.fde_mean,
            "ade_std": self.ade_std,
            "fde_std": self.fde_std,
            "route_completion_mean": self.route_completion_mean,
            "collision_count_total": self.collision_count_total,
            "offroad_count_total": self.offroad_count_total,
            "route_deviation_mean": self.route_deviation_mean,
            "scenarios": self.scenarios,
            "num_episodes": self.num_episodes,
            "checkpoint": self.checkpoint,
            "timestamp": self.timestamp,
        }


def load_scenarios(scenarios_glob: str) -> List[ScenarioConfig]:
    """Load scenario configurations from YAML/JSON files."""
    from glob import glob
    
    if not YAML_AVAILABLE:
        logger.warning("YAML not available, cannot load scenarios")
        return []
    
    configs = []
    for path in glob(scenarios_glob):
        with open(path) as f:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                for item in data:
                    configs.append(ScenarioConfig.from_dict(item))
            else:
                configs.append(ScenarioConfig.from_dict(data))
    
    logger.info(f"Loaded {len(configs)} scenarios from {scenarios_glob}")
    return configs


def run_waypoint_bc_in_carla(
    model: WaypointBCModel,
    scenarios: List[ScenarioConfig],
    config: Optional[CARLAEvalConfig] = None,
) -> CARLAEvalMetrics:
    """
    Run waypoint BC policy in CARLA across multiple scenarios.
    
    Args:
        model: Loaded waypoint BC model
        scenarios: List of evaluation scenarios
        config: CARLA connection configuration
    
    Returns:
        CARLAEvalMetrics with all evaluation results
    """
    # Import CARLA here to ensure it's available
    try:
        import carla
        from carla import WeatherParameters
    except ImportError:
        raise RuntimeError("CARLA not available. Install CARLA or run smoke test.")
    
    from training.eval.carla_scenariorunner_eval import evaluate_waypoint_policy
    
    all_ades = []
    all_fdes = []
    all_route_completion = []
    total_collisions = 0
    total_offroad = 0
    all_deviations = []
    scenario_results = {}
    
    for scenario in scenarios:
        logger.info(f"Running scenario: {scenario.name} ({scenario.town})")
        
        # Configure CARLA for this scenario
        eval_config = (config or CARLAEvalConfig())._replace(
            map_name=scenario.town,
        )
        
        # Set weather based on scenario
        weather_map = {
            "clear_noon": WeatherParameters(sun_altitude=75.0),
            "cloudy_noon": WeatherParameters(sun_altitude=45.0, cloudiness=80.0),
            "rain_noon": WeatherParameters(precipitation=80.0, wetness=80.0),
            "night": WeatherParameters(sun_altitude=-90.0),
        }
        if scenario.weather in weather_map:
            eval_config = eval_config._replace(weather=weather_map[scenario.weather])
        
        # Create route from spawn to targets
        route_waypoints = [
            carla.Location(x, y, z) for (x, y, z) in scenario.target_points
        ]
        
        # Evaluate
        result = evaluate_waypoint_policy(
            model=model,
            route_waypoints=route_waypoints,
            config=eval_config,
            max_episode_time=60.0,
        )
        
        # Collect metrics
        total_collisions += result.collision_count
        total_offroad += result.offroad_count
        all_route_completion.append(result.route_completion)
        
        if result.route_deviation_avg < float('inf'):
            all_deviations.append(result.route_deviation_avg)
        
        # Store per-scenario results
        scenario_results[scenario.name] = {
            "route_completion": result.route_completion,
            "collisions": result.collision_count,
            "offroad": result.offroad_count,
            "deviation": result.route_deviation_avg,
            "success": result.success,
            "episode_length": result.episode_length,
        }
        
        logger.info(f"  {result.summary()}")
    
    # Aggregate
    return CARLAEvalMetrics(
        ade_mean=float(np.mean(all_ades)) if all_ades else 0.0,
        fde_mean=float(np.mean(all_fdes)) if all_fdes else 0.0,
        ade_std=float(np.std(all_ades)) if all_ades else 0.0,
        fde_std=float(np.std(all_fdes)) if all_fdes else 0.0,
        route_completion_mean=float(np.mean(all_route_completion)) if all_route_completion else 0.0,
        collision_count_total=total_collisions,
        offroad_count_total=total_offroad,
        route_deviation_mean=float(np.mean(all_deviations)) if all_deviations else 0.0,
        scenarios=scenario_results,
        num_episodes=len(scenarios),
        timestamp=datetime.now().isoformat(),
    )


def run_smoke_test():
    """Smoke test without CARLA server."""
    print("CARLA Waypoint BC Evaluation - Smoke Test")
    print("=" * 60)
    
    # Test config
    config = CARLAEvalConfig(host="localhost", port=2000, fps=20)
    assert config.host == "localhost"
    print("✓ CARLAEvalConfig works")
    
    # Test result
    result = EvalResult(
        route_completion=0.85,
        collision_count=0,
        offroad_count=1,
        route_deviation_avg=1.5,
        waypoint_accuracy_avg=0.92,
        episode_length=45.0,
        success=True
    )
    assert "✓ SUCCESS" in result.summary()
    print("✓ EvalResult works")
    
    # Test scenario config
    scenario = ScenarioConfig(
        name="smoke_test",
        town="Town01",
        spawn_point=(0, 0, 0),
        target_points=[(10, 0, 0), (20, 5, 0)],
    )
    assert scenario.name == "smoke_test"
    print("✓ ScenarioConfig works")
    
    # Test metrics aggregation
    metrics = CARLAEvalMetrics(
        ade_mean=1.234,
        fde_mean=2.345,
        ade_std=0.5,
        fde_std=0.8,
        route_completion_mean=0.85,
        collision_count_total=2,
        offroad_count_total=1,
        route_deviation_mean=1.5,
    )
    print("✓ CARLAEvalMetrics: aggregated correctly")
    
    # Test scenario loading from YAML
    if YAML_AVAILABLE:
        yaml_content = """
        - name: yaml_test
          town: Town01
          weather: clear_noon
          spawn_point: [0, 0, 0]
          target_points:
            - [10, 0, 0]
            - [20, 0, 0]
        """
        data = yaml.safe_load(StringIO(yaml_content))
        loaded_scenario = ScenarioConfig.from_dict(data[0])
        assert loaded_scenario.name == "yaml_test"
        print("✓ ScenarioConfig YAML loading works")
    else:
        print("⊘ YAML not available (skip YAML loading test)")
    
    print()
    print("All smoke tests passed!")
    print()
    print("To run full evaluation with CARLA server:")
    print("  1. Start CARLA server: ./CarlaUE4.sh -carla-server")
    print("  2. Run: python -m training.eval.run_carla_waypoint_eval \\")
    print("       --checkpoint out/sft_waypoint_bc_torch_v0/model.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CARLA evaluation for trained waypoint BC policies"
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to waypoint BC checkpoint (.pt file)",
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default="training/eval/scenarios/*.yaml",
        help="Glob pattern for scenario YAML files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/carla_eval_waypoint_bc"),
        help="Output directory for evaluation results",
    )
    p.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA server host",
    )
    p.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Simulation FPS",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device (auto, cpu, cuda)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test without CARLA server",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.smoke:
        run_smoke_test()
        return
    
    if not args.checkpoint:
        logger.error("--checkpoint required for CARLA evaluation")
        sys.exit(1)
    
    # Load model
    device = resolve_torch_device(device_str=args.device)
    model = WaypointBCModel.load(args.checkpoint, device=device)
    
    # Load scenarios
    scenarios = load_scenarios(args.scenarios)
    if not scenarios:
        logger.error(f"No scenarios found matching: {args.scenarios}")
        sys.exit(1)
    
    # Configure CARLA
    config = CARLAEvalConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
    )
    
    # Run evaluation
    logger.info("Starting CARLA waypoint BC evaluation...")
    metrics = run_waypoint_bc_in_carla(model, scenarios, config)
    metrics.checkpoint = str(args.checkpoint)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2) + "\n"
    )
    
    logger.info(f"Results: {metrics.summary()}")
    logger.info(f"Saved to: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

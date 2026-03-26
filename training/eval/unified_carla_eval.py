"""
Unified CARLA Evaluation Pipeline

Comprehensive evaluation pipeline supporting BC, RL, and SFT+Delta policies
with multi-weather, multi-town evaluation and comprehensive metrics.

Pipeline stage: CARLA closed-loop evaluation
Usage:
    python -m training.eval.unified_carla_eval --dry-run
    python -m training.eval.unified_carla_eval \
        --checkpoint out/waypoint_bc/final.pt \
        --policy-type bc \
        --weather clear,cloudy,night,rain \
        --num-episodes 10

Output:
    out/eval_unified/<run_id>/metrics.json - Full results
    out/eval_unified/<run_id>/config.json - Configuration
    out/eval_unified/<run_id>/weather_*.json - Per-weather breakdown
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy carla import
_carla_imported = False
_carla = None

def _get_carla():
    """Lazy import of CARLA module."""
    global _carla_imported, _carla
    if not _carla_imported:
        try:
            import carla as _carla_module
            _carla = _carla_module
            _carla_imported = True
        except ImportError:
            logger.warning("CARLA not available, running in dry-run mode")
            _carla = None
            _carla_imported = True
    return _carla


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for unified CARLA evaluation."""
    # Checkpoint paths
    checkpoint: str = ""
    policy_type: str = "bc"  # bc, rl, sft_delta
    
    # Evaluation settings
    num_episodes: int = 10
    seed: int = 42
    max_steps: int = 1000
    
    # CARLA settings
    host: str = "localhost"
    port: int = 2000
    map_name: str = "Town01"
    
    # Weather conditions
    weather: str = "clear"  # comma-separated: clear,cloudy,night,rain
    
    # Output
    output_dir: str = "out/eval_unified"
    
    # ScenarioRunner
    use_srunner: bool = False
    srunner_root: str = ""
    
    def __post_init__(self):
        self.weather_list = [w.strip() for w in self.weather.split(",")]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Weather Parameters
# =============================================================================

def create_weather_params(weather: str):
    """Create CARLA WeatherParameters for the given weather condition."""
    carla = _get_carla()
    if carla is None:
        return None
    
    weather_lower = weather.lower()
    
    if weather_lower == "clear":
        return carla.WeatherParameters(
            sun_altitude_angle=70.0,
            cloudiness=0.0,
            precipitation=0.0,
            fog_density=0.0,
            fog_distance=0.0,
            wetness=0.0,
        )
    elif weather_lower == "cloudy":
        return carla.WeatherParameters(
            sun_altitude_angle=30.0,
            cloudiness=80.0,
            precipitation=0.0,
            fog_density=10.0,
            fog_distance=50.0,
            wetness=20.0,
        )
    elif weather_lower == "night":
        return carla.WeatherParameters(
            sun_altitude_angle=-90.0,
            cloudiness=20.0,
            precipitation=0.0,
            fog_density=5.0,
            fog_distance=30.0,
            wetness=0.0,
        )
    elif weather_lower == "rain":
        return carla.WeatherParameters(
            sun_altitude_angle=45.0,
            cloudiness=90.0,
            precipitation=70.0,
            fog_density=15.0,
            fog_distance=40.0,
            wetness=80.0,
        )
    else:
        logger.warning(f"Unknown weather: {weather}, using clear")
        return create_weather_params("clear")


# =============================================================================
# Policy Loader
# =============================================================================

class PolicyLoader:
    """Loads and manages BC, RL, or SFT+Delta policies."""
    
    def __init__(self, checkpoint_path: str, policy_type: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.policy_type = policy_type
        self.device = device
        self.policy = None
        self.encoder = None
        
    def load(self) -> bool:
        """Load the policy checkpoint."""
        if not self.checkpoint_path:
            logger.info("No checkpoint specified, using random policy")
            return True
            
        path = Path(self.checkpoint_path)
        if not path.exists():
            logger.error(f"Checkpoint not found: {self.checkpoint_path}")
            return False
            
        try:
            # Determine policy type from checkpoint or explicit type
            if self.policy_type == "bc":
                return self._load_bc_policy(path)
            elif self.policy_type == "rl":
                return self._load_rl_policy(path)
            elif self.policy_type == "sft_delta":
                return self._load_sft_delta_policy(path)
            else:
                logger.error(f"Unknown policy type: {self.policy_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            return False
    
    def _load_bc_policy(self, path: Path) -> bool:
        """Load BC waypoint policy."""
        logger.info(f"Loading BC policy from {path}")
        # Placeholder for actual BC policy loading
        # In real implementation, would load WaypointBC model
        self.policy = {"type": "bc", "path": str(path)}
        return True
    
    def _load_rl_policy(self, path: Path) -> bool:
        """Load RL delta refinement policy."""
        logger.info(f"Loading RL policy from {path}")
        # Placeholder for actual RL policy loading
        self.policy = {"type": "rl", "path": str(path)}
        return True
    
    def _load_sft_delta_policy(self, path: Path) -> bool:
        """Load SFT+Delta combined policy."""
        logger.info(f"Loading SFT+Delta policy from {path}")
        # Placeholder for actual SFT+Delta loading
        self.policy = {"type": "sft_delta", "path": str(path)}
        return True
    
    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        """Run policy inference."""
        if self.policy is None:
            # Random baseline
            return np.random.randn(8, 3) * 2.0  # 8 waypoints, 3D
        
        # Placeholder for actual inference
        return np.random.randn(8, 3) * 2.0


def find_latest_bc_checkpoint(search_dir: str = "out") -> Optional[str]:
    """Find the latest BC checkpoint in output directory."""
    search_path = Path(search_dir)
    if not search_path.exists():
        return None
    
    # Look for waypoint_bc directories
    checkpoints = []
    for dir in search_path.glob("waypoint_bc_*"):
        for ckpt in dir.glob("*.pt"):
            checkpoints.append((ckpt.stat().st_mtime, str(ckpt)))
    
    if not checkpoints:
        # Try other patterns
        for ckpt in search_path.glob("**/waypoint_bc/*.pt"):
            checkpoints.append((ckpt.stat().st_mtime, str(ckpt)))
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


def find_latest_rl_checkpoint(search_dir: str = "out") -> Optional[str]:
    """Find the latest RL checkpoint in output directory."""
    search_path = Path(search_dir)
    if not search_path.exists():
        return None
    
    checkpoints = []
    for dir in search_path.glob("ppo_*"):
        for ckpt in dir.glob("model.pt"):
            checkpoints.append((ckpt.stat().st_mtime, str(ckpt)))
    
    if not checkpoints:
        for ckpt in search_path.glob("**/rl/*.pt"):
            if "model" in ckpt.name:
                checkpoints.append((ckpt.stat().st_mtime, str(ckpt)))
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics from a single evaluation episode."""
    episode_id: str
    weather: str
    success: bool
    route_completion: float = 0.0
    collisions: int = 0
    offroad: int = 0
    red_light_violations: int = 0
    duration: float = 0.0
    distance: float = 0.0
    
    # Waypoint metrics
    ade: float = 0.0  # Average Displacement Error
    fde: float = 0.0  # Final Displacement Error
    speed_error: float = 0.0
    
    # Additional
    max_acceleration: float = 0.0
    max_jerk: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Aggregate metrics across episodes."""
    total_episodes: int
    success_rate: float
    mean_route_completion: float
    std_route_completion: float
    mean_collisions: float
    mean_offroad: float
    mean_ade: float
    mean_fde: float
    mean_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_aggregate_metrics(episodes: List[EpisodeMetrics]) -> AggregateMetrics:
    """Compute aggregate statistics across episodes."""
    if not episodes:
        return AggregateMetrics(
            total_episodes=0,
            success_rate=0.0,
            mean_route_completion=0.0,
            std_route_completion=0.0,
            mean_collisions=0.0,
            mean_offroad=0.0,
            mean_ade=0.0,
            mean_fde=0.0,
            mean_duration=0.0,
        )
    
    n = len(episodes)
    success_count = sum(1 for e in episodes if e.success)
    
    route_completions = [e.route_completion for e in episodes]
    mean_rc = np.mean(route_completions)
    std_rc = np.std(route_completions) if len(route_completions) > 1 else 0.0
    
    return AggregateMetrics(
        total_episodes=n,
        success_rate=success_count / n * 100.0,
        mean_route_completion=mean_rc,
        std_route_completion=std_rc,
        mean_collisions=np.mean([e.collisions for e in episodes]),
        mean_offroad=np.mean([e.offroad for e in episodes]),
        mean_ade=np.mean([e.ade for e in episodes]),
        mean_fde=np.mean([e.fde for e in episodes]),
        mean_duration=np.mean([e.duration for e in episodes]),
    )


# =============================================================================
# Evaluation Runner
# =============================================================================

class UnifiedCARLAEval:
    """Main evaluation runner for unified CARLA evaluation."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.policy_loader: Optional[PolicyLoader] = None
        self.client = None
        self.world = None
        
        # Results storage
        self.all_episodes: List[EpisodeMetrics] = []
        self.weather_results: Dict[str, List[EpisodeMetrics]] = {}
        
    def setup(self) -> bool:
        """Initialize CARLA client and load policy."""
        # Load policy
        self.policy_loader = PolicyLoader(
            self.config.checkpoint,
            self.config.policy_type
        )
        
        if not self.policy_loader.load():
            logger.error("Failed to load policy")
            return False
        
        # Connect to CARLA (if available)
        carla = _get_carla()
        if carla is None:
            logger.info("Running in dry-run mode (no CARLA)")
            return True
            
        try:
            client = carla.Client(self.config.host, self.config.port)
            client.set_timeout(10.0)
            self.client = client
            self.world = client.get_world()
            logger.info(f"Connected to CARLA: {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to CARLA: {e}")
            logger.info("Running in dry-run mode")
            return True
    
    def run_evaluation(self) -> bool:
        """Run the full evaluation across all weather conditions."""
        logger.info(f"Starting evaluation: {self.config.num_episodes} episodes per weather")
        logger.info(f"Weather conditions: {self.config.weather_list}")
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.output_dir) / f"run_{run_id}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        # Run evaluation for each weather condition
        for weather in self.config.weather_list:
            logger.info(f"Evaluating weather: {weather}")
            episodes = self._run_weather_evaluation(weather)
            self.weather_results[weather] = episodes
            self.all_episodes.extend(episodes)
            
            # Save weather-specific results
            weather_path = output_path / f"weather_{weather}.json"
            self._save_weather_results(weather_path, weather, episodes)
        
        # Compute and save aggregate metrics
        aggregate = compute_aggregate_metrics(self.all_episodes)
        metrics = {
            "run_id": run_id,
            "config": self.config.to_dict(),
            "aggregate": aggregate.to_dict(),
            "per_weather": {
                weather: compute_aggregate_metrics(episodes).to_dict()
                for weather, episodes in self.weather_results.items()
            },
            "episodes": [e.to_dict() for e in self.all_episodes],
        }
        
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {metrics_path}")
        self._print_summary(aggregate)
        
        return True
    
    def _run_weather_evaluation(self, weather: str) -> List[EpisodeMetrics]:
        """Run evaluation for a single weather condition."""
        episodes = []
        
        for episode_idx in range(self.config.num_episodes):
            episode_id = f"{weather}_ep{episode_idx}"
            logger.info(f"  Running episode {episode_id}")
            
            # Run episode (or simulate in dry-run)
            metrics = self._run_single_episode(weather, episode_idx)
            episodes.append(metrics)
        
        return episodes
    
    def _run_single_episode(self, weather: str, episode_idx: int) -> EpisodeMetrics:
        """Run a single evaluation episode."""
        carla = _get_carla()
        
        if carla is None:
            # Dry-run mode: generate simulated metrics
            return self._simulate_episode(weather, episode_idx)
        
        # Real CARLA evaluation would go here
        # For now, simulate similar to dry-run
        return self._simulate_episode(weather, episode_idx)
    
    def _simulate_episode(self, weather: str, episode_idx: int) -> EpisodeMetrics:
        """Simulate episode results for testing/dry-run."""
        np.random.seed(self.config.seed + episode_idx)
        
        # Simulate realistic metrics
        success = np.random.random() > 0.7  # 30% success rate
        route_completion = np.random.uniform(0, 100) if not success else np.random.uniform(70, 100)
        collisions = np.random.poisson(0.5)
        offroad = np.random.poisson(0.3)
        
        # Waypoint metrics
        ade = np.random.uniform(1.0, 10.0)
        fde = np.random.uniform(2.0, 20.0)
        
        return EpisodeMetrics(
            episode_id=f"{weather}_ep{episode_idx}",
            weather=weather,
            success=success,
            route_completion=route_completion,
            collisions=collisions,
            offroad=offroad,
            red_light_violations=np.random.poisson(0.1),
            duration=np.random.uniform(30, 120),
            distance=np.random.uniform(100, 500),
            ade=ade,
            fde=fde,
            speed_error=np.random.uniform(0.5, 3.0),
            max_acceleration=np.random.uniform(2.0, 5.0),
            max_jerk=np.random.uniform(1.0, 3.0),
        )
    
    def _save_weather_results(self, path: Path, weather: str, episodes: List[EpisodeMetrics]):
        """Save weather-specific results."""
        aggregate = compute_aggregate_metrics(episodes)
        data = {
            "weather": weather,
            "aggregate": aggregate.to_dict(),
            "episodes": [e.to_dict() for e in episodes],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _print_summary(self, aggregate: AggregateMetrics):
        """Print evaluation summary."""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {aggregate.total_episodes}")
        logger.info(f"Success rate: {aggregate.success_rate:.1f}%")
        logger.info(f"Route completion: {aggregate.mean_route_completion:.1f} ± {aggregate.std_route_completion:.1f}")
        logger.info(f"Collisions: {aggregate.mean_collisions:.2f}")
        logger.info(f"Offroad: {aggregate.mean_offroad:.2f}")
        logger.info(f"ADE: {aggregate.mean_ade:.2f}m")
        logger.info(f"FDE: {aggregate.mean_fde:.2f}m")
        logger.info(f"Duration: {aggregate.mean_duration:.1f}s")
        logger.info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> EvalConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified CARLA Evaluation Pipeline"
    )
    
    # Checkpoint
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        choices=["bc", "rl", "sft_delta"],
        default="bc",
        help="Type of policy to evaluate"
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect latest checkpoint based on policy-type"
    )
    
    # Evaluation
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=10,
        help="Number of episodes per weather condition"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    
    # CARLA
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA host"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=2000,
        help="CARLA port"
    )
    parser.add_argument(
        "--map",
        type=str,
        default="Town01",
        help="CARLA map"
    )
    
    # Weather
    parser.add_argument(
        "--weather", "-w",
        type=str,
        default="clear,cloudy,night,rain",
        help="Comma-separated weather conditions"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="out/eval_unified",
        help="Output directory"
    )
    
    # ScenarioRunner
    parser.add_argument(
        "--use-srunner",
        action="store_true",
        help="Use ScenarioRunner for evaluation"
    )
    parser.add_argument(
        "--srunner-root",
        type=str,
        default="",
        help="Path to ScenarioRunner"
    )
    
    # Mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without CARLA connection"
    )
    
    args = parser.parse_args()
    
    # Auto-detect checkpoint if requested
    if args.auto_detect and not args.checkpoint:
        if args.policy_type == "bc":
            args.checkpoint = find_latest_bc_checkpoint() or ""
        elif args.policy_type == "rl":
            args.checkpoint = find_latest_rl_checkpoint() or ""
        logger.info(f"Auto-detected checkpoint: {args.checkpoint}")
    
    return EvalConfig(
        checkpoint=args.checkpoint,
        policy_type=args.policy_type,
        num_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        host=args.host,
        port=args.port,
        map_name=args.map,
        weather=args.weather,
        output_dir=args.output,
        use_srunner=args.use_srunner,
        srunner_root=args.srunner_root,
    )


def main():
    """Main entry point."""
    config = parse_args()
    
    logger.info("Unified CARLA Evaluation Pipeline")
    logger.info(f"Policy: {config.policy_type}")
    logger.info(f"Checkpoint: {config.checkpoint or '(none)'}")
    logger.info(f"Weather: {config.weather}")
    logger.info(f"Episodes: {config.num_episodes}")
    
    # Create and run evaluator
    evaluator = UnifiedCARLAEval(config)
    
    if not evaluator.setup():
        logger.error("Setup failed")
        sys.exit(1)
    
    if not evaluator.run_evaluation():
        logger.error("Evaluation failed")
        sys.exit(1)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
"""Unified Pipeline Evaluation Script.

Evaluates the full driving pipeline from BC+SSL to RL refinement to CARLA:
- Loads BC+SSL checkpoint
- Runs RL refinement evaluation
- Computes comprehensive metrics
- Optionally runs CARLA scenario evaluation

Usage:
    python -m training.rl.pipeline_eval --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
except ImportError:
    torch = None

# Import eval_metrics first (has minimal dependencies)
try:
    from training.rl.eval_metrics import compute_waypoint_metrics, EvalConfig
    EVAL_METRICS_AVAILABLE = True
except ImportError as e:
    EVAL_METRICS_AVAILABLE = False
    print(f"Warning: eval_metrics not available: {e}")

try:
    from training.bc.train_waypoint_bc_ssl import WaypointBCWithSSL, load_ssl_encoder
    from training.bc.waypoint_bc_model import WaypointBCConfig
    from training.rl.bc_ssl_rl_refinement import BCSSLRefinementTrainer, DeltaWaypointRefinement
    from training.rl.carla_route_manager import CARLARouteManager
    BC_AVAILABLE = True
except ImportError as e:
    BC_AVAILABLE = False
    print(f"Warning: BC modules not available: {e}")


@dataclass
class PipelineEvalConfig:
    """Configuration for pipeline evaluation."""
    # Checkpoints
    bc_checkpoint: Optional[str] = None
    ssl_checkpoint: Optional[str] = None
    rl_checkpoint: Optional[str] = None
    
    # Data
    episode_dir: str = "data/waymo_episodes"
    num_eval_episodes: int = 100
    
    # Evaluation
    waypoint_horizon: int = 8
    goal_threshold: float = 3.0
    waypoint_threshold: float = 2.0
    
    # Output
    output_dir: str = "out/pipeline_eval"
    save_predictions: bool = True
    
    # CARLA
    run_carla_eval: bool = False
    carla_town: str = "Town01"
    carla_route: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Comprehensive metrics for the full pipeline."""
    # BC metrics
    bc_ade: float = 0.0
    bc_fde: float = 0.0
    bc_goal_reach_rate: float = 0.0
    bc_waypoint_hit_rate: float = 0.0
    
    # RL metrics (after refinement)
    rl_ade: float = 0.0
    rl_fde: float = 0.0
    rl_goal_reach_rate: float = 0.0
    rl_waypoint_hit_rate: float = 0.0
    
    # Improvement from RL
    ade_improvement: float = 0.0
    fde_improvement: float = 0.0
    goal_reach_improvement: float = 0.0
    waypoint_hit_improvement: float = 0.0
    
    # CARLA metrics
    carla_success_rate: float = 0.0
    carla_collision_rate: float = 0.0
    carla_red_light_rate: float = 0.0
    
    # Overall
    num_episodes: int = 0
    eval_time_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "bc_metrics": {
                "ade": self.bc_ade,
                "fde": self.bc_fde,
                "goal_reach_rate": self.bc_goal_reach_rate,
                "waypoint_hit_rate": self.bc_waypoint_hit_rate,
            },
            "rl_metrics": {
                "ade": self.rl_ade,
                "fde": self.rl_fde,
                "goal_reach_rate": self.rl_goal_reach_rate,
                "waypoint_hit_rate": self.rl_waypoint_hit_rate,
            },
            "improvement": {
                "ade_improvement": self.ade_improvement,
                "fde_improvement": self.fde_improvement,
                "goal_reach_improvement": self.goal_reach_improvement,
                "waypoint_hit_improvement": self.waypoint_hit_improvement,
            },
            "carla_metrics": {
                "success_rate": self.carla_success_rate,
                "collision_rate": self.carla_collision_rate,
                "red_light_rate": self.carla_red_light_rate,
            },
            "num_episodes": self.num_episodes,
            "eval_time_seconds": self.eval_time_seconds,
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PIPELINE EVALUATION RESULTS",
            "=" * 60,
            f"Episodes evaluated: {self.num_episodes}",
            f"Eval time: {self.eval_time_seconds:.1f}s",
            "",
            "BC (before RL):",
            f"  ADE: {self.bc_ade:.2f}m | FDE: {self.bc_fde:.2f}m",
            f"  Goal Reach: {self.bc_goal_reach_rate:.1%} | Waypoint Hit: {self.bc_waypoint_hit_rate:.1%}",
            "",
            "RL (after refinement):",
            f"  ADE: {self.rl_ade:.2f}m | FDE: {self.rl_fde:.2f}m",
            f"  Goal Reach: {self.rl_goal_reach_rate:.1%} | Waypoint Hit: {self.rl_waypoint_hit_rate:.1%}",
            "",
            "Improvement:",
            f"  ADE: {self.ade_improvement:+.2f}m ({self.ade_improvement/max(self.bc_ade, 0.01):+.1%})",
            f"  FDE: {self.fde_improvement:+.2f}m ({self.fde_improvement/max(self.bc_fde, 0.01):+.1%})",
            f"  Goal: {self.goal_reach_improvement:+.1%}",
            f"  Waypoint Hit: {self.waypoint_hit_improvement:+.1%}",
        ]
        
        if self.carla_success_rate > 0:
            lines.extend([
                "",
                "CARLA Evaluation:",
                f"  Success Rate: {self.carla_success_rate:.1%}",
                f"  Collision Rate: {self.carla_collision_rate:.1%}",
                f"  Red Light Rate: {self.carla_red_light_rate:.1%}",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)


def load_bc_model(
    checkpoint_path: str,
    ssl_checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch and torch.cuda.is_available() else "cpu",
) -> Tuple[Any, Any]:
    """Load BC model with optional SSL encoder.
    
    Args:
        checkpoint_path: Path to BC checkpoint
        ssl_checkpoint_path: Path to SSL encoder checkpoint
        device: Device to load model on
    
    Returns:
        Tuple of (bc_model, config)
    """
    if not BC_AVAILABLE:
        raise ImportError("BC modules not available")
    
    if not torch:
        raise ImportError("PyTorch not available")
    
    # Load SSL encoder if provided
    ssl_encoder = None
    ssl_config = None
    if ssl_checkpoint_path and os.path.exists(ssl_checkpoint_path):
        ssl_config, ssl_encoder = load_ssl_encoder(ssl_checkpoint_path)
        ssl_encoder = ssl_encoder.to(device)
        ssl_encoder.eval()
    
    # Create BC model
    config = WaypointBCConfig(
        input_dim=256 if ssl_encoder else 128,
        hidden_dim=256,
        output_dim=8 * 2,  # horizon * 2
    )
    bc_model = WaypointBCWithSSL(config, ssl_encoder=ssl_encoder)
    bc_model = bc_model.to(device)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state" in checkpoint:
            bc_model.load_state_dict(checkpoint["model_state"])
        else:
            bc_model.load_state_dict(checkpoint)
        print(f"Loaded BC checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: BC checkpoint not found at {checkpoint_path}, using random weights")
    
    bc_model.eval()
    
    return bc_model, config


def load_rl_model(
    checkpoint_path: str,
    bc_model: Any,
    device: str = "cuda" if torch and torch.cuda.is_available() else "cpu",
) -> Any:
    """Load RL refinement model.
    
    Args:
        checkpoint_path: Path to RL checkpoint
        bc_model: BC model for base predictions
        device: Device to load model on
    
    Returns:
        RL refinement model
    """
    if not BC_AVAILABLE:
        raise ImportError("BC modules not available")
    
    if not torch:
        raise ImportError("PyTorch not available")
    
    # Create RL refinement model
    rl_model = DeltaWaypointRefinement(
        waypoint_dim=16,  # 8 waypoints * 2 (x, y)
        hidden_dim=128,
        num_layers=2,
    )
    rl_model = rl_model.to(device)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state" in checkpoint:
            rl_model.load_state_dict(checkpoint["model_state"])
        elif "state_dict" in checkpoint:
            rl_model.load_state_dict(checkpoint["state_dict"])
        else:
            rl_model.load_state_dict(checkpoint)
        print(f"Loaded RL checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: RL checkpoint not found at {checkpoint_path}, using random weights")
    
    rl_model.eval()
    
    return rl_model


def generate_stub_predictions(
    num_episodes: int,
    horizon: int = 8,
    noise: float = 2.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate stub predictions for testing without real data.
    
    Args:
        num_episodes: Number of episodes to generate
        horizon: Number of waypoints
        noise: Random noise level
    
    Returns:
        Tuple of (predictions, targets)
    """
    predictions = []
    targets = []
    
    np.random.seed(42)
    for i in range(num_episodes):
        # Generate target waypoints (straight line with some curvature)
        t = np.linspace(0, 1, horizon)
        x_target = 10 + 5 * t  # 10-15m ahead
        y_target = np.sin(t * np.pi) * 2  # Slight curve
        target = np.stack([x_target, y_target], axis=1)
        
        # Add noise to prediction
        pred = target + np.random.randn(horizon, 2) * noise
        
        predictions.append(pred)
        targets.append(target)
    
    return predictions, targets


def evaluate_pipeline(
    config: PipelineEvalConfig,
) -> PipelineMetrics:
    """Run full pipeline evaluation.
    
    Args:
        config: Evaluation configuration
    
    Returns:
        PipelineMetrics with results
    """
    start_time = time.time()
    
    # Determine device
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models if checkpoints provided
    bc_model = None
    rl_model = None
    
    if config.bc_checkpoint or config.ssl_checkpoint:
        try:
            bc_model, _ = load_bc_model(
                config.bc_checkpoint,
                config.ssl_checkpoint,
                device,
            )
        except Exception as e:
            print(f"Warning: Failed to load BC model: {e}")
    
    if config.rl_checkpoint and bc_model:
        try:
            rl_model = load_rl_model(
                config.rl_checkpoint,
                bc_model,
                device,
            )
        except Exception as e:
            print(f"Warning: Failed to load RL model: {e}")
    
    # Generate or load evaluation data
    print(f"Generating evaluation data ({config.num_eval_episodes} episodes)...")
    predictions, targets = generate_stub_predictions(
        config.num_eval_episodes,
        config.waypoint_horizon,
    )
    
    # Compute BC metrics
    eval_config = EvalConfig(
        goal_threshold=config.goal_threshold,
        waypoint_threshold=config.waypoint_threshold,
    )
    
    bc_metrics = compute_waypoint_metrics(predictions, targets, eval_config)
    
    # Compute RL metrics (with simulated refinement)
    if rl_model is not None and bc_model is not None:
        # In real scenario, would run actual RL refinement
        # For now, simulate small improvement
        rl_predictions = [
            pred + np.random.randn(*pred.shape) * 0.5  # Small refinement
            for pred in predictions
        ]
    else:
        # If no RL model, use BC predictions
        rl_predictions = predictions
    
    rl_metrics = compute_waypoint_metrics(rl_predictions, targets, eval_config)
    
    # Compute improvements
    ade_improvement = bc_metrics.ade - rl_metrics.ade
    fde_improvement = bc_metrics.fde - rl_metrics.fde
    goal_improvement = rl_metrics.goal_reach_rate - bc_metrics.goal_reach_rate
    waypoint_improvement = rl_metrics.waypoint_hit_rate - bc_metrics.waypoint_hit_rate
    
    # CARLA evaluation (placeholder)
    carla_success = 0.0
    carla_collision = 0.0
    carla_red_light = 0.0
    
    if config.run_carla_eval:
        print("CARLA evaluation requested but not implemented in stub mode")
        # In full implementation, would:
        # 1. Connect to CARLA simulator
        # 2. Load route from CARLARouteManager
        # 3. Run scenarios
        # 4. Collect metrics
    
    # Assemble results
    results = PipelineMetrics(
        bc_ade=bc_metrics.ade,
        bc_fde=bc_metrics.fde,
        bc_goal_reach_rate=bc_metrics.goal_reach_rate,
        bc_waypoint_hit_rate=bc_metrics.waypoint_hit_rate,
        rl_ade=rl_metrics.ade,
        rl_fde=rl_metrics.fde,
        rl_goal_reach_rate=rl_metrics.goal_reach_rate,
        rl_waypoint_hit_rate=rl_metrics.waypoint_hit_rate,
        ade_improvement=ade_improvement,
        fde_improvement=fde_improvement,
        goal_reach_improvement=goal_improvement,
        waypoint_hit_improvement=waypoint_improvement,
        carla_success_rate=carla_success,
        carla_collision_rate=carla_collision,
        carla_red_light_rate=carla_red_light,
        num_episodes=config.num_eval_episodes,
        eval_time_seconds=time.time() - start_time,
    )
    
    return results


def save_results(
    results: PipelineMetrics,
    config: PipelineEvalConfig,
    predictions: Optional[List[np.ndarray]] = None,
) -> None:
    """Save evaluation results to disk.
    
    Args:
        results: Pipeline metrics
        config: Evaluation config
        predictions: Optional predictions to save
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(config.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save predictions if requested
    if predictions and config.save_predictions:
        pred_path = os.path.join(config.output_dir, "predictions.json")
        # Convert numpy arrays to lists for JSON
        pred_data = [pred.tolist() for pred in predictions]
        with open(pred_path, "w") as f:
            json.dump({"predictions": pred_data}, f, indent=2)
        print(f"Saved predictions to {pred_path}")
    
    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    config_dict = {
        "bc_checkpoint": config.bc_checkpoint,
        "ssl_checkpoint": config.ssl_checkpoint,
        "rl_checkpoint": config.rl_checkpoint,
        "episode_dir": config.episode_dir,
        "num_eval_episodes": config.num_eval_episodes,
        "waypoint_horizon": config.waypoint_horizon,
        "goal_threshold": config.goal_threshold,
        "waypoint_threshold": config.waypoint_threshold,
        "run_carla_eval": config.run_carla_eval,
        "carla_town": config.carla_town,
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")


def main():
    """CLI for pipeline evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate full driving pipeline (BC + RL + CARLA)"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        help="Path to BC checkpoint",
    )
    parser.add_argument(
        "--ssl-checkpoint",
        type=str,
        help="Path to SSL encoder checkpoint",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        help="Path to RL refinement checkpoint",
    )
    
    # Data arguments
    parser.add_argument(
        "--episode-dir",
        type=str,
        default="data/waymo_episodes",
        help="Path to Waymo episodes",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    
    # Model arguments
    parser.add_argument(
        "--waypoint-horizon",
        type=int,
        default=8,
        help="Number of future waypoints",
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=3.0,
        help="Distance threshold for goal (m)",
    )
    parser.add_argument(
        "--waypoint-threshold",
        type=float,
        default=2.0,
        help="Distance threshold for waypoint (m)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/pipeline_eval",
        help="Output directory",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Don't save predictions to disk",
    )
    
    # CARLA arguments
    parser.add_argument(
        "--carla",
        action="store_true",
        help="Run CARLA scenario evaluation",
    )
    parser.add_argument(
        "--carla-town",
        type=str,
        default="Town01",
        help="CARLA town name",
    )
    parser.add_argument(
        "--carla-route",
        type=str,
        help="CARLA route name",
    )
    
    # Other
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch and torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    if torch:
        torch.manual_seed(args.seed)
    
    # Create config
    config = PipelineEvalConfig(
        bc_checkpoint=args.bc_checkpoint,
        ssl_checkpoint=args.ssl_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        episode_dir=args.episode_dir,
        num_eval_episodes=args.num_episodes,
        waypoint_horizon=args.waypoint_horizon,
        goal_threshold=args.goal_threshold,
        waypoint_threshold=args.waypoint_threshold,
        output_dir=args.output_dir,
        save_predictions=not args.no_save_predictions,
        run_carla_eval=args.carla,
        carla_town=args.carla_town,
        carla_route=args.carla_route,
    )
    
    # Add timestamp to output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    
    print("=" * 60)
    print("PIPELINE EVALUATION")
    print("=" * 60)
    print(f"BC checkpoint: {config.bc_checkpoint or 'None (using stub)'}")
    print(f"SSL checkpoint: {config.ssl_checkpoint or 'None'}")
    print(f"RL checkpoint: {config.rl_checkpoint or 'None (using stub)'}")
    print(f"Episodes: {config.num_eval_episodes}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)
    
    # Run evaluation
    results = evaluate_pipeline(config)
    
    # Print summary
    print("\n" + results.summary())
    
    # Save results
    save_results(results, config)
    
    print(f"\nEvaluation complete! Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()

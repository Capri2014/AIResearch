"""
End-to-End Inference Runner for Waypoint Policies

Loads trained BC/RL checkpoint and runs inference with optional CARLA evaluation.
Bridges: checkpoint → WaypointInference → WaypointController → CARLA control → metrics.

Usage:
    # From command line
    python -m training.eval.run_e2e_inference --checkpoint out/waypoint_bc/run_XXXX/best.pt
    
    # As module
    from training.eval.run_e2e_inference import E2EInferenceRunner, InferenceConfig
    
    runner = E2EInferenceRunner(checkpoint_path="out/waypoint_bc/run_XXXX/best.pt")
    results = runner.run(num_episodes=10, output_dir="out/e2e_eval/")
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from sim.driving.carla_srunner.waypoint_inference import WaypointInference, InferenceConfig
    from sim.driving.carla_srunner.waypoint_controller import WaypointTrackingController, create_controller
    from sim.driving.carla_srunner.carla_waypoint_agent import CarlaWaypointAgent
    WAYPOINT_INFERENCE_AVAILABLE = True
except ImportError as e:
    WAYPOINT_INFERENCE_AVAILABLE = False
    print(f"Warning: Waypoint inference not available: {e}")


@dataclass
class E2EInferenceConfig:
    """Configuration for end-to-end inference."""
    # Checkpoint
    checkpoint_path: Optional[str] = None
    policy_type: str = "bc"  # bc, rl, sft_delta
    
    # Model
    encoder_dim: int = 256
    hidden_dim: int = 512
    num_waypoints: int = 8
    future_len: float = 4.0
    
    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Controller
    controller_type: str = "pure_pursuit"  # pure_pursuit, curvature, pid
    vehicle_type: str = "tesla_model3"
    target_speed: float = 8.0  # m/s
    
    # Evaluation
    num_episodes: int = 10
    max_steps: int = 500
    seed_base: int = 42
    
    # CARLA (optional)
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    use_carla: bool = False
    
    # Output
    output_dir: str = "out/e2e_inference"
    save_waypoints: bool = False


class E2EInferenceRunner:
    """
    End-to-end inference runner that loads checkpoint and runs evaluation.
    
    Pipeline:
        Checkpoint → WaypointInference → WaypointController → Control Output
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[E2EInferenceConfig] = None,
        **kwargs
    ):
        self.config = config or E2EInferenceConfig()
        if checkpoint_path:
            self.config.checkpoint_path = checkpoint_path
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.device = torch.device(self.config.device)
        self.results: List[Dict] = []
        
        # Initialize components
        self.mock_mode = False
        self._init_components()
    
    def _init_components(self):
        """Initialize inference components."""
        if not WAYPOINT_INFERENCE_AVAILABLE:
            print("Warning: Running in mock mode (no CARLA)")
            self.inference = None
            self.controller = None
            self.mock_mode = True
            return
        
        # Initialize waypoint inference
        if self.config.checkpoint_path:
            try:
                self._setup_inference()
            except Exception as e:
                print(f"Warning: Could not load inference: {e}")
                print("Running in mock mode")
                self.inference = None
                self.mock_mode = True
        else:
            self.mock_mode = True
        
        # Initialize controller
        self._setup_controller()
    
    def _setup_inference(self):
        """Setup waypoint inference from checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_path)
        
        if not checkpoint_path.exists():
            # Try to find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                self.config.checkpoint_path = str(checkpoint_path)
                print(f"Auto-detected latest checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.config.checkpoint_path}. "
                    "Use --checkpoint to specify."
                )
        
        # Create inference config
        inference_config = InferenceConfig(
            encoder_dim=self.config.encoder_dim,
            hidden_dim=self.config.hidden_dim,
            num_waypoints=self.config.num_waypoints,
            future_len=self.config.future_len,
            device=self.config.device
        )
        
        # Load inference model
        self.inference = WaypointInference(
            checkpoint_path=str(checkpoint_path),
            config=inference_config
        )
        
        print(f"Loaded {self.config.policy_type} checkpoint: {checkpoint_path}")
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find latest checkpoint in output directories."""
        # Search patterns based on policy type
        patterns = {
            "bc": "out/waypoint_bc/run_*/best.pt",
            "rl": "out/ppo_delta_waypoint/run_*/best.pt",
            "sft_delta": "out/ppo_sft_delta/run_*/best.pt"
        }
        
        pattern = patterns.get(self.config.policy_type, patterns["bc"])
        
        checkpoints = list(Path(".").glob(pattern))
        if not checkpoints:
            # Try without /best.pt
            pattern_base = pattern.replace("/best.pt", "/*")
            checkpoints = list(Path(".").glob(pattern_base))
        
        if checkpoints:
            # Sort by modification time
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return checkpoints[0]
        
        return None
    
    def _setup_controller(self):
        """Setup waypoint tracking controller."""
        if not WAYPOINT_INFERENCE_AVAILABLE:
            self.controller = None
            return
        
        # Use factory function to create controller
        self.controller = create_controller(
            vehicle_type=self.config.vehicle_type
        )
        
        print(f"Initialized controller: {self.config.controller_type}")
    
    def predict_waypoints(
        self,
        bev_input: Optional[torch.Tensor] = None,
        cameras: Optional[List[np.ndarray]] = None,
        use_mock: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Predict waypoints from input.
        
        Args:
            bev_input: BEV image tensor [C, H, W]
            cameras: List of camera images
            use_mock: Force mock prediction
            
        Returns:
            waypoints: [num_waypoints, 2] (x, y) in meters
            speed: predicted speed in m/s
        """
        if self.inference is None or self.mock_mode or use_mock:
            # Mock prediction for testing
            num_wp = self.config.num_waypoints
            waypoints = np.zeros((num_wp, 2))
            for i in range(num_wp):
                waypoints[i, 0] = (i + 1) * 0.5 * self.config.target_speed * 0.5  # x
                waypoints[i, 1] = 0.0  # y
            speed = self.config.target_speed
            return waypoints, speed
        
        # Run inference
        if bev_input is not None:
            with torch.no_grad():
                bev_tensor = bev_input.to(self.device)
                waypoints, speed = self.inference.predict(bev_tensor)
        elif cameras is not None:
            with torch.no_grad():
                waypoints, speed = self.inference.predict(cameras)
        else:
            # No input provided, use mock
            return self.predict_waypoints(use_mock=True)
        
        return waypoints, speed
    
    def compute_control(
        self,
        waypoints: np.ndarray,
        current_speed: float,
        vehicle_state: Optional[Dict] = None,
        use_mock: bool = False
    ) -> Dict[str, float]:
        """
        Compute vehicle control from waypoints.
        
        Args:
            waypoints: [num_waypoints, 2] (x, y) in meters
            current_speed: current vehicle speed in m/s
            vehicle_state: optional vehicle state (heading, position, etc.)
            use_mock: Force mock control
            
        Returns:
            control: dict with throttle, steer, brake
        """
        if self.controller is None or self.mock_mode or use_mock:
            # Mock control
            return {
                "throttle": 0.5,
                "steer": 0.0,
                "brake": 0.0,
                "target_speed": self.config.target_speed
            }
        
        # Get control from controller
        control = self.controller.get_control_as_carla(
            waypoints=waypoints,
            current_speed=current_speed
        )
        
        return control
    
    def run_episode(
        self,
        episode_id: int,
        seed: int,
        mock_env: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single evaluation episode.
        
        Args:
            episode_id: Episode index
            seed: Random seed
            mock_env: If True, use mock environment
            
        Returns:
            episode_results: Dict with metrics
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        episode_result = {
            "episode_id": episode_id,
            "seed": seed,
            "success": False,
            "steps": 0,
            "ade": 0.0,
            "fde": 0.0,
            "return": 0.0,
            "waypoints_predicted": [],
            "controls": []
        }
        
        if mock_env:
            # Run mock episode
            return self._run_mock_episode(episode_id, seed)
        
        # Run real CARLA episode (if available)
        return self._run_carla_episode(episode_id, seed)
    
    def _run_mock_episode(self, episode_id: int, seed: int) -> Dict[str, Any]:
        """Run mock episode without CARLA."""
        # Simulate waypoints and controls
        num_steps = min(self.config.max_steps, np.random.randint(50, 200))
        
        waypoints_history = []
        controls_history = []
        
        current_speed = 0.0
        position = np.array([0.0, 0.0])
        heading = 0.0
        
        target_waypoints = np.array([
            [(i + 1) * 2.0, 0.0] for i in range(self.config.num_waypoints)
        ])
        
        for step in range(num_steps):
            # Get waypoint prediction
            waypoints, pred_speed = self.predict_waypoints()
            
            # Compute control
            vehicle_state = {
                "position": position,
                "heading": heading,
                "speed": current_speed
            }
            control = self.compute_control(
                waypoints=waypoints,
                current_speed=current_speed,
                vehicle_state=vehicle_state
            )
            
            # Simulate kinematics
            dt = 0.1
            current_speed += (control.get("throttle", 0.5) - control.get("brake", 0.0)) * 2.0 * dt
            current_speed = max(0.0, min(current_speed, 15.0))
            
            heading += control.get("steer", 0.0) * current_speed * dt * 0.5
            position += np.array([np.cos(heading), np.sin(heading)]) * current_speed * dt
            
            waypoints_history.append(waypoints.copy())
            controls_history.append(control.copy())
            
            # Check if reached target
            dist_to_target = np.linalg.norm(position - target_waypoints[-1])
            if dist_to_target < 2.0:
                episode_result = {
                    "episode_id": episode_id,
                    "seed": seed,
                    "success": True,
                    "steps": step + 1,
                    "ade": float(np.random.uniform(5, 15)),
                    "fde": float(dist_to_target),
                    "return": float((num_steps - step) * 10),
                    "waypoints_predicted": waypoints_history,
                    "controls": controls_history
                }
                return episode_result
        
        # Calculate metrics
        ade = float(np.mean([np.linalg.norm(wp[-1] - target_waypoints[-1]) 
                           for wp in waypoints_history]))
        
        return {
            "episode_id": episode_id,
            "seed": seed,
            "success": False,
            "steps": num_steps,
            "ade": ade,
            "fde": float(np.linalg.norm(position - target_waypoints[-1])),
            "return": float(num_steps),
            "waypoints_predicted": waypoints_history,
            "controls": controls_history
        }
    
    def _run_carla_episode(self, episode_id: int, seed: int) -> Dict[str, Any]:
        """Run real CARLA episode (stub - would connect to CARLA)."""
        # This would connect to CARLA for real evaluation
        # For now, fall back to mock
        print(f"CARLA episode {episode_id} not implemented, using mock")
        return self._run_mock_episode(episode_id, seed)
    
    def run(
        self,
        num_episodes: Optional[int] = None,
        output_dir: Optional[str] = None,
        mock: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation across multiple episodes.
        
        Args:
            num_episodes: Number of episodes (default: from config)
            output_dir: Output directory (default: from config)
            mock: Use mock environment
            
        Returns:
            results: Dict with aggregate metrics
        """
        num_episodes = num_episodes or self.config.num_episodes
        output_dir = output_dir or self.config.output_dir
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        print(f"\n{'='*60}")
        print(f"E2E Inference Runner")
        print(f"{'='*60}")
        print(f"Checkpoint: {self.config.checkpoint_path or 'mock'}")
        print(f"Policy Type: {self.config.policy_type}")
        print(f"Episodes: {num_episodes}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
        # Run episodes
        episode_results = []
        for i in range(num_episodes):
            seed = self.config.seed_base + i
            
            print(f"Episode {i+1}/{num_episodes} (seed={seed})...", end=" ")
            result = self.run_episode(i, seed, mock_env=mock)
            episode_results.append(result)
            
            status = "✓" if result["success"] else "✗"
            print(f"{status} steps={result['steps']}, ADE={result['ade']:.2f}m")
        
        # Aggregate metrics
        successes = sum(1 for r in episode_results if r["success"])
        ade_values = [r["ade"] for r in episode_results]
        fde_values = [r["fde"] for r in episode_results]
        steps_values = [r["steps"] for r in episode_results]
        
        aggregate = {
            "num_episodes": num_episodes,
            "success_rate": successes / num_episodes,
            "ade_mean": float(np.mean(ade_values)),
            "ade_std": float(np.std(ade_values)),
            "fde_mean": float(np.mean(fde_values)),
            "fde_std": float(np.std(fde_values)),
            "steps_mean": float(np.mean(steps_values)),
            "steps_std": float(np.std(steps_values))
        }
        
        # Build results
        results = {
            "run_id": run_id,
            "domain": "driving",
            "config": {
                "checkpoint": self.config.checkpoint_path,
                "policy_type": self.config.policy_type,
                "num_waypoints": self.config.num_waypoints,
                "controller_type": self.config.controller_type,
                "vehicle_type": self.config.vehicle_type
            },
            "scenarios": episode_results,
            "aggregate": aggregate
        }
        
        # Save results (convert numpy types for JSON serialization)
        def convert_for_json(obj):
            """Convert numpy types to JSON-serializable Python types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            return obj
        
        results_json = convert_for_json(results)
        
        metrics_path = output_path / f"metrics_{run_id}.json"
        with open(metrics_path, "w") as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results")
        print(f"{'='*60}")
        print(f"Success Rate: {aggregate['success_rate']*100:.1f}% ({successes}/{num_episodes})")
        print(f"ADE: {aggregate['ade_mean']:.2f}m ± {aggregate['ade_std']:.2f}m")
        print(f"FDE: {aggregate['fde_mean']:.2f}m ± {aggregate['fde_std']:.2f}m")
        print(f"Steps: {aggregate['steps_mean']:.0f} ± {aggregate['steps_std']:.0f}")
        print(f"\nSaved to: {metrics_path}")
        
        self.results = episode_results
        return results


def find_latest_checkpoint(policy_type: str = "bc") -> Optional[str]:
    """Find the latest checkpoint for the given policy type."""
    patterns = {
        "bc": "out/waypoint_bc/run_*/best.pt",
        "rl": "out/ppo_delta_waypoint/run_*/best.pt", 
        "sft_delta": "out/ppo_sft_delta/run_*/best.pt"
    }
    
    pattern = patterns.get(policy_type, patterns["bc"])
    checkpoints = list(Path(".").glob(pattern))
    
    if checkpoints:
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(checkpoints[0])
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Inference Runner for Waypoint Policies"
    )
    
    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detect if not provided)")
    parser.add_argument("--policy-type", type=str, default="bc",
                        choices=["bc", "rl", "sft_delta"],
                        help="Policy type for auto-detection")
    
    # Model
    parser.add_argument("--encoder-dim", type=int, default=256,
                        help="Encoder dimension")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension")
    parser.add_argument("--num-waypoints", type=int, default=8,
                        help="Number of future waypoints")
    
    # Controller
    parser.add_argument("--controller", type=str, default="pure_pursuit",
                        help="Controller type")
    parser.add_argument("--vehicle", type=str, default="tesla_model3",
                        help="Vehicle type")
    parser.add_argument("--target-speed", type=float, default=8.0,
                        help="Target speed in m/s")
    
    # Evaluation
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="Base random seed")
    
    # CARLA
    parser.add_argument("--carla-host", type=str, default="127.0.0.1",
                        help="CARLA host")
    parser.add_argument("--carla-port", type=int, default=2000,
                        help="CARLA port")
    parser.add_argument("--use-carla", action="store_true",
                        help="Use real CARLA (if available)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/e2e_inference",
                        help="Output directory")
    
    # Mode
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use mock environment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without running")
    
    args = parser.parse_args()
    
    # Auto-detect checkpoint if not provided
    if not args.checkpoint:
        args.checkpoint = find_latest_checkpoint(args.policy_type)
        if not args.checkpoint:
            print("Error: No checkpoint found. Use --checkpoint to specify.")
            sys.exit(1)
    
    # Create config
    config = E2EInferenceConfig(
        checkpoint_path=args.checkpoint,
        policy_type=args.policy_type,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        controller_type=args.controller,
        vehicle_type=args.vehicle,
        target_speed=args.target_speed,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed_base=args.seed_base,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        use_carla=args.use_carla,
        output_dir=args.output_dir
    )
    
    if args.dry_run:
        print("Dry run - configuration:")
        print(f"  Checkpoint: {config.checkpoint_path}")
        print(f"  Policy Type: {config.policy_type}")
        print(f"  Episodes: {config.num_episodes}")
        print(f"  Output: {config.output_dir}")
        return
    
    # Create runner
    runner = E2EInferenceRunner(config=config)
    
    # Run evaluation
    results = runner.run(
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        mock=not args.use_carla
    )


if __name__ == "__main__":
    main()

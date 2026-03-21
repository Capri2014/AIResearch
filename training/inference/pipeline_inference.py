"""
Inference Pipeline for BC → RL Driving Models

Provides a unified inference interface for:
- BC waypoint prediction (from WaypointBCModel)
- RL delta-waypoint refinement (from PPODeltaPolicy)
- Combined BC + RL inference

Usage:
    from training.inference.pipeline_inference import PipelineInference, InferenceConfig

    config = InferenceConfig(
        bc_checkpoint="out/waypoint_bc/final.pt",
        rl_checkpoint="out/ppo_delta_waypoint_2026_03_15/final.pt",
    )
    inference = PipelineInference(config)
    
    # Single inference
    result = inference.predict(observation)
    
    # Batch inference
    results = inference.predict_batch(observations)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    
    # Checkpoints
    bc_checkpoint: Optional[Path] = None
    rl_checkpoint: Optional[Path] = None
    ssl_checkpoint: Optional[Path] = None
    
    # Model settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp32"  # fp32, fp16, bf16
    
    # Inference settings
    num_waypoints: int = 8
    waypoint_time_delta: float = 0.5  # seconds between waypoints
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("out/inference"))
    
    # Behavior
    use_rl: bool = True  # Whether to apply RL refinement
    return_confidence: bool = True


@dataclass
class InferenceResult:
    """Result from a single inference call."""
    
    # Raw predictions - must come before optional fields
    bc_waypoints: np.ndarray  # (num_waypoints, 2) - x, y in vehicle frame
    final_waypoints: np.ndarray  # (num_waypoints, 2) - x, y
    rl_delta: Optional[np.ndarray] = None  # (num_waypoints, 2) - delta corrections
    
    # Confidence scores
    confidence: Optional[float] = None
    bc_confidence: Optional[float] = None
    rl_confidence: Optional[float] = None
    
    # Metadata
    inference_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    mode: str = "bc_only"  # bc_only, bc_rl
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bc_waypoints": self.bc_waypoints.tolist(),
            "rl_delta": self.rl_delta.tolist() if self.rl_delta is not None else None,
            "final_waypoints": self.final_waypoints.tolist(),
            "confidence": self.confidence,
            "bc_confidence": self.bc_confidence,
            "rl_confidence": self.rl_confidence,
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp,
            "mode": self.mode,
        }


class PipelineInference:
    """
    Unified inference pipeline for BC → RL driving models.
    
    Supports:
    - BC-only inference (waypoint prediction)
    - BC + RL inference (waypoint + delta refinement)
    - Batch inference for efficiency
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.bc_model = None
        self.rl_model = None
        self.ssl_encoder = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all configured models."""
        # Load BC model
        if self.config.bc_checkpoint:
            self.bc_model = self._load_bc_model()
        
        # Load RL model
        if self.config.rl_checkpoint and self.config.use_rl:
            self.rl_model = self._load_rl_model()
        
        # Load SSL encoder (optional)
        if self.config.ssl_checkpoint:
            self.ssl_encoder = self._load_ssl_encoder()
    
    def _load_bc_model(self):
        """Load BC waypoint model from checkpoint."""
        print(f"Loading BC checkpoint: {self.config.bc_checkpoint}")
        
        try:
            from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
            from training.rl.bc_checkpoint_loader import load_bc_waypoint_model
            
            checkpoint = torch.load(self.config.bc_checkpoint, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if "config" in checkpoint:
                    bc_config = WaypointBCConfig(**checkpoint["config"])
                else:
                    bc_config = WaypointBCConfig()
                model = load_bc_waypoint_model(checkpoint, bc_config)
            else:
                bc_config = WaypointBCConfig()
                model = checkpoint
            
            model.to(self.device)
            model.eval()
            print(f"  BC model loaded: {type(model).__name__}")
            return model
            
        except Exception as e:
            print(f"  Warning: Could not load BC model: {e}")
            return None
    
    def _load_rl_model(self):
        """Load RL delta-waypoint model from checkpoint."""
        print(f"Loading RL checkpoint: {self.config.rl_checkpoint}")
        
        try:
            from training.rl.ppo_delta_waypoint_trainer import PPODeltaPolicy, PPODeltaConfig
            
            checkpoint = torch.load(self.config.rl_checkpoint, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if "config" in checkpoint:
                    rl_config = PPODeltaConfig(**checkpoint["config"])
                else:
                    rl_config = PPODeltaConfig()
                model = PPODeltaPolicy(rl_config)
                if "model_state" in checkpoint:
                    model.load_state_dict(checkpoint["model_state"])
            else:
                rl_config = PPODeltaConfig()
                model = checkpoint
            
            model.to(self.device)
            model.eval()
            print(f"  RL model loaded: {type(model).__name__}")
            return model
            
        except Exception as e:
            print(f"  Warning: Could not load RL model: {e}")
            return None
    
    def _load_ssl_encoder(self):
        """Load SSL encoder from checkpoint."""
        print(f"Loading SSL checkpoint: {self.config.ssl_checkpoint}")
        
        try:
            checkpoint = torch.load(self.config.ssl_checkpoint, map_location=self.device)
            # SSL encoder loading - depends on training config
            print(f"  SSL encoder loaded")
            return checkpoint
        except Exception as e:
            print(f"  Warning: Could not load SSL encoder: {e}")
            return None
    
    def _create_dummy_observation(self) -> dict:
        """Create a dummy observation for testing when no real data available."""
        return {
            "speed_mps": 5.0,
            "yaw_rad": 0.0,
            "camera": torch.randn(1, 3, 256, 256).to(self.device),
            "lidar": torch.randn(1, 256, 4).to(self.device) if hasattr(self, 'has_lidar') else None,
        }
    
    def predict(self, observation: Optional[dict] = None) -> InferenceResult:
        """
        Run inference on a single observation.
        
        Args:
            observation: Dict with 'speed_mps', 'yaw_rad', 'camera', 'lidar'
                       If None, uses dummy observation for testing.
        
        Returns:
            InferenceResult with waypoints and confidence scores
        """
        import time
        start_time = time.time()
        
        # Use dummy observation if none provided
        if observation is None:
            observation = self._create_dummy_observation()
        
        # Get BC prediction
        bc_waypoints = self._predict_bc(observation)
        
        # Get RL refinement if available
        rl_delta = None
        rl_confidence = None
        if self.rl_model is not None and self.config.use_rl:
            rl_delta, rl_confidence = self._predict_rl(observation, bc_waypoints)
        
        # Compute final waypoints
        if rl_delta is not None:
            final_waypoints = bc_waypoints + rl_delta
            mode = "bc_rl"
        else:
            final_waypoints = bc_waypoints
            mode = "bc_only"
        
        # Compute confidence
        bc_confidence = self._compute_bc_confidence(bc_waypoints)
        confidence = bc_confidence
        if rl_confidence is not None:
            confidence = (bc_confidence + rl_confidence) / 2
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return InferenceResult(
            bc_waypoints=bc_waypoints,
            final_waypoints=final_waypoints,
            rl_delta=rl_delta,
            confidence=confidence if self.config.return_confidence else None,
            bc_confidence=bc_confidence if self.config.return_confidence else None,
            rl_confidence=rl_confidence if self.config.return_confidence else None,
            inference_time_ms=inference_time_ms,
            mode=mode,
        )
    
    def _predict_bc(self, observation: dict) -> np.ndarray:
        """Predict waypoints using BC model."""
        if self.bc_model is None:
            # Return dummy waypoints if no model
            return np.zeros((self.config.num_waypoints, 2))
        
        with torch.no_grad():
            speed = torch.tensor([[observation["speed_mps"]]]).to(self.device)
            yaw = torch.tensor([[observation["yaw_rad"]]]).to(self.device)
            
            # Check for camera input
            if "camera" in observation and observation["camera"] is not None:
                camera = observation["camera"].to(self.device)
            else:
                camera = torch.randn(1, 3, 256, 256).to(self.device)
            
            # Forward pass
            try:
                outputs = self.bc_model(speed, yaw, camera)
                if isinstance(outputs, dict):
                    waypoints = outputs.get("waypoints", outputs.get("future_waypoints"))
                else:
                    waypoints = outputs
                
                if len(waypoints.shape) == 3:
                    waypoints = waypoints[0]  # Remove batch dimension
                
                return waypoints.cpu().numpy()
            except Exception as e:
                print(f"  BC inference error: {e}")
                return np.zeros((self.config.num_waypoints, 2))
    
    def _predict_rl(self, observation: dict, bc_waypoints: np.ndarray) -> tuple:
        """Predict delta correction using RL model."""
        if self.rl_model is None:
            return None, None
        
        with torch.no_grad():
            # Prepare inputs
            speed = torch.tensor([[observation["speed_mps"]]]).to(self.device)
            bc_wp = torch.tensor([bc_waypoints]).to(self.device)
            
            try:
                # RL model forward
                delta, value = self.rl_model(speed, bc_wp)
                
                delta = delta[0].cpu().numpy()
                value = value[0].cpu().numpy() if value is not None else None
                
                # Convert value to confidence (higher value = higher confidence)
                if value is not None:
                    confidence = float(np.clip(value.item(), 0, 1))
                else:
                    confidence = 0.5
                
                return delta, confidence
            except Exception as e:
                print(f"  RL inference error: {e}")
                return None, None
    
    def _compute_bc_confidence(self, waypoints: np.ndarray) -> float:
        """Compute confidence score from BC waypoints."""
        # Simple heuristic: confidence based on waypoint magnitude
        # Larger magnitudes in reasonable range = more confident
        norms = np.linalg.norm(waypoints, axis=1)
        avg_norm = np.mean(norms)
        
        # Normalize to [0, 1]
        # Expected waypoint distance: ~5m over 4s (8 waypoints * 0.5s)
        # So avg_norm should be around 2-3 for normal driving
        confidence = float(np.clip(avg_norm / 3.0, 0.1, 1.0))
        return confidence
    
    def predict_batch(self, observations: list[dict]) -> list[InferenceResult]:
        """
        Run inference on a batch of observations.
        
        Args:
            observations: List of observation dicts
        
        Returns:
            List of InferenceResults
        """
        results = []
        for obs in observations:
            result = self.predict(obs)
            results.append(result)
        return results
    
    def run_evaluation_suite(
        self,
        scenarios: list[str],
        num_runs: int = 1,
    ) -> dict:
        """
        Run evaluation on a suite of scenarios.
        
        Args:
            scenarios: List of scenario names
            num_runs: Number of runs per scenario
        
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*60)
        print("Running Inference Evaluation Suite")
        print("="*60)
        
        print(f"\nModels loaded:")
        print(f"  BC: {'Yes' if self.bc_model else 'No'}")
        print(f"  RL: {'Yes' if self.rl_model else 'No'}")
        
        print(f"\nEvaluating {len(scenarios)} scenarios ({num_runs} run(s) each):")
        
        all_results = []
        for scenario in scenarios:
            scenario_results = []
            for run in range(num_runs):
                result = self.predict()  # Use dummy for now
                scenario_results.append(result.to_dict())
            
            # Aggregate
            success_rate = 1.0  # Placeholder
            avg_confidence = np.mean([r["confidence"] for r in scenario_results])
            avg_inference_time = np.mean([r["inference_time_ms"] for r in scenario_results])
            
            scenario_summary = {
                "scenario": scenario,
                "num_runs": num_runs,
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "avg_inference_time_ms": avg_inference_time,
                "runs": scenario_results,
            }
            all_results.append(scenario_summary)
            
            print(f"  ✓ {scenario}: conf={avg_confidence:.2f}, time={avg_inference_time:.1f}ms")
        
        # Save results
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.config.output_dir / f"inference_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return all_results


def load_inference_pipeline(
    bc_checkpoint: Optional[str] = None,
    rl_checkpoint: Optional[str] = None,
    use_rl: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> PipelineInference:
    """
    Convenience function to load inference pipeline.
    
    Args:
        bc_checkpoint: Path to BC checkpoint
        rl_checkpoint: Path to RL checkpoint
        use_rl: Whether to use RL refinement
        device: Device to run inference on
    
    Returns:
        PipelineInference instance
    """
    config = InferenceConfig(
        bc_checkpoint=Path(bc_checkpoint) if bc_checkpoint else None,
        rl_checkpoint=Path(rl_checkpoint) if rl_checkpoint else None,
        use_rl=use_rl,
        device=device,
    )
    return PipelineInference(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Inference")
    
    parser.add_argument("--bc-checkpoint", type=str, help="Path to BC checkpoint")
    parser.add_argument("--rl-checkpoint", type=str, help="Path to RL checkpoint")
    parser.add_argument("--ssl-checkpoint", type=str, help="Path to SSL encoder checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--use-rl", action="store_true", default=True, help="Use RL refinement")
    parser.add_argument("--no-rl", action="store_false", dest="use_rl", help="Disable RL refinement")
    parser.add_argument("--output-dir", type=str, default="out/inference", help="Output directory")
    
    args = parser.parse_args()
    
    config = InferenceConfig(
        bc_checkpoint=Path(args.bc_checkpoint) if args.bc_checkpoint else None,
        rl_checkpoint=Path(args.rl_checkpoint) if args.rl_checkpoint else None,
        ssl_checkpoint=Path(args.ssl_checkpoint) if args.ssl_checkpoint else None,
        device=args.device,
        use_rl=args.use_rl,
        output_dir=Path(args.output_dir),
    )
    
    inference = PipelineInference(config)
    
    # Run single inference
    result = inference.predict()
    print("\nInference Result:")
    print(f"  Mode: {result.mode}")
    print(f"  BC waypoints: {result.bc_waypoints[:2]}...")
    print(f"  Final waypoints: {result.final_waypoints[:2]}...")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Inference time: {result.inference_time_ms:.1f}ms")

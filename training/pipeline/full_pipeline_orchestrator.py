"""
Full Driving Pipeline Orchestrator

End-to-end orchestrator combining:
1. SSL pretrained encoder (MoCo/SimCLR)
2. Waypoint BC policy
3. RL delta refinement
4. CARLA ScenarioRunner evaluation

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval

Usage:
    orchestrator = FullPipelineOrchestrator(
        ssl_encoder_path="out/moco_waymo/final.pt",
        bc_policy_path="out/waypoint_bc/final.pt",
        rl_delta_path="out/ppo_delta_waypoint/model.pt"
    )
    results = orchestrator.run_full_pipeline(episode_path, eval_carla=True)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import importlib

logger = logging.getLogger(__name__)


def _import_or_none(module_path: str, class_names: List[str]):
    """Try to import module, return None if not found."""
    try:
        module = importlib.import_module(module_path)
        return {name: getattr(module, name) for name in class_names if hasattr(module, name)}
    except (ImportError, ModuleNotFoundError):
        return None


# Try importing optional modules
_MOCO = _import_or_none("training.pretrain.moco_waymo_ssl", ["load_moco_checkpoint", "MoCoEncoder"])
_BC = _import_or_none("training.bc.train_waypoint_bc_ssl", ["load_bc_checkpoint", "WaypointBC"])
_RL = _import_or_none("training.rl.ppo_delta_waypoint_trainer", ["load_ppo_checkpoint", "PPOActor"])
_CARLA = _import_or_none("training.eval.carla_scenariorunner_eval", ["CARLAScenarioRunner", "CARLAEvalConfig"])

# Set to None if not available
load_moco_checkpoint = _MOCO.get("load_moco_checkpoint") if _MOCO else None
MoCoEncoder = _MOCO.get("MoCoEncoder") if _MOCO else None
load_bc_checkpoint = _BC.get("load_bc_checkpoint") if _BC else None
WaypointBC = _BC.get("WaypointBC") if _BC else None
load_ppo_checkpoint = _RL.get("load_ppo_checkpoint") if _RL else None
PPOActor = _RL.get("PPOActor") if _RL else None
CARLAScenarioRunner = _CARLA.get("CARLAScenarioRunner") if _CARLA else None
CARLAEvalConfig = _CARLA.get("CARLAEvalConfig") if _CARLA else None

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Model paths
    ssl_encoder_path: str = "out/moco_waymo/final.pt"
    bc_policy_path: str = "out/waypoint_bc/final.pt"
    rl_delta_path: str = "out/ppo_delta_waypoint/model.pt"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pipeline modes
    use_ssl: bool = True
    use_rl: bool = True
    eval_carla: bool = False
    
    # Waypoint settings
    num_waypoints: int = 8
    waypoint_dim: int = 3
    
    # CARLA settings
    carla_host: str = "localhost"
    carla_port: int = 2000


@dataclass
class PipelineStepResult:
    """Result from a single pipeline step."""
    step_name: str
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        msg = f"{status} {self.step_name}"
        if self.error:
            msg += f" (error: {self.error})"
        return msg


@dataclass
class FullPipelineResult:
    """Result from the full pipeline."""
    episode_id: str
    step_results: List[PipelineStepResult]
    waypoints: Optional[np.ndarray] = None
    carla_result: Optional[Any] = None
    total_inference_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        return all(r.success for r in self.step_results)
    
    def summary(self) -> str:
        lines = [f"Full Pipeline Result: {self.episode_id}"]
        lines.append("-" * 50)
        for r in self.step_results:
            lines.append(f"  {r.summary()}")
        if self.carla_result:
            lines.append(f"  CARLA: {self.carla_result.summary()}")
        lines.append(f"Total inference: {self.total_inference_time_ms:.1f}ms")
        return "\n".join(lines)


class FullPipelineOrchestrator:
    """
    Orchestrates the full driving pipeline from perception to control.
    
    Integrates SSL encoder, BC waypoint predictor, and RL delta refiner
    into a unified inference pipeline.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = torch.device(self.config.device)
        
        # Models (lazy loaded)
        self.ssl_encoder: Optional[MoCoEncoder] = None
        self.bc_policy: Optional[WaypointBC] = None
        self.rl_actor: Optional[PPOActor] = None
        
        # CARLA runner (lazy connect)
        self.carla_runner: Optional[CARLAScenarioRunner] = None
        
        # Statistics
        self.stats = {
            "inference_count": 0,
            "total_time_ms": 0.0,
            "step_times": {k: 0.0 for k in ["ssl", "bc", "rl", "carla"]}
        }
    
    def _load_ssl_encoder(self) -> PipelineStepResult:
        """Load SSL pretrained encoder."""
        import time
        start = time.time()
        
        try:
            if not self.config.use_ssl:
                return PipelineStepResult("SSL_Encoder", True, {"skipped": True})
            
            if load_moco_checkpoint is None or MoCoEncoder is None:
                logger.warning("MoCo SSL module not available, skipping")
                return PipelineStepResult("SSL_Encoder", True, {"skipped": True, "reason": "module unavailable"})
            
            checkpoint = load_moco_checkpoint(self.config.ssl_encoder_path, self.device)
            self.ssl_encoder = MoCoEncoder(
                feature_dim=checkpoint.get("feature_dim", 256),
                queue_size=0  # Inference mode
            )
            self.ssl_encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.ssl_encoder.eval()
            
            elapsed = (time.time() - start) * 1000
            self.stats["step_times"]["ssl"] += elapsed
            
            return PipelineStepResult(
                "SSL_Encoder",
                True,
                {"feature_dim": checkpoint.get("feature_dim", 256), "load_time_ms": elapsed}
            )
        except Exception as e:
            logger.warning(f"SSL encoder load failed: {e}")
            return PipelineStepResult("SSL_Encoder", True, {"skipped": True, "reason": str(e)})
    
    def _load_bc_policy(self) -> PipelineStepResult:
        """Load BC waypoint policy."""
        import time
        start = time.time()
        
        try:
            if load_bc_checkpoint is None or WaypointBC is None:
                logger.warning("BC module not available, skipping")
                return PipelineStepResult("BC_WaypointPolicy", True, {"skipped": True, "reason": "module unavailable"})
            
            checkpoint = load_bc_checkpoint(self.config.bc_policy_path, self.device)
            self.bc_policy = WaypointBC(
                encoder_dim=checkpoint.get("encoder_dim", 256),
                num_waypoints=self.config.num_waypoints,
                waypoint_dim=self.config.waypoint_dim
            )
            self.bc_policy.load_state_dict(checkpoint["model_state_dict"])
            self.bc_policy.eval()
            
            elapsed = (time.time() - start) * 1000
            self.stats["step_times"]["bc"] += elapsed
            
            return PipelineStepResult(
                "BC_WaypointPolicy",
                True,
                {"num_waypoints": self.config.num_waypoints, "load_time_ms": elapsed}
            )
        except Exception as e:
            return PipelineStepResult("BC_WaypointPolicy", False, error=str(e))
    
    def _load_rl_actor(self) -> PipelineStepResult:
        """Load RL delta actor."""
        import time
        start = time.time()
        
        try:
            if not self.config.use_rl:
                return PipelineStepResult("RL_DeltaActor", True, {"skipped": True})
            
            if load_ppo_checkpoint is None or PPOActor is None:
                logger.warning("RL module not available, skipping")
                return PipelineStepResult("RL_DeltaActor", True, {"skipped": True, "reason": "module unavailable"})
            
            checkpoint = load_ppo_checkpoint(self.config.rl_delta_path, self.device)
            self.rl_actor = PPOActor(
                state_dim=checkpoint.get("state_dim", 512),
                action_dim=checkpoint.get("action_dim", self.config.num_waypoints * self.config.waypoint_dim)
            )
            self.rl_actor.load_state_dict(checkpoint["actor_state_dict"])
            self.rl_actor.eval()
            
            elapsed = (time.time() - start) * 1000
            self.stats["step_times"]["rl"] += elapsed
            
            return PipelineStepResult(
                "RL_DeltaActor",
                True,
                {"action_dim": self.config.waypoint_dim * self.config.num_waypoints, "load_time_ms": elapsed}
            )
        except Exception as e:
            logger.warning(f"RL actor load failed: {e}")
            return PipelineStepResult("RL_DeltaActor", True, {"skipped": True, "reason": str(e)})
    
    def _connect_carla(self) -> PipelineStepResult:
        """Connect to CARLA for evaluation."""
        try:
            if not self.config.eval_carla:
                return PipelineStepResult("CARLA_Connect", True, {"skipped": True})
            
            if CARLAScenarioRunner is None or CARLAEvalConfig is None:
                logger.warning("CARLA module not available, skipping")
                return PipelineStepResult("CARLA_Connect", True, {"skipped": True, "reason": "module unavailable"})
            
            config = CARLAEvalConfig(
                host=self.config.carla_host,
                port=self.config.carla_port
            )
            self.carla_runner = CARLAScenarioRunner(config)
            
            if self.carla_runner.connect():
                return PipelineStepResult("CARLA_Connect", True, {"connected": True})
            else:
                return PipelineStepResult("CARLA_Connect", False, error="Connection failed")
        except Exception as e:
            return PipelineStepResult("CARLA_Connect", False, error=str(e))
    
    def load_models(self) -> List[PipelineStepResult]:
        """Load all models in the pipeline."""
        results = []
        results.append(self._load_ssl_encoder())
        results.append(self._load_bc_policy())
        results.append(self._load_rl_actor())
        results.append(self._connect_carla())
        return results
    
    def predict_waypoints(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict waypoints from observation using BC policy.
        
        Args:
            observation: Shape (H, W, C) or (C,) image/feature tensor
            
        Returns:
            waypoints: Shape (num_waypoints, waypoint_dim)
        """
        import time
        start = time.time()
        
        # Extract features if SSL encoder available
        if self.ssl_encoder is not None:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
                if obs_tensor.dim() == 3:
                    obs_tensor = obs_tensor.permute(2, 0, 1)  # HWC -> CHW
                obs_tensor = obs_tensor.unsqueeze(0).to(self.device)
                features = self.ssl_encoder(obs_tensor)
        else:
            features = observation
        
        # BC waypoint prediction
        if self.bc_policy is not None:
            with torch.no_grad():
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features).float()
                if features.dim() == 3:
                    features = features.flatten()
                features = features.unsqueeze(0).to(self.device)
                waypoints = self.bc_policy(features)
                waypoints = waypoints.cpu().numpy().squeeze()
        else:
            waypoints = np.zeros((self.config.num_waypoints, self.config.waypoint_dim))
        
        # RL delta refinement
        if self.rl_actor is not None:
            with torch.no_grad():
                # Concatenate waypoints with features for delta prediction
                combined = torch.cat([
                    features.squeeze(),
                    torch.from_numpy(waypoints).float().flatten().to(self.device)
                ]).unsqueeze(0)
                delta = self.rl_actor(combined)
                delta = delta.cpu().numpy().squeeze()
                delta = delta.reshape(self.config.num_waypoints, self.config.waypoint_dim)
                waypoints = waypoints + delta
        
        self.stats["inference_count"] += 1
        self.stats["total_time_ms"] += (time.time() - start) * 1000
        
        return waypoints
    
    def run_full_pipeline(
        self,
        episode_id: str,
        observation: np.ndarray,
        route: Optional[np.ndarray] = None
    ) -> FullPipelineResult:
        """
        Run the full pipeline on a single observation.
        
        Args:
            episode_id: Identifier for this episode
            observation: Input observation (image or features)
            route: Optional route for CARLA evaluation
            
        Returns:
            FullPipelineResult with all step results and final waypoints
        """
        import time
        total_start = time.time()
        
        # Load models if not already loaded
        if self.bc_policy is None:
            load_results = self.load_models()
        else:
            load_results = []
        
        # Predict waypoints
        waypoints = self.predict_waypoints(observation)
        inference_result = PipelineStepResult(
            "Inference",
            True,
            {"waypoints_shape": waypoints.shape}
        )
        
        # Run CARLA evaluation if configured
        carla_result = None
        if self.config.eval_carla and self.carla_runner is not None and route is not None:
            import time
            start = time.time()
            try:
                carla_result = self.carla_runner.evaluate_trajectory(waypoints, route)
                self.stats["step_times"]["carla"] += (time.time() - start) * 1000
            except Exception as e:
                logger.error(f"CARLA evaluation failed: {e}")
        
        total_time = (time.time() - total_start) * 1000
        
        return FullPipelineResult(
            episode_id=episode_id,
            step_results=load_results + [inference_result],
            waypoints=waypoints,
            carla_result=carla_result,
            total_inference_time_ms=total_time
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.stats.copy()
        if stats["inference_count"] > 0:
            stats["avg_inference_ms"] = stats["total_time_ms"] / stats["inference_count"]
        else:
            stats["avg_inference_ms"] = 0.0
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        if self.carla_runner:
            self.carla_runner.disconnect()


def run_pipeline_demo():
    """Demo the full pipeline with synthetic data."""
    print("Full Pipeline Orchestrator Demo")
    print("=" * 50)
    
    # Create config
    config = PipelineConfig(
        use_ssl=False,  # Skip SSL for demo (no checkpoint)
        use_rl=False,   # Skip RL for demo
        eval_carla=False,
        num_waypoints=8,
        waypoint_dim=3
    )
    
    orchestrator = FullPipelineOrchestrator(config)
    
    # Load models (will skip missing checkpoints)
    print("\n1. Loading models...")
    results = orchestrator.load_models()
    for r in results:
        print(f"   {r.summary()}")
    
    # Run inference with synthetic observation
    print("\n2. Running inference on synthetic observation...")
    synthetic_obs = np.random.randn(256).astype(np.float32)  # Feature vector
    result = orchestrator.run_full_pipeline("demo_episode_001", synthetic_obs)
    print(result.summary())
    
    # Print stats
    print("\n3. Pipeline Statistics:")
    stats = orchestrator.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    orchestrator.cleanup()
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    run_pipeline_demo()

"""
Pipeline Integration: BC → RL Refinement → CARLA Evaluation.

This script demonstrates the full driving-first pipeline:
1. Load BC checkpoint (with optional SSL encoder)
2. Create RL refinement model (BC + delta head)
3. Wrap for CARLA ScenarioRunner evaluation

Usage:
    # Full pipeline with BC checkpoint
    python -m training.rl.pipeline_integration \
        --bc-checkpoint out/bc/model.pt \
        --ssl-checkpoint out/ssl/model.pt \
        --output out/pipeline
        
    # BC only (for inference testing)
    python -m training.rl.pipeline_integration \
        --bc-checkpoint out/bc/model.pt \
        --output out/pipeline
        
    # Test mode (no checkpoints)
    python -m training.rl.pipeline_integration --test
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# Pipeline Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Stage 1: BC
    bc_checkpoint: Optional[Path] = None
    ssl_checkpoint: Optional[Path] = None
    
    # Stage 2: RL (optional)
    rl_checkpoint: Optional[Path] = None
    
    # Model config
    num_waypoints: int = 8
    waypoint_dim: int = 2
    predict_speed: bool = True
    
    # Delta head config
    delta_hidden_dims: List[int] = None
    
    # Output
    output_dir: Path = Path("out/pipeline")
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.delta_hidden_dims is None:
            self.delta_hidden_dims = [256, 128, 64]


# ============================================================================
# Pipeline Stages
# ============================================================================

class PipelineStage:
    """Base class for pipeline stages."""
    
    name: str = "base"
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
    
    def load(self) -> nn.Module:
        """Load the model for this stage."""
        raise NotImplementedError
    
    def save_config(self, path: Path):
        """Save stage configuration."""
        pass


class Stage1_BC(PipelineStage):
    """Stage 1: Load BC waypoint model."""
    
    name = "bc"
    
    def load(self) -> Tuple[nn.Module, Dict]:
        from training.rl.bc_checkpoint_loader import load_bc_waypoint_model
        from training.pretrain.train_waymo_ssl import load_ssl_encoder
        
        config = self.config
        
        # Load SSL encoder if provided
        ssl_encoder = None
        ssl_config = None
        if config.ssl_checkpoint is not None:
            ssl_config, ssl_encoder = load_ssl_encoder(config.ssl_checkpoint)
            ssl_encoder = ssl_encoder.to(config.device)
            ssl_encoder.eval()
        
        # Load BC model
        if config.bc_checkpoint is not None:
            model, bc_config = load_bc_waypoint_model(
                checkpoint=config.bc_checkpoint,
                ssl_encoder=ssl_encoder,
                device=config.device,
            )
        else:
            # Create stub model
            from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
            bc_config = WaypointBCConfig(
                num_waypoints=config.num_waypoints,
                waypoint_dim=config.waypoint_dim,
                predict_speed=config.predict_speed,
            )
            model = WaypointBCModel(bc_config, ssl_encoder=ssl_encoder)
        
        info = {
            "ssl_checkpoint": str(config.ssl_checkpoint) if config.ssl_checkpoint else None,
            "bc_checkpoint": str(config.bc_checkpoint) if config.bc_checkpoint else None,
        }
        
        return model, info


class Stage2_RL(PipelineStage):
    """Stage 2: RL refinement with residual delta."""
    
    name = "rl"
    
    def __init__(self, config: PipelineConfig, bc_model: nn.Module):
        super().__init__(config)
        self.bc_model = bc_model
    
    def load(self) -> Tuple[nn.Module, Dict]:
        from training.rl.waypoint_policy_wrapper import (
            WaypointPolicyWithDelta,
            create_rl_refinement_model,
        )
        
        config = self.config
        
        # If RL checkpoint provided, load it
        if config.rl_checkpoint is not None:
            from training.rl.waypoint_policy_wrapper import load_rl_model
            model = load_rl_model(config.rl_checkpoint, self.bc_model, config.device)
            info = {"rl_checkpoint": str(config.rl_checkpoint)}
        else:
            # Create new RL model from BC
            model, info = create_rl_refinement_model(
                bc_checkpoint=config.bc_checkpoint,
                delta_hidden_dims=config.delta_hidden_dims,
                num_waypoints=config.num_waypoints,
                waypoint_dim=config.waypoint_dim,
                predict_speed=config.predict_speed,
                freeze_bc=True,
                device=config.device,
            )
        
        return model, info


class Stage3_CARLA(PipelineStage):
    """Stage 3: Wrap for CARLA evaluation."""
    
    name = "carla"
    
    def __init__(self, config: PipelineConfig, rl_model: nn.Module, bc_model: Optional[nn.Module] = None):
        super().__init__(config)
        self.rl_model = rl_model
        self.bc_model = bc_model
    
    def load(self) -> Tuple[nn.Module, Dict]:
        config = self.config
        
        # Use the loaded models directly if available
        if self.rl_model is not None:
            # Return the RL model for inference
            info = {
                "device": config.device,
                "num_waypoints": config.num_waypoints,
                "type": "rl_refinement",
            }
            return self.rl_model, info
        elif self.bc_model is not None:
            # Return BC model for inference
            info = {
                "device": config.device,
                "num_waypoints": config.num_waypoints,
                "type": "bc_only",
            }
            return self.bc_model, info
        else:
            raise ValueError("No model available for CARLA evaluation")


# ============================================================================
# Full Pipeline
# ============================================================================

class DrivingPipeline:
    """
    Full driving-first pipeline: BC → RL → CARLA.
    
    Usage:
        pipeline = DrivingPipeline(config)
        pipeline.run()
        
        # Or load specific stages
        bc_model = pipeline.load_bc()
        rl_model = pipeline.load_rl(bc_model)
        carla_interface = pipeline.load_carla(rl_model)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.bc_model = None
        self.rl_model = None
        self.carla_interface = None
    
    def load_bc(self) -> Tuple[nn.Module, Dict]:
        """Load Stage 1: BC model."""
        stage = Stage1_BC(self.config)
        self.bc_model, info = stage.load()
        return self.bc_model, info
    
    def load_rl(self, bc_model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict]:
        """Load Stage 2: RL refinement model."""
        if bc_model is None:
            bc_model = self.bc_model
        if bc_model is None:
            raise ValueError("Must load BC model first or pass bc_model")
        
        stage = Stage2_RL(self.config, bc_model)
        self.rl_model, info = stage.load()
        return self.rl_model, info
    
    def load_carla(self, rl_model: Optional[nn.Module] = None, bc_model: Optional[nn.Module] = None) -> Tuple[any, Dict]:
        """Load Stage 3: CARLA interface."""
        if rl_model is None:
            rl_model = self.rl_model
        if bc_model is None:
            bc_model = self.bc_model
        if rl_model is None and bc_model is None:
            raise ValueError("Must load RL or BC model first")
        
        stage = Stage3_CARLA(self.config, rl_model, bc_model)
        self.carla_interface, info = stage.load()
        return self.carla_interface, info
    
    def run(self) -> Dict:
        """Run full pipeline."""
        # Stage 1: BC
        bc_model, bc_info = self.load_bc()
        
        # Stage 2: RL
        rl_model, rl_info = self.load_rl(bc_model)
        
        # Stage 3: CARLA
        carla_interface, carla_info = self.load_carla(rl_model)
        
        # Compile info
        pipeline_info = {
            "stages": {
                "bc": bc_info,
                "rl": rl_info,
                "carla": carla_info,
            },
            "config": asdict(self.config),
        }
        
        # Save config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_dir / "pipeline_config.json", "w") as f:
            json.dump(pipeline_info, f, indent=2)
        
        return pipeline_info
    
    def get_model_info(self) -> Dict:
        """Get model parameter counts."""
        info = {}
        
        if self.bc_model is not None:
            info["bc_params"] = sum(p.numel() for p in self.bc_model.parameters())
        
        if self.rl_model is not None:
            # Delta head params only
            info["delta_params"] = sum(
                p.numel() for p in self.rl_model.delta_head.parameters()
            )
            info["frozen_bc_params"] = sum(
                p.numel() for p in self.rl_model.bc_model.parameters()
                if not p.requires_grad
            )
        
        return info


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Driving-first pipeline: BC → RL → CARLA"
    )
    parser.add_argument("--bc-checkpoint", type=Path, help="Path to BC checkpoint")
    parser.add_argument("--ssl-checkpoint", type=Path, help="Path to SSL encoder checkpoint")
    parser.add_argument("--rl-checkpoint", type=Path, help="Path to RL checkpoint")
    parser.add_argument("--output", type=Path, default=Path("out/pipeline"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test", action="store_true", help="Test mode with stub models")
    
    args = parser.parse_args()
    
    # Build config
    config = PipelineConfig(
        bc_checkpoint=args.bc_checkpoint if not args.test else None,
        ssl_checkpoint=args.ssl_checkpoint if not args.test else None,
        rl_checkpoint=args.rl_checkpoint if not args.test else None,
        output_dir=args.output,
        device=args.device,
    )
    
    # Run pipeline
    pipeline = DrivingPipeline(config)
    
    if args.test:
        # Test mode - verify imports and model creation
        print("Test mode: verifying model creation")
        
        from training.rl.bc_checkpoint_loader import load_bc_waypoint_model, BCCheckpointConfig
        from training.rl.waypoint_policy_wrapper import (
            WaypointPolicyWithDelta,
            create_rl_refinement_model,
            WaypointPolicyInference,
        )
        from training.bc.waypoint_bc_model import WaypointBCModel, WaypointBCConfig
        import torch
        
        # Create test models directly
        bc_config = WaypointBCConfig(
            bev_feature_dim=768,
            bev_height=1,
            bev_width=1,
            num_waypoints=8,
            waypoint_dim=2,
            predict_speed=False,
            use_temporal=False,
        )
        bc_model = WaypointBCModel(bc_config)
        
        # Test BC forward
        bev = torch.randn(1, 768, 1, 1)
        bc_out = bc_model(bev)
        waypoints = bc_out[0] if isinstance(bc_out, tuple) else bc_out
        print(f"  BC output: {waypoints.shape}")
        
        # Test RL model creation
        rl_model, info = create_rl_refinement_model(
            bc_checkpoint=None,
            delta_hidden_dims=[128, 64],
            num_waypoints=8,
            predict_speed=False,  # Match BC config
            device="cpu",
        )
        print(f"  RL model created: {info['trainable_params']} trainable params")
        
        # Test RL forward
        state = torch.tensor([[0, 0, 0, 5.0]])
        waypoints, delta = rl_model(bev, state)
        print(f"  RL output: waypoints {waypoints.shape}, delta {delta.shape}")
        
        print("Test passed!")
    else:
        info = pipeline.run()
        print(f"Pipeline complete!")
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

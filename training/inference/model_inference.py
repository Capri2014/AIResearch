#!/usr/bin/env python3
"""
Model Inference API - Unified interface for running single forward passes through pipeline models.

Provides a clean interface for:
- SSL encoder embeddings from images
- Waypoint predictions from observation sequences  
- RL refinement delta corrections
- Combined SFT + RL final waypoints

Usage:
    # CLI
    python training/inference/model_inference.py \
        --checkpoint out/waypoint_bc/final.pt \
        --input observation.npy \
        --output predictions.json

    # As library
    from training.inference.model_inference import WaypointInferenceAPI
    
    api = WaypointInferenceAPI(checkpoint="out/waypoint_bc/final.pt")
    waypoints = api.predict(observation)  # (num_waypoints, 2)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pipeline components
from training.sft.train_waypoint_bc import WaypointBCConfig, WaypointBCModel
from training.rl.run_refine_delta_waypoint import DeltaHead, RefinementPolicy


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    # Model checkpoint path
    checkpoint: str = ""
    
    # Model stage: ssl | waypoint_bc | rl_refinement
    stage: str = "waypoint_bc"
    
    # RL refinement options (for rl_refinement stage)
    sft_checkpoint: str = ""
    rl_checkpoint: str = ""
    delta_scale: float = 0.5
    
    # Model architecture
    num_waypoints: int = 8
    encoder_dim: int = 256
    waypoint_dim: int = 2
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class WaypointPrediction:
    """Single waypoint prediction result."""
    
    # Shape: (num_waypoints, 2) for positions, (num_waypoints,) for speeds
    waypoints: np.ndarray      # (N, 2) - x, y positions
    speeds: np.ndarray        # (N,) - speed at each waypoint
    progress: np.ndarray     # (N,) - route progress [0, 1]
    
    # Metadata
    confidence: Optional[float] = None
    raw_logits: Optional[np.ndarray] = None


@dataclass
class InferenceResult:
    """Result from running inference."""
    
    # Predictions
    waypoints: Optional[WaypointPrediction] = None
    
    # Raw embeddings (if stage=ssl)
    embeddings: Optional[np.ndarray] = None
    
    # Delta corrections (if stage=rl_refinement)
    delta_waypoints: Optional[np.ndarray] = None
    sft_waypoints: Optional[np.ndarray] = None
    final_waypoints: Optional[np.ndarray] = None
    
    # Metadata
    success: bool = True
    error: Optional[str] = None
    inference_time_ms: float = 0.0


class SSLEncoderInference:
    """SSL encoder for generating embeddings from images."""
    
    def __init__(
        self,
        checkpoint: str,
        encoder_dim: int = 256,
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.encoder_dim = encoder_dim
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load SSL encoder model from checkpoint."""
        import training.pretrain.run_combined_ssl as combined_ssl
        import training.pretrain.run_unified_ssl as unified_ssl
        
        ckpt_path = Path(self.checkpoint)
        if not ckpt_path.exists():
            print(f"[WARN] Checkpoint {self.checkpoint} not found, using toy model")
            self.model = None
            return
        
        try:
            # Try to load unified SSL model
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
                # Create model
                self.model = unified_ssl.ConvEncoder(
                    encoder_dim=self.encoder_dim,
                )
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print(f"[INFO] Loaded UnifiedSSL model from {self.checkpoint}")
            else:
                print(f"[WARN] Unknown checkpoint format, using toy model")
                self.model = None
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            self.model = None
    
    @torch.no_grad()
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Generate embeddings from images.
        
        Args:
            images: (B, C, H, W) or (C, H, W) tensor or array
            
        Returns:
            embeddings: (B, encoder_dim) or (encoder_dim,)
        """
        # Convert to tensor
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        # Add batch dimension if needed
        if images.ndim == 3:
            images = images.unsqueeze(0)
        
        images = images.to(self.device).float() / 255.0
        
        # Run model
        if self.model is None:
            # Toy embeddings
            embeddings = np.random.randn(
                images.shape[0], self.encoder_dim
            ).astype(np.float32)
        else:
            self.model.eval()
            embeddings = self.model(images).cpu().numpy()
        
        return embeddings


class WaypointBCInference:
    """Waypoint BC model for inference."""
    
    def __init__(
        self,
        checkpoint: str,
        num_waypoints: int = 8,
        encoder_dim: int = 256,
        waypoint_dim: int = 2,
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.num_waypoints = num_waypoints
        self.encoder_dim = encoder_dim
        self.waypoint_dim = waypoint_dim
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load waypoint BC model from checkpoint."""
        ckpt_path = Path(self.checkpoint)
        if not ckpt_path.exists():
            print(f"[WARN] Checkpoint {self.checkpoint} not found, using toy model")
            self.model = None
            return
        
        try:
            # Create model
            self.model = WaypointBCModel(
                encoder_dim=self.encoder_dim,
                waypoint_dim=self.waypoint_dim,
                num_waypoints=self.num_waypoints,
            )
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"[INFO] Loaded WaypointBC model from {self.checkpoint}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            self.model = None
    
    @torch.no_grad()
    def predict(
        self,
        observations: np.ndarray,
        return_dict: bool = False,
    ) -> Union[WaypointPrediction, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Predict waypoints from observation sequence.
        
        Args:
            observations: (B, T, obs_dim) or (T, obs_dim) - sequence of observations
            return_dict: If True, return WaypointPrediction object
            
        Returns:
            WaypointPrediction or (waypoints, speeds, progress) arrays
        """
        # Convert to tensor
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations)
        
        # Add batch dimension if needed
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        
        observations = observations.to(self.device).float()
        
        # Run model
        if self.model is None:
            # Toy predictions: simple straight line with constant speed
            B = observations.shape[0]
            waypoints = np.zeros((B, self.num_waypoints, 2), dtype=np.float32)
            speeds = np.ones((B, self.num_waypoints), dtype=np.float32) * 5.0  # 5 m/s
            progress = np.linspace(0, 1, self.num_waypoints)[None, :]
            
            waypoints[:, :, 0] = np.linspace(0, 50, self.num_waypoints)[None, :]
            waypoints[:, :, 1] = np.random.randn(B, 1) * 0.5
        else:
            self.model.eval()
            output = self.model(observations)
            
            waypoints = output["waypoints"].cpu().numpy()
            speeds = output["speeds"].cpu().numpy()
            progress = output["progress"].cpu().numpy()
        
        if return_dict:
            return WaypointPrediction(
                waypoints=waypoints[0],
                speeds=speeds[0],
                progress=progress[0],
            )
        
        return waypoints[0], speeds[0], progress[0]


class RLRefinementInference:
    """RL refinement model for inference."""
    
    def __init__(
        self,
        sft_checkpoint: str,
        rl_checkpoint: str,
        delta_scale: float = 0.5,
        encoder_dim: int = 256,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
        device: str = "cuda",
    ):
        self.sft_checkpoint = sft_checkpoint
        self.rl_checkpoint = rl_checkpoint
        self.delta_scale = delta_scale
        self.encoder_dim = encoder_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.device = device
        
        self.sft_model = None
        self.delta_head = None
        self._load_models()
    
    def _load_models(self):
        """Load SFT and RL refinement models."""
        # Load SFT model
        sft_path = Path(self.sft_checkpoint)
        if sft_path.exists():
            try:
                self.sft_model = WaypointBCModel(
                    encoder_dim=self.encoder_dim,
                    waypoint_dim=self.waypoint_dim,
                    num_waypoints=self.num_waypoints,
                )
                checkpoint = torch.load(sft_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.sft_model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.sft_model.load_state_dict(checkpoint)
                self.sft_model.to(self.device)
                self.sft_model.eval()
                print(f"[INFO] Loaded SFT model from {self.sft_checkpoint}")
            except Exception as e:
                print(f"[WARN] Failed to load SFT: {e}")
                self.sft_model = None
        else:
            print(f"[WARN] SFT checkpoint {self.sft_checkpoint} not found")
            self.sft_model = None
        
        # Load RL delta head
        if self.rl_checkpoint:
            rl_path = Path(self.rl_checkpoint)
            if rl_path.exists():
                try:
                    self.delta_head = DeltaHead(
                        encoder_dim=self.encoder_dim,
                        num_waypoints=self.num_waypoints,
                    )
                    checkpoint = torch.load(rl_path, map_location=self.device)
                    self.delta_head.load_state_dict(checkpoint)
                    self.delta_head.to(self.device)
                    self.delta_head.eval()
                    print(f"[INFO] Loaded RL delta head from {self.rl_checkpoint}")
                except Exception as e:
                    print(f"[WARN] Failed to load RL delta: {e}")
                    self.delta_head = None
            else:
                print(f"[WARN] RL checkpoint {self.rl_checkpoint} not found")
                self.delta_head = None
    
    @torch.no_grad()
    def predict(
        self,
        observations: np.ndarray,
        return_dict: bool = False,
    ) -> Union[InferenceResult, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Predict refined waypoints: final = sft + delta_scale * delta(obs)
        
        Args:
            observations: (B, T, obs_dim) or (T, obs_dim) - sequence of observations
            return_dict: If True, return InferenceResult object
            
        Returns:
            InferenceResult or (sft_waypoints, delta_waypoints, final_waypoints)
        """
        # Convert to tensor
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations)
        
        # Add batch dimension if needed
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        
        observations = observations.to(self.device).float()
        
        # Get SFT waypoints
        if self.sft_model is None:
            B = observations.shape[0]
            sft_waypoints = np.zeros((B, self.num_waypoints, 2), dtype=np.float32)
            sft_waypoints[:, :, 0] = np.linspace(0, 50, self.num_waypoints)
        else:
            self.sft_model.eval()
            output = self.sft_model(observations)
            sft_waypoints = output["waypoints"].cpu().numpy()
        
        # Get delta waypoints
        if self.delta_head is None:
            delta_waypoints = np.zeros_like(sft_waypoints)
        else:
            self.delta_head.eval()
            delta_waypoints = self.delta_head(observations).cpu().numpy()
        
        # Final waypoints = SFT + delta_scale * delta
        final_waypoints = sft_waypoints + self.delta_scale * delta_waypoints
        
        if return_dict:
            return InferenceResult(
                waypoints=None,  # Deprecated, use sft/final instead
                delta_waypoints=delta_waypoints[0],
                sft_waypoints=sft_waypoints[0],
                final_waypoints=final_waypoints[0],
                success=True,
            )
        
        return sft_waypoints[0], delta_waypoints[0], final_waypoints[0]


class WaypointInferenceAPI:
    """
    Unified inference API for pipeline models.
    
    Provides a single interface for running inference across all pipeline stages.
    """
    
    def __init__(
        self,
        checkpoint: str = "",
        sft_checkpoint: str = "",
        rl_checkpoint: str = "",
        stage: str = "waypoint_bc",
        delta_scale: float = 0.5,
        num_waypoints: int = 8,
        encoder_dim: int = 256,
        waypoint_dim: int = 2,
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.sft_checkpoint = sft_checkpoint or checkpoint
        self.rl_checkpoint = rl_checkpoint
        self.stage = stage
        self.delta_scale = delta_scale
        self.num_waypoints = num_waypoints
        self.encoder_dim = encoder_dim
        self.waypoint_dim = waypoint_dim
        self.device = device
        
        # Initialize model
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the appropriate model based on stage."""
        if self.stage == "ssl":
            self.model = SSLEncoderInference(
                checkpoint=self.checkpoint,
                encoder_dim=self.encoder_dim,
                device=self.device,
            )
        elif self.stage == "waypoint_bc":
            self.model = WaypointBCInference(
                checkpoint=self.checkpoint,
                num_waypoints=self.num_waypoints,
                encoder_dim=self.encoder_dim,
                waypoint_dim=self.waypoint_dim,
                device=self.device,
            )
        elif self.stage == "rl_refinement":
            self.model = RLRefinementInference(
                sft_checkpoint=self.sft_checkpoint,
                rl_checkpoint=self.rl_checkpoint,
                delta_scale=self.delta_scale,
                encoder_dim=self.encoder_dim,
                num_waypoints=self.num_waypoints,
                waypoint_dim=self.waypoint_dim,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
    
    @torch.no_grad()
    def predict(
        self,
        observations: np.ndarray,
    ) -> np.ndarray:
        """
        Run inference on observations.
        
        Args:
            observations: Input observations (batch, seq, obs_dim) or (seq, obs_dim)
            
        Returns:
            Predictions appropriate for the stage
        """
        if self.stage == "ssl":
            return self.model.predict(observations)
        elif self.stage == "waypoint_bc":
            wp, speeds, progress = self.model.predict(observations)
            return wp
        elif self.stage == "rl_refinement":
            sft, delta, final = self.model.predict(observations)
            return final
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
    
    def predict_detailed(
        self,
        observations: np.ndarray,
    ) -> InferenceResult:
        """
        Run inference with full metadata.
        
        Args:
            observations: Input observations
            
        Returns:
            InferenceResult with full prediction details
        """
        if self.stage == "rl_refinement":
            return self.model.predict(observations, return_dict=True)
        else:
            # Use WaypointBC for other stages
            wp_pred = self.model.predict(observations, return_dict=True)
            return InferenceResult(
                waypoints=wp_pred,
                success=True,
            )


def find_latest_checkpoint(
    stage: str,
    out_dir: str = "out",
) -> Optional[str]:
    """Find the latest checkpoint for a given stage."""
    import glob
    
    out_path = Path(out_dir)
    if not out_path.exists():
        return None
    
    if stage == "waypoint_bc":
        patterns = [
            "waypoint_bc/*/final.pt",
            "waypoint_bc/*/best.pt",
            "bc/*/final.pt",
            "bc/*/best.pt",
        ]
    elif stage == "rl_refinement":
        patterns = [
            "rl/*/final.pt",
            "rl/*/best_reward.pt",
            "rl/*/best.pt",
        ]
    elif stage == "ssl":
        patterns = [
            "ssl/*/final.pt",
            "pretrain/*/final.pt",
            "pretrain/*/best.pt",
        ]
    else:
        return None
    
    for pattern in patterns:
        matches = sorted(
            glob.glob(str(out_path / pattern)),
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    
    return None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference on pipeline models"
    )
    parser.add_argument(
        "--checkpoint",
        "--model",
        "-m",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--stage",
        "-s",
        default="waypoint_bc",
        choices=["ssl", "waypoint_bc", "rl_refinement"],
        help="Model stage",
    )
    parser.add_argument(
        "--sft-checkpoint",
        help="SFT checkpoint (for rl_refinement stage)",
    )
    parser.add_argument(
        "--rl-checkpoint",
        help="RL checkpoint (for rl_refinement stage)",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=0.5,
        help="Delta scale for RL refinement",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input observations (numpy .npy file or JSON)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output predictions (JSON file)",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=8,
        help="Number of waypoints",
    )
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=256,
        help="Encoder dimension",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--find-latest",
        action="store_true",
        help="Automatically find latest checkpoint",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Find latest checkpoint if requested
    if args.find_latest and not args.checkpoint:
        args.checkpoint = find_latest_checkpoint(args.stage)
        if args.checkpoint:
            print(f"[INFO] Found latest checkpoint: {args.checkpoint}")
        else:
            print("[WARN] No checkpoint found, using toy model")
    
    # Create API
    api = WaypointInferenceAPI(
        checkpoint=args.checkpoint or "",
        sft_checkpoint=args.sft_checkpoint or args.checkpoint or "",
        rl_checkpoint=args.rl_checkpoint or "",
        stage=args.stage,
        delta_scale=args.delta_scale,
        num_waypoints=args.num_waypoints,
        encoder_dim=args.encoder_dim,
        device=args.device,
    )
    
    # Load input
    if args.input:
        input_path = Path(args.input)
        if input_path.suffix == ".npy":
            observations = np.load(input_path)
        elif input_path.suffix == ".json":
            with open(input_path) as f:
                observations = np.array(json.load(f))
        else:
            raise ValueError(f"Unknown input format: {input_path.suffix}")
    else:
        # Default: random observation sequence
        # (seq_len, obs_dim=17): [x, y, vx, vy, heading, target_x, target_y, ... waypoints]
        obs_dim = 17
        seq_len = 4
        observations = np.random.randn(seq_len, obs_dim).astype(np.float32)
        print(f"[INFO] Using random observations: {observations.shape}")
    
    # Run inference
    import time
    start = time.perf_counter()
    
    if args.stage == "rl_refinement":
        result = api.predict_detailed(observations)
        predictions = {
            "sft_waypoints": result.sft_waypoints.tolist() if result.sft_waypoints is not None else None,
            "delta_waypoints": result.delta_waypoints.tolist() if result.delta_waypoints is not None else None,
            "final_waypoints": result.final_waypoints.tolist() if result.final_waypoints is not None else None,
        }
    elif args.stage == "ssl":
        embeddings = api.predict(observations)
        predictions = {
            "embeddings": embeddings.tolist(),
        }
    else:
        # waypoint_bc
        result = api.predict_detailed(observations)
        if result.waypoints:
            predictions = {
                "waypoints": result.waypoints.waypoints.tolist(),
                "speeds": result.waypoints.speeds.tolist(),
                "progress": result.waypoints.progress.tolist(),
            }
        else:
            predictions = {
                "waypoints": api.predict(observations).tolist(),
            }
    
    elapsed = (time.perf_counter() - start) * 1000
    predictions["_metadata"] = {
        "stage": args.stage,
        "checkpoint": args.checkpoint,
        "inference_time_ms": round(elapsed, 2),
    }
    
    if args.verbose:
        print(f"[INFO] Inference took {elapsed:.2f}ms")
        print(f"[INFO] Stage: {args.stage}")
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"[INFO] Saved predictions to {args.output}")
    else:
        print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
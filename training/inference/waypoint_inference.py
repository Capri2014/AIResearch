#!/usr/bin/env python3
"""
Inference wrapper for waypoint policy models.
Loads trained checkpoints and provides inference API for CARLA closed-loop evaluation.

Pipeline stage: Waypoint BC/SFT → Inference → CARLA ScenarioRunner eval
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

import torch
import numpy as np


# --- Configuration ---

@dataclass
class InferenceConfig:
    """Configuration for waypoint inference."""
    checkpoint_path: str
    model_type: str = "bc"  # bc, sft, rl_refined
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_waypoints: int = 8
    waypoint_dt: float = 0.5  # seconds between waypoints
    max_speed: float = 15.0  # m/s
    use_cot: bool = False


@dataclass
class WaypointOutput:
    """Output from waypoint inference."""
    waypoints: np.ndarray  # (num_waypoints, 3) - x, y, heading
    confidence: Optional[np.ndarray] = None  # (num_waypoints,)
    logits: Optional[np.ndarray] = None  # For debugging


# --- Model Loading ---

def load_waypoint_model(config: InferenceConfig) -> torch.nn.Module:
    """Load waypoint model from checkpoint."""
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    
    # Determine model class from checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Infer model architecture from state_dict keys
    has_cot = any("cot" in k.lower() for k in state_dict.keys())
    config.use_cot = has_cot
    
    if config.model_type == "bc" or has_cot:
        from training.sft.train_waypoint_bc_cot import WaypointBCModel
        model = WaypointBCModel(
            num_waypoints=config.num_waypoints,
            use_cot=config.use_cot
        )
    else:
        from training.sft.train_waypoint_bc_with_metrics import WaypointBCMetricModel
        model = WaypointBCMetricModel(num_waypoints=config.num_waypoints)
    
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    
    return model


def load_sft_model(config: InferenceConfig) -> torch.nn.Module:
    """Load SFT waypoint model (supervised fine-tuned)."""
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    from training.sft.train_waypoint_bc_cot import WaypointBCModel
    model = WaypointBCModel(
        num_waypoints=config.num_waypoints,
        use_cot=True  # SFT typically uses CoT
    )
    
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    
    return model


def load_rl_model(config: InferenceConfig) -> torch.nn.Module:
    """Load RL-refined waypoint model."""
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    
    # RL checkpoints may have different structure
    if "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    from training.rl.train_rl_delta_waypoint_stub import DeltaWaypointHead
    model = DeltaWaypointHead(config.num_waypoints)
    
    # Try to load with strict=False for compatibility
    model.load_state_dict(state_dict, strict=False)
    model.to(config.device)
    model.eval()
    
    return model


# --- Inference ---

def prepare_inputs(
    route_mask: np.ndarray,
    current_pose: np.ndarray,
    speed: float,
    config: InferenceConfig
) -> dict:
    """
    Prepare inputs for waypoint inference.
    
    Args:
        route_mask: (H, W) binary mask of route/lanes
        current_pose: (3,) - x, y, yaw (radians)
        speed: current speed in m/s
        
    Returns:
        Dictionary of model inputs
    """
    # Convert route mask to tensor
    route_tensor = torch.from_numpy(route_mask).float()
    if route_tensor.ndim == 3:
        route_tensor = route_tensor.permute(2, 0, 1)  # (C, H, W)
    else:
        route_tensor = route_tensor.unsqueeze(0)  # (1, H, W)
    
    # Normalize pose
    pose_normalized = current_pose.copy()
    pose_normalized[0] /= 100.0  # Normalize x
    pose_normalized[1] /= 100.0  # Normalize y
    pose_normalized[2] /= np.pi  # Normalize yaw
    
    # Normalize speed
    speed_normalized = speed / config.max_speed
    
    inputs = {
        "route": route_tensor.unsqueeze(0),  # (1, C, H, W)
        "pose": torch.tensor(pose_normalized).float().unsqueeze(0),  # (1, 3)
        "speed": torch.tensor([speed_normalized]).float(),  # (1,)
    }
    
    return inputs


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    inputs: dict,
    config: InferenceConfig
) -> WaypointOutput:
    """Run inference on prepared inputs."""
    # Move inputs to device
    inputs_device = {
        k: v.to(config.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    
    # Run model
    outputs = model(**inputs_device)
    
    # Extract waypoints
    if isinstance(outputs, dict):
        waypoints = outputs.get("waypoints", outputs.get("pred_waypoints"))
        confidence = outputs.get("confidence", None)
        logits = outputs.get("logits", None)
    else:
        waypoints = outputs
        confidence = None
        logits = None
    
    # Convert to numpy
    waypoints_np = waypoints.cpu().numpy()[0]  # (num_waypoints, 3)
    
    # Denormalize waypoints
    waypoints_np[:, 0] *= 100.0  # Denormalize x
    waypoints_np[:, 1] *= 100.0  # Denormalize y
    
    confidence_np = confidence.cpu().numpy()[0] if confidence is not None else None
    logits_np = logits.cpu().numpy() if logits is not None else None
    
    return WaypointOutput(
        waypoints=waypoints_np,
        confidence=confidence_np,
        logits=logits_np
    )


def inference_waypoints(
    checkpoint_path: str,
    route_mask: np.ndarray,
    current_pose: np.ndarray,
    speed: float,
    model_type: str = "bc"
) -> WaypointOutput:
    """
    High-level API for waypoint inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        route_mask: (H, W) binary mask of route/lanes
        current_pose: (3,) - x, y, yaw (radians)
        speed: current speed in m/s
        model_type: "bc", "sft", or "rl_refined"
        
    Returns:
        WaypointOutput with predicted waypoints
    """
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        model_type=model_type
    )
    
    # Load model
    if model_type == "sft":
        model = load_sft_model(config)
    elif model_type == "rl_refined":
        model = load_rl_model(config)
    else:
        model = load_waypoint_model(config)
    
    # Prepare inputs
    inputs = prepare_inputs(route_mask, current_pose, speed, config)
    
    # Run inference
    output = run_inference(model, inputs, config)
    
    return output


# --- Checkpoint Discovery ---

def find_latest_checkpoint(
    base_dir: str = "out",
    model_type: str = "bc"
) -> Optional[str]:
    """Find the latest checkpoint for the given model type."""
    base_path = Path(base_dir)
    
    # Search patterns by model type
    if model_type == "bc":
        patterns = ["**/waypoint_bc_*/final.pt", "**/waypoint_bc_*/best.pt"]
    elif model_type == "sft":
        patterns = ["**/waypoint_sft_*/final.pt", "**/waypoint_sft_*/best.pt"]
    elif model_type == "rl_refined":
        patterns = ["**/rl_delta_waypoint_*/best_model.pt", "**/rl_delta_waypoint_*/final_model.pt"]
    else:
        patterns = ["**/checkpoints/*.pt"]
    
    latest = None
    latest_mtime = 0
    
    for pattern in patterns:
        for path in base_path.glob(pattern):
            if path.is_file():
                mtime = path.stat().st_mtime
                if mtime > latest_mtime:
                    latest = str(path)
                    latest_mtime = mtime
    
    return latest


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Waypoint inference wrapper")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (auto-discover if not provided)")
    parser.add_argument("--model-type", type=str, default="bc",
                        choices=["bc", "sft", "rl_refined"],
                        help="Model type")
    parser.add_argument("--output-dir", type=str, default="out/inference",
                        help="Output directory for predictions")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List available checkpoints")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test with dummy inputs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Auto-discover checkpoint if not provided
    if args.checkpoint is None:
        checkpoint = find_latest_checkpoint(model_type=args.model_type)
        if checkpoint:
            print(f"Auto-discovered checkpoint: {checkpoint}")
        else:
            print("No checkpoint found")
            return
    else:
        checkpoint = args.checkpoint
    
    # List checkpoints mode
    if args.list_checkpoints:
        print(f"=== {args.model_type.upper()} checkpoints ===")
        for model_type in ["bc", "sft", "rl_refined"]:
            ckpt = find_latest_checkpoint(model_type=model_type)
            print(f"  {model_type}: {ckpt}")
        return
    
    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Smoke test mode
    if args.smoke_test:
        print("Running smoke test...")
        
        # Create dummy inputs
        route_mask = np.zeros((128, 128), dtype=np.float32)
        route_mask[32:96, 32:96] = 1.0  # Simple route
        
        current_pose = np.array([50.0, 50.0, 0.0], dtype=np.float32)  # x, y, yaw
        speed = 5.0  # m/s
        
        # Run inference
        output = inference_waypoints(
            checkpoint_path=checkpoint,
            route_mask=route_mask,
            current_pose=current_pose,
            speed=speed,
            model_type=args.model_type
        )
        
        print(f"Output waypoints shape: {output.waypoints.shape}")
        print(f"First waypoint: {output.waypoints[0]}")
        
        # Save output
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "smoke_test_output.json")
        
        output_data = {
            "checkpoint": checkpoint,
            "model_type": args.model_type,
            "waypoints": output.waypoints.tolist(),
            "confidence": output.confidence.tolist() if output.confidence is not None else None
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Output saved to: {output_path}")
        print("Smoke test PASSED")
        return
    
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
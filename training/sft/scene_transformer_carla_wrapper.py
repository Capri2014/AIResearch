"""Scene Transformer model wrapper for CARLA closed-loop evaluation.

This module provides a ModelWrapper that loads a trained SceneTransformerWithWaypointHead
checkpoint and serves predictions for CARLA ScenarioRunner evaluation.

Usage:
    from training.sft.scene_transformer_carla_wrapper import SceneTransformerModelWrapper
    
    wrapper = SceneTransformerModelWrapper(
        checkpoint="out/sft_scene_transformer/model.pt",
        device="cuda"
    )
    
    # For CARLA: get waypoints for current agent state
    waypoints = wrapper.predict_waypoints(
        agent_history=[...],  # (A, T, 7) agent trajectory history
        map_polylines=[...],  # (P, M, 3) map polyline points
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn


@dataclass
class WaypointPrediction:
    """Waypoint prediction output."""
    proposals: torch.Tensor  # (K, horizon_steps, 2) predicted trajectories
    scores: torch.Tensor     # (K,) confidence scores for each mode
    best_index: int          # Index of best mode


class SceneTransformerModelWrapper(nn.Module):
    """Wrapper around SceneTransformerWithWaypointHead for CARLA integration.
    
    Provides a simple API that converts from CARLA-style agent/lane representations
    to the tensor format expected by the Scene Transformer model.
    """
    
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        horizon_steps: int = 20,
        k_proposals: int = 5,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.k_proposals = k_proposals
        
        # Default model config (matching train_scene_transformer.py)
        self.hidden_dim = 256
        self.num_agents = 32
        self.num_polylines = 16
        self.points_per_polyline = 20
        self.history_steps = 20
        
        # Model will be built on first forward pass or when checkpoint loaded
        self.model: Optional[nn.Module] = None
        self._built = False
        
        if checkpoint:
            self.load_checkpoint(checkpoint)
    
    def _build_model(self) -> None:
        """Build the Scene Transformer model architecture."""
        from training.sft.scene_encoder import (
            SceneEncoderConfig,
            SceneTransformerWithWaypointHead,
        )
        
        # Create config matching training config
        config = SceneEncoderConfig(
            hidden_dim=self.hidden_dim,
            num_agents=self.num_agents,
            num_map_points=self.points_per_polyline,
            num_history=self.history_steps,
            num_layers=3,
            output_dim=self.hidden_dim,
        )
        
        self.model = SceneTransformerWithWaypointHead(
            encoder_config=config,
            use_proposal_head=True,
            num_proposals=self.k_proposals,
            horizon_steps=self.horizon_steps,
        ).to(self.device)
        
        self._built = True
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained model checkpoint."""
        if not self._built:
            self._build_model()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                # Try loading directly
                self.model.load_state_dict(checkpoint)
        else:
            # Checkpoint is the state dict directly
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"[SceneTransformerModelWrapper] Loaded checkpoint: {checkpoint_path}")
    
    def _create_dummy_agent_history(
        self,
        ego_position: List[float],
        ego_yaw: float,
        ego_velocity: List[float],
        other_agents: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Create agent history tensor from current state.
        
        Args:
            ego_position: [x, y] position in meters
            ego_yaw: heading in radians
            ego_velocity: [vx, vy] velocity in m/s
            other_agents: List of dicts with position, velocity
        
        Returns:
            agent_history: (A, T, 7) where 7 = [x, y, z, vx, vy, yaw, type]
        """
        A = self.num_agents
        T = self.history_steps
        
        # Initialize with zeros (padding)
        history = torch.zeros(A, T, 7, device=self.device)
        
        # Type encoding: 1.0 = vehicle, 2.0 = pedestrian, 3.0 = cyclist
        VEHICLE_TYPE = 1.0
        
        # Fill in ego agent (index 0)
        # Use current position for all history (static snapshot)
        for t in range(T):
            history[0, t, 0] = ego_position[0]  # x
            history[0, t, 1] = ego_position[1]  # y
            history[0, t, 2] = 0.0  # z
            history[0, t, 3] = ego_velocity[0]  # vx
            history[0, t, 4] = ego_velocity[1]  # vy
            history[0, t, 5] = ego_yaw  # yaw
            history[0, t, 6] = VEHICLE_TYPE  # type
        
        # Fill in other agents
        if other_agents:
            for i, agent in enumerate(other_agents[:A - 1]):
                pos = agent.get("position", [0.0, 0.0])
                vel = agent.get("velocity", [0.0, 0.0])
                yaw = agent.get("yaw", 0.0)
                agent_type = agent.get("type", "vehicle")
                
                type_val = VEHICLE_TYPE if agent_type == "vehicle" else 2.0
                
                for t in range(T):
                    history[i + 1, t, 0] = pos[0]
                    history[i + 1, t, 1] = pos[1]
                    history[i + 1, t, 2] = 0.0
                    history[i + 1, t, 3] = vel[0]
                    history[i + 1, t, 4] = vel[1]
                    history[i + 1, t, 5] = yaw
                    history[i + 1, t, 6] = type_val
        
        return history
    
    def _create_dummy_map_polylines(
        self,
        lanes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create map polylines tensor from lane data.
        
        Args:
            lanes: List of dicts with keys: points (list of [x, y])
        
        Returns:
            polylines: (P, M, 3) where 3 = [x, y, is_endpoint]
            masks: (P, M) boolean mask for valid points
        """
        P = self.num_polylines
        M = self.points_per_polyline
        
        polylines = torch.zeros(P, M, 3, device=self.device)
        masks = torch.zeros(P, M, dtype=torch.bool, device=self.device)
        
        if lanes:
            for i, lane in enumerate(lanes[:P]):
                points = lane.get("points", [])
                for j, pt in enumerate(points[:M]):
                    polylines[i, j, 0] = pt[0]  # x
                    polylines[i, j, 1] = pt[1]  # y
                    polylines[i, j, 2] = 1.0 if j == len(points) - 1 else 0.0  # is_endpoint
                    masks[i, j] = True
        
        return polylines, masks
    
    @torch.no_grad()
    def predict_waypoints(
        self,
        ego_position: List[float],
        ego_yaw: float,
        ego_velocity: List[float],
        other_agents: Optional[List[Dict[str, Any]]] = None,
        lanes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict waypoints for the ego agent.
        
        Args:
            ego_position: [x, y] position in meters
            ego_yaw: heading in radians
            ego_velocity: [vx, vy] velocity in m/s
            other_agents: List of dicts with position, velocity, yaw, type
            lanes: List of dicts with points (list of [x, y])
        
        Returns:
            waypoints: (K, horizon_steps, 2) predicted trajectories
            scores: (K,) confidence scores
        """
        if not self._built:
            self._build_model()
        
        # Create input tensors from current state snapshot
        agent_history = self._create_dummy_agent_history(
            ego_position, ego_yaw, ego_velocity, other_agents
        ).unsqueeze(0)  # (1, A, T, 7)
        
        polylines, polyline_masks = self._create_dummy_map_polylines(lanes)
        polylines = polylines.unsqueeze(0)  # (1, P, M, 3)
        polyline_masks = polyline_masks.unsqueeze(0)  # (1, P, M)
        
        # Create agent mask (all valid)
        agent_masks = torch.ones(1, self.num_agents, self.history_steps, dtype=torch.bool, device=self.device)
        
        # Run model
        output = self.model(
            agent_history,
            polylines,
            agent_masks,
            polyline_masks,
            target_agent_idx=0,
        )
        
        # Extract waypoints and scores
        proposals = output["proposals"]  # (1, K, H, 2)
        scores = output["scores"]       # (1, K)
        
        proposals = proposals.squeeze(0)  # (K, H, 2)
        scores = scores.squeeze(0)       # (K,)
        
        return proposals, scores
    
    @torch.no_grad()
    def predict_best_waypoints(
        self,
        ego_position: List[float],
        ego_yaw: float,
        ego_velocity: List[float],
        other_agents: Optional[List[Dict[str, Any]]] = None,
        lanes: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Predict the single best waypoint trajectory.
        
        Returns: (horizon_steps, 2) predicted trajectory
        """
        proposals, scores = self.predict_waypoints(
            ego_position, ego_yaw, ego_velocity, other_agents, lanes
        )
        
        # Select best mode
        best_idx = scores.argmax().item()
        return proposals[best_idx]  # (horizon_steps, 2)


def load_model_for_carla(
    checkpoint: str,
    device: str = "cuda",
    horizon_steps: int = 20,
) -> SceneTransformerModelWrapper:
    """Convenience function to load a model for CARLA evaluation.
    
    Args:
        checkpoint: Path to trained model checkpoint
        device: Device to run on
        horizon_steps: Number of future waypoints to predict
    
    Returns:
        Loaded SceneTransformerModelWrapper ready for inference
    """
    wrapper = SceneTransformerModelWrapper(
        checkpoint=checkpoint,
        device=device,
        horizon_steps=horizon_steps,
    )
    return wrapper


# Example usage when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon-steps", type=int, default=20)
    args = parser.parse_args()
    
    # Load and test
    wrapper = load_model_for_carla(args.checkpoint, args.device, args.horizon_steps)
    
    # Dummy prediction test
    waypoints = wrapper.predict_best_waypoints(
        ego_position=[0.0, 0.0],
        ego_yaw=0.0,
        ego_velocity=[1.0, 0.0],
        other_agents=[],
        lanes=[],
    )
    
    print(f"Predicted waypoints shape: {waypoints.shape}")
    print(f"Waypoints (first 5):\n{waypoints[:5]}")

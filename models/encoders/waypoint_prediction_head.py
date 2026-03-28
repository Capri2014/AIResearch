"""Waypoint prediction head for SSL pretraining.

This module provides the prediction head that combines with TinyMultiCamEncoder
for end-to-end waypoint prediction SSL pretraining.

Driving-first pipeline:
- Waymo episodes → SSL pretrain (waypoint prediction) → waypoint BC → RL refinement
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class WaypointPredictionHead(nn.Module):
    """MLP head for waypoint prediction.
    
    Takes encoder embeddings and predicts waypoints in ego coordinates.
    
    Attributes:
        embedding_dim: Input embedding dimension
        num_waypoints: Number of waypoints to predict
        hidden_dim: Hidden layer dimension
       num_layers: Number of hidden layers
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_waypoints: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_waypoints = num_waypoints
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build MLP layers
        layers = []
        in_dim = embedding_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output head: predict (num_waypoints, 2) coordinates
        self.output = nn.Linear(hidden_dim, num_waypoints * 2)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from encoder embeddings.
        
        Args:
            embeddings: (batch, embedding_dim)
            
        Returns:
            waypoints: (batch, num_waypoints, 2) in ego coordinates
        """
        x = self.mlp(embeddings)
        x = self.output(x)
        waypoints = x.view(-1, self.num_waypoints, 2)
        return waypoints


class WaypointPredictionEncoder(nn.Module):
    """Combined encoder + waypoint prediction head.
    
    This module combines TinyMultiCamEncoder with WaypointPredictionHead
    for end-to-end SSL pretraining with waypoint regression.
    
    Attributes:
        encoder: TinyMultiCamEncoder backbone
        waypoint_head: WaypointPredictionHead
        num_waypoints: Number of waypoints to predict
    """
    
    def __init__(
        self,
        *,
        out_dim: int = 128,
        num_waypoints: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        # Import here to avoid circular import
        from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
        
        self.encoder = TinyMultiCamEncoder(out_dim=out_dim)
        
        self.waypoint_head = WaypointPredictionHead(
            embedding_dim=out_dim,
            num_waypoints=num_waypoints,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        self.num_waypoints = num_waypoints
    
    def forward(
        self,
        image_tensors: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning embeddings and waypoints.
        
        Args:
            image_tensors: Dict[cam_name, tensor] (batch, 3, H, W)
            
        Returns:
            embeddings: (batch, embedding_dim)
            waypoints: (batch, num_waypoints, 2)
        """
        embeddings = self.encoder(image_tensors)
        waypoints = self.waypoint_head(embeddings)
        return embeddings, waypoints
    
    def predict_waypoints(
        self,
        image_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Shortcut to get just waypoint predictions.
        
        Args:
            image_tensors: Dict[cam_name, tensor]
            
        Returns:
            waypoints: (batch, num_waypoints, 2)
        """
        _, waypoints = self.forward(image_tensors)
        return waypoints
    
    def get_encoder_embedding(
        self,
        image_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get just the encoder embedding (for multi-task learning).
        
        Args:
            image_tensors: Dict[cam_name, tensor]
            
        Returns:
            embeddings: (batch, embedding_dim)
        """
        embeddings = self.encoder(image_tensors)
        return embeddings


def waypoint_prediction_loss(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor
) -> torch.Tensor:
    """Compute waypoint prediction loss (L1 regression).
    
    Args:
        pred_waypoints: Predicted waypoints (batch, num_waypoints, 2)
        target_waypoints: Target waypoints (batch, num_waypoints, 2)
        
    Returns:
        Loss scalar
    """
    # L1 loss per waypoint
    loss = torch.abs(pred_waypoints - target_waypoints).mean()
    return loss


def squared_waypoint_prediction_loss(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor
) -> torch.Tensor:
    """Compute waypoint prediction loss (L2 regression).
    
    Args:
        pred_waypoints: Predicted waypoints (batch, num_waypoints, 2)
        target_waypoints: Target waypoints (batch, num_waypoints, 2)
        
    Returns:
        Loss scalar
    """
    # L2 loss per waypoint
    loss = torch.nn.functional.mse_loss(pred_waypoints, target_waypoints)
    return loss


def run_smoke_test(
    batch_size: int = 4,
    num_cameras: int = 4,
    out_dim: int = 128,
    num_waypoints: int = 8,
    device: str = "cpu",
):
    """Smoke test for WaypointPredictionEncoder.
    
    Args:
        batch_size: Batch size
        num_cameras: Number of cameras
        out_dim: Embedding dimension
        num_waypoints: Number of waypoints
        device: Device to use
    """
    torch.manual_seed(42)
    
    print(f"\n=== WaypointPredictionEncoder Smoke Test ===")
    print(f"Batch size: {batch_size}")
    print(f"Cameras: {num_cameras}")
    print(f"Embedding dim: {out_dim}")
    print(f"Waypoints: {num_waypoints}")
    print(f"Device: {device}")
    
    # Create model
    model = WaypointPredictionEncoder(
        out_dim=out_dim,
        num_waypoints=num_waypoints,
    ).to(device)
    
    # Create dummy input
    cam_names = ["front", "rear", "left", "right"][:num_cameras]
    image_tensors = {
        cam: torch.randn(batch_size, 3, 224, 224).to(device)
        for cam in cam_names
    }
    
    # Forward pass
    embeddings, waypoints = model.forward(image_tensors)
    
    print(f"\nOutput shapes:")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  waypoints: {waypoints.shape}")
    
    # Test loss
    target_waypoints = torch.randn(batch_size, num_waypoints, 2).to(device)
    loss = waypoint_prediction_loss(waypoints, target_waypoints)
    print(f"\nLoss (L1): {loss.item():.4f}")
    
    # Gradient check
    loss.backward()
    print(f"Gradient check passed!")
    
    print(f"\n=== Smoke test passed! ===")
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Waypoint prediction encoder")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    run_smoke_test(
        batch_size=args.batch_size,
        num_cameras=args.num_cameras,
        out_dim=args.out_dim,
        num_waypoints=args.num_waypoints,
        device=args.device,
    )


if __name__ == "__main__":
    main()
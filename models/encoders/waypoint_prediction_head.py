"""Waypoint prediction encoder head for SSL pretraining.

This module adds a waypoint prediction head to the TinyMultiCamEncoder,
enabling end-to-end pretraining with waypoint regression as the objective.

Driving-first pipeline:
- Waymo episodes → SSL pretrain (waypoint prediction) → waypoint BC → RL refinement

Usage:
  from models.encoders.waypoint_prediction_head import WaypointPredictionEncoder
  
  encoder = WaypointPredictionEncoder(encoder_out_dim=256, num_waypoints=8)
  embeddings = encoder(images_by_cam)
  waypoints = encoder.predict_waypoints(embeddings)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("PyTorch is required") from e
    return torch


class WaypointPredictionHead(_require_torch().nn.Module):
    """Waypoint prediction head for regression.
    
    Takes encoder embeddings and predicts future waypoints in ego frame.
    
    Architecture:
    - MLP with 2 hidden layers
    - Output: (num_waypoints, 2) waypoints in ego coordinates
    """
    
    def __init__(
        self,
        encoder_out_dim: int = 128,
        num_waypoints: int = 8,
        hidden_dim: int = 256,
    ):
        torch = _require_torch()
        super().__init__()
        
        self.encoder_out_dim = encoder_out_dim
        self.num_waypoints = num_waypoints
        
        # MLP head
        self.net = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, num_waypoints * 2),  # (num_waypoints, 2)
        )
        
        self.out_dim = num_waypoints * 2
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from encoder embeddings.
        
        Args:
            embeddings: Tensor (batch, encoder_out_dim)
            
        Returns:
            waypoints: Tensor (batch, num_waypoints, 2)
        """
        batch_size = embeddings.shape[0]
        flat = self.net(embeddings)  # (batch, num_waypoints * 2)
        waypoints = flat.view(batch_size, self.num_waypoints, 2)  # (batch, num_waypoints, 2)
        return waypoints


class WaypointPredictionEncoder(_require_torch().nn.Module):
    """Combined encoder for waypoint prediction SSL.
    
    This combines:
    1. TinyMultiCamEncoder for multi-camera perception
    2. WaypointPredictionHead for waypoint regression
    
    The model can be trained end-to-end with waypoint prediction loss.
    """
    
    def __init__(
        self,
        encoder_out_dim: int = 128,
        num_waypoints: int = 8,
        hidden_dim: int = 256,
    ):
        torch = _require_torch()
        super().__init__()
        
        # Import encoder here to avoid circular imports
        from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
        
        self.encoder = TinyMultiCamEncoder(out_dim=encoder_out_dim)
        self.head = WaypointPredictionHead(
            encoder_out_dim=encoder_out_dim,
            num_waypoints=num_waypoints,
            hidden_dim=hidden_dim,
        )
        
        self.encoder_out_dim = encoder_out_dim
        self.num_waypoints = num_waypoints
    
    def forward(
        self,
        images_by_cam: Dict[str, torch.Tensor],
        image_valid_by_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder + head.
        
        Args:
            images_by_cam: Dict[cam_name, tensor] (B, 3, H, W)
            image_valid_by_cam: Optional mask per camera
            
        Returns:
            Tuple of:
            - embeddings: (B, encoder_out_dim)
            - waypoints: (B, num_waypoints, 2)
        """
        embeddings = self.encoder(
            images_by_cam,
            image_valid_by_cam=image_valid_by_cam,
        )
        waypoints = self.head(embeddings)
        return embeddings, waypoints
    
    def predict_waypoints(
        self,
        images_by_cam: Dict[str, torch.Tensor],
        image_valid_by_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Shortcut to get just waypoint predictions.
        
        Args:
            images_by_cam: Dict[cam_name, tensor] (B, 3, H, W)
            image_valid_by_cam: Optional mask per camera
            
        Returns:
            waypoints: (B, num_waypoints, 2)
        """
        _, waypoints = self.forward(images_by_cam, image_valid_by_cam)
        return waypoints
    
    def get_encoder_embedding(
        self,
        images_by_cam: Dict[str, torch.Tensor],
        image_valid_by_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get encoder embedding (for use with other SSL objectives).
        
        Args:
            images_by_cam: Dict[cam_name, tensor] (B, 3, H, W)
            image_valid_by_cam: Optional mask per camera
            
        Returns:
            embeddings: (B, encoder_out_dim)
        """
        embeddings, _ = self.forward(images_by_cam, image_valid_by_cam)
        return embeddings


def waypoint_prediction_loss(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute waypoint prediction loss (L1 regression).
    
    Args:
        pred_waypoints: Predicted waypoints (batch, num_waypoints, 2)
        target_waypoints: Target waypoints (batch, num_waypoints, 2)
        reduction: Loss reduction ("mean", "sum", "none")
        
    Returns:
        Loss scalar (or per-sample if reduction="none")
    """
    torch = _require_torch()
    
    # L1 loss per waypoint
    loss = torch.abs(pred_waypoints - target_waypoints)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss  # (batch, num_waypoints, 2)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def squared_waypoint_prediction_loss(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute waypoint prediction loss (L2/squared regression).
    
    Args:
        pred_waypoints: Predicted waypoints (batch, num_waypoints, 2)
        target_waypoints: Target waypoints (batch, num_waypoints, 2)
        reduction: Loss reduction ("mean", "sum", "none")
        
    Returns:
        Loss scalar (or per-sample if reduction="none")
    """
    torch = _require_torch()
    
    # L2 loss per waypoint
    diff = pred_waypoints - target_waypoints
    loss = torch.sum(diff ** 2, dim=-1)  # (batch, num_waypoints)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss  # (batch, num_waypoints)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def run_smoke_test(
    batch_size: int = 4,
    encoder_out_dim: int = 128,
    num_waypoints: int = 8,
    device: str = "cpu",
):
    """Smoke test for waypoint prediction encoder.
    
    Args:
        batch_size: Batch size
        encoder_out_dim: Encoder output dimension
        num_waypoints: Number of waypoints to predict
        device: Device to use
    """
    torch = _require_torch()
    
    print(f"\n=== Waypoint Prediction Encoder Smoke Test ===")
    print(f"Batch size: {batch_size}")
    print(f"Encoder out dim: {encoder_out_dim}")
    print(f"Num waypoints: {num_waypoints}")
    print(f"Device: {device}")
    
    # Create model
    model = WaypointPredictionEncoder(
        encoder_out_dim=encoder_out_dim,
        num_waypoints=num_waypoints,
    ).to(device)
    
    model.eval()
    
    # Create dummy inputs (simulate 2 cameras)
    images_by_cam = {
        "front": torch.randn(batch_size, 3, 224, 224).to(device),
        "front_left": torch.randn(batch_size, 3, 224, 224).to(device),
    }
    
    # Forward pass
    with torch.no_grad():
        embeddings, waypoints = model(images_by_cam)
    
    print(f"\nOutputs:")
    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  waypoints shape: {waypoints.shape}")
    print(f"  waypoints[0, 0]: {waypoints[0, 0].tolist()}")
    
    # Test loss
    target_waypoints = torch.randn(batch_size, num_waypoints, 2).to(device)
    loss = waypoint_prediction_loss(waypoints, target_waypoints)
    print(f"  loss (L1): {loss.item():.4f}")
    
    loss_sq = squared_waypoint_prediction_loss(waypoints, target_waypoints)
    print(f"  loss (L2): {loss_sq.item():.4f}")
    
    # Test gradient flow
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(3):
        optimizer.zero_grad()
        _, waypoints_pred = model(images_by_cam)
        loss = waypoint_prediction_loss(waypoints_pred, target_waypoints)
        loss.backward()
        optimizer.step()
        print(f"  step {step}: loss = {loss.item():.4f}")
    
    print(f"\n=== Smoke test passed! ===")
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Waypoint prediction encoder smoke test"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-out-dim", type=int, default=128)
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    run_smoke_test(
        batch_size=args.batch_size,
        encoder_out_dim=args.encoder_out_dim,
        num_waypoints=args.num_waypoints,
        device=args.device,
    )


if __name__ == "__main__":
    main()
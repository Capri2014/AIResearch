"""Waypoint prediction head + encoder for SSL pretraining.

This module provides:
- WaypointPredictionHead: MLP that regresses waypoints from encoder embeddings
- WaypointPredictionEncoder: Combined encoder + waypoint head for multi-task learning

Usage:
  from models.encoders.waypoint_prediction_head import WaypointPredictionEncoder, WaypointPredictionHead
  
  encoder = WaypointPredictionEncoder(out_dim=128, num_waypoints=8)
  embeddings, waypoints = encoder(images_dict)  # Forward pass
  wp = encoder.predict_waypoints(images_dict)   # Just waypoints
"""

from __future__ import annotations

from typing import Dict, Optional

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required") from e
    return torch


class WaypointPredictionHead(torch.nn.Module):
    """MLP head that regresses waypoints from encoder embeddings."""

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        num_waypoints: int = 8,
    ):
        torch = _require_torch()
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, num_waypoints * 2),  # (x, y) per waypoint
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Regress waypoints from embeddings.

        Args:
          x: (B, in_dim) encoder embeddings

        Returns:
          waypoints: (B, num_waypoints, 2) in ego coordinates (x forward, y left)
        """
        out = self.net(x)  # (B, num_waypoints * 2)
        return out.view(-1, self.num_waypoints, 2)


class WaypointPredictionEncoder(torch.nn.Module):
    """Combined multi-camera encoder + waypoint prediction head.

    This enables end-to-end pretraining where:
    1. The encoder learns from contrastive SSL across camera views
    2. The waypoint head learns from ground-truth waypoint supervision
    """

    def __init__(
        self,
        out_dim: int = 128,
        num_waypoints: int = 8,
        encoder_hidden_dim: Optional[int] = None,
    ):
        torch = _require_torch()
        super().__init__()
        self.out_dim = out_dim
        self.num_waypoints = num_waypoints

        # Multi-camera encoder.
        self.encoder = TinyMultiCamEncoder(out_dim=out_dim)

        # Waypoint prediction head.
        self.waypoint_head = WaypointPredictionHead(
            in_dim=out_dim,
            hidden_dim=encoder_hidden_dim or out_dim * 2,
            num_waypoints=num_waypoints,
        )

    def forward(
        self,
        images_by_cam: Dict[str, torch.Tensor],
        *,
        image_valid_by_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning embeddings and predicted waypoints.

        Args:
          images_by_cam: dict[cam, tensor] where each tensor is (B,3,H,W)
          image_valid_by_cam: Optional dict[cam, bool tensor] of shape (B,)

        Returns:
          embeddings: (B, out_dim) encoder embeddings
          waypoints: (B, num_waypoints, 2) predicted waypoints
        """
        embeddings = self.encoder(images_by_cam, image_valid_by_cam=image_valid_by_cam)
        waypoints = self.waypoint_head(embeddings)
        return embeddings, waypoints

    def predict_waypoints(
        self,
        images_by_cam: Dict[str, torch.Tensor],
        *,
        image_valid_by_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Shortcut: just predict waypoints without returning embeddings.

        Args:
          images_by_cam: dict[cam, tensor] where each tensor is (B,3,H,W)
          image_valid_by_cam: Optional dict[cam, bool tensor] of shape (B,)

        Returns:
          waypoints: (B, num_waypoints, 2) predicted waypoints
        """
        _, waypoints = self.forward(images_by_cam, image_valid_by_cam=image_valid_by_cam)
        return waypoints

    def get_encoder_embedding(
        self,
        images_by_cam: Dict[str, torch.Tensor],
        *,
        image_valid_by_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Extract just the encoder embedding (for multi-task learning)."""
        embeddings, _ = self.forward(images_by_cam, image_valid_by_cam=image_valid_by_cam)
        return embeddings


# ============================================================================
# Loss functions for waypoint prediction.
# ============================================================================

def waypoint_prediction_loss(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 loss (smooth L1 / Huber) for waypoint regression.

    Args:
      pred_waypoints: (B, num_waypoints, 2) predictions
      target_waypoints: (B, num_waypoints, 2) ground truth
      reduction: "mean", "sum", or "none"

    Returns:
      loss: scalar tensor
    """
    torch = _require_torch()
    diff = (pred_waypoints - target_waypoints).abs()
    loss = diff.sum(dim=-1)  # (B, num_waypoints) - sum over (x, y)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # (B, num_waypoints)


def squared_waypoint_prediction_loss(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """L2 loss (squared Euclidean) for waypoint regression.

    Args:
      pred_waypoints: (B, num_waypoints, 2) predictions
      target_waypoints: (B, num_waypoints, 2) ground truth
      reduction: "mean", "sum", or "none"

    Returns:
      loss: scalar tensor
    """
    torch = _require_torch()
    diff_sq = (pred_waypoints - target_waypoints) ** 2
    loss = diff_sq.sum(dim=-1)  # (B, num_waypoints) - sum over (x, y)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # (B, num_waypoints)


# ============================================================================
# Smoke test.
# ============================================================================

if __name__ == "__main__":
    import sys

    print("[waypoint_prediction_head] Running smoke test...")

    torch = _require_torch()
    device = torch.device("cpu")
    bs, c, h, w = 4, 3, 224, 224

    # Dummy batch: 2 cameras.
    images = {
        "front": torch.randn(bs, c, h, w),
        "front_left": torch.randn(bs, c, h, w),
    }
    valid = {
        "front": torch.ones(bs, dtype=torch.bool),
        "front_left": torch.ones(bs, dtype=torch.bool),
    }

    # Create encoder.
    enc = WaypointPredictionEncoder(out_dim=128, num_waypoints=8)
    embeddings, waypoints = enc(images, image_valid_by_cam=valid)

    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  waypoints shape: {waypoints.shape}")

    # Dummy ground truth.
    gt_waypoints = torch.randn(bs, 8, 2)

    # Loss.
    loss_l1 = waypoint_prediction_loss(waypoints, gt_waypoints)
    loss_l2 = squared_waypoint_prediction_loss(waypoints, gt_waypoints)
    print(f"  loss (L1): {float(loss_l1):.4f}")
    print(f"  loss (L2): {float(loss_l2):.4f}")

    # Gradient check.
    loss = loss_l1
    loss.backward()

    # Check that gradients flowed to encoder.
    has_grad = False
    for name, param in enc.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    print(f"  gradient check: {'PASSED' if has_grad else 'FAILED'}")

    print("[waypoint_prediction_head] Smoke test complete.")
    sys.exit(0 if has_grad else 1)
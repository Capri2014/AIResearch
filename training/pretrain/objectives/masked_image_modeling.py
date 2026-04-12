"""Masked Image Modeling (MIM) objective for SSL pretraining.

This provides a reconstruction-based objective that complements contrastive learning.
While contrastive objectives learn invariant representations, MIM learns generative
representations that understand structure and detail.

Approach:
- Mask random patches of input
- Train encoder to predict masked pixel values (or latent tokens)
- Uses MSE loss for pixel reconstruction

Usage:
  python -m training.pretrain.run_mim_pretrain \
    --episodes-glob "out/episodes/**/*.json" \
    --mask-ratio 0.4

This can be combined with contrastive objectives (multi-positive loss)
for a richer pretraining signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def random_masking(
    x: Tensor,
    mask_ratio: float = 0.4,
    mask_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Apply random masking to input tensor.

    Args:
        x: Input tensor of shape (B, C, H, W) or (B, N, C)
        mask_ratio: Fraction of tokens to mask (0.0 to 1.0)
        mask_value: Value to fill masked positions

    Returns:
        masked_x: Input with masked patches
        mask: Boolean mask (True = kept, False = masked)
    """
    B = x.shape[0]
    if x.ndim == 4:
        # Image: (B, C, H, W)
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        N = H * W
        # Flatten to (B, C, N)
        x_flat = x.view(B, C, N)
        # Generate mask
        mask = torch.rand(B, N, device=x.device) > mask_ratio
        # Apply mask
        masked_x = x_flat.clone()
        masked_x[:, :, ~mask] = mask_value
        # Reshape back
        masked_x = masked_x.view(B, C, H, W)
        return masked_x, mask
    else:
        # Already flattened: (B, N, C) or (B, C, N)
        N = x.shape[1] if x.ndim == 3 else x.shape[1]
        mask = torch.rand(B, N, device=x.device) > mask_ratio
        masked_x = x.clone()
        masked_x[~mask] = mask_value
        return masked_x, mask


def mim_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute MIM (masked image modeling) loss.

    Args:
        pred: Predicted values (B, N, C) or (B, C, N)
        target: Target values (B, N, C) or (B, C, N)
        mask: Boolean mask, True = positions to compute loss on
        reduction: "mean" | "sum" | "none"

    Returns:
        Loss tensor
    """
    # Ensure shapes match
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    
    # Compute MSE loss only on masked positions
    loss = (pred - target) ** 2
    
    # Apply mask (invert since we want loss on masked positions)
    mask_inverse = ~mask
    
    if reduction == "none":
        return loss[mask_inverse]
    elif reduction == "sum":
        return (loss * mask_inverse.float()).sum()
    else:  # mean
        # Compute mean only over valid masked positions
        valid_count = mask_inverse.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device)
        return (loss * mask_inverse.float()).sum() / valid_count


class MIMObjective(nn.Module):
    """Masked Image Modeling objective for SSL pretraining."""

    def __init__(
        self,
        mask_ratio: float = 0.4,
        mask_value: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute MIM loss.

        Args:
            pred: Predicted tokens (B, N, C)
            target: Target tokens (B, N, C)

        Returns:
            Loss tensor
        """
        B, N, _ = pred.shape
        mask = torch.rand(B, N, device=pred.device) > self.mask_ratio
        return mim_loss(pred, target, mask, self.reduction)


def combine_contrastive_and_mim(
    z_anchor: Tensor,
    z_positive: Tensor,
    pred_mim: Tensor,
    target_mim: Tensor,
    alpha: float = 0.5,
    temperature: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Combine contrastive and MIM objectives.

    This allows training with both invariant (contrastive) and generative (MIM)
    objectives simultaneously for richer representations.

    Args:
        z_anchor: Anchor embeddings (B, D)
        z_positive: Positive embeddings (B, D)
        pred_mim: MIM predictions (B, N, C)
        target_mim: MIM targets (B, N, C)
        alpha: Weight for MIM loss (0 = pure contrastive, 1 = pure MIM)
        temperature: Temperature for contrastive loss

    Returns:
        Tuple of (contrastive_loss, mim_loss)
    """
    # Contrastive loss (InfoNCE)
    logits = z_anchor @ z_positive.T / temperature
    labels = torch.arange(len(z_anchor), device=z_anchor.device)
    contrastive_loss = nn.CrossEntropyLoss()(logits, labels)
    
    # MIM loss
    B, N, _ = pred_mim.shape
    mask = torch.rand(B, N, device=pred_mim.device) > 0.4
    mim_loss_val = mim_loss(pred_mim, target_mim, mask, "mean")
    
    # Combined
    total = alpha * mim_loss_val + (1 - alpha) * contrastive_loss
    
    return contrastive_loss, mim_loss_val, total


if __name__ == "__main__":
    # Basic smoke test
    B, C, H, W = 4, 3, 64, 64
    x = torch.randn(B, C, H, W)
    
    masked_x, mask = random_masking(x, mask_ratio=0.4)
    print(f"Input shape: {x.shape}")
    print(f"Masked shape: {masked_x.shape}")
    print(f"Mask kept: {mask.sum().item()}/{mask.numel()} ({mask.float().mean():.2%})")
    
    # Test loss computation
    pred = torch.randn(B, H * W, C)
    target = torch.randn(B, H * W, C)
    loss = mim_loss(pred, target, mask)
    print(f"MIM loss: {loss.item():.4f}")
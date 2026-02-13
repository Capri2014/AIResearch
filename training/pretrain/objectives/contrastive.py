"""Contrastive SSL objective (InfoNCE) for multi-view (multi-camera) alignment.

We treat different cameras at the same timestep as positive pairs.
Negatives are other samples in the batch.

This is intentionally simple and self-contained.
"""

from __future__ import annotations


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required") from e
    return torch


def info_nce_loss(z_a, z_b, temperature: float = 0.1):
    """Compute symmetric InfoNCE between two embedding batches.

    Args:
      z_a: (B, D)
      z_b: (B, D)

    Returns:
      scalar loss
    """
    torch = _require_torch()

    # Normalize
    z_a = torch.nn.functional.normalize(z_a, dim=1)
    z_b = torch.nn.functional.normalize(z_b, dim=1)

    logits = (z_a @ z_b.t()) / temperature  # (B,B)
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_ab = torch.nn.functional.cross_entropy(logits, labels)
    loss_ba = torch.nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)

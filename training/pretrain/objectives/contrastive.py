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


def multi_pair_info_nce_loss(
    z_by_cam: dict[str, "torch.Tensor"],
    valid_by_cam: dict[str, "torch.Tensor"],
    *,
    temperature: float = 0.1,
    max_pairs_per_step: int | None = None,
):
    """Average InfoNCE across multiple camera pairs in the same timestep.

    This is a simple extension of pairwise InfoNCE:
    - for each (cam_i, cam_j) pair, compute InfoNCE on samples where both
      cameras are valid.
    - return the mean over pairs.

    Args:
      z_by_cam: dict[cam, (B,D)] embeddings.
      valid_by_cam: dict[cam, (B,)] bool masks.
      temperature: InfoNCE temperature.
      max_pairs_per_step: If set, limit to at most this many cam pairs.

    Returns:
      scalar loss, or None if no valid pairs exist.
    """
    torch = _require_torch()

    cams = sorted(set(z_by_cam.keys()) & set(valid_by_cam.keys()))
    if len(cams) < 2:
        return None

    # Deterministic pair order; optionally truncate.
    pairs: list[tuple[str, str]] = []
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            pairs.append((cams[i], cams[j]))
    if max_pairs_per_step is not None:
        pairs = pairs[: int(max_pairs_per_step)]

    losses = []
    for ca, cb in pairs:
        va = valid_by_cam[ca]
        vb = valid_by_cam[cb]
        v = va & vb
        # Need at least 2 for cross-entropy (otherwise logits is 1x1).
        if int(v.sum().item()) < 2:
            continue
        losses.append(info_nce_loss(z_by_cam[ca][v], z_by_cam[cb][v], temperature=temperature))

    if not losses:
        return None
    return torch.stack(losses).mean()

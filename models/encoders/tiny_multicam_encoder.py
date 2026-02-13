"""Tiny multi-camera encoder (PyTorch).

This is intentionally small and simple:
- per-camera CNN stem
- average pool across cameras that are present

Input:
- images: dict[str, torch.Tensor] with each tensor shaped (B, 3, H, W)

Output:
- embedding: torch.Tensor shaped (B, D)

This is a scaffold to unblock real SSL objectives.
"""

from __future__ import annotations


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required") from e
    return torch


class TinyCNNEncoder(_require_torch().nn.Module):
    def __init__(self, out_dim: int = 128):
        torch = _require_torch()
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TinyMultiCamEncoder(_require_torch().nn.Module):
    def __init__(self, out_dim: int = 128):
        torch = _require_torch()
        super().__init__()
        self.per_cam = TinyCNNEncoder(out_dim=out_dim)
        self.out_dim = out_dim

    def forward(self, images_by_cam, *, image_valid_by_cam=None):
        """Encode and fuse across cameras.

        Args:
          images_by_cam:
            dict[cam, tensor] where each tensor is (B,3,H,W), OR None for cams
            missing in the entire batch.
          image_valid_by_cam:
            Optional dict[cam, bool tensor] where each tensor is (B,). If
            provided, fusion will ignore invalid cameras *per sample*.

        Returns:
          embedding: torch.Tensor shaped (B, out_dim)

        Notes:
          - If `image_valid_by_cam` is not provided, we treat all provided
            cameras as valid.
          - If a sample has zero valid cameras, its output embedding will be
            zeros (denominator clamped).
        """
        torch = _require_torch()

        feats_per_cam = []
        weights_per_cam = []

        for cam, x in sorted(images_by_cam.items()):
            if x is None:
                continue

            f = self.per_cam(x)  # (B, D)
            feats_per_cam.append(f)

            if image_valid_by_cam is None:
                w = torch.ones((f.shape[0],), dtype=torch.float32, device=f.device)
            else:
                v = image_valid_by_cam.get(cam)
                if v is None:
                    # Be permissive: if mask missing for a cam, treat as valid.
                    w = torch.ones((f.shape[0],), dtype=torch.float32, device=f.device)
                else:
                    w = v.to(device=f.device, dtype=torch.float32)

            weights_per_cam.append(w)

        if not feats_per_cam:
            raise ValueError("No cameras provided to encoder")

        # Stack
        feats = torch.stack(feats_per_cam, dim=0)  # (C, B, D)
        weights = torch.stack(weights_per_cam, dim=0).unsqueeze(-1)  # (C, B, 1)

        # Weighted mean across cams per sample.
        weighted = feats * weights
        denom = weights.sum(dim=0).clamp(min=1.0)  # (B,1)
        out = weighted.sum(dim=0) / denom  # (B,D)
        return out

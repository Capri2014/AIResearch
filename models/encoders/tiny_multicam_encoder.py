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

    def forward(self, images_by_cam):
        """Encode and fuse across cameras.

        Args:
          images_by_cam: dict[cam, tensor] where each tensor is (B,3,H,W)

        Returns:
          (B, out_dim)
        """
        torch = _require_torch()
        feats = []
        for _, x in sorted(images_by_cam.items()):
            feats.append(self.per_cam(x))

        if not feats:
            raise ValueError("No cameras provided to encoder")

        # (num_cam, B, D) -> (B, D)
        stack = torch.stack(feats, dim=0)
        return stack.mean(dim=0)

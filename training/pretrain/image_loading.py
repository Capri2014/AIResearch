"""Image loading utilities for episodes-backed pretraining.

Design goals
------------
- Keep dependencies optional.
- Avoid forcing torchvision.

Behavior
--------
- If `path` is None: return None.
- If pillow is missing: raise a clear RuntimeError.

Returned tensor
---------------
A float32 tensor of shape (3, H, W) with values in [0, 1].

Note: This is intentionally minimal. For real training you likely want torchvision
transforms + augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for image loading") from e
    return torch


def _require_pil():
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PIL (pillow) is required for image decoding. Install pillow, or run in path-only mode."
        ) from e
    return Image


@dataclass(frozen=True)
class ImageConfig:
    size: Tuple[int, int] = (224, 224)


def load_image_tensor(path: Optional[str], cfg: ImageConfig = ImageConfig()):
    """Load an image file path into a float tensor (C,H,W) in [0,1]."""
    if path is None:
        return None

    torch = _require_torch()
    Image = _require_pil()

    img = Image.open(path).convert("RGB")
    img = img.resize(cfg.size)

    w, h = img.size
    # img.getdata() yields (R,G,B) tuples, length = w*h
    data = list(img.getdata())
    t = torch.tensor(data, dtype=torch.uint8).reshape(h, w, 3)
    t = t.to(torch.float32) / 255.0
    t = t.permute(2, 0, 1).contiguous()  # (3,H,W)
    return t

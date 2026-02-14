from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    *,
    torch: object,
    out_dir: Path,
    step: int,
    cfg: Any,
    model_state: dict[str, Any],
    optim_state: dict[str, Any] | None = None,
    name: str = "latest.pt",
) -> Path:
    """Save a training checkpoint.

    model_state is expected to contain torch state_dict()-compatible blobs.
    cfg can be a dataclass or plain dict.
    """

    ckpt_dir = _ensure_dir(Path(out_dir) / "checkpoints")
    path = ckpt_dir / name

    if hasattr(cfg, "__dataclass_fields__"):
        cfg_blob = asdict(cfg)
    elif isinstance(cfg, dict):
        cfg_blob = dict(cfg)
    else:
        cfg_blob = {"repr": repr(cfg)}

    payload = {
        "step": int(step),
        "cfg": cfg_blob,
        "model": model_state,
    }
    if optim_state is not None:
        payload["optim"] = optim_state

    torch.save(payload, path)
    return path


def maybe_load_checkpoint(*, torch: object, resume: str | None) -> dict[str, Any] | None:
    """Load a checkpoint path if provided.

    resume supports:
      - None: return None
      - "latest": expects <out_dir>/checkpoints/latest.pt handled by caller
      - explicit path
    """

    if resume is None:
        return None

    p = Path(resume)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    ckpt = torch.load(p, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format: {p}")
    return ckpt

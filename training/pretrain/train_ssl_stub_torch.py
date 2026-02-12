"""SSL pretrain stub (PyTorch) — episodes backend.

This is the first PyTorch-oriented pretrain scaffold.

What it does today:
- loads frames from episode.json files via `EpisodesFrameDataset`
- builds batches (paths + state)
- runs a placeholder "loss" that only depends on state

Why a placeholder loss?
- lets us validate distributed/data plumbing without committing to a specific
  objective/architecture.

Next:
- add image decoding + an encoder
- implement a real SSL objective (multi-view / temporal contrastive / masked modeling)
- save checkpoints
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from training.pretrain.dataloader_episodes import EpisodesFrameDataset, collate_batch


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch


@dataclass
class Config:
    episodes_glob: str = "out/episodes/**/*.json"
    batch_size: int = 8
    num_steps: int = 20
    lr: float = 1e-3
    out_dir: Path = Path("out/pretrain_ssl_stub")


def main() -> None:
    torch = _require_torch()

    cfg = Config()
    ds = EpisodesFrameDataset(cfg.episodes_glob)

    # Simple parameter to prove optimization runs.
    w = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    opt = torch.optim.Adam([w], lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Manual batching for simplicity.
    step = 0
    idx = 0
    while step < cfg.num_steps:
        batch = []
        for _ in range(cfg.batch_size):
            batch.append(ds[idx % len(ds)])
            idx += 1

        b = collate_batch(batch)

        # Placeholder loss: encourage w to match mean speed (nonsense but tests the pipe).
        speed = b["state"]["speed_mps"]
        loss = (w - speed.mean()) ** 2

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 5 == 0:
            print(f"[pretrain/ssl_stub] step={step} loss={float(loss):.6f} w={float(w.detach()):.4f}")

        step += 1

    # Save a tiny artifact.
    (cfg.out_dir / "stub.txt").write_text("ssl stub ran successfully\n")
    print(f"[pretrain/ssl_stub] wrote: {cfg.out_dir / 'stub.txt'}")


if __name__ == "__main__":
    main()

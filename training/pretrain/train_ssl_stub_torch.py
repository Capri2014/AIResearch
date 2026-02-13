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
from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder


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
    # Enable decode + caching in the dataset to keep the training loop simple.
    ds = EpisodesFrameDataset(cfg.episodes_glob, decode_images=True)

    encoder = TinyMultiCamEncoder(out_dim=128)
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Manual batching for simplicity.
    step = 0
    idx = 0
    while step < cfg.num_steps:
        batch = []
        for _ in range(cfg.batch_size):
            batch.append(ds[idx % len(ds)])
            idx += 1

        b = collate_batch(batch, stack_images=True)

        # Keep only cameras that are fully valid for this batch.
        # (TinyMultiCamEncoder has no masking yet.)
        images_by_cam = {}
        for cam, x in b.get("images_by_cam", {}).items():
            if x is None:
                continue
            valid = b.get("image_valid_by_cam", {}).get(cam)
            if valid is None or not bool(valid.all()):
                continue
            images_by_cam[cam] = x

        if not images_by_cam:
            # No decodable images; skip.
            if step % 5 == 0:
                print(f"[pretrain/ssl_stub] step={step} (no fully-valid cameras; skipping)")
            step += 1
            continue

        emb = encoder(images_by_cam)  # (B,D)
        # Placeholder loss: keep embeddings bounded.
        loss = (emb ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 5 == 0:
            print(f"[pretrain/ssl_stub] step={step} loss={float(loss):.6f} cams={list(images_by_cam.keys())}")

        step += 1

    # Save a tiny artifact.
    (cfg.out_dir / "stub.txt").write_text("ssl stub ran successfully\n")
    ckpt = {"encoder": encoder.state_dict(), "out_dim": encoder.out_dim}
    torch.save(ckpt, cfg.out_dir / "encoder.pt")
    print(f"[pretrain/ssl_stub] wrote: {cfg.out_dir / 'stub.txt'}")
    print(f"[pretrain/ssl_stub] wrote: {cfg.out_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()

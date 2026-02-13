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
from training.pretrain.image_loading import ImageConfig, load_image_tensor
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
    ds = EpisodesFrameDataset(cfg.episodes_glob)

    encoder = TinyMultiCamEncoder(out_dim=128)
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)

    img_cfg = ImageConfig(size=(224, 224))

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

        # Decode images for cameras that are present for this batch.
        images_by_cam = {}
        for cam, paths in b["image_paths_by_cam"].items():
            imgs = []
            ok = True
            for p in paths:
                try:
                    t = load_image_tensor(p, cfg=img_cfg)
                except RuntimeError:
                    # pillow missing; fall back to path-only mode.
                    ok = False
                    break
                if t is None:
                    ok = False
                    break
                imgs.append(t)
            if not ok:
                continue
            images_by_cam[cam] = torch.stack(imgs, dim=0)  # (B,3,H,W)

        if not images_by_cam:
            # No decodable images; skip.
            if step % 5 == 0:
                print(f"[pretrain/ssl_stub] step={step} (no images decoded; skipping)")
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

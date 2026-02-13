"""SSL pretrain v0: multi-view contrastive alignment across cameras.

Pipeline:
- episodes-backed dataset yields `image_path` per camera
- decode images (pillow)
- encode each camera with a tiny CNN
- apply InfoNCE between two selected camera views

This is a first real objective; it is still intentionally minimal.

Usage:
  python3 -m training.pretrain.train_ssl_contrastive_v0 --episodes-glob "out/episodes/**/*.json"

Deps:
- torch
- pillow

If torch/pillow are not installed, this script will raise a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

from models.encoders.tiny_multicam_encoder import TinyCNNEncoder
from training.pretrain.dataloader_episodes import EpisodesFrameDataset, collate_batch
# image decoding is handled inside EpisodesFrameDataset(decode_images=True)
from training.pretrain.objectives.contrastive import info_nce_loss


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch


@dataclass
class Config:
    episodes_glob: str
    batch_size: int = 16
    num_steps: int = 200
    lr: float = 1e-3
    out_dir: Path = Path("out/pretrain_contrastive_v0")
    temperature: float = 0.1
    cam_a: str = "front"
    cam_b: str = "front_left"


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=Path, default=Path("out/pretrain_contrastive_v0"))
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--cam-a", type=str, default="front")
    p.add_argument("--cam-b", type=str, default="front_left")
    a = p.parse_args()
    return Config(
        episodes_glob=a.episodes_glob,
        batch_size=a.batch_size,
        num_steps=a.num_steps,
        lr=a.lr,
        out_dir=a.out_dir,
        temperature=a.temperature,
        cam_a=a.cam_a,
        cam_b=a.cam_b,
    )


def main() -> None:
    torch = _require_torch()
    cfg = parse_args()

    # Use dataset-managed decoding so we also get per-sample camera validity masks.
    ds = EpisodesFrameDataset(cfg.episodes_glob, decode_images=True)
    enc = TinyCNNEncoder(out_dim=128)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    idx = 0
    while step < cfg.num_steps:
        batch = [ds[(idx + i) % len(ds)] for i in range(cfg.batch_size)]
        idx += cfg.batch_size
        # Stack images into dense tensors + validity masks so we can ignore padded zeros.
        b = collate_batch(batch, stack_images=True)

        xa = b.get("images_by_cam", {}).get(cfg.cam_a)
        xb = b.get("images_by_cam", {}).get(cfg.cam_b)
        va = b.get("image_valid_by_cam", {}).get(cfg.cam_a)
        vb = b.get("image_valid_by_cam", {}).get(cfg.cam_b)

        if xa is None or xb is None or va is None or vb is None:
            if step % 20 == 0:
                print(
                    f"[ssl/contrastive] step={step} missing cameras in batch (a={cfg.cam_a}, b={cfg.cam_b}); skipping"
                )
            step += 1
            continue

        valid = va & vb
        n_valid = int(valid.sum().item())
        if n_valid < 2:
            # InfoNCE needs at least 2 examples to be meaningful.
            if step % 20 == 0:
                print(
                    f"[ssl/contrastive] step={step} too few valid pairs (n_valid={n_valid}); skipping"
                )
            step += 1
            continue

        za_all = enc(xa)
        zb_all = enc(xb)
        za = za_all[valid]
        zb = zb_all[valid]

        loss = info_nce_loss(za, zb, temperature=cfg.temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            print(f"[ssl/contrastive] step={step} loss={float(loss):.6f}")

        step += 1

    torch.save({"encoder": enc.state_dict(), "out_dim": 128}, cfg.out_dir / "encoder.pt")
    (cfg.out_dir / "done.txt").write_text("contrastive v0 finished\n")
    print(f"[ssl/contrastive] wrote: {cfg.out_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()

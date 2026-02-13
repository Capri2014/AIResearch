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
from training.pretrain.image_loading import ImageConfig, load_image_tensor
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

    ds = EpisodesFrameDataset(cfg.episodes_glob)
    enc = TinyCNNEncoder(out_dim=128)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)

    img_cfg = ImageConfig(size=(224, 224))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    idx = 0
    while step < cfg.num_steps:
        batch = [ds[(idx + i) % len(ds)] for i in range(cfg.batch_size)]
        idx += cfg.batch_size
        b = collate_batch(batch)

        # decode two camera views
        def decode_cam(cam: str):
            paths = b["image_paths_by_cam"].get(cam)
            if paths is None:
                return None
            imgs = []
            for p in paths:
                t = load_image_tensor(p, cfg=img_cfg)
                if t is None:
                    return None
                imgs.append(t)
            return torch.stack(imgs, dim=0)

        xa = decode_cam(cfg.cam_a)
        xb = decode_cam(cfg.cam_b)
        if xa is None or xb is None:
            if step % 20 == 0:
                print(f"[ssl/contrastive] step={step} missing cam paths (a={cfg.cam_a}, b={cfg.cam_b}); skipping")
            step += 1
            continue

        za = enc(xa)
        zb = enc(xb)

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

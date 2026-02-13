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

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
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
    # Loader settings.
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True

    # Training device.
    device: str = "cuda"


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
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--no-persistent-workers", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--no-drop-last", action="store_true")
    a = p.parse_args()

    if a.drop_last and a.no_drop_last:
        raise ValueError("Pass only one of --drop-last or --no-drop-last")
    drop_last = True
    if a.no_drop_last:
        drop_last = False
    if a.drop_last:
        drop_last = True

    if a.pin_memory and a.no_pin_memory:
        raise ValueError("Pass only one of --pin-memory or --no-pin-memory")
    pin_memory = True
    if a.no_pin_memory:
        pin_memory = False
    if a.pin_memory:
        pin_memory = True

    if a.persistent_workers and a.no_persistent_workers:
        raise ValueError("Pass only one of --persistent-workers or --no-persistent-workers")
    persistent_workers = True
    if a.no_persistent_workers:
        persistent_workers = False
    if a.persistent_workers:
        persistent_workers = True

    return Config(
        episodes_glob=a.episodes_glob,
        batch_size=a.batch_size,
        num_steps=a.num_steps,
        lr=a.lr,
        out_dir=a.out_dir,
        temperature=a.temperature,
        cam_a=a.cam_a,
        cam_b=a.cam_b,
        num_workers=a.num_workers,
        prefetch_factor=a.prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        device=a.device,
    )


def main() -> None:
    torch = _require_torch()
    cfg = parse_args()

    # Use dataset-managed decoding so we also get per-sample camera validity masks.
    ds = EpisodesFrameDataset(cfg.episodes_glob, decode_images=True)
    enc = TinyMultiCamEncoder(out_dim=128).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        collate_fn=lambda batch: collate_batch(batch, stack_images=True),
    )

    # DataLoader perf knobs (valid only when num_workers > 0)
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)

    loader_kwargs["pin_memory"] = bool(cfg.pin_memory)

    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    step = 0
    it = iter(loader)
    while step < cfg.num_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)

        xa = b.get("images_by_cam", {}).get(cfg.cam_a)
        xb = b.get("images_by_cam", {}).get(cfg.cam_b)
        va = b.get("image_valid_by_cam", {}).get(cfg.cam_a)
        vb = b.get("image_valid_by_cam", {}).get(cfg.cam_b)

        # Move to device.
        if xa is not None:
            xa = xa.to(device, non_blocking=True)
        if xb is not None:
            xb = xb.to(device, non_blocking=True)
        if va is not None:
            va = va.to(device, non_blocking=True)
        if vb is not None:
            vb = vb.to(device, non_blocking=True)

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

        # Build a 2-cam batch and use masks to select a single view per pass.
        # This exercises TinyMultiCamEncoder's mask-aware fusion path.
        images = {cfg.cam_a: xa, cfg.cam_b: xb}
        zeros = torch.zeros_like(va)
        mask_a = {cfg.cam_a: va, cfg.cam_b: zeros}
        mask_b = {cfg.cam_a: zeros, cfg.cam_b: vb}

        za_all = enc(images, image_valid_by_cam=mask_a)
        zb_all = enc(images, image_valid_by_cam=mask_b)
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

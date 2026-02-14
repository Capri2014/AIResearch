"""SSL pretrain: temporal contrastive alignment (t vs t+k) within the same camera.

This complements multi-camera contrastive (train_ssl_contrastive_v0.py) by adding
*temporal positives*.

Objective:
  - pick a camera (default: front)
  - embed anchor frame z_t and positive frame z_{t+k}
  - InfoNCE between the two sets (in-batch negatives)

Usage:
  python3 -m training.pretrain.train_ssl_temporal_contrastive_v0 \
    --episodes-glob "out/episodes/**/*.json" \
    --cam front \
    --dt-frames 1

Deps:
  - torch
  - pillow (via EpisodesTemporalPairDataset when decode_images=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
from training.pretrain.dataloader_temporal_pairs import EpisodesTemporalPairDataset, collate_temporal_pair_batch
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
    out_dir: Path = Path("out/pretrain_temporal_contrastive_v0")
    temperature: float = 0.1
    cam: str = "front"
    dt_frames: int = 1

    # Loader settings.
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True

    device: str = "cuda"


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=Path, default=Path("out/pretrain_temporal_contrastive_v0"))
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--cam", type=str, default="front")
    p.add_argument("--dt-frames", type=int, default=1)
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
        cam=a.cam,
        dt_frames=a.dt_frames,
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

    ds = EpisodesTemporalPairDataset(cfg.episodes_glob, dt_frames=cfg.dt_frames, decode_images=True)

    device = torch.device(cfg.device)
    enc = TinyMultiCamEncoder(out_dim=128).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        collate_fn=lambda batch: collate_temporal_pair_batch(batch, stack_images=True),
    )

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

        xa = b.get("anchor", {}).get("images_by_cam", {}).get(cfg.cam)
        xp = b.get("pos", {}).get("images_by_cam", {}).get(cfg.cam)
        va = b.get("anchor", {}).get("image_valid_by_cam", {}).get(cfg.cam)
        vp = b.get("pos", {}).get("image_valid_by_cam", {}).get(cfg.cam)

        if xa is not None:
            xa = xa.to(device, non_blocking=True)
        if xp is not None:
            xp = xp.to(device, non_blocking=True)
        if va is not None:
            va = va.to(device, non_blocking=True)
        if vp is not None:
            vp = vp.to(device, non_blocking=True)

        if xa is None or xp is None or va is None or vp is None:
            if step % 20 == 0:
                print(f"[ssl/temporal] step={step} missing camera='{cfg.cam}' in batch; skipping")
            step += 1
            continue

        valid = va & vp
        n_valid = int(valid.sum().item())
        if n_valid < 2:
            if step % 20 == 0:
                print(f"[ssl/temporal] step={step} too few valid pairs (n_valid={n_valid}); skipping")
            step += 1
            continue

        # Mask-aware usage of TinyMultiCamEncoder (single cam active).
        images_a = {cfg.cam: xa}
        images_p = {cfg.cam: xp}
        mask_a = {cfg.cam: va}
        mask_p = {cfg.cam: vp}

        za_all = enc(images_a, image_valid_by_cam=mask_a)
        zp_all = enc(images_p, image_valid_by_cam=mask_p)
        za = za_all[valid]
        zp = zp_all[valid]

        loss = info_nce_loss(za, zp, temperature=cfg.temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            dt = b.get("meta", {}).get("dt")
            dt_str = f" dt~{float(dt[0]):.3f}s" if isinstance(dt, list) and dt else ""
            print(f"[ssl/temporal] step={step} loss={float(loss):.6f}{dt_str}")

        step += 1

    torch.save(
        {"encoder": enc.state_dict(), "out_dim": 128, "cam": cfg.cam, "dt_frames": cfg.dt_frames},
        cfg.out_dir / "encoder.pt",
    )
    (cfg.out_dir / "done.txt").write_text("temporal contrastive v0 finished\n")
    print(f"[ssl/temporal] wrote: {cfg.out_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()

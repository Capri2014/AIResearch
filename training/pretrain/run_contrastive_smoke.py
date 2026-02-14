"""Smoke runner for SSL contrastive v0.

Purpose:
- Run a short end-to-end training loop on *real* episodes and print a compact
  performance/health report (throughput, skips, GPU memory).

This is deliberately not a benchmark harness. It's a quick sanity check to run
whenever we change the dataloader/encoder/objective.

Usage (GPU):
  python3 -m training.pretrain.run_contrastive_smoke \
    --episodes-glob "out/episodes/**/*.json" \
    --device cuda --num-workers 4 --steps 50

Usage (CPU):
  python3 -m training.pretrain.run_contrastive_smoke \
    --episodes-glob "out/episodes/**/*.json" \
    --device cpu --num-workers 4 --steps 20

Deps:
- torch
- pillow (for image decoding)

If deps are missing, a clear error should be raised.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import time

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
from training.pretrain.dataloader_episodes import EpisodesFrameDataset, collate_batch
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
    steps: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    temperature: float = 0.1
    cam_a: str = "front"
    cam_b: str = "front_left"

    device: str = "cuda"
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--cam-a", type=str, default="front")
    p.add_argument("--cam-b", type=str, default="front_left")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--no-persistent-workers", action="store_true")
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--no-drop-last", action="store_true")

    a = p.parse_args()

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

    if a.drop_last and a.no_drop_last:
        raise ValueError("Pass only one of --drop-last or --no-drop-last")
    drop_last = True
    if a.no_drop_last:
        drop_last = False
    if a.drop_last:
        drop_last = True

    return Config(
        episodes_glob=a.episodes_glob,
        steps=a.steps,
        batch_size=a.batch_size,
        lr=a.lr,
        temperature=a.temperature,
        cam_a=a.cam_a,
        cam_b=a.cam_b,
        device=a.device,
        num_workers=a.num_workers,
        prefetch_factor=a.prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


def _maybe_cuda_mem_report(torch, device) -> str:
    if device.type != "cuda":
        return ""
    if not torch.cuda.is_available():
        return " (cuda not available)"
    allocated = int(torch.cuda.max_memory_allocated(device=device))
    reserved = int(torch.cuda.max_memory_reserved(device=device))

    def fmt(n: int) -> str:
        return f"{n/1024/1024:.1f} MiB"

    return f" | cuda_max_alloc={fmt(allocated)} cuda_max_reserved={fmt(reserved)}"


def main() -> None:
    torch = _require_torch()
    cfg = parse_args()

    device = torch.device(cfg.device)

    ds = EpisodesFrameDataset(cfg.episodes_glob, decode_images=True)
    enc = TinyMultiCamEncoder(out_dim=128).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        collate_fn=lambda batch: collate_batch(batch, stack_images=True),
        pin_memory=bool(cfg.pin_memory),
    )
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)

    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    # Counters
    n_total = 0
    n_missing_cam = 0
    n_too_few_pairs = 0
    n_optim = 0

    t0 = time.perf_counter()
    it = iter(loader)
    for step in range(cfg.steps):
        n_total += 1
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)

        xa = b.get("images_by_cam", {}).get(cfg.cam_a)
        xb = b.get("images_by_cam", {}).get(cfg.cam_b)
        va = b.get("image_valid_by_cam", {}).get(cfg.cam_a)
        vb = b.get("image_valid_by_cam", {}).get(cfg.cam_b)

        if xa is None or xb is None or va is None or vb is None:
            n_missing_cam += 1
            continue

        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        va = va.to(device, non_blocking=True)
        vb = vb.to(device, non_blocking=True)

        valid = va & vb
        if int(valid.sum().item()) < 2:
            n_too_few_pairs += 1
            continue

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
        n_optim += 1

    dt = time.perf_counter() - t0
    steps_per_s = (cfg.steps / dt) if dt > 0 else float("inf")

    mem = _maybe_cuda_mem_report(torch, device)

    print(
        "[smoke/contrastive] "
        f"steps={cfg.steps} batch={cfg.batch_size} device={cfg.device} "
        f"workers={cfg.num_workers} iters_per_s={steps_per_s:.2f}"
        f" | optim_steps={n_optim}/{n_total} missing_cam={n_missing_cam} too_few_pairs={n_too_few_pairs}"
        f"{mem}"
    )


if __name__ == "__main__":
    main()

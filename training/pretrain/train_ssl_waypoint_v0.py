"""SSL + Waypoint Prediction: Multi-task pretraining combines contrastive and waypoint regression.

Pipeline:
- episodes-backed dataset yields frames with camera images + ground-truth waypoints
- encode with WaypointPredictionEncoder (multi-camera encoder + waypoint head)
- joint training: contrastive SSL + waypoint regression

This enables end-to-end pretraining where the encoder learns from both:
1. Contrastive alignment across camera views
2. Waypoint regression supervision

Usage:
  python3 -m training.pretrain.train_ssl_waypoint_v0 --episodes-glob "out/episodes/**/*.json"

Deps:
- torch
- pillow

If torch/pillow are not installed, this script will raise a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
from models.encoders.waypoint_prediction_head import WaypointPredictionEncoder
from training.pretrain.dataloader_episodes import EpisodesFrameDataset, collate_batch
from training.pretrain.objectives.contrastive import info_nce_loss
from training.pretrain.objectives.waypoint_prediction import (
    waypoint_prediction_loss,
    squared_waypoint_prediction_loss,
)


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
    out_dir: Path = Path("out/pretrain_ssl_waypoint_v0")
    temperature: float = 0.1
    cam_a: str = "front"
    cam_b: str = "front_left"
    # Multi-task loss weights.
    waypoint_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.1
    # Waypoint regression settings.
    num_waypoints: int = 8
    waypoint_loss_type: str = "l1"  # "l1" or "l2"
    # Encoder settings.
    encoder_out_dim: int = 128
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
    p.add_argument("--out-dir", type=Path, default=Path("out/pretrain_ssl_waypoint_v0"))
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--cam-a", type=str, default="front")
    p.add_argument("--cam-b", type=str, default="front_left")
    p.add_argument("--waypoint-loss-weight", type=float, default=1.0)
    p.add_argument("--contrastive-loss-weight", type=float, default=0.1)
    p.add_argument("--num-waypoints", type=int, default=8)
    p.add_argument("--waypoint-loss-type", type=str, default="l1")
    p.add_argument("--encoder-out-dim", type=int, default=128)
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

    # Handle mutual exclusivity flags.
    def resolve_bool(store_var: bool, no_var: bool, default: bool) -> bool:
        if store_var and no_var:
            raise ValueError(f"Pass only one of --{store_var} or --{no_var}")
        if no_var:
            return False
        return default

    pin_memory = resolve_bool(a.pin_memory, a.no_pin_memory, True)
    persistent_workers = resolve_bool(a.persistent_workers, a.no_persistent_workers, True)
    drop_last = resolve_bool(a.drop_last, a.no_drop_last, True)

    if a.waypoint_loss_type not in ("l1", "l2"):
        raise ValueError(f"Invalid --waypoint-loss-type: {a.waypoint_loss_type}")

    return Config(
        episodes_glob=a.episodes_glob,
        batch_size=a.batch_size,
        num_steps=a.num_steps,
        lr=a.lr,
        out_dir=a.out_dir,
        temperature=a.temperature,
        cam_a=a.cam_a,
        cam_b=a.cam_b,
        waypoint_loss_weight=a.waypoint_loss_weight,
        contrastive_loss_weight=a.contrastive_loss_weight,
        num_waypoints=a.num_waypoints,
        waypoint_loss_type=a.waypoint_loss_type,
        encoder_out_dim=a.encoder_out_dim,
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

    # Create dataset - this will provide images AND ground-truth waypoints.
    ds = EpisodesFrameDataset(cfg.episodes_glob, decode_images=True)

    device = torch.device(cfg.device)
    # WaypointPredictionEncoder = TinyMultiCamEncoder + WaypointPredictionHead
    enc = WaypointPredictionEncoder(
        out_dim=cfg.encoder_out_dim,
        num_waypoints=cfg.num_waypoints,
    ).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # DataLoader kwargs.
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        collate_fn=lambda batch: collate_batch(batch, stack_images=True),
    )
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    loader_kwargs["pin_memory"] = bool(cfg.pin_memory)

    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    # Loss functions.
    if cfg.waypoint_loss_type == "l1":
        wp_loss_fn = waypoint_prediction_loss
    else:
        wp_loss_fn = squared_waypoint_prediction_loss

    step = 0
    it = iter(loader)
    accumulated_wp_loss = 0.0
    accumulated_contr_loss = 0.0

    while step < cfg.num_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)

        # Get camera images.
        xa = b.get("images_by_cam", {}).get(cfg.cam_a)
        xb = b.get("images_by_cam", {}).get(cfg.cam_b)
        va = b.get("image_valid_by_cam", {}).get(cfg.cam_a)
        vb = b.get("image_valid_by_cam", {}).get(cfg.cam_b)

        # Skip batch if cameras unavailable.
        if xa is None or xb is None or va is None or vb is None:
            if step % 20 == 0:
                print(
                    f"[ssl+wp] step={step} missing cameras (a={cfg.cam_a}, b={cfg.cam_b}); skipping"
                )
            step += 1
            continue

        # Get ground-truth waypoints from batch.
        gt_waypoints = b.get("waypoints")  # (B, num_waypoints, 2) or None
        if gt_waypoints is None:
            if step % 20 == 0:
                print(f"[ssl+wp] step={step} no ground-truth waypoints in batch; skipping")
            step += 1
            continue

        # Move data to device.
        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        va = va.to(device, non_blocking=True)
        vb = vb.to(device, non_blocking=True)
        gt_waypoints = gt_waypoints.to(device, non_blocking=True)

        # Build image dicts for contrastive learning.
        zeros = torch.zeros_like(va)
        mask_a = {cfg.cam_a: va, cfg.cam_b: zeros}
        mask_b = {cfg.cam_a: zeros, cfg.cam_b: vb}

        # Create images dict for encoder.
        images = {cfg.cam_a: xa, cfg.cam_b: xb}

        # Forward pass through WaypointPredictionEncoder.
        # This returns (embeddings, waypoints).
        embeddings_all, pred_waypoints_all = enc(images, image_valid_by_cam=mask_a)

        # Build valid mask for filtering.
        valid = va & vb
        n_valid = int(valid.sum().item())
        if n_valid < 2:
            if step % 20 == 0:
                print(f"[ssl+wp] step={step} too few valid pairs (n_valid={n_valid}); skipping")
            step += 1
            continue

        # Apply valid mask to get embeddings for contrastive loss.
        embeddings_valid = embeddings_all[valid]

        # Compute multi-task loss.
        # 1. Waypoint regression loss (full batch, both views).
        wp_loss = wp_loss_fn(pred_waypoints_all, gt_waypoints)

        # 2. Contrastive loss (valid samples only).
        # Re-run encoder with mask_b to get second view.
        embeddings_all_b, _ = enc(images, image_valid_by_cam=mask_b)
        embeddings_valid_b = embeddings_all_b[valid]
        contr_loss = info_nce_loss(embeddings_valid, embeddings_valid_b, temperature=cfg.temperature)

        # Combined loss.
        total_loss = (
            cfg.waypoint_loss_weight * wp_loss +
            cfg.contrastive_loss_weight * contr_loss
        )

        # Backward and step.
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        # Track losses for logging.
        accumulated_wp_loss += float(wp_loss)
        accumulated_contr_loss += float(contr_loss)

        if step % 20 == 0:
            avg_wp = accumulated_wp_loss / max(1, step % 20)
            avg_contr = accumulated_contr_loss / max(1, step % 20)
            print(
                f"[ssl+wp] step={step} "
                f"wp_loss={float(wp_loss):.4f} (avg={avg_wp:.4f}) "
                f"contr_loss={float(contr_loss):.4f} (avg={avg_contr:.4f}) "
                f"total={float(total_loss):.4f}"
            )
            accumulated_wp_loss = 0.0
            accumulated_contr_loss = 0.0

        step += 1

    # Save checkpoint.
    torch.save({
        "encoder": enc.state_dict(),
        "out_dim": cfg.encoder_out_dim,
        "num_waypoints": cfg.num_waypoints,
    }, cfg.out_dir / "encoder.pt")

    # Save training summary.
    summary = {
        "num_steps": step,
        "batch_size": cfg.batch_size,
        "waypoint_loss_weight": cfg.waypoint_loss_weight,
        "contrastive_loss_weight": cfg.contrastive_loss_weight,
        "waypoint_loss_type": cfg.waypoint_loss_type,
    }
    import json
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    (cfg.out_dir / "done.txt").write_text("ssl+waypoint v0 finished\n")
    print(f"[ssl+wp] wrote: {cfg.out_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()
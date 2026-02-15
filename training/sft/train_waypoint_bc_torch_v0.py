"""Waypoint behavior cloning (BC) with PyTorch (v0).

This is the next step after the episode contract + SSL pretrain plumbing:
  Waymo episode shards -> image encoder -> waypoint head.

This trainer is intentionally minimal:
- single-camera input (default: front)
- predicts the full flattened waypoint vector (H,2)
- MSE loss
- ADE/FDE evaluation metrics after training

Usage
-----
python -m training.sft.train_waypoint_bc_torch_v0 \
  --episodes-glob "out/episodes/**/*.json" \
  --cam front \
  --batch-size 32 \
  --num-steps 200

Optionally initialize the encoder from temporal SSL:
python -m training.sft.train_waypoint_bc_torch_v0 \
  --episodes-glob "out/episodes/**/*.json" \
  --pretrained-encoder out/pretrain_temporal_contrastive_v0/encoder.pt

Outputs
-------
- out/sft_waypoint_bc_torch_v0/model.pt
- out/sft_waypoint_bc_torch_v0/train_metrics.json (includes ADE/FDE)

Notes
-----
- For now we keep the encoder trainable (no freezing) to keep the script simple.
- Episode images are resized to 224x224 via PIL.
- ADE = Average Displacement Error (mean over all waypoints)
- FDE = Final Displacement Error (error at last waypoint only)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset, collate_waypoint_bc_batch
from training.utils.checkpointing import save_checkpoint
from training.utils.device import resolve_torch_device


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch


def compute_ade_fde(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).

    Args:
        preds: Predicted waypoints of shape (B, H, 2)
        targets: Ground truth waypoints of shape (B, H, 2)

    Returns:
        ade: Mean Euclidean distance across all waypoints
        fde: Euclidean distance at the final waypoint only
    """
    errors = torch.norm(preds - targets, dim=2)  # (B, H)
    ade = float(torch.mean(errors).item())
    fde = float(errors[:, -1].mean().item())
    return ade, fde


class WaypointHead:
    """Small MLP head that maps encoder embeddings -> flattened waypoint vector."""

    def __init__(self, *, torch: object, in_dim: int, horizon_steps: int):
        nn = torch.nn  # type: ignore
        out_dim = int(horizon_steps) * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )
        self.horizon_steps = int(horizon_steps)

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def __call__(self, z):
        # z: (B,D) -> (B,H,2)
        y = self.net(z)
        b = y.shape[0]
        return y.view(b, self.horizon_steps, 2)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        return self.net.load_state_dict(sd)


@dataclass
class Config:
    episodes_glob: str
    out_dir: Path = Path("out/sft_waypoint_bc_torch_v0")

    cam: str = "front"
    horizon_steps: int = 20

    batch_size: int = 32
    num_steps: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4

    seed: int = 0
    save_every: int = 100
    freeze_encoder: bool = False

    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True

    device: str = "auto"

    # Optional: init encoder weights from SSL.
    pretrained_encoder: Path | None = None


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    p.add_argument("--out-dir", type=Path, default=Path("out/sft_waypoint_bc_torch_v0"))
    p.add_argument("--cam", type=str, default="front")
    p.add_argument("--horizon-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--no-persistent-workers", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--no-drop-last", action="store_true")
    p.add_argument("--pretrained-encoder", type=Path, default=None)
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
        out_dir=a.out_dir,
        cam=a.cam,
        horizon_steps=int(a.horizon_steps),
        batch_size=int(a.batch_size),
        num_steps=int(a.num_steps),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        num_workers=int(a.num_workers),
        prefetch_factor=int(a.prefetch_factor),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        drop_last=bool(drop_last),
        device=a.device,
        seed=int(a.seed),
        save_every=int(a.save_every),
        freeze_encoder=bool(a.freeze_encoder),
        pretrained_encoder=a.pretrained_encoder,
    )


def main() -> None:
    torch = _require_torch()
    cfg = parse_args()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    ds = EpisodesWaypointBCDataset(
        cfg.episodes_glob,
        cam=cfg.cam,
        horizon_steps=cfg.horizon_steps,
        decode_images=True,
    )

    # Repro.
    torch.manual_seed(int(cfg.seed))
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(int(cfg.seed))

    device = resolve_torch_device(torch=torch, device_str=cfg.device)

    enc = TinyMultiCamEncoder(out_dim=128).to(device)
    head = WaypointHead(torch=torch, in_dim=128, horizon_steps=cfg.horizon_steps).to(device)

    if cfg.pretrained_encoder is not None:
        ckpt = torch.load(cfg.pretrained_encoder, map_location="cpu")
        sd = ckpt.get("encoder") if isinstance(ckpt, dict) else None
        if not isinstance(sd, dict):
            raise SystemExit(f"Unexpected pretrained encoder checkpoint format: {cfg.pretrained_encoder}")
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        print(f"[sft/waypoint_bc_torch] loaded encoder: missing={len(missing)} unexpected={len(unexpected)}")

    if bool(cfg.freeze_encoder):
        for p in enc.parameters():
            p.requires_grad = False
        print("[sft/waypoint_bc_torch] encoder frozen")

    params = [p for p in list(enc.parameters()) + list(head.parameters()) if bool(getattr(p, "requires_grad", True))]
    opt = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
        collate_fn=collate_waypoint_bc_batch,
    )
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    loader_kwargs["pin_memory"] = bool(cfg.pin_memory)

    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    step = 0
    it = iter(loader)
    losses = []
    eval_preds = []
    eval_targets = []

    while step < cfg.num_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)

        x = b.get("image")
        xv = b.get("image_valid")
        y = b.get("waypoints")
        yv = b.get("waypoints_valid")

        if x is None or y is None or xv is None or yv is None:
            step += 1
            continue

        # Keep only fully-valid examples.
        valid = xv & yv
        n_valid = int(valid.sum().item())
        if n_valid < 2:
            step += 1
            continue

        x = x.to(device, non_blocking=True)[valid]
        y = y.to(device, non_blocking=True)[valid]

        # TinyMultiCamEncoder expects a dict of cameras.
        z = enc({cfg.cam: x}, image_valid_by_cam={cfg.cam: torch.ones((n_valid,), dtype=torch.bool, device=device)})
        yhat = head(z)

        loss = torch.mean((yhat - y) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))

        # Collect evaluation samples (last 100 valid samples)
        if len(eval_preds) < 100:
            eval_preds.append(yhat.detach().cpu())
            eval_targets.append(y.detach().cpu())

        if step % 20 == 0:
            print(f"[sft/waypoint_bc_torch] step={step} loss={float(loss):.6f} n_valid={n_valid}")

        if int(cfg.save_every) > 0 and (step + 1) % int(cfg.save_every) == 0:
            save_checkpoint(
                torch=torch,
                out_dir=cfg.out_dir,
                step=step + 1,
                cfg=cfg,
                model_state={"encoder": enc.state_dict(), "head": head.state_dict()},
                optim_state=opt.state_dict(),
            )

        step += 1

    # Compute ADE/FDE evaluation metrics
    eval_ade, eval_fde = None, None
    if eval_preds:
        eval_preds_cat = torch.cat(eval_preds, dim=0)
        eval_targets_cat = torch.cat(eval_targets, dim=0)
        eval_ade, eval_fde = compute_ade_fde(eval_preds_cat, eval_targets_cat)
        print(f"[sft/waypoint_bc_torch] eval ADE={eval_ade:.4f} FDE={eval_fde:.4f}")

    metrics = {
        "loss_mean": float(sum(losses) / max(1, len(losses))),
        "num_steps": int(cfg.num_steps),
        "cam": cfg.cam,
        "horizon_steps": int(cfg.horizon_steps),
        "n_examples": int(len(ds)),
    }

    if eval_ade is not None:
        metrics["ade"] = round(eval_ade, 4)
        metrics["fde"] = round(eval_fde, 4)

    ckpt_path = cfg.out_dir / "model.pt"
    torch.save(
        {
            "encoder": enc.state_dict(),
            "head": head.state_dict(),
            "out_dim": 128,
            "cam": cfg.cam,
            "horizon_steps": int(cfg.horizon_steps),
        },
        ckpt_path,
    )
    (cfg.out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[sft/waypoint_bc_torch] wrote: {ckpt_path}")


if __name__ == "__main__":
    main()

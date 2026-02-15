"""Waypoint BC evaluation script.

Computes ADE (Average Displacement Error) and FDE (Final Displacement Error)
for trained waypoint policies on held-out episode frames.

Usage
-----
python -m training.sft.eval_waypoint_bc \
  --episodes-glob "out/episodes/**/*.json" \
  --checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --output-dir out/eval_waypoint_bc

Outputs
-------
- out/eval_waypoint_bc/metrics.json (ADE, FDE, summary stats)
- out/eval_waypoint_bc/predictions.jsonl (per-frame predictions vs ground truth)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset, collate_waypoint_bc_batch
from training.utils.device import resolve_torch_device


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch


@dataclass
class Config:
    episodes_glob: str
    checkpoint: Path
    output_dir: Path

    cam: str = "front"
    horizon_steps: int = 20
    batch_size: int = 64
    seed: int = 42
    device: str = "auto"
    eval_fraction: float = 0.2  # fraction of dataset to eval (for speed)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cam", type=str, default="front")
    p.add_argument("--horizon-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval-fraction", type=float, default=0.2)
    a = p.parse_args()
    return Config(
        episodes_glob=a.episodes_glob,
        checkpoint=Path(a.checkpoint),
        output_dir=Path(a.output_dir),
        cam=a.cam,
        horizon_steps=int(a.horizon_steps),
        batch_size=int(a.batch_size),
        seed=int(a.seed),
        device=a.device,
        eval_fraction=float(a.eval_fraction),
    )


def main() -> None:
    torch = _require_torch()
    cfg = parse_args()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(torch=torch, device_str=cfg.device)

    # Load checkpoint.
    ckpt = torch.load(cfg.checkpoint, map_location="cpu")
    cam = ckpt.get("cam", cfg.cam)
    horizon_steps = ckpt.get("horizon_steps", cfg.horizon_steps)
    out_dim = ckpt.get("out_dim", 128)

    print(f"[eval/waypoint_bc] checkpoint cam={cam!r} horizon_steps={horizon_steps}")

    enc = TinyMultiCamEncoder(out_dim=out_dim).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()

    # Build simple head inline (mirrors training script structure).
    class WaypointHead:
        def __init__(self, in_dim, horizon):
            import torch.nn as nn
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, horizon * 2),
            )
            self.horizon = horizon

        def to(self, device):
            self.net = self.net.to(device)
            return self

        def __call__(self, z):
            y = self.net(z)
            return y.view(-1, self.horizon, 2)

    head = WaypointHead(out_dim, horizon_steps).to(device)
    head.load_state_dict(ckpt["head"])
    head.eval()

    # Dataset (use eval_fraction for speed).
    ds = EpisodesWaypointBCDataset(
        cfg.episodes_glob,
        cam=cam,
        horizon_steps=horizon_steps,
        decode_images=True,
    )

    # Subsample for eval.
    eval_size = max(1, int(len(ds) * cfg.eval_fraction))
    indices = torch.randperm(len(ds), device="cpu")[:eval_size].tolist()
    from torch.utils.data import Subset
    eval_ds = Subset(ds, indices)

    loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_waypoint_bc_batch,
    )

    # Evaluation loop.
    all_ades = []
    all_fdes = []
    predictions = []

    with torch.no_grad():
        for b in loader:
            x = b.get("image")
            xv = b.get("image_valid")
            y = b.get("waypoints")
            yv = b.get("waypoints_valid")

            if x is None or y is None:
                continue

            valid = xv & yv
            if valid.sum() < 1:
                continue

            x = x.to(device)[valid]
            z = enc({cam: x}, image_valid_by_cam={cam: torch.ones((x.shape[0],), dtype=torch.bool, device=device)})
            yhat = head(z)

            # Compute per-example ADE and FDE.
            y_cpu = y.cpu()
            yhat_cpu = yhat.cpu()

            for i in range(y_cpu.shape[0]):
                gt = y_cpu[i].numpy()  # (H,2)
                pred = yhat_cpu[i].numpy()  # (H,2)

                # L2 distance per timestep.
                dists = ((gt - pred) ** 2).sum(axis=1) ** 0.5  # (H,)
                ade = float(dists.mean())
                fde = float(dists[-1])

                all_ades.append(ade)
                all_fdes.append(fde)

                predictions.append({
                    "ade": ade,
                    "fde": fde,
                    "meta": {
                        "episode_id": b["meta"]["episode_id"][i],
                        "frame_index": b["meta"]["frame_index"][i],
                    },
                })

    # Aggregate metrics.
    metrics = {
        "ade_mean": float(sum(all_ades) / max(1, len(all_ades))),
        "ade_std": float((sum((x - metrics["ade_mean"]) ** 2 for x in all_ades) / max(1, len(all_ades))) ** 0.5),
        "fde_mean": float(sum(all_fdes) / max(1, len(all_fdes))),
        "fde_std": float((sum((x - metrics["fde_mean"]) ** 2 for x in all_fdes) / max(1, len(all_fdes))) ** 0.5),
        "num_examples": len(all_ades),
        "cam": cam,
        "horizon_steps": horizon_steps,
        "checkpoint": str(cfg.checkpoint),
    }

    # Write outputs.
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    with (cfg.output_dir / "predictions.jsonl").open("w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"[eval/waypoint_bc] ade_mean={metrics['ade_mean']:.4f} fde_mean={metrics['fde_mean']:.4f} n={metrics['num_examples']}")
    print(f"[eval/waypoint_bc] wrote: {cfg.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

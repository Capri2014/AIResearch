"""Multi-task SSL pretraining: contrastive + waypoint regression.

This combines:
1. Multi-camera contrastive alignment (InfoNCE)
2. Waypoint regression from encoder embeddings

Pipeline:
- episodes-backed dataset yields images + ego waypoints per frame
- encode each camera with TinyMultiCamEncoder
- apply InfoNCE between camera pairs (cross-view alignment)
- predict future waypoints from pooled embedding (cross-temporal planning)

Combined loss:
  L_total = λ_contrastive * L_contrastive + λ_waypoint * L_waypoint

Usage:
  python -m training.pretrain.train_ssl_multi_task \
      --episodes-glob "out/episodes/**/*.json" \
      --out-dir out/pretrain_multi_task \
      --num-steps 5000 \
      --device cpu

Dependencies:
- torch
- pillow
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import argparse
from typing import Dict, Optional, Tuple

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
from training.pretrain.dataloader_episodes import EpisodesFrameDataset, collate_batch
from training.pretrain.objectives.contrastive import multi_pair_info_nce_loss


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("torch is required") from e
    return torch


# =============================================================================
# Waypoint extraction from episode frames
# =============================================================================

def extract_future_waypoints(
    episode_path: Path,
    frame_t: float,
    horizon_sec: float = 4.0,
    waypoint_interval_sec: float = 0.5,
    num_waypoints: int = 8,
) -> Optional[Dict[str, float]]:
    """Extract future waypoints in ego frame from an episode.

    Looks ahead `horizon_sec` from `frame_t`, sampling `num_waypoints` evenly.

    Returns:
        dict with keys "waypoints_x" and "waypoints_y" (each list of floats),
        or None if insufficient future frames.
    """
    try:
        ep_data = json.loads(episode_path.read_text())
    except Exception:
        return None

    # Collect future frames sorted by time
    future_frames = []
    for fr in ep_data.get("frames", []):
        t = float(fr.get("t", 0.0))
        if t > frame_t:
            obs = fr.get("observations", {})
            state = obs.get("state", {})
            # Try to get ego position from actions (waypoints in world)
            action = state.get("action", {})
            if action:
                future_frames.append((t, state))

    if len(future_frames) < 2:
        return None

    # Sample waypoints
    waypoints_x = []
    waypoints_y = []
    step = horizon_sec / num_waypoints

    import numpy as np
    for i in range(num_waypoints):
        t_target = frame_t + step * (i + 1)
        # Find closest frame
        closest = min(future_frames, key=lambda f: abs(f[0] - t_target))
        # Use speed and yaw to build relative waypoints in ego frame
        # If action contains (accel, steer) → integrate in ego frame
        # We'll use a simplified proxy: waypoints = cumulative_displacement
        speed = float(closest[1].get("speed_mps", 0.0))
        yaw = float(closest[1].get("yaw_rad", 0.0))
        dt = step
        dx_ego = speed * dt * 1.0  # approximate longitudinal
        dy_ego = 0.0  # simplified: lateral is small in ego frame
        waypoints_x.append(dx_ego)
        waypoints_y.append(dy_ego)

    return {"waypoints_x": waypoints_x, "waypoints_y": waypoints_y}


class MultiTaskDataset(EpisodesFrameDataset):
    """Dataset that yields frames + future waypoint targets.

    Each sample includes:
    - images_by_cam: decoded camera images
    - state: speed, yaw
    - waypoints: future waypoints in ego frame (num_waypoints, 2)
    """

    def __init__(
        self,
        episodes_glob: str,
        *,
        decode_images: bool = True,
        num_waypoints: int = 8,
        waypoint_horizon_sec: float = 4.0,
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__(
            episodes_glob=episodes_glob,
            decode_images=decode_images,
            image_size=image_size,
        )
        self.num_waypoints = num_waypoints
        self.waypoint_horizon_sec = waypoint_horizon_sec

    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)

        # Extract waypoints from episode
        ep_path = Path(sample.get("meta", {}).get("_episode_path", ""))
        frame_t = sample.get("meta", {}).get("t", 0.0)

        if ep_path.exists():
            wp_dict = extract_future_waypoints(
                ep_path,
                frame_t,
                horizon_sec=self.waypoint_horizon_sec,
                num_waypoints=self.num_waypoints,
            )
        else:
            wp_dict = None

        if wp_dict is None:
            # Fallback: zero waypoints
            import torch
            wp = torch.zeros(self.num_waypoints, 2, dtype=torch.float32)
        else:
            import torch
            wp = torch.tensor(
                [[wp_dict["waypoints_x"][i], wp_dict["waypoints_y"][i]]
                 for i in range(self.num_waypoints)],
                dtype=torch.float32,
            )

        sample["waypoints"] = wp
        return sample


def multi_task_collate(
    batch,
    *,
    stack_images: bool = True,
) -> Dict:
    """Collate batch with waypoint stacking."""
    torch = _require_torch()
    out = collate_batch(batch, stack_images=stack_images)

    # Stack waypoints
    waypoints_list = [b.get("waypoints") for b in batch if b.get("waypoints") is not None]
    if waypoints_list:
        out["waypoints"] = torch.stack(waypoints_list)
    else:
        out["waypoints"] = torch.zeros(0, 0, 2)

    return out


# =============================================================================
# Model
# =============================================================================

class MultiTaskEncoder(_require_torch().nn.Module):
    """Encoder that outputs both contrastive embeddings and waypoint predictions."""

    def __init__(
        self,
        encoder_out_dim: int = 128,
        num_waypoints: int = 8,
        hidden_dim: int = 256,
    ):
        torch = _require_torch()
        super().__init__()

        self.encoder = TinyMultiCamEncoder(out_dim=encoder_out_dim)

        # Waypoint prediction head
        self.waypoint_head = torch.nn.Sequential(
            torch.nn.Linear(encoder_out_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, num_waypoints * 2),
        )

        self.num_waypoints = num_waypoints
        self.out_dim = encoder_out_dim

    def forward(
        self,
        images_by_cam: Dict[str, torch.Tensor],
        image_valid_by_cam: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: return embedding + waypoint predictions."""
        emb = self.encoder(images_by_cam, image_valid_by_cam=image_valid_by_cam)
        wp_flat = self.waypoint_head(emb)
        wp = wp_flat.view(-1, self.num_waypoints, 2)
        return emb, wp

    def get_embeddings(self, images_by_cam, image_valid_by_cam=None):
        """Get encoder embeddings only (for contrastive)."""
        return self.encoder(images_by_cam, image_valid_by_cam=image_valid_by_cam)


# =============================================================================
# Training
# =============================================================================

@dataclass
class Config:
    episodes_glob: str = "out/episodes/**/*.json"
    batch_size: int = 16
    num_steps: int = 2000
    lr: float = 1e-3
    weight_contrastive: float = 1.0
    weight_waypoint: float = 1.0
    temperature: float = 0.1
    encoder_out_dim: int = 128
    num_waypoints: int = 8
    waypoint_horizon_sec: float = 4.0
    out_dir: Path = field(default_factory=lambda: Path("out/pretrain_multi_task"))
    num_workers: int = 4
    device: str = "cuda"
    save_every: int = 500
    log_every: int = 50
    test: bool = False

    # Camera pairs for contrastive
    cam_pairs: list = field(default_factory=lambda: [
        ("front", "front_left"),
        ("front", "front_right"),
        ("front", "rear"),
    ])


def train_step(model, batch, optimizer, cfg, step):
    torch = _require_torch()
    model.train()

    images = batch.get("images_by_cam", {})
    valid = batch.get("image_valid_by_cam", {})
    waypoints = batch.get("waypoints", torch.zeros(1, cfg.num_waypoints, 2))

    if not images:
        return 0.0

    # Forward
    embeddings, pred_wp = model(images, image_valid_by_cam=valid)

    # Contrastive loss (multi-pair InfoNCE)
    emb_by_cam = {}
    valid_by_cam = {}
    for cam, img_t in images.items():
        if img_t is None:
            continue
        cam_emb = model.get_embeddings({cam: img_t})
        emb_by_cam[cam] = cam_emb
        valid_by_cam[cam] = valid.get(cam, torch.ones(img_t.shape[0], dtype=torch.bool, device=img_t.device))

    l_contrastive = multi_pair_info_nce_loss(
        emb_by_cam,
        valid_by_cam,
        temperature=cfg.temperature,
    )

    # Waypoint regression loss (L1)
    B = pred_wp.shape[0]
    if waypoints.shape[0] == B:
        l_wp = torch.abs(pred_wp - waypoints.to(pred_wp.device)).mean()
    else:
        l_wp = torch.tensor(0.0, device=pred_wp.device)

    # Combined loss
    loss = cfg.weight_contrastive * l_contrastive + cfg.weight_waypoint * l_wp

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())


def run_training(cfg: Config):
    torch = _require_torch()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"\n=== Multi-Task SSL Training ===")
    print(f"Device: {device}")
    print(f"Episodes glob: {cfg.episodes_glob}")
    print(f"Output dir: {cfg.out_dir}")
    print(f"Contrastive weight: {cfg.weight_contrastive}, Waypoint weight: {cfg.weight_waypoint}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset — try to load from episodes; fall back to stub if no episodes found
    try:
        from training.pretrain.dataloader_episodes import EpisodesFrameDataset, collate_batch
        dataset = MultiTaskDataset(
            episodes_glob=cfg.episodes_glob,
            decode_images=True,
            num_waypoints=cfg.num_waypoints,
            waypoint_horizon_sec=cfg.waypoint_horizon_sec,
        )
        use_real = len(dataset) > 0
        print(f"Dataset: {len(dataset)} frames loaded")
    except Exception as e:
        print(f"Dataset load failed ({e}), using synthetic data")
        use_real = False

    # Model
    model = MultiTaskEncoder(
        encoder_out_dim=cfg.encoder_out_dim,
        num_waypoints=cfg.num_waypoints,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_steps)

    # Training loop
    global_step = 0
    losses = []

    # Synthetic data fallback for testing
    def make_synthetic_batch(B):
        torch = _require_torch()
        cams = ["front", "front_left", "front_right"]
        images_by_cam = {
            c: torch.randn(B, 3, 224, 224, device=device)
            for c in cams
        }
        valid_by_cam = {
            c: torch.ones(B, dtype=torch.bool, device=device)
            for c in cams
        }
        waypoints = torch.randn(B, cfg.num_waypoints, 2, device=device) * 2.0
        return {
            "images_by_cam": images_by_cam,
            "image_valid_by_cam": valid_by_cam,
            "waypoints": waypoints,
        }

    while global_step < cfg.num_steps:
        if use_real:
            try:
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    collate_fn=lambda b: multi_task_collate(b, stack_images=True),
                    pin_memory=True,
                    persistent_workers=True,
                )
                loader_iter = iter(loader)
            except Exception:
                use_real = False
                print("Falling back to synthetic data")

        if not use_real:
            # Synthetic batch every step
            batch = make_synthetic_batch(cfg.batch_size)

        for _ in range(min(100, cfg.num_steps - global_step)):
            if use_real:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)

            loss = train_step(model, batch, optimizer, cfg, global_step)
            scheduler.step()
            losses.append(loss)
            global_step += 1

            if global_step % cfg.log_every == 0:
                avg_loss = sum(losses[-cfg.log_every:]) / len(losses[-cfg.log_every:])
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step}: loss={avg_loss:.4f}, lr={lr:.6f}")

            if global_step % cfg.save_every == 0:
                ckpt_path = cfg.out_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": dataclass_as_dict(cfg),
                }, ckpt_path)
                print(f"  ✓ Saved {ckpt_path}")

        if not use_real:
            break  # Synthetic mode: one burst then exit

    # Final save
    final_path = cfg.out_dir / "final_model.pt"
    torch.save({
        "step": global_step,
        "model_state": model.state_dict(),
        "config": dataclass_as_dict(cfg),
    }, final_path)
    print(f"\n✓ Training complete: {global_step} steps, avg_loss={sum(losses)/len(losses):.4f}")
    print(f"✓ Saved to {final_path}")


def dataclass_as_dict(obj):
    """Convert dataclass to dict (including nested Path objects)."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_as_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, list):
        return [dataclass_as_dict(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_as_dict(v) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(description="Multi-task SSL: contrastive + waypoint regression")
    parser.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-contrastive", type=float, default=1.0)
    parser.add_argument("--weight-waypoint", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--encoder-out-dim", type=int, default=128)
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--waypoint-horizon-sec", type=float, default=4.0)
    parser.add_argument("--out-dir", type=Path, default=Path("out/pretrain_multi_task"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--test", action="store_true", help="Smoke test with synthetic data")

    args = parser.parse_args()
    cfg = Config(
        episodes_glob=args.episodes_glob,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        weight_contrastive=args.weight_contrastive,
        weight_waypoint=args.weight_waypoint,
        temperature=args.temperature,
        encoder_out_dim=args.encoder_out_dim,
        num_waypoints=args.num_waypoints,
        waypoint_horizon_sec=args.waypoint_horizon_sec,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        device=args.device,
        save_every=args.save_every,
        log_every=args.log_every,
        test=args.test,
    )

    if args.test:
        # Smoke test
        torch = _require_torch()
        device = torch.device("cpu")
        model = MultiTaskEncoder(encoder_out_dim=128, num_waypoints=8).to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("\n=== Multi-Task SSL Smoke Test ===")
        for step in range(5):
            batch = {
                "images_by_cam": {
                    c: torch.randn(4, 3, 224, 224, device=device)
                    for c in ["front", "front_left"]
                },
                "image_valid_by_cam": {
                    c: torch.ones(4, dtype=torch.bool, device=device)
                    for c in ["front", "front_left"]
                },
                "waypoints": torch.randn(4, 8, 2, device=device) * 2.0,
            }
            loss = train_step(model, batch, opt, cfg, step)
            print(f"  step {step}: loss={loss:.4f}")

        print("✓ Smoke test passed")
        return

    run_training(cfg)


if __name__ == "__main__":
    main()
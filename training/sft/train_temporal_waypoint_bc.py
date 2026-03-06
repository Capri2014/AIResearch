"""
Temporal Waypoint Behavior Cloning (BC) with PyTorch.

This extends the basic waypoint BC to use temporal context:
  - Input: sequence of frames (T, H, W, C) instead of single frame
  - Encoder: LSTM/GRU over frame embeddings
  - Output: waypoints (H, 2) predicted from temporal context

This improves waypoint predictions by leveraging:
  1. Temporal consistency in driving
  2. Motion cues from consecutive frames
  3. Better context for predicting future waypoints

Usage
-----
python -m training.sft.train_temporal_waypoint_bc \
  --episodes-glob "out/episodes/**/*.json" \
  --sequence-length 4 \
  --batch-size 16 \
  --num-steps 200

Optionally initialize from SSL pretrained encoder:
python -m training.sft.train_temporal_waypoint_bc \
  --episodes-glob "out/episodes/**/*.json" \
  --pretrained-encoder out/pretrain_temporal/encoder.pt

Outputs
-------
- out/temporal_waypoint_bc/model.pt
- out/temporal_waypoint_bc/train_metrics.json (ADE/FDE)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset
from training.utils.checkpointing import save_checkpoint
from training.utils.device import resolve_torch_device


def compute_ade_fde(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Compute ADE and FDE for waypoint predictions.
    
    Args:
        preds: Predicted waypoints of shape (B, H, 2)
        targets: Ground truth waypoints of shape (B, H, 2)
    
    Returns:
        ade: Mean Euclidean distance across all waypoints
        fde: Euclidean distance at the final waypoint
    """
    errors = torch.norm(preds - targets, dim=2)  # (B, H)
    ade = float(torch.mean(errors).item())
    fde = float(errors[:, -1].mean().item())
    return ade, fde


class TemporalEncoder(nn.Module):
    """Encodes temporal sequence of frames using LSTM over frame embeddings.
    
    Architecture:
        1. Per-frame CNN encoder (TinyMultiCamEncoder)
        2. LSTM/GRU to aggregate temporal context
        3. Output: single embedding representing the sequence
    """
    
    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        encoder_dim = encoder.out_dim if hasattr(encoder, 'out_dim') else 256
        
        # Project encoder output to hidden dim
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        
        # LSTM for temporal aggregation
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, T, C, H, W) - batch of temporal sequences
        
        Returns:
            embedding: (B, hidden_dim) - temporal context embedding
        """
        B, T, C, H, W = images.shape
        
        # Encode each frame using the multi-cam encoder
        # The encoder expects dict of camera -> (B, C, H, W)
        embeddings = []
        for t in range(T):
            frame = images[:, t]  # (B, C, H, W)
            # Create dict for encoder (single camera)
            frame_dict = {"front": frame}
            emb = self.encoder(frame_dict)  # (B, D)
            embeddings.append(emb)
        
        # Stack embeddings: (B, T, D)
        embeddings = torch.stack(embeddings, dim=1)
        
        # Project to hidden dim
        embeddings = self.encoder_proj(embeddings)
        
        # Temporal aggregation with LSTM
        output, (hidden, cell) = self.rnn(embeddings)  # output: (B, T, hidden)
        
        # Use final hidden state as the sequence representation
        return hidden[-1]  # (B, hidden_dim)


class WaypointHead(nn.Module):
    """MLP head that maps encoder embeddings -> waypoints."""
    
    def __init__(self, in_dim: int, horizon_steps: int, hidden_dim: int = 256):
        super().__init__()
        out_dim = horizon_steps * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.horizon_steps = horizon_steps
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) -> waypoints (B, H, 2)"""
        B = z.shape[0]
        y = self.net(z)
        return y.view(B, self.horizon_steps, 2)


class TemporalWaypointBC(nn.Module):
    """Complete temporal waypoint BC model."""
    
    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_dim: int = 256,
        horizon_steps: int = 20,
        num_rnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_layers=num_rnn_layers,
            dropout=dropout,
        )
        self.waypoint_head = WaypointHead(
            in_dim=hidden_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
        )
        self.horizon_steps = horizon_steps
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, T, C, H, W) - temporal sequence of frames
        
        Returns:
            waypoints: (B, H, 2) - predicted future waypoints
        """
        embedding = self.temporal_encoder(images)
        waypoints = self.waypoint_head(embedding)
        return waypoints


@dataclass
class Config:
    episodes_glob: str
    sequence_length: int = 4
    batch_size: int = 16
    num_steps: int = 200
    lr: float = 1e-3
    hidden_dim: int = 256
    num_rnn_layers: int = 2
    dropout: float = 0.1
    horizon_steps: int = 20
    cam: str = "front"
    out_dir: Path = Path("out/temporal_waypoint_bc")
    pretrained_encoder: Path | None = None
    device: str = "cuda"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Temporal Waypoint BC Training")
    p.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-rnn-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--horizon-steps", type=int, default=20)
    p.add_argument("--cam", type=str, default="front")
    p.add_argument("--out-dir", type=Path, default=Path("out/temporal_waypoint_bc"))
    p.add_argument("--pretrained-encoder", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    a = p.parse_args()
    return Config(**vars(a))


def collate_temporal_batch(batch):
    """Collate function for temporal waypoint BC.
    
    Args:
        batch: List of (images_seq, waypoints) where:
            - images_seq: (T, C, H, W) tensor
            - waypoints: (H, 2) tensor
    
    Returns:
        images: (B, T, C, H, W)
        targets: (B, H, 2)
    """
    images_list = []
    targets_list = []
    
    for images_seq, waypoints in batch:
        images_list.append(images_seq)
        targets_list.append(waypoints)
    
    images = torch.stack(images_list)
    targets = torch.stack(targets_list)
    
    return images, targets


class TemporalEpisodesDataset:
    """Dataset that yields temporal sequences from episodes.
    
    Samples consecutive frames from the same episode to create
    temporal sequences for the LSTM-based waypoint predictor.
    """
    
    def __init__(
        self,
        episodes_glob: str,
        sequence_length: int = 4,
        cam: str = "front",
        horizon_steps: int = 20,
        decode_images: bool = True,
        image_size: tuple[int, int] = (224, 224),
    ):
        from training.episodes.episode_paths import glob_episode_paths
        import json
        from training.pretrain.image_loading import ImageConfig, load_image_tensor
        
        self._torch = _require_torch()
        self.cam = cam
        self.sequence_length = sequence_length
        self.horizon_steps = horizon_steps
        self.decode_images = decode_images
        self.image_size = image_size
        self._img_cfg = ImageConfig(size=image_size)
        
        # Load episode paths
        self.episode_paths = glob_episode_paths(episodes_glob)
        
        # Build index: (episode_path, frame_index) for valid starting points
        self.index: List[Tuple[Path, int]] = []
        
        for ep_path in self.episode_paths:
            ep = json.loads(ep_path.read_text())
            frames = ep.get("frames", [])
            
            # Only index frames that have enough future frames for sequence
            max_start = len(frames) - sequence_length
            for frame_idx in range(max_start + 1):
                self.index.append((ep_path, frame_idx))
        
        self._episode_cache: Dict[Path, Any] = {}
        self._img_cache: dict = {}
    
    def __len__(self):
        return len(self.index)
    
    def _load_episode(self, ep_path: Path) -> dict:
        """Load episode JSON with caching."""
        if ep_path not in self._episode_cache:
            self._episode_cache[ep_path] = json.loads(ep_path.read_text())
        return self._episode_cache[ep_path]
    
    def __getitem__(self, idx):
        ep_path, frame_start = self.index[idx]
        ep = self._load_episode(ep_path)
        frames = ep.get("frames", [])
        
        # Load sequence of frames
        images_list = []
        for i in range(self.sequence_length):
            frame = frames[frame_start + i]
            image_path = frame.get("image_paths", {}).get(self.cam)
            
            if image_path and self.decode_images:
                # Resolve relative path
                img_full_path = ep_path.parent / image_path
                try:
                    img_tensor = load_image_tensor(str(img_full_path), self._img_cfg)
                except Exception:
                    # Fallback to zeros if image loading fails
                    img_tensor = torch.zeros(3, self.image_size[0], self.image_size[1])
            else:
                img_tensor = torch.zeros(3, self.image_size[0], self.image_size[1])
            
            images_list.append(img_tensor)
        
        # Stack into (T, C, H, W)
        images_seq = torch.stack(images_list)
        
        # Get waypoints from the last frame in sequence
        target_frame = frames[frame_start + self.sequence_length - 1]
        waypoints = target_frame.get("waypoints")
        
        if waypoints is None:
            waypoints = torch.zeros(self.horizon_steps, 2)
        else:
            waypoints = torch.tensor(waypoints, dtype=torch.float32)
        
        return images_seq, waypoints


def main():
    cfg = parse_args()
    torch = torch
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[temporal_waypoint_bc] Device: {device}")
    
    # Create output directory
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build model
    print(f"[temporal_waypoint_bc] Building model...")
    base_encoder = TinyMultiCamEncoder(cam_names=[cfg.cam])
    
    # Optionally load pretrained encoder
    if cfg.pretrained_encoder and cfg.pretrained_encoder.exists():
        print(f"[temporal_waypoint_bc] Loading pretrained encoder: {cfg.pretrained_encoder}")
        state = torch.load(cfg.pretrained_encoder, map_location=device)
        if 'encoder' in state:
            base_encoder.load_state_dict(state['encoder'])
        elif 'model' in state:
            base_encoder.load_state_dict(state['model'])
    
    model = TemporalWaypointBC(
        encoder=base_encoder,
        hidden_dim=cfg.hidden_dim,
        horizon_steps=cfg.horizon_steps,
        num_rnn_layers=cfg.num_rnn_layers,
        dropout=cfg.dropout,
    )
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    
    # Create temporal dataset
    print(f"[temporal_waypoint_bc] Loading dataset: {cfg.episodes_glob}")
    try:
        dataset = TemporalEpisodesDataset(
            episodes_glob=cfg.episodes_glob,
            sequence_length=cfg.sequence_length,
            cam=cfg.cam,
            horizon_steps=cfg.horizon_steps,
        )
        print(f"[temporal_waypoint_bc] Temporal dataset: {len(dataset)} sequences")
    except Exception as e:
        print(f"[temporal_waypoint_bc] Failed to create temporal dataset: {e}")
        print(f"[temporal_waypoint_bc] Using fallback single-frame mode")
        # Fallback to single frame
        dataset = EpisodesWaypointBCDataset(
            episodes_glob=cfg.episodes_glob,
            cam=cfg.cam,
            horizon_steps=cfg.horizon_steps,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_temporal_batch,
    )
    
    # Training loop
    print(f"[temporal_waypoint_bc] Starting training for {cfg.num_steps} steps...")
    metrics_history = []
    
    for step, (images, targets) in enumerate(dataloader):
        if step >= cfg.num_steps:
            break
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        ade, fde = compute_ade_fde(preds, targets)
        
        if step % 20 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, ADE={ade:.3f}m, FDE={fde:.3f}m")
        
        metrics_history.append({
            "step": step,
            "loss": float(loss.item()),
            "ade": ade,
            "fde": fde,
        })
    
    # Save model and metrics
    print(f"[temporal_waypoint_bc] Saving to {cfg.out_dir}")
    save_checkpoint(
        model,
        optimizer,
        step=cfg.num_steps,
        path=cfg.out_dir / "model.pt",
    )
    
    # Save metrics
    metrics_path = cfg.out_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config": {
                "sequence_length": cfg.sequence_length,
                "hidden_dim": cfg.hidden_dim,
                "num_rnn_layers": cfg.num_rnn_layers,
                "horizon_steps": cfg.horizon_steps,
            },
            "final_metrics": metrics_history[-1] if metrics_history else {},
            "history": metrics_history,
        }, f, indent=2)
    
    print(f"[temporal_waypoint_bc] Done!")
    print(f"  Model: {cfg.out_dir / 'model.pt'}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Final ADE: {metrics_history[-1]['ade']:.3f}m")
    print(f"  Final FDE: {metrics_history[-1]['fde']:.3f}m")


if __name__ == "__main__":
    main()

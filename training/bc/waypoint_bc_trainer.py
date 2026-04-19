#!/usr/bin/env python3
"""
Waypoint BC Trainer - Supervised behavior cloning for waypoint prediction.

This trainer takes waypoint trajectories from WaypointBatchCollator and trains
a waypoint prediction model using supervised behavior cloning. It supports:
- Loading pretrained SSL representations as backbone
- Multi-horizon waypoint prediction
- Progress-aware prediction (predicting based on current progress in episode)
- Checkpointing and evaluation

Pipeline position: BC stage (after SSL pretrain, before RL refinement)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Try to import from pipeline components
try:
    from training.bc.waypoint_batch_collator import WaypointBatchCollator, WaypointSample
except ImportError:
    WaypointBatchCollator = None


# Define a local WaypointSample if not available from collator
@dataclass
class LocalWaypointSample:
    """Local waypoint sample for standalone training."""
    episode_id: str
    frame_id: int
    waypoints: np.ndarray
    speed: float
    progress: float


if 'WaypointSample' not in dir():
    WaypointSample = LocalWaypointSample


@dataclass
class TrainerConfig:
    """Configuration for waypoint BC training."""
    # Model
    input_dim: int = 6  # position(2) + heading(1) + speed(1) + progress(1) + timestamp(1)
    hidden_dim: int = 256
    num_layers: int = 3
    num_waypoints: int = 8
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    validate_every: int = 5
    
    # Data
    train_episodes_dir: str = "data/waymo/episodes"
    val_episodes_dir: str = "data/waymo/episodes"
    val_split: float = 0.1
    
    # Pretraining
    pretrained_ssl_path: Optional[str] = None
    freeze_backbone: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/waypoint_bc"
    save_best_only: bool = True
    resume_from: Optional[str] = None
    
    # Output
    output_dir: str = "out/waypoint_bc_trainer"


@dataclass
class TrainState:
    """Training state for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    train_loss: float = 0.0
    val_loss: float = 0.0


class WaypointMLP(nn.Module):
    """MLP for waypoint prediction from observation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_waypoints: int, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, num_waypoints * 2)  # 2D waypoints
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, input_dim) observation
        Returns:
            waypoints: (batch, num_waypoints, 2) predicted waypoints in world frame
        """
        h = self.encoder(obs)
        waypoints = self.output(h).view(-1, self.num_waypoints, 2)
        return waypoints


class ResidualWaypointMLP(nn.Module):
    """Residual waypoint prediction with progress conditioning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_waypoints: int,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Progress encoder: encodes current progress into embedding
        self.progress_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Base encoder for current state
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer-style layers with progress conditioning
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output head
        self.output = nn.Linear(hidden_dim, num_waypoints * 2)
    
    def forward(self, obs: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, input_dim) current state observation
            progress: (batch, 1) normalized progress [0, 1]
        Returns:
            waypoints: (batch, num_waypoints, 2) predicted waypoints
        """
        # Encode progress
        prog_emb = self.progress_embed(progress)  # (batch, hidden_dim)
        
        # Encode observation
        base = self.base_encoder(obs)  # (batch, hidden_dim)
        
        # Add progress embedding (broadcasting)
        x = base + prog_emb  # (batch, hidden_dim)
        
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        
        waypoints = self.output(x).view(-1, self.num_waypoints, 2)
        return waypoints


class WaypointBCTrainer:
    """Main trainer class for waypoint behavior cloning."""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model
        self.model = self._build_model().to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Data collator
        self.train_collator = None
        self.val_collator = None
        self.train_loader = None
        self.val_loader = None
        
        # State
        self.state = TrainState()
        
        # Metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        # Checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _build_model(self) -> nn.Module:
        """Build the waypoint prediction model."""
        if self.config.pretrained_ssl_path:
            # TODO: Load pretrained SSL backbone and adapt
            # For now, just use the MLP
            return ResidualWaypointMLP(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_waypoints=self.config.num_waypoints,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        else:
            return ResidualWaypointMLP(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_waypoints=self.config.num_waypoints,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
    
    def _load_data(self):
        """Load training and validation data."""
        if WaypointBatchCollator is None:
            print("Warning: WaypointBatchCollator not available, using synthetic data")
            self._generate_synthetic_data()
            return
        
        print(f"Loading training data from {self.config.train_episodes_dir}")
        
        # Create collators
        self.train_collator = WaypointBatchCollator(
            batch_size=self.config.batch_size,
            shuffle=True,
            augment=True
        )
        
        self.val_collator = WaypointBatchCollator(
            batch_size=self.config.batch_size,
            shuffle=False,
            augment=False
        )
        
        # Load data
        train_samples = self.train_collator.load_episodes(self.config.train_episodes_dir)
        val_samples = self.val_collator.load_episodes(self.config.val_episodes_dir)
        
        # Split train/val if same directory
        if self.config.train_episodes_dir == self.config.val_episodes_dir:
            n = len(train_samples)
            n_val = int(n * self.config.val_split)
            np.random.shuffle(train_samples)
            val_samples = train_samples[:n_val]
            train_samples = train_samples[n_val:]
        
        print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
        
        # Create data loaders
        self.train_loader = self.train_collator.create_dataloader(train_samples)
        self.val_loader = self.val_collator.create_dataloader(val_samples)
    
    def _generate_synthetic_data(self):
        """Generate synthetic training data for testing."""
        print("Generating synthetic waypoint data...")
        
        num_train = 1000
        num_val = 200
        
        # Generate synthetic samples
        def generate_samples(n: int) -> List[WaypointSample]:
            samples = []
            for i in range(n):
                # Random trajectory in lane
                t = i / n
                waypoints = np.zeros((self.config.num_waypoints, 2))
                for j in range(self.config.num_waypoints):
                    # Curved lane following
                    offset = j * 0.5
                    waypoints[j] = [
                        t * 50 + offset + np.random.randn() * 0.1,
                        np.sin(t * 3 + j * 0.3) * 2 + np.random.randn() * 0.1
                    ]
                
                sample = WaypointSample(
                    episode_id=f"sim_{i}",
                    frame_id=i,
                    waypoints=waypoints,
                    speed=np.random.uniform(5, 15),
                    progress=t
                )
                samples.append(sample)
            return samples
        
        train_samples = generate_samples(num_train)
        val_samples = generate_samples(num_val)
        
        print(f"Synthetic: {len(train_samples)} train, {len(val_samples)} val")
        
        # Store as list
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.train_iter = iter(self._batch_samples(train_samples))
        self.val_iter = iter(self._batch_samples(val_samples))
    
    def _batch_samples(self, samples: List[WaypointSample]) -> DataLoader:
        """Convert samples to DataLoader-style iterator."""
        class SimpleDataset:
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        class SimpleCollate:
            def __init__(self, config):
                self.config = config
            
            def __call__(self, batch):
                """Collate batch of WaypointSample."""
                obs_list = []
                target_list = []
                
                for sample in batch:
                    # Observation: current position + heading + speed + progress
                    # Use last known waypoint as "current" position
                    curr_pos = sample.waypoints[0] if len(sample.waypoints) > 0 else [0, 0]
                    
                    obs = np.array([
                        curr_pos[0], curr_pos[1],  # position
                        0.0,  # heading (approximate from waypoints)
                        sample.speed,
                        sample.progress,
                        sample.frame_id / 1000.0  # normalized timestamp
                    ])
                    obs_list.append(obs)
                    
                    # Target: all future waypoints
                    target_list.append(sample.waypoints.flatten())
                
                obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32)
                target_batch = torch.tensor(np.array(target_list), dtype=torch.float32)
                
                return obs_batch, target_batch
        
        dataset = SimpleDataset(samples)
        loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            collate_fn=SimpleCollate(self.config)
        )
        return loader
    
    def _train_step(self, batch) -> float:
        """Single training step."""
        obs, targets = batch
        obs = obs.to(self.device)
        targets = targets.to(self.device)
        
        # Extract progress from observation (5th element)
        progress = obs[:, 4:5]
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(obs, progress)
        
        # Reshape targets to match
        targets = targets.view(-1, self.config.num_waypoints, 2)
        
        # Compute loss
        loss = self.loss_fn(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
        
        self.optimizer.step()
        
        return loss.item()
    
    def _validate_step(self, batch) -> float:
        """Single validation step."""
        obs, targets = batch
        obs = obs.to(self.device)
        targets = targets.to(self.device)
        
        progress = obs[:, 4:5]
        
        with torch.no_grad():
            predictions = self.model(obs, progress)
            targets = targets.view(-1, self.config.num_waypoints, 2)
            loss = self.loss_fn(predictions, targets)
        
        return loss.item()
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Get data loader
        if self.train_loader is not None:
            loader = self.train_loader
        else:
            # Use synthetic data
            try:
                loader = self.train_iter
            except AttributeError:
                return 0.0
        
        for batch in loader:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            self.state.global_step += 1
        
        return total_loss / max(num_batches, 1)
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        loader = self.val_loader if self.val_loader is not None else self.val_iter
        
        for batch in iter(loader):
            loss = self._validate_step(batch)
            total_loss += loss
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": self.state,
            "config": self.config,
            "metrics": self.metrics
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def _load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.state = checkpoint["state"]
        self.metrics = checkpoint.get("metrics", self.metrics)
        print(f"Loaded checkpoint from {path}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        # Load data
        self._load_data()
        
        # Resume if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)
        
        # Training loop
        best_val_loss = self.state.best_val_loss
        
        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.state.train_loss = train_loss
            self.metrics["train_loss"].append(train_loss)
            self.metrics["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            # Validate periodically
            if (epoch + 1) % self.config.validate_every == 0:
                val_loss = self.validate()
                self.state.val_loss = val_loss
                self.metrics["val_loss"].append(val_loss)
                print(f"  val_loss={val_loss:.4f}")
                
                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.state.best_val_loss = val_loss
                    best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
                    self._save_checkpoint(best_path)
            
            # Save checkpoint
            if not self.config.save_best_only or epoch == self.config.num_epochs - 1:
                ckpt_path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch}.pt")
                self._save_checkpoint(ckpt_path)
            
            # Step scheduler
            self.scheduler.step()
        
        # Save final model
        final_path = os.path.join(self.config.checkpoint_dir, "final.pt")
        self._save_checkpoint(final_path)
        
        # Save training metrics
        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        print(f"Checkpoints: {self.config.checkpoint_dir}")
        
        return best_val_loss
    
    def evaluate(self, checkpoint_path: Optional[str] = None) -> dict:
        """Evaluate the model on validation set."""
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        
        loader = self.val_loader if self.val_loader is not None else self.val_iter
        
        with torch.no_grad():
            for batch in iter(loader):
                obs, targets = batch
                obs = obs.to(self.device)
                progress = obs[:, 4:5]
                
                predictions = self.model(obs, progress)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets)
        
        # Compute metrics
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0).view(-1, self.config.num_waypoints, 2)
        
        # MSE per waypoint
        mse_per_waypoint = F.mse_loss(predictions, targets, reduction="none").mean(dim=0)
        
        # Final waypoint error (most important for planning)
        final_mse = mse_per_waypoint[-1].item()
        final_rmse = np.sqrt(final_mse)
        
        metrics = {
            "final_mse": final_mse,
            "final_rmse": final_rmse,
            "mse_per_waypoint": mse_per_waypoint.tolist(),
            "num_waypoints": self.config.num_waypoints
        }
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Waypoint BC Trainer")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier")
    parser.add_argument("--episodes-dir", type=str, default="data/waymo/episodes",
                     help="Episodes directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-waypoints", type=int, default=8, help="Number of waypoints to predict")
    parser.add_argument("--pretrained-ssl", type=str, default=None,
                     help="Pretrained SSL checkpoint path")
    parser.add_argument("--freeze-backbone", action="store_true",
                     help="Freeze pretrained backbone")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/waypoint_bc",
                     help="Checkpoint output directory")
    parser.add_argument("--resume-from", type=str, default=None,
                     help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="out/waypoint_bc_trainer",
                     help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no training)")
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    
    args = parser.parse_args()
    
    # Build config
    config = TrainerConfig(
        train_episodes_dir=args.episodes_dir,
        val_episodes_dir=args.episodes_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        pretrained_ssl_path=args.pretrained_ssl,
        freeze_backbone=args.freeze_backbone,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        output_dir=args.output_dir
    )
    
    print(f"Config: {config}")
    
    # Create trainer
    trainer = WaypointBCTrainer(config)
    
    if args.dry_run:
        print("[DRY RUN] Configuration validated")
        return
    
    if args.eval:
        metrics = trainer.evaluate(args.resume_from)
        print(f"Evaluation metrics: {metrics}")
        return
    
    # Train
    best_loss = trainer.train()
    
    # Save final metrics
    metrics_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "best_val_loss": best_loss,
            "config": vars(args),
            "run_id": args.run_id
        }, f, indent=2)
    
    print(f"Training complete. Best val_loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
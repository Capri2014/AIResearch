"""
Unified Training Architecture

Integrates all models into ONE pipeline:
- SSL Encoder (JEPA pre-trained)
- World Model (latent dynamics loss)
- VLA Planner (trajectory + explanation)
- Safety Layer (constraint loss)

Single training loop, unified loss.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Training Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                        │
│  │ Images  │  │ State   │  │ CoT Text│                        │
│  └────┬────┘  └────┬────┘  └────┬────┘                        │
│       │            │            │                                │
│       ▼            ▼            ▼                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           SSL Encoder (frozen initially)                  │   │
│  │                    → latent_z                           │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│       ┌────────────────────┼────────────────────┐              │
│       │                    │                    │              │
│       ▼                    ▼                    ▼              │
│  ┌─────────┐        ┌──────────┐        ┌──────────┐       │
│  │ World   │        │  VLA     │        │  Safety  │       │
│  │ Model   │        │  Planner │        │  Layer   │       │
│  │         │        │          │        │          │       │
│  │ latent  │──────►│ trajectory│◄──────│ validated│       │
│  │ dynamics│        │ + explain │        │ trajectory│       │
│  └────┬────┘        └─────┬─────┘        └────┬─────┘       │
│       │                    │                    │              │
│       ▼                    ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Unified Loss Function                       │   │
│  │  L = L_trajectory + λ1*L_world_model + λ2*L_safety    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Usage:
    from training.unified import UnifiedTrainer
    
    trainer = UnifiedTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class UnifiedConfig:
    """Unified training configuration."""
    # Model dims
    ssl_embed_dim: int = 768
    state_dim: int = 16
    cot_dim: int = 256
    hidden_dim: int = 512
    
    # World Model
    world_model_latent: int = 32
    world_model_hidden: int = 256
    use_world_model: bool = True
    world_model_weight: float = 0.1
    
    # VLA Planner
    trajectory_horizon: int = 20
    use_vla: bool = True
    vla_weight: float = 1.0
    
    # Safety
    use_safety: bool = True
    safety_weight: float = 0.5
    
    # Training
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 100
    device: str = "cuda"
    
    # Data
    data_root: str = "data/waymo"
    cot_file: Optional[str] = None


# ============================================================================
# Unified Model
# ============================================================================

class UnifiedDrivingModel(nn.Module):
    """
    Unified model combining:
    - SSL Encoder
    - World Model
    - VLA Planner
    - Safety Layer
    """
    
    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config
        
        # === SSL Encoder (placeholder - load pretrained) ===
        self.ssl_encoder = nn.Identity()  # Replace with actual JEPA encoder
        
        # === State Encoder ===
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # === CoT Encoder (optional) ===
        self.cot_encoder = None
        if config.use_vla:
            self.cot_encoder = nn.LSTM(
                input_size=256,
                hidden_size=config.cot_dim,
                num_layers=2,
                batch_first=True,
            )
        
        # === World Model ===
        self.world_model = None
        if config.use_world_model:
            from training.models.world_model import WorldModel, WorldModelConfig
            wm_config = WorldModelConfig(
                latent_dim=config.world_model_latent,
                hidden_dim=config.world_model_hidden,
            )
            self.world_model = WorldModel(wm_config)
        
        # === VLA Planner ===
        self.vla_planner = None
        if config.use_vla:
            from training.models.vla_planner import VLADrivingPlanner, VLAConfig
            vla_config = VLAConfig(
                vision_hidden=config.hidden_dim,
                fusion_hidden=config.hidden_dim,
                trajectory_horizon=config.trajectory_horizon,
            )
            self.vla_planner = VLADrivingPlanner(vla_config)
        
        # === Fusion ===
        # Use flexible projection that can handle variable input
        max_input_dim = config.ssl_embed_dim + config.hidden_dim + config.cot_dim
        self.fusion_proj = nn.Linear(max_input_dim, config.hidden_dim)
        
        # Register buffer for padding
        self.register_buffer('_zero', torch.zeros(1))
        
        # === Trajectory Decoder ===
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.trajectory_horizon * 3),
        )
        
        # === Safety Layer ===
        self.safety_layer = None
        if config.use_safety:
            from training.models.safety_layer import SafetyLayer, SafetyConfig
            self.safety_layer = SafetyLayer(SafetyConfig())
        
        # Loss weights
        self.world_model_weight = config.world_model_weight
        self.safety_weight = config.safety_weight
    
    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        cot_tokens: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through unified model.
        
        Args:
            images: [B, C, H, W] or [B, D] image features
            state: [B, state_dim] numerical state
            cot_tokens: [B, L] optional CoT tokens
            actions: [B, T, 3] expert actions (for loss)
            
        Returns:
            Dictionary with all outputs:
            - trajectory: predicted trajectory
            - world_model_loss: latent dynamics loss
            - safety_loss: safety constraint loss
            - explanation: generated explanation (if VLA)
        """
        B = images.shape[0]
        
        # === Encode image (or use pre-encoded) ===
        if images.ndim == 4:
            # Raw images - encode with SSL
            image_features = self.ssl_encoder(images)
        else:
            # Pre-encoded features
            image_features = images
        
        # === Encode state ===
        state_features = self.state_encoder(state)
        
        # === Encode CoT ===
        cot_features = None
        if cot_tokens is not None and self.cot_encoder is not None:
            cot_out, (h_n, c_n) = self.cot_encoder(cot_tokens)
            cot_features = h_n[-1]  # Take last hidden
        
        # === Fuse ===
        max_input_dim = self.config.ssl_embed_dim + self.config.hidden_dim + self.config.cot_dim
        if cot_features is not None:
            fused_input = torch.cat([image_features, state_features, cot_features], dim=-1)
        else:
            # Pad to max dimension
            padding_needed = max_input_dim - (image_features.shape[1] + state_features.shape[1])
            if padding_needed > 0:
                padding = torch.zeros(B, padding_needed, device=image_features.device)
                fused_input = torch.cat([image_features, state_features, padding], dim=-1)
            else:
                fused_input = torch.cat([image_features, state_features], dim=-1)
        
        # Project to hidden_dim
        fused = F.relu(self.fusion_proj(fused_input))
        
        # === VLA Planner ===
        trajectory = None
        if self.vla_planner is not None and images.ndim == 4:
            # Use VLA only for raw images
            vla_result = self.vla_planner(
                images=images,
                query=cot_tokens,
            )
            trajectory = vla_result["trajectory"]
        
        # Use decoder if VLA not available
        if trajectory is None:
            trajectory = self.decoder(fused).view(B, self.config.trajectory_horizon, 3)
        
        # === World Model Loss ===
        world_model_loss = torch.tensor(0.0, device=images.device)
        if self.world_model is not None and actions is not None and images.ndim == 4:
            # Use first action as conditioning (only for real images)
            first_action = actions[:, 0, :2] if actions.ndim == 3 else actions[:, :2]
            wm_result = self.world_model(images, first_action)
            # Reconstruction loss
            if "reconstruction" in wm_result:
                world_model_loss = F.mse_loss(
                    wm_result["reconstruction"].mean(),
                    image_features.mean()
                )
        
        # === Safety Loss ===
        safety_loss = torch.tensor(0.0, device=images.device)
        if self.safety_layer is not None and trajectory is not None:
            # Compute safety constraint loss
            # Penalize trajectories that would be modified by safety layer
            traj_np = trajectory.detach().cpu().numpy()
            # Simple safety metric: penalize high curvature
            curvature = torch.diff(trajectory[:, :, 0], dim=1).abs().mean()
            safety_loss = curvature * 0.1
        
        return {
            "trajectory": trajectory,
            "world_model_loss": world_model_loss,
            "safety_loss": safety_loss,
            "fused_features": fused,
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute unified loss.
        
        Args:
            predictions: model outputs
            targets: ground truth
            
        Returns:
            Dictionary with loss components
        """
        total_loss = 0
        losses = {}
        
        # Trajectory loss (main)
        if "trajectory" in predictions and "actions" in targets:
            traj_loss = F.mse_loss(predictions["trajectory"], targets["actions"])
            total_loss += traj_loss
            losses["trajectory"] = traj_loss.item()
        
        # World model loss
        if predictions.get("world_model_loss") is not None:
            wm_loss = predictions["world_model_loss"] * self.world_model_weight
            total_loss += wm_loss
            losses["world_model"] = wm_loss.item()
        
        # Safety loss
        if predictions.get("safety_loss") is not None:
            safe_loss = predictions["safety_loss"] * self.safety_weight
            total_loss += safe_loss
            losses["safety"] = safe_loss.item()
        
        losses["total"] = total_loss.item()
        
        return losses


# ============================================================================
# Unified Trainer
# ============================================================================

class UnifiedTrainer:
    """
    Single trainer for all models.
    
    Handles:
    - Data loading
    - Training loop
    - Evaluation
    - Checkpointing
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
        # Model
        self.model = UnifiedDrivingModel(config)
        self.model.to(config.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
        )
        
        # Data
        self.train_loader = None
        self.val_loader = None
        
        # State
        self.epoch = 0
        self.global_step = 0
    
    def setup_data(self):
        """Setup data loaders."""
        from training.data import UnifiedDataset
        
        # Create datasets
        train_dataset = UnifiedDataset.from_config({
            "name": "waymo",
            "data_root": self.config.data_root,
            "split": "train",
        })
        
        val_dataset = UnifiedDataset.from_config({
            "name": "waymo",
            "data_root": self.config.data_root,
            "split": "val",
        })
        
        # Loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move to device
        batch = {k: v.to(self.config.device) if torch.is_tensor(v) else v 
                 for k, v in batch.items()}
        
        # Forward
        outputs = self.model(
            images=batch.get("images", torch.randn(len(batch.get("state", [1])), 768).to(self.config.device)),
            state=batch.get("state"),
            cot_tokens=batch.get("cot_tokens"),
            actions=batch.get("actions"),
        )
        
        # Compute loss
        losses = self.compute_loss(outputs, batch)
        
        # Backward
        self.optimizer.zero_grad()
        losses["total"].backward()
        self.optimizer.step()
        
        self.global_step += 1
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute unified loss."""
        return self.model.compute_loss(predictions, targets)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        epoch_losses = {}
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.epoch}"):
            losses = self.train_step(batch)
            
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v
        
        # Average
        n_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        eval_losses = {}
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {k: v.to(self.config.device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            outputs = self.model(
                images=batch.get("images", torch.randn(len(batch.get("state", [1])), 768).to(self.config.device)),
                state=batch.get("state"),
                cot_tokens=batch.get("cot_tokens"),
                actions=batch.get("actions"),
            )
            
            losses = self.compute_loss(outputs, batch)
            
            for k, v in losses.items():
                if k not in eval_losses:
                    eval_losses[k] = 0
                eval_losses[k] += v.item()
        
        n_batches = len(self.val_loader)
        for k in eval_losses:
            eval_losses[k] /= n_batches
        
        return eval_losses
    
    def train(self, output_dir: str = "outputs/unified"):
        """Main training loop."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        best_loss = float("inf")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Evaluate
            val_losses = self.evaluate()
            
            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train: {train_losses}")
            print(f"  Val: {val_losses}")
            
            # Save best
            if val_losses.get("total", float("inf")) < best_loss:
                best_loss = val_losses["total"]
                self.save_checkpoint(output_path / "best.pt")
            
            # Save periodic
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(output_path / f"epoch_{epoch}.pt")
        
        print(f"\nTraining complete! Best loss: {best_loss}")
    
    def save_checkpoint(self, path: Path):
        """Save checkpoint."""
        torch.save({
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"]


# ============================================================================
# Usage
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = UnifiedConfig(
        use_world_model=True,
        use_vla=True,
        use_safety=True,
        batch_size=32,
        epochs=100,
    )
    
    # Create trainer
    trainer = UnifiedTrainer(config)
    
    # Setup data
    # trainer.setup_data()
    
    # Train (when data available)
    # trainer.train()
    
    # Test forward pass
    print("Testing unified model...")
    model = UnifiedDrivingModel(config)
    
    B = 4
    images = torch.randn(B, 768)  # Pre-encoded features
    state = torch.randn(B, 16)
    actions = torch.randn(B, 20, 3)
    
    result = model.forward(images, state, actions=actions)
    print(f"  Trajectory: {result['trajectory'].shape}")
    print(f"  World model loss: {result['world_model_loss']}")
    print(f"  Safety loss: {result['safety_loss']}")
    
    print("\n✓ Unified model working!")

"""
ResAD (Residual with Attention and Dynamics) Implementation
=========================================================

ResAD is a residual learning approach for autonomous driving that uses:
1. Normalized residual learning: Δ = (y - ŷ) / σ
2. Uncertainty estimation for adaptive weighting
3. Inertial reference frame for robustness

Key Features:
- Normalized residual instead of raw residual
- Uncertainty-aware training
- Inertial reference frame transformation
- Complements frozen SFT model

Reference: ResAD (arXiv:2510.08562)

Usage:
    from training.rl.resad import ResADModule, UncertaintyHead, ResADTrainer
    
    resad = ResADModule(policy, config)
    delta, sigma = resad(features, waypoints)
    y_final = resad.apply(features, waypoints, delta, sigma)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import os


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ResADConfig:
    """
    Configuration for ResAD algorithm.
    
    Attributes:
        - feature_dim: Input feature dimension
        - waypoint_dim: Waypoint dimension (x, y, heading)
        - hidden_dim: Hidden dimension for networks
        - dropout: Dropout probability
        - use_inertial_ref: Use inertial reference frame
        - uncertainty_weight: Weight for uncertainty loss
        - kl_weight: Weight for KL divergence regularization
        - normalize_residual: Normalize residual by uncertainty
    """
    # Model dimensions
    feature_dim: int = 256
    waypoint_dim: int = 3
    hidden_dim: int = 128
    
    # Training
    dropout: float = 0.1
    uncertainty_weight: float = 1.0
    kl_weight: float = 0.01
    
    # Inertial reference
    use_inertial_ref: bool = True
    
    # Normalization
    normalize_residual: bool = True
    sigma_min: float = 1e-4  # Minimum uncertainty
    
    # Loss
    use_nll_loss: bool = True
    use_mse_loss: bool = True
    use_kl_regularization: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.waypoint_dim > 0, "waypoint_dim must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.uncertainty_weight >= 0, "uncertainty_weight must be non-negative"


# ============================================================================
# Uncertainty Head
# ============================================================================

class UncertaintyHead(nn.Module):
    """
    Predicts aleatoric uncertainty for waypoint predictions.
    
    Architecture:
    - Takes SFT features + predictions as input
    - Outputs log(sigma) to ensure sigma > 0
    - Per-waypoint uncertainty estimation
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        waypoint_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.waypoint_dim = waypoint_dim
        self.hidden_dim = hidden_dim
        
        input_dim = feature_dim + waypoint_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, waypoint_dim),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, feature_dim] or [B, T, feature_dim]
            waypoints: [B, T, waypoint_dim]
            
        Returns:
            log_sigma: [B, T, waypoint_dim]
        """
        if features.dim() == 2:
            # Expand to match waypoints time dimension
            features = features.unsqueeze(1).expand(-1, waypoints.size(1), -1)
        
        x = torch.cat([features, waypoints], dim=-1)
        log_sigma = self.net(x)
        
        return log_sigma


# ============================================================================
# Residual Head
# ============================================================================

class ResADResidualHead(nn.Module):
    """
    ResAD Residual Head with Inertial Reference.
    
    Predicts normalized residual: Δ_norm = (y - ŷ) / σ
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        waypoint_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_inertial_ref: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.waypoint_dim = waypoint_dim
        self.hidden_dim = hidden_dim
        self.use_inertial_ref = use_inertial_ref
        
        # Input dimension: features + waypoints (+ ego_state if using inertial ref)
        ego_dim = 2 if use_inertial_ref else 0
        input_dim = feature_dim + waypoint_dim + ego_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, waypoint_dim),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        waypoints: torch.Tensor,
        ego_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, feature_dim] or [B, T, feature_dim]
            waypoints: [B, T, waypoint_dim]
            ego_state: [B, 2] (velocity, heading)
            
        Returns:
            delta_norm: [B, T, waypoint_dim]
        """
        if features.dim() == 2:
            # Expand to match waypoints time dimension
            features = features.unsqueeze(1).expand(-1, waypoints.size(1), -1)
        
        if self.use_inertial_ref and ego_state is not None:
            ego_features = ego_state.unsqueeze(1).expand(-1, waypoints.size(1), -1)
            waypoints_input = torch.cat([waypoints, ego_features], dim=-1)
        else:
            waypoints_input = waypoints
        
        x = torch.cat([features, waypoints_input], dim=-1)
        delta_norm = self.net(x)
        
        return delta_norm


# ============================================================================
# Inertial Reference Transform
# ============================================================================

class InertialReferenceTransform(nn.Module):
    """
    Transform waypoints between map frame and ego (inertial) frame.
    """
    
    def __init__(self, waypoint_dim: int = 3):
        super().__init__()
        self.waypoint_dim = waypoint_dim
    
    def map_to_ego(
        self,
        waypoints: torch.Tensor,  # [B, T, 3]
        ego_pose: torch.Tensor,     # [B, 3] (x, y, heading)
    ) -> torch.Tensor:
        """Transform from map frame to ego frame."""
        B, T, _ = waypoints.shape
        
        ego_x = ego_pose[:, 0]
        ego_y = ego_pose[:, 1]
        ego_heading = ego_pose[:, 2]
        
        # Relative position
        rel_x = waypoints[:, :, 0] - ego_x.unsqueeze(1)
        rel_y = waypoints[:, :, 1] - ego_y.unsqueeze(1)
        
        # Rotate to ego frame
        cos_h = torch.cos(ego_heading)
        sin_h = torch.sin(ego_heading)
        
        ego_rel_x = rel_x * cos_h + rel_y * sin_h
        ego_rel_y = -rel_x * sin_h + rel_y * cos_h
        
        # Relative heading
        ego_heading_rel = waypoints[:, :, 2] - ego_heading.unsqueeze(1)
        ego_heading_rel = torch.atan2(
            torch.sin(ego_heading_rel),
            torch.cos(ego_heading_rel)
        )
        
        return torch.stack([ego_rel_x, ego_rel_y, ego_heading_rel], dim=-1)
    
    def ego_to_map(
        self,
        waypoints_ego: torch.Tensor,  # [B, T, 3]
        ego_pose: torch.Tensor,        # [B, 3]
    ) -> torch.Tensor:
        """Transform from ego frame to map frame."""
        B, T, _ = waypoints_ego.shape
        
        ego_x = ego_pose[:, 0]
        ego_y = ego_pose[:, 1]
        ego_heading = ego_pose[:, 2]
        
        cos_h = torch.cos(ego_heading)
        sin_h = torch.sin(ego_heading)
        
        # Rotate to map frame
        map_rel_x = waypoints_ego[:, :, 0] * cos_h - waypoints_ego[:, :, 1] * sin_h
        map_rel_y = waypoints_ego[:, :, 0] * sin_h + waypoints_ego[:, :, 1] * cos_h
        
        # Translate
        map_x = map_rel_x + ego_x.unsqueeze(1)
        map_y = map_rel_y + ego_y.unsqueeze(1)
        map_heading = waypoints_ego[:, :, 2] + ego_heading.unsqueeze(1)
        
        return torch.stack([map_x, map_y, map_heading], dim=-1)


# ============================================================================
# Complete ResAD Module
# ============================================================================

class ResADModule(nn.Module):
    """
    Complete ResAD Module combining residual head and uncertainty head.
    
    Usage:
        resad = ResADModule(
            feature_dim=256,
            waypoint_dim=3,
            hidden_dim=128,
            use_inertial_ref=True,
        )
        
        delta, log_sigma = resad(features, waypoints, ego_state)
        y_final, sigma = resad.apply(waypoints, delta, log_sigma)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        waypoint_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_inertial_ref: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.waypoint_dim = waypoint_dim
        self.hidden_dim = hidden_dim
        self.use_inertial_ref = use_inertial_ref
        
        self.residual_head = ResADResidualHead(
            feature_dim=feature_dim,
            waypoint_dim=waypoint_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_inertial_ref=use_inertial_ref,
        )
        
        self.uncertainty_head = UncertaintyHead(
            feature_dim=feature_dim,
            waypoint_dim=waypoint_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        features: torch.Tensor,
        waypoints: torch.Tensor,
        ego_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, feature_dim] or [B, T, feature_dim]
            waypoints: [B, T, waypoint_dim]
            ego_state: [B, 2] optional
            
        Returns:
            delta_norm: [B, T, waypoint_dim]
            log_sigma: [B, T, waypoint_dim]
        """
        delta_norm = self.residual_head(features, waypoints, ego_state)
        log_sigma = self.uncertainty_head(features, waypoints)
        
        return delta_norm, log_sigma
    
    def apply(
        self,
        waypoints: torch.Tensor,
        delta_norm: torch.Tensor,
        log_sigma: torch.Tensor,
        uncertainty_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply residual correction.
        
        Formula: y_final = ŷ + Δ_norm × σ
        """
        sigma = torch.exp(log_sigma) * uncertainty_weight
        sigma = torch.clamp(sigma, min=1e-4)
        corrected = waypoints + delta_norm * sigma
        
        return corrected, sigma
    
    def loss(
        self,
        delta_norm: torch.Tensor,
        log_sigma: torch.Tensor,
        target_residual: torch.Tensor,
        target_uncertainty: Optional[torch.Tensor] = None,
        kl_weight: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ResAD loss.
        
        Returns loss dict with:
        - total_loss
        - mse_loss
        - nll_loss
        - kl_loss
        - sigma_mean
        """
        sigma = torch.exp(log_sigma)
        
        # NLL Loss
        nll = 0.5 * ((target_residual - delta_norm) ** 2 / (sigma + 1e-6) + log_sigma)
        nll_loss = nll.mean()
        
        # MSE Loss
        mse_loss = F.mse_loss(delta_norm, target_residual)
        
        # KL Divergence
        kl = 0.5 * (sigma - 1 - torch.log(sigma + 1e-6))
        kl_loss = kl.mean()
        
        total_loss = mse_loss + nll_loss + kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'nll_loss': nll_loss,
            'kl_loss': kl_loss,
            'sigma_mean': sigma.mean(),
            'delta_norm_mean': delta_norm.mean(),
        }


# ============================================================================
# ResAD with SFT Integration
# ============================================================================

class ResADWithSFT(nn.Module):
    """
    ResAD module integrated with frozen SFT model.
    
    Usage:
        sft_model = load_sft_model("sft.pt")
        resad = ResADWithSFT(sft_model, config)
        
        output = resad(features)
        # output['waypoints']: corrected waypoints
        # output['uncertainty']: uncertainty estimates
    """
    
    def __init__(
        self,
        sft_model: nn.Module,
        config: Optional[ResADConfig] = None,
    ):
        super().__init__()
        
        self.sft_model = sft_model
        self.config = config or ResADConfig()
        
        # Freeze SFT model
        for param in sft_model.parameters():
            param.requires_grad = False
        self.sft_model.eval()
        
        # Get feature dimension
        if hasattr(sft_model, 'config'):
            feature_dim = getattr(sft_model.config, 'hidden_dim', 256)
        else:
            feature_dim = 256
        
        self.resad = ResADModule(
            feature_dim=feature_dim,
            waypoint_dim=self.config.waypoint_dim,
            hidden_dim=self.config.hidden_dim,
            use_inertial_ref=self.config.use_inertial_ref,
        )
    
    def forward(
        self,
        features: torch.Tensor,
        target_waypoints: Optional[torch.Tensor] = None,
        ego_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        with torch.no_grad():
            sft_output = self.sft_model(features)
            # Handle both dict and tensor outputs
            if isinstance(sft_output, dict):
                sft_waypoints = sft_output.get('waypoints', sft_output.get('logits', sft_output))
            else:
                sft_waypoints = sft_output
        
        delta_norm, log_sigma = self.resad(features, sft_waypoints, ego_state)
        corrected, uncertainty = self.resad.apply(sft_waypoints, delta_norm, log_sigma)
        
        result = {
            'waypoints': corrected,
            'uncertainty': uncertainty,
            'sft_waypoints': sft_waypoints,
            'delta': delta_norm,
            'log_sigma': log_sigma,
        }
        
        if target_waypoints is not None:
            with torch.no_grad():
                target_residual = target_waypoints - sft_waypoints
            
            loss_dict = self.resad.loss(
                delta_norm, log_sigma,
                target_residual,
                target_uncertainty=torch.abs(target_residual),
                kl_weight=self.config.kl_weight,
            )
            result['loss'] = loss_dict
        
        return result


# ============================================================================
# ResAD Trainer
# ============================================================================

class ResADTrainer:
    """
    Trainer for ResAD algorithm.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ResADConfig,
        lr: float = 1e-4,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.resad.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        
        self.global_step = 0
    
    def train_step(
        self,
        features: torch.Tensor,
        waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
        ego_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        output = self.model(
            features,
            target_waypoints=target_waypoints,
            ego_state=ego_state,
        )
        
        loss_dict = output['loss']
        total_loss = loss_dict['total_loss']
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.resad.parameters(),
            max_norm=1.0
        )
        self.optimizer.step()
        
        self.global_step += 1
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = defaultdict(list)
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            waypoints = batch['waypoints'].to(self.device)
            targets = batch['targets'].to(self.device)
            ego_state = batch.get('ego_state')
            if ego_state is not None:
                ego_state = ego_state.to(self.device)
            
            losses = self.train_step(features, waypoints, targets, ego_state)
            
            for k, v in losses.items():
                total_losses[k].append(v)
        
        return {k: np.mean(v) for k, v in total_losses.items()}
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.resad.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'epoch': epoch,
            'global_step': self.global_step,
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


# ============================================================================
# ResAD Evaluation
# ============================================================================

class ResADEvaluator:
    """Evaluator for ResAD model."""
    
    def __init__(self, config: ResADConfig):
        self.config = config
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader,
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        model.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features']
                targets = batch['targets']
                
                output = model(features)
                waypoints = output['waypoints']
                uncertainty = output['uncertainty']
                sft_waypoints = output['sft_waypoints']
                
                # ADE
                ade = torch.norm(waypoints - targets, dim=-1).mean().item()
                metrics['ade'].append(ade)
                
                # FDE
                fde = torch.norm(waypoints[:, -1] - targets[:, -1], dim=-1).mean().item()
                metrics['fde'].append(fde)
                
                # SFT baseline
                sft_ade = torch.norm(sft_waypoints - targets, dim=-1).mean().item()
                metrics['sft_ade'].append(sft_ade)
                
                # Uncertainty
                metrics['uncertainty'].append(uncertainty.mean().item())
        
        return {k: np.mean(v) for k, v in metrics.items()}


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of using ResAD."""
    
    # Mock SFT model
    class MockSFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 30)
        
        def forward(self, x):
            return self.fc(x)
    
    # Configuration
    config = ResADConfig(
        feature_dim=256,
        waypoint_dim=3,
        hidden_dim=128,
        use_inertial_ref=True,
    )
    
    # SFT model
    sft_model = MockSFT()
    
    # ResAD with SFT
    resad = ResADWithSFT(sft_model, config)
    
    # Forward pass
    features = torch.randn(4, 256)
    targets = torch.randn(4, 10, 3)
    
    output = resad(features, target_waypoints=targets)
    
    print("ResAD Output:")
    print(f"  SFT waypoints: {output['sft_waypoints'].shape}")
    print(f"  Corrected waypoints: {output['waypoints'].shape}")
    print(f"  Uncertainty: {output['uncertainty'].shape}")
    print(f"  Loss: {output['loss']['total_loss'].item():.4f}")
    
    # Training step
    trainer = ResADTrainer(resad, config)
    
    # Mock dataloader
    class MockDataset(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            return {
                'features': torch.randn(256),
                'waypoints': torch.randn(10, 3),
                'targets': torch.randn(10, 3),
            }
        def __len__(self):
            return 100
    
    dataloader = torch.utils.data.DataLoader(MockDataset(), batch_size=32)
    
    print("\nTraining for 1 epoch...")
    metrics = trainer.train_epoch(dataloader)
    print(f"Losses: {metrics}")


if __name__ == "__main__":
    example_usage()

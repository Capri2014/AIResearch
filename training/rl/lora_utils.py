#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) Utilities for RL Delta Head Training.

This module provides LoRA implementations for efficient fine-tuning of delta heads
in the RL-after-SFT pipeline. LoRA reduces trainable parameters while maintaining
performance, making it ideal for adapting pretrained SFT models.

Usage:
    from lora_utils import LoRALinear, apply_lora_to_model
    
    # Apply LoRA to a linear layer
    lora_linear = LoRALinear(in_features, out_features, rank=8)
    
    # Apply LoRA to an existing model
    apply_lora_to_model(model, rank=8, alpha=16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 8  # LoRA rank (r)
    alpha: int = 16  # LoRA scaling factor
    dropout: float = 0.1  # Dropout probability
    target_modules: Optional[List[str]] = None  # Module names to apply LoRA
    bias: str = "none"  # Bias: "none", "all", or "lora_only"


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Original: y = Wx + b
    LoRA: y = Wx + (BA)x where B,A are low-rank matrices
    
    This adds a parallel branch with rank-r decomposition:
    - A: down-project from in_features to rank
    - B: up-project from rank to out_features
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        bias: Optional[torch.Tensor] = None,
        freeze_original: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original weight (frozen if requested)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)
        
        if freeze_original:
            self.weight.requires_grad = False
        else:
            self.weight.requires_grad = True
        
        # LoRA branches (A down-project, B up-project)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA parameters
        self._init_lora_parameters()
    
    def _init_lora_parameters(self):
        """Initialize LoRA matrices with Kaiming/He initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with original + LoRA branch."""
        # Original forward pass
        original = F.linear(x, self.weight, self.bias)
        
        # LoRA branch: (x @ A.T @ B.T) * scaling
        if self.training:
            lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        else:
            # During eval, no dropout
            lora = x @ self.lora_A.T @ self.lora_B.T
        
        return original + lora * self.scaling
    
    def merge_weights(self):
        """Merge LoRA weights into original for inference."""
        if self.weight.requires_grad:
            raise RuntimeError("Cannot merge when original weight is trainable")
        
        # W_merged = W + (B @ A) * scaling
        merged = self.weight + self.lora_B @ self.lora_A * self.scaling
        return merged
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}'


class LoRAConv2d(nn.Module):
    """
    Conv2d layer with LoRA adaptation.
    
    Similar to LoRALinear but for convolutional layers.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        freeze_original: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original conv layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        
        if freeze_original:
            for param in self.conv.parameters():
                param.requires_grad = False
        
        # LoRA branches
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_channels * kernel_size * kernel_size)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_channels * kernel_size * kernel_size, rank)
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        self._init_lora_parameters()
    
    def _init_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.conv(x)
        
        # Reshape input for LoRA computation
        b, c, h, w = x.shape
        x_flat = x.reshape(b, -1)  # (batch, c*h*w)
        
        if self.training:
            lora = self.dropout(x_flat) @ self.lora_A.T @ self.lora_B.T
        else:
            lora = x_flat @ self.lora_A.T @ self.lora_B.T
        
        # Reshape back to (batch, out_channels, h, w)
        # Assuming stride/padding preserve spatial dimensions
        lora = lora.reshape(b, self.out_channels, h, w)
        
        return original + lora * self.scaling


class LoRADeltaHead(nn.Module):
    """
    LoRA-adapted delta head for RL refinement.
    
    Takes frozen SFT waypoints and learns residual corrections via LoRA.
    Architecture: final_waypoints = sft_waypoints + lora_delta_head(state)
    
    Benefits:
    - Fewer trainable parameters than full delta head
    - Modular: can swap LoRA adapters without retraining SFT
    - Efficient: frozen SFT backbone, only LoRA params train
    """
    
    def __init__(
        self,
        state_dim: int,
        waypoint_dim: int,
        n_waypoints: int,
        hidden_dim: int = 128,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.waypoint_dim = waypoint_dim
        self.n_waypoints = n_waypoints
        self.output_dim = n_waypoints * waypoint_dim
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LoRA-adapted output head
        # Instead of full linear: use LoRA linear for efficiency
        self.delta_head = LoRALinear(
            hidden_dim,
            self.output_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            freeze_original=True  # No original weight, just LoRA
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute delta waypoints.
        
        Args:
            state: (batch, state_dim) current state
            
        Returns:
            delta: (batch, n_waypoints * waypoint_dim) delta to add to SFT waypoints
        """
        hidden = self.encoder(state)
        delta = self.delta_head(hidden)
        return delta
    
    def get_delta_matrix(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get delta as 2D matrix (n_waypoints, waypoint_dim).
        
        Args:
            state: (batch, state_dim)
            
        Returns:
            delta_matrix: (batch, n_waypoints, waypoint_dim)
        """
        delta_flat = self.forward(state)
        return delta_flat.reshape(-1, self.n_waypoints, self.waypoint_dim)


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_layer_types: tuple = (nn.Linear,)
) -> nn.Module:
    """
    Apply LoRA to all layers of a given type in a model.
    
    This replaces the original layers with LoRA-adapted versions while
    keeping the original weights frozen.
    
    Args:
        model: PyTorch model to adapt
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA
        target_layer_types: Tuple of layer types to adapt (default: Linear)
        
    Returns:
        Modified model with LoRA applied
    """
    for name, module in model.named_children():
        if isinstance(module, target_layer_types):
            # Replace with LoRA version
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=module.bias,
                freeze_original=True
            )
            setattr(model, name, lora_layer)
        else:
            # Recursively apply to child modules
            apply_lora_to_model(
                module, rank=rank, alpha=alpha, 
                dropout=dropout, target_layer_types=target_layer_types
            )
    
    return model


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in LoRA components vs original.
    
    Returns:
        Dictionary with 'lora_params', 'frozen_params', 'total_params'
    """
    lora_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params += param.numel()
        elif not param.requires_grad:
            frozen_params += param.numel()
    
    total_params = lora_params + frozen_params + sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    return {
        'lora_params': lora_params,
        'frozen_params': frozen_params,
        'trainable_params': lora_params,  # LoRA params are trainable
        'total_params': total_params,
        'lora_ratio': lora_params / total_params if total_params > 0 else 0
    }


class LoRAWrapper(nn.Module):
    """
    Wrapper that adds LoRA adaptation to any module.
    
    Usage:
        # Wrap a pretrained model
        base_model = WaypointBCModel(...)
        lora_model = LoRAWrapper(base_model, rank=8, alpha=16)
        
        # Only LoRA parameters are trainable
        # base_model weights remain frozen
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Apply LoRA to specified modules or all Linear layers
        if target_modules is None:
            apply_lora_to_model(
                self, rank=rank, alpha=alpha, 
                dropout=dropout, target_layer_types=(nn.Linear,)
            )
        else:
            # Apply only to named modules
            for name, module in base_model.named_modules():
                if any(target in name for target in target_modules):
                    if isinstance(module, nn.Linear):
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = base_model.get_submodule(parent_name) if parent_name else base_model
                        
                        lora_module = LoRALinear(
                            module.in_features,
                            module.out_features,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                            bias=module.bias,
                            freeze_original=True
                        )
                        setattr(parent, child_name, lora_module)
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self):
        """Return only trainable (LoRA) parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def merge_and_unload(self):
        """
        Merge LoRA weights into base model and unload LoRA adapters.
        
        This creates a single model with merged weights for inference.
        """
        # This would require implementing merge for all LoRA layers
        # For now, raise NotImplementedError
        raise NotImplementedError("Merge functionality coming soon")


# Utility function for integration with existing training pipelines
def create_lora_delta_head(
    state_dim: int,
    waypoint_dim: int,
    n_waypoints: int,
    config: Optional[LoRAConfig] = None
) -> LoRADeltaHead:
    """
    Factory function to create a LoRA-adapted delta head.
    
    Args:
        state_dim: Dimension of state input
        waypoint_dim: Dimension per waypoint (2 for x,y)
        n_waypoints: Number of waypoints to predict
        config: LoRA configuration (uses defaults if None)
        
    Returns:
        LoRADeltaHead module
    """
    if config is None:
        config = LoRAConfig()
    
    return LoRADeltaHead(
        state_dim=state_dim,
        waypoint_dim=waypoint_dim,
        n_waypoints=n_waypoints,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout
    )


if __name__ == "__main__":
    # Demo usage
    print("LoRA Utilities Demo")
    print("=" * 50)
    
    # Create a simple LoRA-adapted delta head
    delta_head = LoRADeltaHead(
        state_dim=6,  # x, y, heading, speed, goal_x, goal_y
        waypoint_dim=2,
        n_waypoints=5,
        hidden_dim=128,
        rank=8,
        alpha=16
    )
    
    # Test forward pass
    batch_size = 4
    state = torch.randn(batch_size, 6)
    delta = delta_head(state)
    delta_matrix = delta_head.get_delta_matrix(state)
    
    print(f"Input state shape: {state.shape}")
    print(f"Delta output shape: {delta.shape}")
    print(f"Delta matrix shape: {delta_matrix.shape}")
    
    # Count parameters
    param_counts = count_lora_parameters(delta_head)
    print(f"\nParameter counts:")
    for key, val in param_counts.items():
        print(f"  {key}: {val:,}")
    
    # Test LoRA config
    config = LoRAConfig(rank=4, alpha=8, dropout=0.05)
    print(f"\nLoRAConfig: rank={config.rank}, alpha={config.alpha}, dropout={config.dropout}")
    print(f"  scaling: {config.alpha / config.rank}")

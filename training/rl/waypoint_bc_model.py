"""
Waypoint BC Model for RL After SFT Pipeline.

This module provides a simple neural network waypoint predictor that can be
used as the SFT baseline in the residual delta learning framework.

Architecture:
    - Encoder: MLP that processes state (position, velocity, goal)
    - Decoder: Predicts waypoints as offsets from current position
    
The SFT model is frozen, and RL trains a delta head to refine predictions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class WaypointBCConfig:
    """Configuration for Waypoint BC model."""
    
    def __init__(
        self,
        state_dim: int = 6,
        waypoint_dim: int = 2,
        horizon: int = 20,
        hidden_dims: Tuple[int, ...] = (128, 256, 128),
        dropout: float = 0.1,
    ):
        self.state_dim = state_dim
        self.waypoint_dim = waypoint_dim
        self.horizon = horizon
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class WaypointBCEncoder(nn.Module):
    """Encoder for waypoint prediction."""
    
    def __init__(self, config: WaypointBCConfig):
        super().__init__()
        self.config = config
        
        # Build encoder MLP
        dims = [config.state_dim] + list(config.hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
        self.encoder = nn.Sequential(*layers)
        
        # Output projection
        self.z_dim = config.hidden_dims[-1]
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state to latent representation.
        
        Args:
            state: Tensor of shape (batch, state_dim) or (state_dim,)
            
        Returns:
            Tensor of shape (batch, z_dim) or (z_dim,)
        """
        return self.encoder(state)


class WaypointBCDecoder(nn.Module):
    """Decoder that predicts waypoints from latent representation."""
    
    def __init__(self, config: WaypointBCConfig):
        super().__init__()
        self.config = config
        
        # Build decoder MLP
        # Input: latent z + waypoint index (normalized)
        input_dim = config.hidden_dims[-1] + 1
        output_dim = config.waypoint_dim
        
        layers = []
        hidden_dims = list(config.hidden_dims[-2:]) + [output_dim]
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode latent representation to waypoints.
        
        Args:
            z: Latent tensor of shape (batch, z_dim) or (z_dim,)
            indices: Optional waypoint indices of shape (batch, horizon) or (horizon,)
            
        Returns:
            Waypoints of shape (batch, horizon, waypoint_dim) or (horizon, waypoint_dim)
        """
        batch_size = z.shape[0] if z.dim() > 1 else 1
        horizon = self.config.horizon
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
            single = True
        else:
            single = False
        
        # Create waypoint indices (normalized 0 to 1)
        if indices is None:
            indices = torch.arange(horizon, device=z.device).float() / horizon
            indices = indices.unsqueeze(0).expand(batch_size, -1)  # (batch, horizon)
        
        # Expand z for each waypoint
        z_expanded = z.unsqueeze(1).expand(-1, horizon, -1)  # (batch, horizon, z_dim)
        
        # Add index as feature
        indices_expanded = indices.unsqueeze(-1)  # (batch, horizon, 1)
        x = torch.cat([z_expanded, indices_expanded], dim=-1)  # (batch, horizon, z_dim + 1)
        
        # Flatten for MLP
        x = x.reshape(batch_size * horizon, -1)  # (batch * horizon, z_dim + 1)
        
        # Decode
        waypoints = self.decoder(x)  # (batch * horizon, waypoint_dim)
        
        # Reshape
        waypoints = waypoints.reshape(batch_size, horizon, self.config.waypoint_dim)
        
        if single:
            waypoints = waypoints.squeeze(0)
            
        return waypoints


class WaypointBCModel(nn.Module):
    """
    Complete Waypoint BC Model.
    
    Predicts waypoints from state using encoder-decoder architecture.
    """
    
    def __init__(self, config: WaypointBCConfig):
        super().__init__()
        self.config = config
        self.encoder = WaypointBCEncoder(config)
        self.decoder = WaypointBCDecoder(config)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from state.
        
        Args:
            state: Tensor of shape (batch, state_dim) or (state_dim,)
            
        Returns:
            Waypoints of shape (batch, horizon, waypoint_dim) or (horizon, waypoint_dim)
        """
        z = self.encoder(state)
        waypoints = self.decoder(z)
        return waypoints
    
    def predict_single(self, state: np.ndarray) -> np.ndarray:
        """
        Predict waypoints for a single state (numpy interface).
        
        Args:
            state: Array of shape (state_dim,)
            
        Returns:
            Waypoints of shape (horizon, waypoint_dim)
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float()
            waypoints = self.forward(state_tensor)
            return waypoints.numpy()


class ResidualDeltaHead(nn.Module):
    """
    Residual delta head for RL refinement.
    
    Takes SFT waypoints and learns to add delta corrections.
    Architecture: delta_head(z) where z is the latent state representation.
    """
    
    def __init__(
        self,
        z_dim: int,
        horizon: int = 20,
        waypoint_dim: int = 2,
        hidden_dims: Tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        self.z_dim = z_dim
        self.horizon = horizon
        self.waypoint_dim = waypoint_dim
        
        # Delta prediction network
        # Input: latent z + waypoint index
        # Output: delta (dx, dy) for each waypoint
        input_dim = z_dim + 1
        dims = [input_dim] + list(hidden_dims) + [waypoint_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.delta_net = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor, sft_waypoints: torch.Tensor) -> torch.Tensor:
        """
        Compute delta corrections to SFT waypoints.
        
        Args:
            z: Latent representation from encoder (batch, z_dim) or (z_dim,)
            sft_waypoints: SFT waypoints (batch, horizon, waypoint_dim) or (horizon, waypoint_dim)
            
        Returns:
            Delta waypoints of same shape
        """
        single = False
        if z.dim() == 1:
            z = z.unsqueeze(0)
            sft_waypoints = sft_waypoints.unsqueeze(0)
            single = True
            
        batch_size, horizon, _ = sft_waypoints.shape
        
        # Create waypoint indices
        indices = torch.arange(horizon, device=z.device).float() / horizon
        indices = indices.unsqueeze(0).expand(batch_size, -1)
        
        # Expand z for each waypoint
        z_expanded = z.unsqueeze(1).expand(-1, horizon, -1)
        
        # Concatenate z + index
        x = torch.cat([z_expanded, indices.unsqueeze(-1)], dim=-1)
        
        # Flatten for MLP
        x = x.reshape(batch_size * horizon, -1)
        
        # Predict deltas
        deltas = self.delta_net(x).reshape(batch_size, horizon, self.waypoint_dim)
        
        if single:
            deltas = deltas.squeeze(0)
            
        return deltas


class WaypointBCWithResidual(nn.Module):
    """
    Combined SFT + RL model for residual delta learning.
    
    Architecture:
        final_waypoints = sft_waypoints + delta_head(z)
    
    The SFT model is frozen, only delta_head is trained with RL.
    """
    
    def __init__(self, config: WaypointBCConfig, train_delta_only: bool = True):
        super().__init__()
        self.config = config
        self.train_delta_only = train_delta_only
        
        # SFT waypoint predictor (frozen)
        self.sft_model = WaypointBCModel(config)
        
        # Delta head for RL refinement (trainable)
        self.delta_head = ResidualDeltaHead(
            z_dim=config.hidden_dims[-1],
            horizon=config.horizon,
            waypoint_dim=config.waypoint_dim,
        )
        
        # Freeze SFT model if needed
        if train_delta_only:
            for param in self.sft_model.parameters():
                param.requires_grad = False
                
    def forward(self, state: torch.Tensor, use_residual: bool = True) -> torch.Tensor:
        """
        Predict final waypoints.
        
        Args:
            state: State tensor
            use_residual: If True, add delta corrections; if False, return SFT only
            
        Returns:
            Final waypoints
        """
        # Get SFT waypoints
        sft_waypoints = self.sft_model(state)
        
        if not use_residual:
            return sft_waypoints
            
        # Get latent representation
        z = self.sft_model.encoder(state)
        
        # Compute delta
        delta = self.delta_head(z, sft_waypoints)
        
        # Final = SFT + delta
        return sft_waypoints + delta
    
    def get_sft_waypoints(self, state: torch.Tensor) -> torch.Tensor:
        """Get SFT-only waypoints (no residual)."""
        return self.sft_model(state)
    
    def get_delta(self, state: torch.Tensor) -> torch.Tensor:
        """Get delta corrections only."""
        z = self.sft_model.encoder(state)
        sft_waypoints = self.sft_model(state)
        return self.delta_head(z, sft_waypoints)


def create_waypoint_bc_model(
    state_dim: int = 6,
    waypoint_dim: int = 2,
    horizon: int = 20,
    hidden_dims: Tuple[int, ...] = (128, 256, 128),
    device: str = 'cpu',
) -> WaypointBCModel:
    """Factory function to create a waypoint BC model."""
    config = WaypointBCConfig(
        state_dim=state_dim,
        waypoint_dim=waypoint_dim,
        horizon=horizon,
        hidden_dims=hidden_dims,
    )
    model = WaypointBCModel(config)
    return model.to(device)


def create_residual_model(
    state_dim: int = 6,
    waypoint_dim: int = 2,
    horizon: int = 20,
    hidden_dims: Tuple[int, ...] = (128, 256, 128),
    train_delta_only: bool = True,
    device: str = 'cpu',
) -> WaypointBCWithResidual:
    """Factory function to create a residual model (SFT + delta head)."""
    config = WaypointBCConfig(
        state_dim=state_dim,
        waypoint_dim=waypoint_dim,
        horizon=horizon,
        hidden_dims=hidden_dims,
    )
    model = WaypointBCWithResidual(config, train_delta_only=train_delta_only)
    return model.to(device)


# Simple training loop for waypoint BC (SFT stage)
def train_waypoint_bc(
    model: WaypointBCModel,
    states: np.ndarray,
    target_waypoints: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> dict:
    """
    Simple supervised training for waypoint BC model.
    
    Args:
        model: WaypointBCModel to train
        states: Array of shape (N, state_dim)
        target_waypoints: Array of shape (N, horizon, waypoint_dim)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Training metrics dict
    """
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Convert to tensors
    states_tensor = torch.from_numpy(states).float().to(device)
    targets_tensor = torch.from_numpy(target_waypoints).float().to(device)
    
    n_samples = states_tensor.shape[0]
    losses = []
    
    for epoch in range(epochs):
        # Shuffle
        indices = torch.randperm(n_samples)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_states = states_tensor[batch_idx]
            batch_targets = targets_tensor[batch_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_waypoints = model(batch_states)
            
            # Compute loss
            loss = loss_fn(pred_waypoints, batch_targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return {'losses': losses, 'final_loss': losses[-1] if losses else None}


if __name__ == '__main__':
    # Quick test
    config = WaypointBCConfig(state_dim=6, waypoint_dim=2, horizon=20)
    
    # Test SFT model
    sft_model = WaypointBCModel(config)
    state = torch.randn(4, 6)
    waypoints = sft_model(state)
    print(f"SFT model output shape: {waypoints.shape}")
    
    # Test residual model
    residual_model = WaypointBCWithResidual(config, train_delta_only=True)
    final_waypoints = residual_model(state, use_residual=True)
    print(f"Residual model output shape: {final_waypoints.shape}")
    
    sft_only = residual_model.get_sft_waypoints(state)
    print(f"SFT only shape: {sft_only.shape}")
    
    delta = residual_model.get_delta(state)
    print(f"Delta shape: {delta.shape}")
    
    # Test with numpy
    state_np = np.random.randn(6).astype(np.float32)
    waypoints_np = sft_model.predict_single(state_np)
    print(f"Numpy interface output shape: {waypoints_np.shape}")
    
    print("\nAll tests passed!")

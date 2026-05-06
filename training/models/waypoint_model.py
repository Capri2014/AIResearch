#!/usr/bin/env python3
"""
Unified Waypoint Prediction Model

A shared model architecture used across BC, RL, and SSL pipeline stages.
Provides a consistent interface for waypoint prediction that can be:
- Pre-trained with SSL (JEPA/Contrastive)
- Fine-tuned with BC (supervised waypoint prediction)
- Refined with RL (PPO/PPO delta)

This model predicts future waypoints from current observation (agent pose + route).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class WaypointModelConfig:
    """Configuration for the unified waypoint prediction model."""
    # Observation
    obs_dim: int = 64           # Agent observation dimension (pose + surrounding)
    route_dim: int = 128        # Route/waypoints input dimension
    
    # Output
    num_waypoints: int = 8     # Number of future waypoints to predict
    waypoint_dim: int = 2       # 2D (x, y) or 3D (x, y, heading)
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # SSL pretraining
    latent_dim: int = 128      # Latent representation for SSL
    
    @property
    def output_dim(self) -> int:
        return self.num_waypoints * self.waypoint_dim


class ResidualBlock(nn.Module):
    """Residual block with layer norm."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


class ObservationEncoder(nn.Module):
    """Encodes agent observation into latent representation."""
    
    def __init__(self, config: WaypointModelConfig):
        super().__init__()
        self.config = config
        
        self.net = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            *[ResidualBlock(config.hidden_dim, config.dropout) 
              for _ in range(config.num_layers)],
            nn.LayerNorm(config.hidden_dim),
        )
        self.output_dim = config.hidden_dim
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) agent observation
        Returns:
            (batch, hidden_dim) latent observation
        """
        return self.net(obs)


class RouteEncoder(nn.Module):
    """Encodes route/waypoints into latent representation."""
    
    def __init__(self, config: WaypointModelConfig):
        super().__init__()
        self.config = config
        
        # Route is a sequence of waypoints
        self.net = nn.Sequential(
            nn.Linear(config.route_dim, config.hidden_dim),
            *[ResidualBlock(config.hidden_dim, config.dropout) 
              for _ in range(config.num_layers)],
            nn.LayerNorm(config.hidden_dim),
        )
        self.output_dim = config.hidden_dim
    
    def forward(self, route: torch.Tensor) -> torch.Tensor:
        """
        Args:
            route: (batch, route_dim) route/waypoints input
        Returns:
            (batch, hidden_dim) latent route
        """
        return self.net(route)


class WaypointDecoder(nn.Module):
    """Decodes latent representation into future waypoints."""
    
    def __init__(self, config: WaypointModelConfig):
        super().__init__()
        self.config = config
        
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            *[ResidualBlock(config.hidden_dim, config.dropout) 
              for _ in range(config.num_layers)],
            nn.Linear(config.hidden_dim, config.output_dim),
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch, hidden_dim) latent representation
        Returns:
            (batch, num_waypoints * waypoint_dim) predicted waypoints
        """
        return self.net(latent)


class WaypointPredictionModel(nn.Module):
    """
    Unified model for waypoint prediction.
    
    Architecture:
        obs_encoder -> [concat] -> hidden -> waypoint_decoder -> waypoints
        route_encoder -> /
    
    This model can be:
    1. Pre-trained with SSL objectives (JEPA, Contrastive)
    2. Fine-tuned with BC for waypoint prediction
    3. Used in RL as the policy network
    """
    
    def __init__(self, config: WaypointModelConfig):
        super().__init__()
        self.config = config
        
        self.obs_encoder = ObservationEncoder(config)
        self.route_encoder = RouteEncoder(config)
        self.waypoint_decoder = WaypointDecoder(config)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Latent projection for SSL
        self.ssl_projection = nn.Linear(config.hidden_dim, config.latent_dim)
    
    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode just the observation (for SSL)."""
        return self.obs_encoder(obs)
    
    def encode_route(self, route: torch.Tensor) -> torch.Tensor:
        """Encode just the route (for SSL)."""
        return self.route_encoder(route)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        route: torch.Tensor,
        return_latent: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict future waypoints from observation and route.
        
        Args:
            obs: (batch, obs_dim) agent observation
            route: (batch, route_dim) route/waypoints
            return_latent: if True, also return latent representation
            
        Returns:
            waypoints: (batch, num_waypoints * waypoint_dim) predicted waypoints
            latent: (batch, hidden_dim) latent representation (if return_latent=True)
        """
        # Encode both inputs
        obs_latent = self.obs_encoder(obs)      # (batch, hidden_dim)
        route_latent = self.route_encoder(route)  # (batch, hidden_dim)
        
        # Fuse
        fused = torch.cat([obs_latent, route_latent], dim=-1)  # (batch, hidden_dim * 2)
        fused = self.fusion(fused)  # (batch, hidden_dim)
        
        # Decode to waypoints
        waypoints = self.waypoint_decoder(fused)  # (batch, output_dim)
        
        if return_latent:
            latent = self.ssl_projection(fused)
            return waypoints, latent
        return waypoints, None
    
    def get_latent(self, obs: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        """Get latent representation for SSL pretraining."""
        obs_latent = self.obs_encoder(obs)
        route_latent = self.route_encoder(route)
        fused = torch.cat([obs_latent, route_latent], dim=-1)
        fused = self.fusion(fused)
        return self.ssl_projection(fused)
    
    def predict(
        self, 
        obs: torch.Tensor, 
        route: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method that returns only waypoints.
        
        Args:
            obs: (batch, obs_dim) or (obs_dim,) agent observation
            route: (batch, route_dim) or (route_dim,) route
            
        Returns:
            waypoints: (num_waypoints, waypoint_dim) or (num_waypoints * waypoint_dim,)
        """
        was_1d = obs.dim() == 1
        if was_1d:
            obs = obs.unsqueeze(0)
            route = route.unsqueeze(0)
        
        waypoints, _ = self.forward(obs, route)
        
        if was_1d:
            waypoints = waypoints.squeeze(0)
        
        # Reshape to (num_waypoints, waypoint_dim)
        num_wpts = self.config.num_waypoints
        wpt_dim = self.config.waypoint_dim
        return waypoints.view(-1, num_wpts, wpt_dim)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'config': self.config,
            'model_state': self.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'WaypointPredictionModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'])
        return model


def create_model(
    obs_dim: int = 64,
    route_dim: int = 128,
    num_waypoints: int = 8,
    waypoint_dim: int = 2,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.1,
    latent_dim: int = 128,
) -> WaypointPredictionModel:
    """Factory function to create a model with custom dimensions."""
    config = WaypointModelConfig(
        obs_dim=obs_dim,
        route_dim=route_dim,
        num_waypoints=num_waypoints,
        waypoint_dim=waypoint_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        latent_dim=latent_dim,
    )
    return WaypointPredictionModel(config)


# Example usage and smoke test
if __name__ == "__main__":
    import sys
    
    # Smoke test
    print("Smoke testing WaypointPredictionModel...", file=sys.stderr)
    
    config = WaypointModelConfig()
    model = WaypointPredictionModel(config)
    
    # Dummy输入
    batch = 4
    obs = torch.randn(batch, config.obs_dim)
    route = torch.randn(batch, config.route_dim)
    
    # Forward pass
    waypoints, latent = model.forward(obs, route, return_latent=True)
    
    # Check shapes
    assert waypoints.shape == (batch, config.output_dim), f"Expected {(batch, config.output_dim)}, got {waypoints.shape}"
    assert latent.shape == (batch, config.latent_dim), f"Expected {(batch, config.latent_dim)}, got {latent.shape}"
    
    # Latent for SSL
    ssl_latent = model.get_latent(obs, route)
    assert ssl_latent.shape == (batch, config.latent_dim), f"Expected {(batch, config.latent_dim)}, got {ssl_latent.shape}"
    
    # Save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    model.save(path)
    loaded = WaypointPredictionModel.load(path)
    
    # Verify same output
    out1, _ = model.forward(obs, route)
    out2, _ = loaded.forward(obs, route)
    assert torch.allclose(out1, out2), "Save/load produced different outputs!"
    
    print(f"✓ Model: {config.hidden_dim}d, {config.num_layers}l, {config.latent_dim}d latent", file=sys.stderr)
    print(f"✓ Forward: obs ({obs.shape}) + route ({route.shape}) -> waypoints ({waypoints.shape})", file=sys.stderr)
    print(f"✓ SSL latent: {latent.shape}", file=sys.stderr)
    print(f"✓ Save/load: verified", file=sys.stderr)
    print("PASS", file=sys.stderr)
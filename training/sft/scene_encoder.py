"""Scene Transformer encoder for motion prediction.

This module implements a scene-centric encoder based on:
- Scene Transformer (Google 2021): Agent queries + map queries + temporal attention
- Paper: https://arxiv.org/abs/2106.08417

The encoder produces agent embeddings that can be used with waypoint heads
for motion prediction. Designed to integrate with the existing waypoint BC pipeline.

Usage
-----
from training.sft.scene_encoder import SceneTransformerEncoder, SceneEncoderConfig

config = SceneEncoderConfig(
    num_agents=32,
    num_map_points=256,
    num_history=20,
    hidden_dim=256,
)
encoder = SceneTransformerEncoder(config)
agent_embeddings = encoder(agent_history, map_polylines, agent_types)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SceneEncoderConfig:
    """Configuration for Scene Transformer encoder."""
    # Scene dimensions
    num_agents: int = 32  # Max number of agents
    num_map_points: int = 256  # Max polyline points
    num_history: int = 20  # Historical timesteps
    num_future: int = 30  # Future timesteps to predict
    
    # Model dimensions
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    
    # Agent query
    num_agent_queries: int = 32
    
    # Map processing
    map_hidden_dim: int = 128
    
    # Output
    output_dim: int = 256


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        # x: (B, T, D)
        T = x.shape[1]
        return x + self.pe[:T].unsqueeze(0)


class MapPolylineEncoder(nn.Module):
    """Encodes map polylines (roads, lanes, boundaries).
    
    Each polyline is a sequence of (x, y) points with optional attributes.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # x, y, is_endpoint
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Point-level encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Polyline-level pooling
        self.polyline_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        map_points: torch.Tensor,  # (B, num_polylines, max_points, input_dim)
        polyline_masks: torch.Tensor,  # (B, num_polylines, max_points)
    ) -> torch.Tensor:
        """
        Encode map polylines.
        
        Args:
            map_points: (B, P, M, 3) - B batches, P polylines, M points each
            polyline_masks: (B, P, M) - valid point masks
        
        Returns:
            polyline_embeddings: (B, P, output_dim)
        """
        B, P, M, _ = map_points.shape
        
        # Encode each point
        points_emb = self.point_encoder(map_points)  # (B, P, M, hidden_dim)
        
        # Mask invalid points
        points_emb = points_emb * polyline_masks.unsqueeze(-1)
        
        # Pool to polyline level (mean pooling)
        points_emb = points_emb.sum(dim=2) / (polyline_masks.sum(dim=2, keepdim=True) + 1e-8)
        
        # Final polyline embedding
        polyline_emb = self.polyline_pool(points_emb)  # (B, P, output_dim)
        
        return polyline_emb


class AgentHistoryEncoder(nn.Module):
    """Encodes agent trajectory history."""
    
    def __init__(
        self,
        input_dim: int = 7,  # x, y, heading, speed, type, etc.
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal attention for history
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
    
    def forward(
        self,
        agent_history: torch.Tensor,  # (B, A, T, input_dim)
        agent_masks: torch.Tensor,  # (B, A, T)
    ) -> torch.Tensor:
        """
        Encode agent trajectory history.
        
        Args:
            agent_history: (B, A, T, 7) - B batches, A agents, T timesteps
            agent_masks: (B, A, T) - valid agent/timestep masks
        
        Returns:
            agent_embeddings: (B, A, output_dim)
        """
        B, A, T, _ = agent_history.shape
        
        # Project input
        x = self.input_proj(agent_history)  # (B, A, T, hidden_dim)
        
        # Reshape for temporal attention: (B*A, T, hidden_dim)
        x = x.view(B * A, T, self.hidden_dim)
        masks = agent_masks.view(B * A, T)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Temporal self-attention
        # Create key padding mask (True = invalid/masked)
        key_padding_mask = ~masks.bool()
        x_attn, _ = self.temporal_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.temporal_norm(x + x_attn)
        
        # Global average pooling over time
        # Expand masks for pooling
        masks_exp = masks.unsqueeze(-1).float()  # (B*A, T, 1)
        x = (x * masks_exp).sum(dim=1) / (masks_exp.sum(dim=1) + 1e-8)
        
        # Output projection
        x = self.output_proj(x)  # (B*A, output_dim)
        
        # Reshape back to (B, A, output_dim)
        x = x.view(B, A, self.output_dim)
        
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention between agents and map."""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        query: torch.Tensor,  # (B, A, D)
        key_value: torch.Tensor,  # (B, M, D)
        query_mask: Optional[torch.Tensor] = None,  # (B, A)
        kv_mask: Optional[torch.Tensor] = None,  # (B, M)
    ) -> torch.Tensor:
        """
        Cross-attention from agents to map.
        
        Args:
            query: (B, A, D) - agent queries
            key_value: (B, M, D) - map key/values
            query_mask: (B, A) - valid agent mask
            kv_mask: (B, M) - valid map mask
        
        Returns:
            output: (B, A, D)
        """
        # Cross attention
        attn_out, _ = self.cross_attn(
            query, key_value, key_value,
            key_padding_mask=kv_mask if kv_mask is not None else None
        )
        x = self.norm1(query + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x


class SceneTransformerEncoder(nn.Module):
    """Scene Transformer encoder for motion prediction.
    
    This implements the core Scene Transformer architecture:
    1. Agent history encoding with temporal attention
    2. Map polyline encoding
    3. Cross-attention between agents and map
    4. Agent-to-agent interaction attention
    
    The output can be used with waypoint prediction heads.
    """
    
    def __init__(self, config: SceneEncoderConfig):
        super().__init__()
        
        self.config = config
        
        # Agent history encoder
        self.agent_encoder = AgentHistoryEncoder(
            input_dim=7,  # x, y, heading, speed, type, length, width
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_heads=config.num_heads,
        )
        
        # Map encoder
        self.map_encoder = MapPolylineEncoder(
            input_dim=3,  # x, y, is_endpoint
            hidden_dim=config.map_hidden_dim,
            output_dim=config.output_dim,
        )
        
        # Agent-to-map cross attention layers
        self.agent_to_map_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=config.output_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # Agent-to-agent self attention
        self.agent_attn = nn.MultiheadAttention(
            config.output_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.agent_norm = nn.LayerNorm(config.output_dim)
        
        # FFN for agents
        self.agent_ffn = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim * 4, config.output_dim),
            nn.Dropout(config.dropout),
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.output_dim, config.output_dim)
    
    def forward(
        self,
        agent_history: torch.Tensor,  # (B, A, T, 7)
        map_polylines: torch.Tensor,  # (B, P, M, 3)
        agent_masks: Optional[torch.Tensor] = None,  # (B, A, T)
        polyline_masks: Optional[torch.Tensor] = None,  # (B, P, M)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Scene Transformer encoder.
        
        Args:
            agent_history: (B, A, T, 7) - agent trajectory history
            map_polylines: (B, P, M, 3) - map polylines
            agent_masks: (B, A, T) - valid agent history mask
            polyline_masks: (B, P, M) - valid polyline points mask
        
        Returns:
            Dict with:
                - agent_embeddings: (B, A, output_dim) - agent queries
                - map_embeddings: (B, P, output_dim) - map polylines
                - scene_embedding: (B, output_dim) - global scene context
        """
        B, A, T, _ = agent_history.shape
        _, P, _, _ = map_polylines.shape
        
        # Default masks if not provided
        if agent_masks is None:
            agent_masks = torch.ones(B, A, T, device=agent_history.device, dtype=torch.bool)
        if polyline_masks is None:
            polyline_masks = torch.ones(B, P, map_polylines.shape[2], device=map_polylines.device, dtype=torch.bool)
        
        # Encode agent history
        agent_emb = self.agent_encoder(agent_history, agent_masks)  # (B, A, D)
        
        # Encode map polylines
        map_emb = self.map_encoder(map_polylines, polyline_masks)  # (B, P, D)
        
        # Create polyline mask for attention (B, P)
        polyline_valid = polyline_masks.any(dim=2)  # (B, P)
        
        # Agent-to-map cross attention (multiple layers)
        for layer in self.agent_to_map_layers:
            agent_emb = layer(agent_emb, map_emb, kv_mask=~polyline_valid)
        
        # Agent-to-agent self attention
        # Create valid agent mask
        agent_valid = agent_masks.any(dim=2)  # (B, A)
        
        agent_emb_attn, _ = self.agent_attn(
            agent_emb, agent_emb, agent_emb,
            key_padding_mask=~agent_valid
        )
        agent_emb = self.agent_norm(agent_emb + agent_emb_attn)
        
        # FFN
        agent_emb = self.agent_norm(agent_emb + self.agent_ffn(agent_emb))
        
        # Output projection
        agent_emb = self.output_proj(agent_emb)
        
        # Global scene embedding (mean pooling over valid agents)
        agent_mask_exp = agent_valid.unsqueeze(-1).float()
        scene_emb = (agent_emb * agent_mask_exp).sum(dim=1) / (agent_mask_exp.sum(dim=1) + 1e-8)
        
        return {
            "agent_embeddings": agent_emb,  # (B, A, D)
            "map_embeddings": map_emb,  # (B, P, D)
            "scene_embedding": scene_emb,  # (B, D)
        }
    
    @torch.jit.export
    def get_agent_embedding(
        self,
        agent_history: torch.Tensor,
        map_polylines: torch.Tensor,
        agent_masks: Optional[torch.Tensor] = None,
        polyline_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get agent embeddings for waypoint prediction."""
        outputs = self.forward(agent_history, map_polylines, agent_masks, polyline_masks)
        return outputs["agent_embeddings"]
    
    @torch.jit.export
    def get_scene_embedding(self, *args, **kwargs) -> torch.Tensor:
        """Get global scene embedding."""
        outputs = self.forward(*args, **kwargs)
        return outputs["scene_embedding"]


class SceneTransformerWithWaypointHead(nn.Module):
    """Scene Transformer encoder + waypoint prediction head.
    
    End-to-end model that combines:
    1. Scene Transformer encoder (this module)
    2. Waypoint prediction head (from proposal_waypoint_head.py)
    
    This is the complete prediction model.
    """
    
    def __init__(
        self,
        encoder_config: SceneEncoderConfig,
        use_proposal_head: bool = True,
        num_proposals: int = 5,
        horizon_steps: int = 20,
    ):
        super().__init__()
        
        # Scene encoder
        self.encoder = SceneTransformerEncoder(encoder_config)
        
        # Waypoint head
        if use_proposal_head:
            from training.sft.proposal_waypoint_head import ProposalWaypointHead
            self.waypoint_head = ProposalWaypointHead(
                in_dim=encoder_config.output_dim,
                horizon_steps=horizon_steps,
                num_proposals=num_proposals,
            )
        else:
            # Simple regression head
            self.waypoint_head = nn.Sequential(
                nn.Linear(encoder_config.output_dim, encoder_config.output_dim),
                nn.ReLU(),
                nn.Linear(encoder_config.output_dim, horizon_steps * 2),
            )
        
        self.use_proposal_head = use_proposal_head
    
    def forward(
        self,
        agent_history: torch.Tensor,
        map_polylines: torch.Tensor,
        agent_masks: Optional[torch.Tensor] = None,
        polyline_masks: Optional[torch.Tensor] = None,
        target_agent_idx: int = 0,  # Index of target agent to predict for
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for waypoint prediction.
        
        Args:
            agent_history: (B, A, T, 7)
            map_polylines: (B, P, M, 3)
            agent_masks: (B, A, T)
            polyline_masks: (B, P, M)
            target_agent_idx: Which agent to predict for (default: 0 = ego)
        
        Returns:
            If use_proposal_head:
                - proposals: (B, K, H, 2)
                - scores: (B, K)
            Else:
                - waypoints: (B, H, 2)
        """
        # Encode scene
        outputs = self.encoder(agent_history, map_polylines, agent_masks, polyline_masks)
        
        # Get target agent embedding (e.g., ego vehicle)
        agent_emb = outputs["agent_embeddings"]  # (B, A, D)
        target_emb = agent_emb[:, target_agent_idx]  # (B, D)
        
        # Predict waypoints
        if self.use_proposal_head:
            proposals, scores = self.waypoint_head(target_emb)
            return {
                "proposals": proposals,
                "scores": scores,
                "scene_embedding": outputs["scene_embedding"],
            }
        else:
            waypoints = self.waypoint_head(target_emb)
            waypoints = waypoints.view(-1, 20, 2)  # (B, H, 2)
            return {
                "waypoints": waypoints,
                "scene_embedding": outputs["scene_embedding"],
            }


def demo():
    """Demo of Scene Transformer encoder."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--num-polylines", type=int, default=20)
    parser.add_argument("--history-steps", type=int, default=20)
    parser.add_argument("--map-points-per-line", type=int, default=10)
    args = parser.parse_args()
    
    # Config
    config = SceneEncoderConfig(
        num_agents=args.num_agents,
        num_map_points=args.map_points_per_line,
        num_history=args.history_steps,
        hidden_dim=128,
        output_dim=128,
        num_heads=4,
        num_layers=2,
    )
    
    # Create encoder
    encoder = SceneTransformerEncoder(config)
    
    # Dummy inputs
    B = args.batch_size
    A = args.num_agents
    T = args.history_steps
    P = args.num_polylines
    M = args.map_points_per_line
    
    agent_history = torch.randn(B, A, T, 7)
    map_polylines = torch.randn(B, P, M, 3)
    agent_masks = torch.ones(B, A, T, dtype=torch.bool)
    polyline_masks = torch.ones(B, P, M, dtype=torch.bool)
    
    # Forward
    outputs = encoder(agent_history, map_polylines, agent_masks, polyline_masks)
    
    print(f"Agent embeddings: {outputs['agent_embeddings'].shape}")
    print(f"Map embeddings: {outputs['map_embeddings'].shape}")
    print(f"Scene embedding: {outputs['scene_embedding'].shape}")
    
    # Demo full model with waypoint head
    print("\n--- Full model with waypoint head ---")
    model = SceneTransformerWithWaypointHead(
        config,
        use_proposal_head=True,
        num_proposals=5,
        horizon_steps=20,
    )
    
    outputs = model(agent_history, map_polylines, agent_masks, polyline_masks)
    print(f"Proposals: {outputs['proposals'].shape}")
    print(f"Scores: {outputs['scores'].shape}")


if __name__ == "__main__":
    demo()

"""
VLA (Vision-Language-Action) Planner for Autonomous Driving

Vision-Language-Action model that:
- Takes visual observations
- Optionally accepts language queries/instructions
- Outputs both trajectory and natural language explanation

Usage:
    from training.models.vla_planner import VLADrivingPlanner
    
    planner = VLADrivingPlanner(config)
    
    # Plan and explain
    trajectory, explanation = planner(
        images=front_camera,
        query="Drive safely considering the pedestrians ahead"
    )
    
    print(f"Trajectory: {trajectory}")
    print(f"Explanation: {explanation}")
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class VLAConfig:
    """Configuration for VLA planner."""
    # Vision
    image_channels: int = 3
    image_size: int = 128
    
    # Language
    vocab_size: int = 32000
    embed_dim: int = 256
    max_seq_len: int = 128
    
    # Encoder
    vision_hidden: int = 512
    lang_hidden: int = 256
    fusion_hidden: int = 512
    
    # Trajectory
    trajectory_horizon: int = 20
    waypoint_dim: int = 3  # x, y, heading
    
    # Explainer
    explanation_max_len: int = 50
    
    # Fusion type
    fusion_type: str = "cross_attention"  # "cross_attention" or "concat"


# ============================================================================
# Vision Encoder
# ============================================================================

class VisionEncoder(nn.Module):
    """
    Encode images to visual features.
    
    Uses CNN backbone + spatial pooling.
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        
        # CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(config.image_channels, 32, 4, stride=2),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),  # 8x8
            nn.ReLU(),
            nn.Conv2d(256, config.vision_hidden, 4, stride=2),  # 4x4
            nn.ReLU(),
        )
        
        # Spatial pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Project to common space
        self.project = nn.Linear(config.vision_hidden, config.fusion_hidden)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] or [B, N, C, H, W] (multi-camera)
        Returns:
            visual_features: [B, fusion_hidden]
        """
        # Handle multi-camera
        if images.ndim == 5:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            features = self.backbone(images)
            features = self.pool(features).squeeze(-1).squeeze(-1)
            features = features.view(B, N, -1)
            # Aggregate across cameras (mean)
            features = features.mean(dim=1)
        else:
            features = self.backbone(images)
            features = self.pool(features).squeeze(-1).squeeze(-1)
        
        # Project
        return self.project(features)


# ============================================================================
# Language Encoder
# ============================================================================

class LanguageEncoder(nn.Module):
    """
    Encode text to language features.
    
    Uses transformer encoder.
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.lang_hidden)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.lang_hidden,
            max_len=config.max_seq_len,
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.lang_hidden,
            nhead=4,
            dim_feedforward=config.lang_hidden * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Project to common space
        self.project = nn.Linear(config.lang_hidden, config.fusion_hidden)
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, L] token IDs
            mask: [B, L] attention mask
        Returns:
            language_features: [B, fusion_hidden]
        """
        # Embed + positional
        x = self.embedding(tokens)
        x = self.pos_encoder(x)
        
        # Transform
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Pool (CLS token or mean)
        if tokens.shape[1] > 0:
            # Use CLS token (first token)
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        
        return self.project(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:, : x.size(1)]


# ============================================================================
# Cross-Attention Fusion
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """
    Fuse vision and language with cross-attention.
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        
        # Project to query, key, value
        self.vision_proj = nn.Linear(config.fusion_hidden, config.fusion_hidden)
        self.lang_proj = nn.Linear(config.fusion_hidden, config.fusion_hidden)
        
        # Cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.fusion_hidden, config.fusion_hidden),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden, config.fusion_hidden),
        )
    
    def forward(
        self,
        vision: torch.Tensor,
        language: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision: [B, vision_hidden]
            language: [B, lang_hidden]
        Returns:
            fused: [B, fusion_hidden]
        """
        # Project
        q = self.vision_proj(vision).unsqueeze(1)  # [B, 1, hidden]
        k = self.vision_proj(vision).unsqueeze(1)
        v = self.lang_proj(language).unsqueeze(1)
        
        # Cross-attention
        attn_out, _ = self.attention(q, k, v)
        attn_out = attn_out.squeeze(1)
        
        # Output
        return self.output_proj(attn_out)


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion.
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(config.fusion_hidden * 2, config.fusion_hidden),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden, config.fusion_hidden),
        )
    
    def forward(
        self,
        vision: torch.Tensor,
        language: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate and project."""
        return self.fusion(torch.cat([vision, language], dim=-1))


# ============================================================================
# Trajectory Head
# ============================================================================

class TrajectoryHead(nn.Module):
    """
    Predict vehicle trajectory from fused features.
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        
        self.horizon = config.trajectory_horizon
        self.waypoint_dim = config.waypoint_dim
        
        self.network = nn.Sequential(
            nn.Linear(config.fusion_hidden, config.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.fusion_hidden, config.fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden // 2, config.trajectory_horizon * config.waypoint_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, fusion_hidden]
        Returns:
            trajectory: [B, horizon, waypoint_dim]
        """
        output = self.network(features)
        return output.view(-1, self.horizon, self.waypoint_dim)


# ============================================================================
# Explanation Head
# ============================================================================

class ExplanationHead(nn.Module):
    """
    Generate natural language explanations.
    
    Uses autoregressive decoding.
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        
        self.embed_dim = config.embed_dim
        self.max_len = config.explanation_max_len
        self.vocab_size = config.vocab_size
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.lang_hidden)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.lang_hidden,
            max_len=config.max_seq_len,
        )
        
        # Decoder LSTM
        self.lstm = nn.LSTMCell(
            input_size=config.lang_hidden + config.fusion_hidden,
            hidden_size=config.lang_hidden,
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.lang_hidden, config.vocab_size)
        
        # Start token
        self.register_buffer("start_token", torch.tensor([1]))  # Assume 1 is start
    
    def forward(
        self,
        features: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate or decode explanations.
        
        Args:
            features: [B, fusion_hidden] fused features
            target_ids: [B, L] target token IDs (for training)
            max_len: maximum generation length
            
        Returns:
            Dictionary with:
            - logits: [B, L, vocab_size] output logits
            - generated_ids: [B, max_len] generated token IDs
            - generated_text: List of generated strings
        """
        B = features.shape[0]
        max_len = max_len or self.max_len
        
        if target_ids is not None:
            # Training: compute logits for target
            return self._decode_train(features, target_ids)
        else:
            # Inference: autoregressive generation
            return self._generate(features, max_len)
    
    def _decode_train(
        self,
        features: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training: predict next token given previous."""
        B, L = target_ids.shape
        
        # Embed + positional
        x = self.embedding(target_ids)
        x = self.pos_encoder(x)
        
        # Decode
        h = torch.zeros(B, self.embed_dim, device=features.device)
        c = torch.zeros(B, self.embed_dim, device=features.device)
        
        logits_list = []
        
        for t in range(L - 1):
            # Combine previous token + features
            lstm_input = torch.cat([x[:, t], features], dim=-1)
            h, c = self.lstm(lstm_input, (h, c))
            logit = self.output_proj(h)
            logits_list.append(logit)
        
        logits = torch.stack(logits_list, dim=1)
        
        return {
            "logits": logits,
            "target_ids": target_ids[:, 1:],
        }
    
    def _generate(
        self,
        features: torch.Tensor,
        max_len: int,
    ) -> Dict[str, torch.Tensor]:
        """Inference: autoregressive generation."""
        B = features.shape[0]
        device = features.device
        
        # Start with start token
        generated = torch.full((B, 1), 1, dtype=torch.long, device=device)
        
        h = torch.zeros(B, self.embed_dim, device=device)
        c = torch.zeros(B, self.embed_dim, device=device)
        
        # Expand features for each step
        features_expanded = features.unsqueeze(1)  # [B, 1, hidden]
        
        for t in range(max_len - 1):
            # Embed current
            x = self.embedding(generated[:, -1]).unsqueeze(1)  # [B, 1, hidden]
            
            # LSTM step (use expanded features)
            lstm_input = torch.cat([x.squeeze(1), features], dim=-1)
            h, c = self.lstm(lstm_input, (h, c))
            
            # Output
            logit = self.output_proj(h)
            next_token = logit.argmax(dim=-1)
            
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        
        return {
            "generated_ids": generated,
            "generated_text": self._decode_tokens(generated),
        }
    
    def _decode_tokens(self, ids: torch.Tensor) -> List[str]:
        """Convert token IDs to strings (placeholder)."""
        # In practice, use tokenizer.detokenize()
        return [f"[generated {i}]" for i in range(ids.shape[0])]


# ============================================================================
# Complete VLA Planner
# ============================================================================

class VLADrivingPlanner(nn.Module):
    """
    Vision-Language-Action Planner for Autonomous Driving.
    
    Takes visual observations and optionally language instructions,
    outputs trajectory and explanation.
    
    Usage:
        planner = VLADrivingPlanner(config)
        
        # With language query
        trajectory, explanation = planner(
            images=camera_input,
            query="Drive safely and yield to pedestrians"
        )
        
        # Without language (pure vision)
        trajectory, explanation = planner(images=camera_input)
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(config)
        
        # Language encoder
        self.language_encoder = LanguageEncoder(config)
        
        # Fusion
        if config.fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(config)
        else:
            self.fusion = ConcatFusion(config)
        
        # Trajectory head
        self.trajectory_head = TrajectoryHead(config)
        
        # Explanation head
        self.explainer = ExplanationHead(config)
        
        # Auxiliary heads (for safety)
        self.speed_head = nn.Sequential(
            nn.Linear(config.fusion_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0-1 normalized speed
        )
        
        self.attention_head = nn.Sequential(
            nn.Linear(config.fusion_hidden, config.fusion_hidden),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        images: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        target_explanation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VLA planner.
        
        Args:
            images: [B, C, H, W] or [B, N, C, H, W] camera input
            query: [B, L] optional language tokens
            query_mask: [B, L] language attention mask
            target_explanation: [B, L] explanation for training
            
        Returns:
            Dictionary with:
            - trajectory: [B, horizon, waypoint_dim]
            - speed: [B, 1] predicted speed
            - explanation_logits: training logits (if target provided)
            - generated_explanation: generated text (if no target)
            - visual_features: raw vision features
            - fused_features: after fusion
        """
        # Encode vision
        vision_features = self.vision_encoder(images)
        
        # Encode language (if provided)
        if query is not None:
            lang_features = self.language_encoder(query, query_mask)
            
            # Fuse vision + language
            fused = self.fusion(vision_features, lang_features)
        else:
            # Use only vision
            lang_features = None
            fused = vision_features
        
        # Predict trajectory
        trajectory = self.trajectory_head(fused)
        
        # Predict speed
        speed = self.speed_head(fused)
        
        # Attention weights (for interpretability)
        attention = self.attention_head(fused)
        
        # Generate or decode explanation
        if target_explanation is not None:
            # Training mode
            explanation_result = self.explainer(fused, target_explanation)
            explanation = explanation_result.get("logits")
        else:
            # Inference mode
            explanation_result = self.explainer(fused)
            explanation = explanation_result.get("generated_ids")
        
        return {
            "trajectory": trajectory,
            "speed": speed,
            "attention": attention,
            "explanation": explanation,
            "visual_features": vision_features,
            "language_features": lang_features,
            "fused_features": fused,
        }
    
    def compute_loss(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        speeds: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        explanations: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VLA loss.
        
        Args:
            images: [B, C, H, W] camera input
            actions: [B, horizon, waypoint_dim] expert actions
            speeds: [B, 1] expert speeds
            query: [B, L] optional language
            explanations: [B, L] explanation tokens
        """
        result = self.forward(
            images=images,
            query=query,
            query_mask=query_mask,
            target_explanation=explanations,
        )
        
        # Trajectory loss
        traj_loss = F.mse_loss(result["trajectory"], actions)
        
        # Speed loss
        speed_loss = F.mse_loss(result["speed"], speeds)
        
        # Explanation loss (if provided)
        if explanations is not None and result["explanation"] is not None:
            exp_loss = F.cross_entropy(
                result["explanation"].view(-1, self.config.vocab_size),
                explanations[:, 1:].contiguous().view(-1),
            )
        else:
            exp_loss = torch.tensor(0.0, device=images.device)
        
        # Total loss
        total_loss = traj_loss + 0.5 * speed_loss + 0.1 * exp_loss
        
        return {
            "total_loss": total_loss,
            "trajectory_loss": traj_loss,
            "speed_loss": speed_loss,
            "explanation_loss": exp_loss,
        }
    
    def explain_decision(
        self,
        images: torch.Tensor,
        query: str,
    ) -> Tuple[torch.Tensor, str]:
        """
        Get trajectory with explanation.
        
        Args:
            images: [B, C, H, W] camera input
            query: str language query
            
        Returns:
            trajectory, explanation_text
        """
        # Tokenize query (placeholder)
        tokens = torch.tensor([[1]], dtype=torch.long, device=images.device)  # [1, 1]
        
        # Forward
        result = self.forward(images, query=tokens)
        
        return result["trajectory"], "Generated explanation"


# ============================================================================
# Simple VLA (for testing)
# ============================================================================

class SimpleVLA(nn.Module):
    """
    Simplified VLA for quick testing.
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        
        # Simple vision encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # Project
        self.project = nn.Linear(64, config.fusion_hidden)
        
        # Trajectory head
        self.traj_head = nn.Linear(config.fusion_hidden, config.trajectory_horizon * 3)
        
        # Explainer head
        self.exp_head = nn.Linear(config.fusion_hidden, config.vocab_size)
    
    def forward(self, images: torch.Tensor, query=None):
        """Simple forward."""
        features = self.project(self.encoder(images))
        
        trajectory = self.traj_head(features).view(-1, self.trajectory_horizon, 3)
        explanation = self.exp_head(features)
        
        return {
            "trajectory": trajectory,
            "explanation": explanation,
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = VLAConfig(
        image_channels=3,
        image_size=128,
        vocab_size=32000,
        embed_dim=256,
        trajectory_horizon=20,
    )
    
    # Create model
    model = VLADrivingPlanner(config)
    
    # Dummy data
    B = 4
    images = torch.randn(B, 3, 128, 128)
    actions = torch.randn(B, 20, 3)  # x, y, heading
    speeds = torch.rand(B, 1)  # 0-1
    
    # Test forward
    print("Testing forward pass...")
    result = model(images)
    print(f"  Trajectory shape: {result['trajectory'].shape}")
    print(f"  Speed shape: {result['speed'].shape}")
    print(f"  Attention shape: {result['attention'].shape}")
    
    # Test loss
    print("\nTesting loss...")
    losses = model.compute_loss(images, actions, speeds)
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Trajectory loss: {losses['trajectory_loss'].item():.4f}")
    print(f"  Speed loss: {losses['speed_loss'].item():.4f}")
    
    print("\n✓ VLA planner working!")

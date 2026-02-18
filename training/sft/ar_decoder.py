"""
AR Decoder and AR+CoT Combination for Waypoint Prediction

Implements:
1. ARDecoder: Autoregressive waypoint prediction
2. ARCoTDecoder: AR + Chain of Thought reasoning combined

Architecture Options:
1. AR Only: Predict waypoints one by one, conditioned on previous
2. AR + CoT (parallel): CoT reasoning output + AR waypoints
3. AR + CoT (interleaved): CoT reasoning between waypoint predictions
4. CoT-conditioned AR: CoT conditions the AR generation

Usage:
    from training.sft.ar_decoder import ARDecoder, ARCoTDecoder
    
    # AR decoder
    ar_decoder = ARDecoder(config)
    waypoints = ar_decoder(features)  # [B, T, 3]
    
    # AR + CoT decoder
    ar_cot_decoder = ARCoTDecoder(config)
    waypoints, explanation = ar_cot_decoder(features, cot_embedding)
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
class ARDecoderConfig:
    """Config for AR decoder."""
    # Feature dimensions
    feature_dim: int = 768  # Input feature dimension
    hidden_dim: int = 512  # Hidden dimension
    
    # Waypoint settings
    num_waypoints: int = 20
    waypoint_dim: int = 3  # x, y, heading
    
    # AR settings
    max_decode_steps: int = 20
    use_embedding: bool = True  # Use learned waypoint embeddings
    
    # Architecture
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Inference
    temperature: float = 1.0
    top_k: int = 1  # Greedy decoding
    do_sample: bool = False


@dataclass
class ARCoTConfig:
    """Config for AR + CoT combined decoder."""
    # Feature dimensions
    feature_dim: int = 768
    hidden_dim: int = 512
    cot_dim: int = 768  # CoT embedding dimension
    
    # Waypoint settings
    num_waypoints: int = 20
    waypoint_dim: int = 3
    
    # CoT settings
    cot_encoder_type: str = "lstm"  # "lstm", "bert", "none"
    interleave_cot: bool = False  # Interleave CoT between waypoints
    
    # AR settings
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Output
    include_explanation: bool = True


# ============================================================================
# AR Decoder
# ============================================================================

class ARWaypointEmbedding(nn.Module):
    """Learned embedding for waypoints with positional encoding."""
    
    def __init__(self, config: ARDecoderConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.waypoint_dim = config.waypoint_dim
        
        # Learnable positional embeddings
        self.embedding = nn.Embedding(
            num_embeddings=config.max_decode_steps,
            embedding_dim=config.hidden_dim,
        )
        
        # Project waypoint_dim to hidden_dim
        if config.waypoint_dim != config.hidden_dim:
            self.proj = nn.Linear(config.waypoint_dim, config.hidden_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Embed waypoints for AR decoding.
        
        Args:
            waypoints: [B, T, waypoint_dim] current waypoints
            
        Returns:
            embeddings: [B, T, hidden_dim]
        """
        B, T, D = waypoints.shape
        
        # Project to hidden dim
        waypoints = self.proj(waypoints)
        
        # Add learned positional encoding
        positions = torch.arange(0, T, dtype=torch.long, device=waypoints.device)
        pos_emb = self.embedding(positions)  # [T, hidden_dim]
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, T, hidden_dim]
        
        return waypoints + pos_emb


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with learned embedding support."""
    
    def __init__(self, d_model: int, max_len: int = 5000, learned: bool = False):
        super().__init__()
        self.d_model = d_model
        self.learned = learned
        
        if learned:
            # Learnable positional embeddings
            self.embedding = nn.Embedding(max_len, d_model)
        else:
            # Sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: [B, T, d_model] input
            
        Returns:
            x + pe: [B, T, d_model]
        """
        B, T, D = x.shape
        
        if self.learned:
            # Use learned embeddings
            positions = torch.arange(0, T, dtype=torch.long, device=x.device)
            pos_emb = self.embedding(positions)  # [T, d_model]
            pos_emb = pos_emb.unsqueeze(0)  # [1, T, d_model]
        else:
            # Use sinusoidal PE (truncate to T)
            pos_emb = self.pe[:, :T, :D]  # [1, T, D]
        
        # Expand to batch
        pos_emb = pos_emb.expand(B, -1, -1)
        
        return x + pos_emb


class ARTransformerDecoderLayer(nn.Module):
    """Single layer of AR transformer decoder."""
    
    def __init__(self, config: ARDecoderConfig):
        super().__init__()
        self.config = config
        
        # Self-attention over previous waypoints
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # Cross-attention to conditioning features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # FFN
        self.linear1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.linear2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for AR decoder layer.
        
        Args:
            tgt: [B, T, hidden_dim] target (previous waypoints)
            memory: [B, 1, hidden_dim] conditioning features
            tgt_mask: [T, T] causal mask
            
        Returns:
            output: [B, T, hidden_dim]
        """
        # Self-attention with causal masking
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))
        
        # Cross-attention to conditioning features
        cross_out, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(cross_out))
        
        # FFN
        ffn_out = F.relu(self.linear1(tgt))
        ffn_out = self.linear2(ffn_out)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        
        return tgt


class ARDecoder(nn.Module):
    """
    Autoregressive decoder for waypoint prediction.
    
    Instead of predicting all waypoints in parallel,
    predict one at a time, conditioned on previous predictions.
    
    Architecture:
    ```
    Features ─────┐
                   │
    ┌──────────────▼──────────────┐
    │   AR Transformer Decoder     │
    │                              │
    │   Waypoint 1: features → wp1 │
    │   Waypoint 2: (features+wp1) → wp2 │
    │   Waypoint 3: (features+wp1+wp2) → wp3 │
    │   ...                          │
    └──────────────┬──────────────┘
                   │
                   ▼
            Waypoints [B, T, 3]
    ```
    
    Advantages over parallel decoding:
    1. Captures sequential dependencies
    2. Natural handling of variable-length trajectories
    3. Can stop early (variable horizon)
    
    Usage:
        config = ARDecoderConfig(feature_dim=768, num_waypoints=20)
        decoder = ARDecoder(config)
        
        # Training (parallel, with teacher forcing)
        waypoints = decoder(features)  # [B, 20, 3]
        
        # Inference (autoregressive)
        waypoints = decoder.generate(features)  # [B, T, 3]
    """
    
    def __init__(self, config: ARDecoderConfig):
        super().__init__()
        self.config = config
        
        # Feature projection
        self.feature_proj = nn.Linear(
            config.feature_dim,
            config.hidden_dim
        )
        
        # Waypoint embedding
        if config.use_embedding:
            self.wp_embedding = ARWaypointEmbedding(config)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=config.hidden_dim,
            max_len=config.max_decode_steps,
        )
        
        # AR decoder layers
        decoder_layer = ARTransformerDecoderLayer(config)
        self.decoder = nn.ModuleList([decoder_layer] * config.num_layers)
        
        # Waypoint output head
        self.waypoint_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.waypoint_dim),
        )
        
        # Causal mask for AR decoding
        self.register_buffer(
            'causal_mask',
            self._make_causal_mask(config.max_decode_steps),
        )
    
    def _make_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal mask to prevent attending to future positions."""
        mask = torch.triu(
            torch.ones(size, size, dtype=torch.bool),
            diagonal=1,
        )
        return mask
    
    def forward(
        self,
        features: torch.Tensor,
        waypoints: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for AR decoder.
        
        Training: Use teacher forcing (parallel decoding)
        Inference: Autoregressive generation
        
        Args:
            features: [B, feature_dim] conditioning features
            waypoints: [B, T, waypoint_dim] target waypoints (optional, for training)
            
        Returns:
            Dictionary with:
            - waypoints: [B, T, 3] predicted waypoints
            - embeddings: [B, T, hidden_dim] intermediate embeddings
        """
        B = features.size(0)
        T = self.config.num_waypoints
        
        # Project features to hidden dim
        memory = self.feature_proj(features)  # [B, hidden_dim]
        memory = memory.unsqueeze(1)  # [B, 1, hidden_dim]
        
        if waypoints is not None:
            # Training: teacher forcing with causal masking
            # Embed waypoints
            if hasattr(self, 'wp_embedding'):
                tgt = self.wp_embedding(waypoints)
            else:
                tgt = waypoints  # [B, T, waypoint_dim]
            
            tgt = self.pos_encoding(tgt)
            
            # Apply causal mask
            tgt_mask = self.causal_mask[:T, :T]
            
            # Pass through decoder layers
            for layer in self.decoder:
                tgt = layer(tgt, memory, tgt_mask)
            
            # Predict waypoints
            waypoint_embeddings = tgt
            waypoint_preds = self.waypoint_head(tgt)
            
        else:
            # Inference: autoregressive generation
            waypoint_preds = self.generate(features)
            waypoint_embeddings = None
        
        return {
            'waypoints': waypoint_preds,
            'embeddings': waypoint_embeddings,
        }
    
    def generate(
        self,
        features: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation of waypoints.
        
        Args:
            features: [B, feature_dim] conditioning features
            max_steps: Maximum number of waypoints to generate
            
        Returns:
            waypoints: [B, T, 3] generated waypoints
        """
        B = features.size(0)
        max_steps = max_steps or self.config.num_waypoints
        
        # Project features
        memory = self.feature_proj(features)  # [B, hidden_dim]
        memory = memory.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Initialize: start with a start token (learned or zeros)
        if hasattr(self, 'wp_embedding'):
            # Use learned start token projection
            start_token = torch.zeros(B, 1, self.config.waypoint_dim, device=features.device)
            tgt = self.wp_embedding(start_token)  # [B, 1, hidden_dim]
        else:
            # Use zeros as start token
            tgt = torch.zeros(B, 1, self.config.hidden_dim, device=features.device)
        
        # Generate waypoints one by one
        all_waypoints = []
        
        for t in range(max_steps):
            # Add positional encoding for ALL positions in tgt
            # Each position i gets positional encoding for position i
            current_len = tgt.size(1)  # Number of positions so far
            
            if self.config.use_embedding:
                # Create position indices [0, 1, ..., t] for current positions
                positions = torch.arange(current_len, device=features.device)  # [t+1]
                pos_emb = self.wp_embedding.embedding(positions)  # [t+1, hidden_dim]
                pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, t+1, hidden_dim]
            else:
                # Use sinusoidal positional encoding
                pos_emb = self.pos_encoding.pe[:, :current_len, :self.config.hidden_dim]
                pos_emb = pos_emb.expand(B, -1, -1)  # [B, t, hidden_dim]
            
            tgt_with_pos = tgt + pos_emb
            
            # Apply decoder layer
            for layer in self.decoder:
                tgt_with_pos = layer(tgt_with_pos, memory, tgt_mask=None)
            
            # Predict next waypoint from the last position
            last_hidden = tgt_with_pos[:, -1:, :]  # [B, 1, hidden_dim]
            wp = self.waypoint_head(last_hidden)  # [B, 1, 3]
            all_waypoints.append(wp)
            
            # Append last hidden to tgt for next iteration
            tgt = torch.cat([tgt, last_hidden.detach()], dim=1)  # [B, t+2, hidden_dim]
        
        # Concatenate all waypoints
        waypoints = torch.cat(all_waypoints, dim=1)  # [B, max_steps, 3]
        
        return waypoints


# ============================================================================
# AR + CoT Combined Decoder
# ============================================================================

class ARCoTDecoder(nn.Module):
    """
    Combined AR decoder with Chain of Thought reasoning.
    
    Options for combining AR and CoT:
    
    1. Parallel (default):
       - CoT produces reasoning text embedding
       - AR produces waypoints
       - Both conditioned on same features
    
    2. Interleaved:
       - CoT reasoning between waypoint predictions
       - Each waypoint is preceded by reasoning
    
    3. CoT-conditioned AR:
       - CoT embedding conditions the entire AR process
       - CoT provides additional context for waypoint generation
    
    Architecture:
    ```
    Features ──────────────────────┐
                                  │
    ┌──────────────────────────────▼──────────────────────────┐
    │                  CoT Encoder                            │
    │  (LSTM/BERT on reasoning traces)                       │
    │                  cot_embedding                          │
    └──────────────────────────────┬──────────────────────────┘
                                   │
    ┌──────────────────────────────▼──────────────────────────┐
    │                  AR Decoder                               │
    │  • CoT can be concatenated to features                   │
    │  • Or used as memory for cross-attention               │
    │  • Or interleaved between waypoints                     │
    └──────────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
                        Waypoints + Explanation
    ```
    
    Usage:
        config = ARCoTConfig(
            feature_dim=768,
            cot_dim=768,
            interleave_cot=False,
        )
        decoder = ARCoTDecoder(config)
        
        output = decoder(features, cot_tokens)
        # output['waypoints']: [B, T, 3]
        # output['explanation']: [B, cot_dim] (if include_explanation=True)
    """
    
    def __init__(self, config: ARCoTConfig):
        super().__init__()
        self.config = config
        
        # CoT encoder - always define embedding layer
        self.cot_embedding = nn.Embedding(30000, config.cot_dim)
        
        if config.cot_encoder_type == "lstm":
            self.cot_encoder = nn.LSTM(
                input_size=config.cot_dim,
                hidden_size=config.hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=config.dropout,
            )
        elif config.cot_encoder_type == "bert":
            # Placeholder for BERT encoder
            self.cot_encoder = nn.Linear(config.cot_dim, config.hidden_dim)
        else:
            self.cot_encoder = nn.Identity()
        
        # Feature fusion: combine image features + CoT
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.feature_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        
        # AR decoder
        ar_config = ARDecoderConfig(
            feature_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_waypoints=config.num_waypoints,
            waypoint_dim=config.waypoint_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.ar_decoder = ARDecoder(ar_config)
        
        # Explanation head (output CoT-style explanation)
        if config.include_explanation:
            self.explanation_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.cot_dim),
            )
    
    def forward(
        self,
        features: torch.Tensor,
        cot_input: Optional[torch.Tensor] = None,
        cot_mask: Optional[torch.Tensor] = None,
        waypoints: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for AR + CoT decoder.
        
        Training: Use teacher forcing (parallel decoding with causal mask)
        Inference: Use autoregressive generation
        
        Args:
            features: [B, feature_dim] image features
            cot_input: [B, L] CoT token IDs (optional)
            cot_mask: [B, L] attention mask for CoT (optional)
            waypoints: [B, T, waypoint_dim] target waypoints (optional, for training)
            
        Returns:
            Dictionary with:
            - waypoints: [B, T, 3] predicted waypoints
            - explanation: [B, cot_dim] CoT-style explanation
            - cot_embedding: [B, hidden_dim] encoded CoT
        """
        B = features.size(0)
        
        # Encode CoT
        if cot_input is not None:
            # Check if cot_input is token IDs (int64) or embeddings (float)
            if cot_input.dtype == torch.int64:
                # Embed token IDs first
                cot_input = self.cot_embedding(cot_input)
            
            if isinstance(self.cot_encoder, nn.LSTM):
                cot_out, (h_n, c_n) = self.cot_encoder(cot_input)
                cot_embedding = h_n[-1, :, :]  # Last layer, last hidden state
            elif isinstance(self.cot_encoder, nn.Identity):
                # No CoT encoder, use raw embeddings
                if cot_input.dim() == 3:
                    cot_embedding = cot_input[:, -1, :]
                else:
                    cot_embedding = cot_input
            else:
                cot_embedding = self.cot_encoder(cot_input)
                if cot_embedding.dim() == 3:
                    cot_embedding = cot_embedding[:, -1, :]
        else:
            # Use zeros if no CoT
            cot_embedding = torch.zeros(
                B, self.config.hidden_dim, device=features.device
            )
        
        # Fuse features + CoT
        fused = torch.cat([features, cot_embedding], dim=-1)
        fused = self.feature_fusion(fused)
        
        # AR decode (training with teacher forcing OR inference)
        if waypoints is not None:
            # Training: teacher forcing with causal masking
            ar_output = self.ar_decoder(fused, waypoints)
            waypoints_pred = ar_output['waypoints']
            embeddings = ar_output.get('embeddings')
        else:
            # Inference: autoregressive generation
            waypoints_pred = self.ar_decoder.generate(fused)
            embeddings = None
        
        # Generate explanation from CoT embedding
        if self.config.include_explanation:
            explanation = self.explanation_head(cot_embedding)
        else:
            explanation = None
        
        return {
            'waypoints': waypoints_pred,
            'explanation': explanation,
            'cot_embedding': cot_embedding,
            'embeddings': embeddings,
        }
    
    def generate(
        self,
        features: torch.Tensor,
        cot_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive generation with CoT.
        
        Args:
            features: [B, feature_dim] image features
            cot_input: [B, L] CoT token IDs
            
        Returns:
            Dictionary with:
            - waypoints: [B, T, 3] generated waypoints
            - explanation: [B, cot_dim] CoT-style explanation
            - cot_embedding: [B, hidden_dim] encoded CoT
        """
        return self.forward(features, cot_input, waypoints=None)


# ============================================================================
# RL Integration
# ============================================================================

class ARPPOWaypointHead(nn.Module):
    """
    PPO-compatible head for AR waypoint prediction.
    
    Differences from SFT:
    1. Outputs logits for sampling (stochastic)
    2. Can predict value function
    3. Supports online learning
    
    Usage with PPO:
        head = ARPPOWaypointHead(config)
        waypoints, log_probs, values = head(features)
    """
    
    def __init__(self, config: ARDecoderConfig):
        super().__init__()
        self.config = config
        
        # Predict mean and log_std for each waypoint dimension
        self.mean_head = nn.Linear(config.hidden_dim, config.waypoint_dim)
        self.log_std_head = nn.Linear(config.hidden_dim, config.waypoint_dim)
        
        # Value function
        self.value_head = nn.Linear(config.hidden_dim, 1)
        
        # Log std as learnable parameter
        self.log_std = nn.Parameter(torch.zeros(config.waypoint_dim))
    
    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for PPO.
        
        Args:
            embeddings: [B, T, hidden_dim] waypoint embeddings
            
        Returns:
            waypoints: [B, T, 3] sampled waypoints
            log_probs: [B, T, 3] log probability of sampled waypoints
            values: [B, T, 1] value function estimate
        """
        B, T, D = embeddings.shape
        
        # Predict mean
        means = self.mean_head(embeddings)  # [B, T, 3]
        
        # Predict log_std (clamped for stability)
        log_stds = self.log_std_head(embeddings).clamp(-5, 2)
        log_stds = log_stds + self.log_std.unsqueeze(0).unsqueeze(0)
        
        stds = torch.exp(log_stds)
        
        # Sample waypoints
        noise = torch.randn_like(means)
        waypoints = means + stds * noise
        
        # Log probability
        log_probs = -0.5 * ((waypoints - means) / stds) ** 2 - log_stds - 0.5 * math.log(2 * math.pi)
        log_probs = log_probs.sum(dim=-1, keepdim=True)  # [B, T, 1]
        
        # Value function
        values = self.value_head(embeddings)  # [B, T, 1]
        
        return waypoints, log_probs, values


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test AR decoder
    config = ARDecoderConfig(
        feature_dim=768,
        hidden_dim=512,
        num_waypoints=20,
        waypoint_dim=3,
    )
    
    ar_decoder = ARDecoder(config)
    
    # Test forward pass
    features = torch.randn(4, 768)
    waypoints = torch.randn(4, 20, 3)
    
    output = ar_decoder(features, waypoints)
    print(f"AR Decoder output: {output['waypoints'].shape}")
    
    # Test AR + CoT decoder
    cot_config = ARCoTConfig(
        feature_dim=768,
        hidden_dim=512,
        cot_dim=768,
        num_waypoints=20,
    )
    
    ar_cot = ARCoTDecoder(cot_config)
    cot_tokens = torch.randint(0, 30000, (4, 50))  # [B, L]
    
    output = ar_cot(features, cot_tokens)
    print(f"AR+CoT waypoints: {output['waypoints'].shape}")
    print(f"AR+CoT explanation: {output['explanation'].shape}")
    
    print("\n✓ AR decoders working!")

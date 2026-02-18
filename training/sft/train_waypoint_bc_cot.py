"""
SFT Training with CoT (Chain of Thought) Reasoning

Complete pipeline for training waypoint prediction model with:
1. SSL Image Encoder (pre-trained JEPA)
2. State Encoder (numerical features)
3. CoT Encoder (reasoning traces)
4. Fusion → Decoder → Waypoints

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    SFT + CoT + SSL Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Images ─────────────────────────────────────────────────►       │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SSL Encoder (JEPA pre-trained on Waymo)                 │ │
│  │  • Input: [B, C, H, W] images                          │ │
│  │  • Output: [B, D] latent features                       │ │
│  │  • Frozen during initial training                         │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │                                  │
│  State (speed, heading...) ─┤                                  │
│                              │                                  │
│  CoT Text (reasoning) ──────┤                                  │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  State Encoder (MLP)                                     │ │
│  │  • Input: [B, 16] numerical features                    │ │
│  │  • Output: [B, D] state features                       │ │
│  │  • Trained from scratch                                 │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │                                  │
│              ┌─────────────┼─────────────┐                  │
│              │             │             │                   │
│              ▼             ▼             ▼                   │
│  ┌───────────────────┐ ┌───────────────────┐                 │
│  │ CoT Encoder       │ │                   │                 │
│  │ • LSTM (scratch)  │ │                   │                 │
│  │ • BERT (pretrained)│ │                   │               │
│  │ • None (optional) │ │                   │                 │
│  └─────────┬─────────┘ │                   │                 │
│            │           │                   │                 │
│            └───────────┼───────────────────┘                 │
│                        ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Fusion Layer                                            │ │
│  │  • Concatenate: [image + state + cot] features         │ │
│  │  • Project to hidden_dim                                │ │
│  │  • ReLU activation                                      │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Transformer Decoder                                    │ │
│  │  • Temporal modeling for waypoint sequence              │ │
│  │  • Self-attention over waypoints                       │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Heads                                                  │ │
│  │  • Waypoint Head: [T, 3] → [x, y, heading]           │ │
│  │  • Control Head: [3] → [steer, throttle, brake]       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Usage:
    # Stage 1: Pre-train SSL encoder
    python -m training.pretrain.jepa_masked_objective --data waymo --output ssl_encoder.pt
    
    # Stage 2: Train SFT with CoT
    python -m training.sft.train_waypoint_bc_cot \
        --ssl-encoder ssl_encoder.pt \
        --data-dir waymo \
        --cot-file cot_traces.jsonl \
        --output out/sft_cot \
        --epochs 10 \
        --batch-size 32
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
from dataclasses import dataclass, field
from datetime import datetime
import sys

# Import AR Decoder
from training.sft.ar_decoder import (
    ARDecoder,
    ARCoTDecoder,
    ARDecoderConfig,
    ARCoTConfig,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SFTCoTConfig:
    """
    Configuration for SFT + CoT training pipeline.
    
    This config controls all aspects of the model architecture and training.
    """
    # ========== Model Architecture ==========
    
    # SSL Encoder (Image encoder)
    ssl_encoder_name: str = 'jepa'  # 'jepa', 'resnet50', 'vit'
    ssl_feature_dim: int = 256  # Output dimension of SSL encoder
    
    # State Encoder (Numerical features)
    state_input_dim: int = 16  # speed, heading, accel, etc.
    state_hidden_dim: int = 128
    
    # CoT Encoder (Reasoning traces)
    cot_encoder_type: str = 'lstm'  # 'none', 'lstm', 'bert'
    cot_hidden_dim: int = 256
    cot_max_length: int = 128
    
    # Fusion
    fusion_hidden_dim: int = 512
    
    # ========== AR Decoder Settings ==========
    
    use_ar_decoder: bool = True  # Use autoregressive decoder instead of parallel
    ar_hidden_dim: int = 256
    ar_num_layers: int = 4
    ar_num_heads: int = 8
    ar_max_waypoints: int = 16
    ar_dropout: float = 0.1
    ar_use_embedding: bool = True
    
    # ========== Pre-training Strategy ==========
    
    freeze_ssl_encoder: bool = True  # Freeze SSL encoder during SFT training
    ssl_encoder_lr_multiplier: float = 0.1  # LR multiplier if unfrozen
    
    # ========== Training ==========
    
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Loss weights
    waypoint_loss_weight: float = 1.0
    control_loss_weight: float = 0.1
    
    # ========== Logging ==========
    
    log_interval: int = 100
    save_interval: int = 1000
    
    def __post_init__(self):
        """Validate and set defaults."""
        assert self.cot_encoder_type in ['none', 'lstm', 'bert'], \
            f"Invalid cot_encoder_type: {self.cot_encoder_type}"
    
    def to_ar_decoder_config(self) -> ARDecoderConfig:
        """Convert to ARDecoderConfig."""
        return ARDecoderConfig(
            feature_dim=self.fusion_hidden_dim,
            hidden_dim=self.ar_hidden_dim,
            num_waypoints=self.ar_max_waypoints,
            waypoint_dim=3,
            num_layers=self.ar_num_layers,
            num_heads=self.ar_num_heads,
            dropout=self.ar_dropout,
        )
    
    def to_ar_cot_config(self) -> ARCoTConfig:
        """Convert to ARCoTConfig."""
        return ARCoTConfig(
            feature_dim=self.fusion_hidden_dim,
            hidden_dim=self.ar_hidden_dim,
            cot_dim=self.cot_hidden_dim,
            num_waypoints=self.ar_max_waypoints,
            waypoint_dim=3,
            num_layers=self.ar_num_layers,
            num_heads=self.ar_num_heads,
            dropout=self.ar_dropout,
            cot_encoder_type=self.cot_encoder_type,
            include_explanation=True,
        )


# ============================================================================
# Encoders
# ============================================================================

class SSLEncoder(nn.Module):
    """
    SSL Image Encoder.
    
    Pre-trained on Waymo using JEPA masked objective.
    Extracts high-level image features for driving perception.
    
    Input: [B, C, H, W] images
    Output: [B, D] latent features
    """
    
    def __init__(self, config: SFTCoTConfig, pretrained: bool = True):
        super().__init__()
        self.config = config
        
        # Initialize based on encoder type
        if config.ssl_encoder_name == 'jepa':
            # JEPA encoder (from our pretrain module)
            try:
                from training.pretrain.jepa_masked_objective import JEPAEncoder
                self.encoder = JEPAEncoder(
                    in_channels=3,
                    hidden_dim=config.ssl_feature_dim,
                    patch_size=16,
                )
                if pretrained:
                    # Try to load pretrained weights
                    ssl_path = 'out/pretrain/jepa_encoder.pt'
                    if os.path.exists(ssl_path):
                        self.encoder.load_state_dict(torch.load(ssl_path))
                        print(f"Loaded JEPA encoder from {ssl_path}")
            except ImportError:
                # Fallback to ResNet
                self._init_resnet(config)
                
        elif config.ssl_encoder_name == 'resnet50':
            self._init_resnet(config)
            
        elif config.ssl_encoder_name == 'vit':
            self._init_vit(config)
        
        # Freeze if configured
        self.frozen = False
        if config.freeze_ssl_encoder:
            self.frozen = True
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _init_resnet(self, config: SFTCoTConfig):
        """Initialize ResNet encoder."""
        from torchvision import models
        resnet = models.resnet50(pretrained=True)
        # Remove final pooling and FC
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d(1),
        )
        # Project to feature dim
        self.projection = nn.Linear(2048, config.ssl_feature_dim)
    
    def _init_vit(self, config: SFTCoTConfig):
        """Initialize ViT encoder."""
        # Placeholder for ViT initialization
        # Would use timm or transformers library
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 768, 16, stride=16, padding=14),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(768, config.ssl_feature_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features.
        
        Args:
            images: [B, C, H, W] input images
            
        Returns:
            features: [B, D] latent features
        """
        if self.frozen:
            with torch.no_grad():
                features = self._extract_features(images)
        else:
            features = self._extract_features(images)
        
        return features
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        if self.config.ssl_encoder_name == 'resnet50':
            x = self.encoder(images)
            x = x.view(x.size(0), -1)
            x = self.projection(x)
            return x
        else:
            x = self.encoder(images)
            x = x.view(x.size(0), -1)
            return x
    
    def train(self, mode: bool = True):
        """Override train to handle freezing."""
        super().train(mode)
        if self.frozen:
            # Keep encoder in eval mode even when model.train()
            self.encoder.eval()
        return self


class StateEncoder(nn.Module):
    """
    State Encoder for numerical features.
    
    Processes ego vehicle state and other numerical inputs.
    
    Input: [B, 16] - speed, heading, acceleration, etc.
    Output: [B, state_hidden_dim] - state features
    """
    
    def __init__(self, config: SFTCoTConfig):
        super().__init__()
        self.config = config
        
        self.encoder = nn.Sequential(
            nn.Linear(config.state_input_dim, config.state_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.state_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(config.state_hidden_dim, config.state_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.state_hidden_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state features.
        
        Args:
            state: [B, state_input_dim] numerical features
            
        Returns:
            features: [B, state_hidden_dim] state features
        """
        return self.encoder(state)


class CoTEncoder(nn.Module):
    """
    CoT Encoder for reasoning traces.
    
    Options:
    - none: No CoT encoding
    - lstm: Trainable LSTM encoder
    - bert: Pre-trained BERT encoder
    
    Input: [B, L] token indices
    Output: [B, cot_hidden_dim] reasoning features
    """
    
    def __init__(self, config: SFTCoTConfig):
        super().__init__()
        self.config = config
        self.use_cot = config.cot_encoder_type != 'none'
        
        if self.use_cot:
            if config.cot_encoder_type == 'lstm':
                # Trainable LSTM encoder
                self.encoder = nn.LSTM(
                    input_size=64,  # Character embedding size
                    hidden_size=config.cot_hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.1 if config.cot_encoder_type == 'lstm' else 0,
                )
                # Character embedding
                self.embedding = nn.Embedding(1000, 64)  # 1000 vocab size
                
            elif config.cot_encoder_type == 'bert':
                # Pre-trained BERT encoder
                try:
                    from transformers import BertModel, BertTokenizer
                    self.bert = BertModel.from_pretrained('bert-base-uncased')
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    self.projection = nn.Linear(768, config.cot_hidden_dim)
                except ImportError:
                    print("Warning: transformers not installed, falling back to LSTM")
                    config.cot_encoder_type = 'lstm'
                    self.encoder = nn.LSTM(
                        input_size=64,
                        hidden_size=config.cot_hidden_dim,
                        num_layers=2,
                        batch_first=True,
                    )
                    self.embedding = nn.Embedding(1000, 64)
    
    def forward(self, cot_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Encode CoT tokens.
        
        Args:
            cot_tokens: [B, L] token indices
            
        Returns:
            features: [B, cot_hidden_dim] or None if not using CoT
        """
        if not self.use_cot:
            return None
        
        if self.config.cot_encoder_type == 'bert':
            outputs = self.bert(cot_tokens)
            # Use [CLS] token representation
            hidden = outputs.last_hidden_state[:, 0, :]
            features = self.projection(hidden)
            return features
        
        elif self.config.cot_encoder_type == 'lstm':
            # Embed tokens
            embedded = self.embedding(cot_tokens)  # [B, L, 64]
            # LSTM forward
            output, (h_n, c_n) = self.encoder(embedded)
            # Use last hidden state
            features = h_n[-1]  # [B, cot_hidden_dim]
            return features
        
        return None


# ============================================================================
# Fusion and Decoder
# ============================================================================

class FusionLayer(nn.Module):
    """
    Fusion layer combining image, state, and CoT features.
    
    Concatenates all available features and projects to hidden dimension.
    """
    
    def __init__(self, config: SFTCoTConfig):
        super().__init__()
        
        # Calculate input dimension
        fusion_input_dim = config.ssl_feature_dim + config.state_hidden_dim
        if config.cot_encoder_type != 'none':
            fusion_input_dim += config.cot_hidden_dim
        
        self.fusion_input_dim = fusion_input_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        ssl_features: torch.Tensor,
        state_features: torch.Tensor,
        cot_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse all features.
        
        Args:
            ssl_features: [B, ssl_feature_dim] image features
            state_features: [B, state_hidden_dim] state features
            cot_features: [B, cot_hidden_dim] or None
            
        Returns:
            fused: [B, fusion_hidden_dim] fused features
        """
        # Concatenate all features
        features = [ssl_features, state_features]
        if cot_features is not None:
            features.append(cot_features)
        
        fused = torch.cat(features, dim=-1)
        return self.fusion(fused)


# ============================================================================
# AR Decoder Wrapper (for SFT training)
# ============================================================================

class ARDecoderWrapper(nn.Module):
    """
    Wrapper for ARDecoder that handles both training and inference modes.
    
    This wraps the ARDecoder to provide a simple interface for waypoint prediction.
    """
    
    def __init__(self, config: SFTCoTConfig, use_cot: bool = False):
        super().__init__()
        self.config = config
        self.use_cot = use_cot
        
        if use_cot:
            ar_config = config.to_ar_cot_config()
            self.decoder = ARCoTDecoder(ar_config)
        else:
            ar_config = config.to_ar_decoder_config()
            self.decoder = ARDecoder(ar_config)
    
    def forward(
        self,
        fused_features: torch.Tensor,
        waypoints: Optional[torch.Tensor] = None,
        cot_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through AR decoder.
        
        Args:
            fused_features: [B, fusion_hidden_dim] conditioning features
            waypoints: [B, T, 3] target waypoints (for training, optional)
            cot_input: [B, L] CoT token IDs (for AR+CoT mode)
            
        Returns:
            Dict with:
                waypoints: [B, T, 3] predicted waypoints
                embeddings: [B, T, hidden_dim] intermediate embeddings (training only)
        """
        if self.use_cot and cot_input is not None:
            return self.decoder(fused_features, cot_input)
        else:
            return self.decoder(fused_features, waypoints)
    
    def generate(
        self,
        fused_features: torch.Tensor,
        cot_input: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            fused_features: [B, fusion_hidden_dim] conditioning features
            cot_input: [B, L] CoT token IDs (for AR+CoT mode)
            max_steps: Maximum number of waypoints to generate
            
        Returns:
            waypoints: [B, T, 3] generated waypoints
        """
        if self.use_cot and cot_input is not None:
            return self.decoder.generate(fused_features, cot_input)
        else:
            return self.decoder.generate(fused_features, max_steps)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for temporal waypoint prediction.
    
    Uses self-attention to model temporal dependencies between waypoints.
    """
    
    def __init__(self, config: SFTCoTConfig):
        super().__init__()
        self.config = config
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.fusion_hidden_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.fusion_hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_num_layers,
        )
    
    def forward(
        self,
        fused_features: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode fused features to waypoint sequence.
        
        Args:
            fused_features: [B, fusion_hidden_dim] fused input
            memory: Optional memory for cross-attention
            
        Returns:
            decoder_output: [B, T, fusion_hidden_dim] temporal features
        """
        B = fused_features.size(0)
        T = self.config.decoder_max_waypoints
        
        # Create decoder input: repeat fused features for each waypoint
        tgt = fused_features.unsqueeze(1).expand(-1, T, -1)
        
        # Memory: use fused features as memory
        if memory is None:
            memory = fused_features.unsqueeze(1)
        
        # Self-attention decoder
        output = self.decoder(tgt, memory)
        
        return output


# ============================================================================
# Complete Model
# ============================================================================

class WaypointBCWithCoT(nn.Module):
    """
    Complete SFT + CoT model for waypoint prediction.
    
    Combines:
    1. SSL Encoder (images → features)
    2. State Encoder (numerical → features)
    3. CoT Encoder (reasoning → features)
    4. Fusion (combine all features)
    5. Decoder (temporal modeling)
    6. Heads (waypoints + control)
    
    Args:
        config: SFTCoTConfig with all model settings
        ssl_pretrained: Whether SSL encoder is pre-trained
        
    Example:
        >>> config = SFTCoTConfig(
        ...     ssl_encoder_name='jepa',
        ...     cot_encoder_type='lstm',
        ...     freeze_ssl_encoder=True
        ... )
        >>> model = WaypointBCWithCoT(config)
        >>> waypoints = model(images, state, cot_tokens)
    """
    
    def __init__(self, config: SFTCoTConfig, ssl_pretrained: bool = True):
        super().__init__()
        self.config = config
        
        # 1. SSL Encoder (Image features)
        self.ssl_encoder = SSLEncoder(config, pretrained=ssl_pretrained)
        
        # 2. State Encoder (Numerical features)
        self.state_encoder = StateEncoder(config)
        
        # 3. CoT Encoder (Reasoning traces)
        self.cot_encoder = CoTEncoder(config)
        
        # 4. Fusion Layer
        self.fusion = FusionLayer(config)
        
        # 5. AR Decoder (Autoregressive waypoint prediction)
        use_cot = config.cot_encoder_type != 'none'
        self.ar_decoder = ARDecoderWrapper(config, use_cot=use_cot)
        
        # 6. Prediction Heads (for parallel training mode)
        # Note: AR decoder has its own waypoint_head, but we keep this for compatibility
        self.waypoint_head = nn.Sequential(
            nn.Linear(config.ar_hidden_dim, config.ar_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.ar_hidden_dim, 3),  # x, y, heading
        )
        
        self.control_head = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, 3),  # steering, throttle, brake
        )
    
    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        cot_input_ids: Optional[torch.Tensor] = None,
        cot_attention_mask: Optional[torch.Tensor] = None,
        waypoint_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, C, H, W] input images
            state: [B, state_input_dim] numerical features
            cot_input_ids: [B, L] CoT token indices (optional)
            cot_attention_mask: [B, L] attention mask for CoT (optional)
            waypoint_mask: [B, T] mask for valid waypoints
            
        Returns:
            Dict with:
                waypoints: [B, T, 3] predicted waypoints
                control: [B, 3] predicted control
                features: [B, fusion_hidden_dim] fused features
        """
        # 1. SSL Encoder
        ssl_features = self.ssl_encoder(images)  # [B, ssl_feature_dim]
        
        # 2. State Encoder
        state_features = self.state_encoder(state)  # [B, state_hidden_dim]
        
        # 3. CoT Encoder (only encode token IDs, not embeddings)
        cot_features = None
        if cot_input_ids is not None:
            # CoT encoder handles tokenization internally
            cot_features = self.cot_encoder(cot_input_ids)  # [B, cot_hidden_dim]
        
        # 4. Fusion
        fused_features = self.fusion(ssl_features, state_features, cot_features)
        # [B, fusion_hidden_dim]
        
        # 5. AR Decoder (handles both training with waypoints and inference)
        if self.config.use_ar_decoder:
            # AR decoder handles both training (teacher forcing) and inference
            if self.config.cot_encoder_type != 'none' and cot_input_ids is not None:
                # AR+CoT mode: pass cot_input_ids for CoT encoding
                ar_output = self.ar_decoder(fused_features, waypoints, cot_input_ids)
            else:
                # AR mode without CoT
                ar_output = self.ar_decoder(fused_features, waypoints, None)
            
            waypoints_pred = ar_output['waypoints']
            # embeddings = ar_output.get('embeddings')  # Available during training
        else:
            # Legacy parallel decoder mode
            decoder_output = self.decoder(fused_features)
            waypoints_pred = self.waypoint_head(decoder_output)
        
        # 6. Predict control (from fused features)
        control = self.control_head(fused_features)  # [B, 3]
        
        # Apply mask if provided
        if waypoint_mask is not None:
            waypoints_pred = waypoints_pred * waypoint_mask.unsqueeze(-1)
        
        return {
            'waypoints': waypoints_pred,
            'control': control,
            'features': fused_features,
        }


# ============================================================================
# Training
# ============================================================================

class SFTWithCoTTrainer:
    """
    Trainer for SFT + CoT model.
    
    Handles training loop, loss computation, and logging.
    """
    
    def __init__(
        self,
        model: WaypointBCWithCoT,
        config: SFTCoTConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Loss functions
        self.waypoint_loss_fn = nn.MSELoss()
        self.control_loss_fn = nn.MSELoss()
        
        # Logging
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_wp_loss = 0
        total_ctrl_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['images'].to(self.device)
            state = batch['state'].to(self.device)
            waypoints = batch['waypoints'].to(self.device)
            
            # CoT inputs (now with separate input_ids and attention_mask)
            cot_input_ids = batch.get('cot_input_ids')
            cot_attention_mask = batch.get('cot_attention_mask')
            control = batch.get('control')
            
            if cot_input_ids is not None:
                cot_input_ids = cot_input_ids.to(self.device)
            if cot_attention_mask is not None:
                cot_attention_mask = cot_attention_mask.to(self.device)
            if control is not None:
                control = control.to(self.device)
            
            # Create mask for valid waypoints
            waypoint_mask = (waypoints.abs().sum(dim=-1) > 0).float()
            
            # Forward pass
            outputs = self.model(images, state, cot_input_ids, cot_attention_mask, waypoint_mask)
            
            # Compute losses
            wp_loss = self.waypoint_loss_fn(
                outputs['waypoints'],
                waypoints
            )
            
            ctrl_loss = torch.tensor(0.0, device=self.device)
            if control is not None:
                ctrl_loss = self.control_loss_fn(
                    outputs['control'],
                    control
                )
            
            # Total loss
            loss = (
                self.config.waypoint_loss_weight * wp_loss +
                self.config.control_loss_weight * ctrl_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_wp_loss += wp_loss.item()
            total_ctrl_loss += ctrl_loss.item() if isinstance(ctrl_loss, torch.Tensor) else ctrl_loss
            
            self.global_step += 1
            
            if batch_idx % self.config.log_interval == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"WP: {wp_loss.item():.4f}")
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'waypoint_loss': total_wp_loss / n_batches,
            'control_loss': total_ctrl_loss / n_batches,
        }
    
    def save_checkpoint(self, output_dir: str, epoch: int):
        """Save model checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.config),
            'global_step': self.global_step,
            'epoch': epoch,
        }
        
        path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='SFT Training with CoT Reasoning'
    )
    
    # Model
    parser.add_argument('--ssl-encoder', type=str, default='jepa',
                       choices=['jepa', 'resnet50', 'vit'],
                       help='SSL encoder type')
    parser.add_argument('--ssl-feature-dim', type=int, default=256,
                       help='SSL encoder output dimension')
    parser.add_argument('--cot-encoder', type=str, default='lstm',
                       choices=['none', 'lstm', 'bert'],
                       help='CoT encoder type')
    parser.add_argument('--freeze-ssl', action='store_true', default=True,
                       help='Freeze SSL encoder during training')
    
    # Training
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--cot-file', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    print("SFT + CoT Training Pipeline")
    print("=" * 50)
    print(f"SSL Encoder: {args.ssl_encoder}")
    print(f"CoT Encoder: {args.cot_encoder}")
    print(f"Freeze SSL: {args.freeze_ssl}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output}")
    
    # Create config
    config = SFTCoTConfig(
        ssl_encoder_name=args.ssl_encoder,
        ssl_feature_dim=args.ssl_feature_dim,
        cot_encoder_type=args.cot_encoder,
        freeze_ssl_encoder=args.freeze_ssl,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    print("\nModel Architecture:")
    print(f"  SSL features: {config.ssl_feature_dim}")
    print(f"  State hidden: {config.state_hidden_dim}")
    print(f"  CoT hidden: {config.cot_hidden_dim}")
    print(f"  Fusion hidden: {config.fusion_hidden_dim}")
    print(f"  Max waypoints: {config.decoder_max_waypoints}")
    
    # Placeholder for actual training
    print("\nTo run training:")
    print(f"  python -m training.sft.train_waypoint_bc_cot \\")
    print(f"    --ssl-encoder {args.ssl_encoder} \\")
    print(f"    --cot-encoder {args.cot_encoder} \\")
    print(f"    --data-dir {args.data_dir} \\")
    print(f"    --cot-file {args.cot_file} \\")
    print(f"    --output {args.output}")


if __name__ == "__main__":
    main()

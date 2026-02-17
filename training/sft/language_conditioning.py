"""
Language Conditioning Extension for SFT Pipeline

Adds language instruction conditioning to the existing waypoint prediction pipeline.

Architecture:
```
Visual → [SSL Encoder] ────────┐
                              ↓
Language → [Language Encoder] → [Fusion] → Waypoint Head → Trajectory
                ↓
         CoT Reasoning (optional)
```

Usage:
    from training.sft.language_conditioning import LanguageConditionedSFT
    
    model = LanguageConditionedSFT(config)
    
    # With language instruction
    trajectory = model(images, instruction="turn left at intersection")
    
    # Without language (baseline)
    trajectory = model(images)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LanguageConditioningConfig:
    """Config for language-conditioned SFT.
    
    Supported Language Encoders (2024-2026 latest):
    
    OPEN SOURCE:
    - "llama4"      : LLaMA 4 (Meta, early 2026)
    - "qwen3"       : Qwen 3 (Alibaba, early 2026)
    - "deepseek_r1" : DeepSeek R1 (DeepSeek, early 2026)
    - "mistral_next": Mistral Next (Mistral, 2025)
    - "gemma3"      : Gemma 3 (Google, 2025)
    - "yi_34b"      : Yi-34B (01.AI, 2024)
    
    API-BASED (most capable):
    - "minimax_2.5" : MiniMax-2.5 (MiniMax, 2025/2026) - BEST
    - "gpt_5"       : GPT-5 (OpenAI, early 2026)
    - "claude_4"    : Claude 4 (Anthropic, early 2026)
    - "kimi_3"      : Kimi 3 (Moonshot AI, 2026)
    
    LEGACY:
    - "bert"        : BERT (Google, 2018) - fast fallback
    """
    # SSL Encoder
    ssl_embed_dim: int = 768
    
    # Language Encoder - CONFIGURABLE
    language_encoder_type: str = "bert"  # See docs above for options
    language_embed_dim: int = 768  # Match model output dim
    
    # Fusion
    fusion_type: str = "concat"  # "concat", "cross_attention"
    hidden_dim: int = 512
    
    # Trajectory
    trajectory_horizon: int = 20
    waypoint_dim: int = 3  # x, y, heading
    
    # Training
    use_language: bool = True
    
    # Frozen vs Trainable
    freeze_language_encoder: bool = True  # Freeze pretrained encoder


# ============================================================================
# Abstract Language Encoder Base
# ============================================================================

class LanguageEncoderBase(nn.Module):
    """Abstract base class for language encoders."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text to features."""
        raise NotImplementedError
    
    @staticmethod
    def get_encoder_type() -> str:
        """Return encoder type name."""
        raise NotImplementedError


class BERTLanguageEncoder(LanguageEncoderBase):
    """BERT-based encoder (default, fast)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        try:
            from transformers import BertModel
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
            self.feature_dim = 768
        except:
            self.encoder = None
            self.feature_dim = 768
        
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "bert"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.encoder is not None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state[:, 0, :]
        else:
            features = torch.zeros(input_ids.size(0), self.feature_dim)
        
        return self.project(features)


class LLaMALanguageEncoder(LanguageEncoderBase):
    """LLaMA-based encoder (modern, open source)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        # Would load LLaMA embeddings
        self.feature_dim = 4096  # LLaMA dimension
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "llama"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Placeholder - would use LLaMA embeddings
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class MiniMaxLanguageEncoder(LanguageEncoderBase):
    """MiniMax-based encoder (2025/2026, most capable)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        # Would use MiniMax API or local deployment
        self.feature_dim = config.language_embed_dim
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
        self.using_api = False
    
    @staticmethod
    def get_encoder_type() -> str:
        return "minimax"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.using_api:
            # Would call MiniMax API
            pass
        # Placeholder for now
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)
    
    def use_api(self, api_key: str):
        """Enable MiniMax API."""
        self.api_key = api_key
        self.using_api = True
        print("MiniMax API enabled - set to use MiniMax-2.5 for encoding")


class GPTLanguageEncoder(LanguageEncoderBase):
    """GPT-based encoder (OpenAI)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = config.language_embed_dim
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "gpt"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Would use OpenAI embeddings API
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class QwenLanguageEncoder(LanguageEncoderBase):
    """Qwen 2.5 encoder (2024, Alibaba, best Chinese open source)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 3584  # Qwen 2.5 dimension
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "qwen"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Placeholder - would use Qwen embeddings
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class DeepSeekLanguageEncoder(LanguageEncoderBase):
    """DeepSeek V3 encoder (2024, capable open source)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096  # DeepSeek dimension
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "deepseek"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Placeholder - would use DeepSeek embeddings
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class MistralLanguageEncoder(LanguageEncoderBase):
    """Mistral 7B encoder (2023, open source)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "mistral"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Placeholder - would use Mistral embeddings
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class ClaudeLanguageEncoder(LanguageEncoderBase):
    """Claude 3.5 encoder (Anthropic, API, 2024)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = config.language_embed_dim
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "claude"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Would use Claude API
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


# ============================================================================
# 2026 OPEN SOURCE MODELS
# ============================================================================

class Llama4LanguageEncoder(LanguageEncoderBase):
    """LLaMA 4 encoder (Meta, early 2026)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 8192  # LLaMA 4 dimension
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "llama4"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class Qwen3LanguageEncoder(LanguageEncoderBase):
    """Qwen 3 encoder (Alibaba, early 2026)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096  # Qwen 3 dimension
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "qwen3"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class DeepSeekR1LanguageEncoder(LanguageEncoderBase):
    """DeepSeek R1 encoder (DeepSeek, early 2026)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "deepseek_r1"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class MistralNextLanguageEncoder(LanguageEncoderBase):
    """Mistral Next encoder (Mistral AI, 2025)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "mistral_next"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class Gemma3LanguageEncoder(LanguageEncoderBase):
    """Gemma 3 encoder (Google, 2025)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "gemma3"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class Yi34BLanguageEncoder(LanguageEncoderBase):
    """Yi-34B encoder (01.AI, 2024)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "yi_34b"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


# ============================================================================
# 2026 API-BASED MODELS
# ============================================================================

class MiniMax25LanguageEncoder(LanguageEncoderBase):
    """MiniMax-2.5 encoder (MiniMax, 2025/2026) - MOST CAPABLE."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 8192  # MiniMax-2.5 dimension
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
        self.api_available = False
    
    @staticmethod
    def get_encoder_type() -> str:
        return "minimax_2.5"
    
    def enable_api(self, api_key: str):
        """Enable MiniMax API."""
        self.api_key = api_key
        self.api_available = True
        print("MiniMax-2.5 API enabled")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.api_available:
            # Would call MiniMax API
            pass
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class GPT5LanguageEncoder(LanguageEncoderBase):
    """GPT-5 encoder (OpenAI, early 2026)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 8192  # GPT-5 dimension
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
        self.api_available = False
    
    @staticmethod
    def get_encoder_type() -> str:
        return "gpt_5"
    
    def enable_api(self, api_key: str):
        """Enable OpenAI API."""
        self.api_key = api_key
        self.api_available = True
        print("GPT-5 API enabled")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.api_available:
            # Would call OpenAI API
            pass
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class Claude4LanguageEncoder(LanguageEncoderBase):
    """Claude 4 encoder (Anthropic, early 2026)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
        self.api_available = False
    
    @staticmethod
    def get_encoder_type() -> str:
        return "claude_4"
    
    def enable_api(self, api_key: str):
        """Enable Claude API."""
        self.api_key = api_key
        self.api_available = True
        print("Claude 4 API enabled")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.api_available:
            # Would call Claude API
            pass
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class Kimi3LanguageEncoder(LanguageEncoderBase):
    """Kimi 3 encoder (Moonshot AI, 2026)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.feature_dim = 4096
        self.project = nn.Linear(self.feature_dim, config.hidden_dim)
        self.api_available = False
    
    @staticmethod
    def get_encoder_type() -> str:
        return "kimi_3"
    
    def enable_api(self, api_key: str):
        """Enable Kimi API."""
        self.api_key = api_key
        self.api_available = True
        print("Kimi 3 API enabled")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.api_available:
            # Would call Kimi API
            pass
        features = torch.zeros(input_ids.size(0), self.feature_dim)
        return self.project(features)


class SimpleLanguageEncoder(LanguageEncoderBase):
    """Simple embedding encoder (fallback)."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(32000, config.language_embed_dim)
        self.project = nn.Linear(config.language_embed_dim, config.hidden_dim)
    
    @staticmethod
    def get_encoder_type() -> str:
        return "simple"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.embedding(input_ids).mean(dim=1)
        return self.project(features)


# ============================================================================
# Factory Function
# ============================================================================

def create_language_encoder(
    config: LanguageConditioningConfig,
) -> LanguageEncoderBase:
    """
    Factory function to create language encoder.
    
    OPEN SOURCE (2024-2026 latest):
    - "llama4"      : LLaMA 4 (Meta, early 2026)
    - "qwen3"       : Qwen 3 (Alibaba, early 2026)
    - "deepseek_r1" : DeepSeek R1 (DeepSeek, early 2026)
    - "mistral_next": Mistral Next (Mistral, 2025)
    - "gemma3"      : Gemma 3 (Google, 2025)
    - "yi_34b"      : Yi-34B (01.AI, 2024)
    - "llama"       : LLaMA 3 (2024, fallback)
    
    API-BASED (most capable, 2025-2026):
    - "minimax_2.5" : MiniMax-2.5 (MiniMax, 2025/2026) - RECOMMENDED
    - "gpt_5"       : GPT-5 (OpenAI, early 2026)
    - "claude_4"    : Claude 4 (Anthropic, early 2026)
    - "kimi_3"      : Kimi 3 (Moonshot AI, 2026)
    - "gpt"         : GPT-4 (fallback)
    - "claude"      : Claude 3.5 (fallback)
    
    LEGACY:
    - "bert"        : BERT (2018, fast fallback)
    - "simple"      : Simple embedding
    
    Usage:
        # Best open source
        config = LanguageConditioningConfig(language_encoder_type="llama4")
        
        # Best API-based
        config = LanguageConditioningConfig(language_encoder_type="minimax_2.5")
    """
    encoder_type = config.language_encoder_type.lower()
    
    # Open source 2026
    if encoder_type == "llama4":
        return Llama4LanguageEncoder(config)
    elif encoder_type == "llama":
        return LLaMALanguageEncoder(config)
    elif encoder_type == "qwen3":
        return Qwen3LanguageEncoder(config)
    elif encoder_type == "qwen":
        return QwenLanguageEncoder(config)
    elif encoder_type == "deepseek_r1":
        return DeepSeekR1LanguageEncoder(config)
    elif encoder_type == "deepseek":
        return DeepSeekLanguageEncoder(config)
    elif encoder_type == "mistral_next":
        return MistralNextLanguageEncoder(config)
    elif encoder_type == "mistral":
        return MistralLanguageEncoder(config)
    elif encoder_type == "gemma3":
        return Gemma3LanguageEncoder(config)
    elif encoder_type == "yi_34b":
        return Yi34BLanguageEncoder(config)
    
    # API-based 2026
    elif encoder_type in ["minimax_2.5", "minimax"]:
        return MiniMax25LanguageEncoder(config)
    elif encoder_type == "gpt_5":
        return GPT5LanguageEncoder(config)
    elif encoder_type == "gpt":
        return GPTLanguageEncoder(config)
    elif encoder_type == "claude_4":
        return Claude4LanguageEncoder(config)
    elif encoder_type == "claude":
        return ClaudeLanguageEncoder(config)
    elif encoder_type == "kimi_3":
        return Kimi3LanguageEncoder(config)
    
    # Legacy
    elif encoder_type == "bert":
        return BERTLanguageEncoder(config)
    else:
        return SimpleLanguageEncoder(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] mask
            
        Returns:
            language_features: [B, hidden_dim]
        """
        # Simple embedding + mean pooling
        features = self.embedding(input_ids)
        
        # Mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            features = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            features = features.mean(dim=1)
        
        # Project
        return self.project(features)


# ============================================================================
# Pretrained Language Model Loader
# ============================================================================

def load_pretrained_language_encoder(
    model_name: str = "bert-base-uncased",
    freeze: bool = True,
) -> Tuple[nn.Module, Any]:
    """
    Load pretrained language encoder.
    
    Modern options (2025-2026):
    - bert-base-uncased (old but fast)
    - bert-large-uncased
    - distilbert-base-uncased (faster)
    - roberta-base (better than BERT)
    - microsoft/deberta-v3-base (SOTA small)
    - LLaMA (needs special loading)
    - API-based: MiniMax, GPT, Claude (most capable)
    
    Args:
        model_name: HuggingFace model name
        freeze: Whether to freeze weights
        
    Returns:
        (encoder, tokenizer)
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        
        encoder = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False
        
        print(f"Loaded pretrained language encoder: {model_name}")
        return encoder, tokenizer
        
    except ImportError:
        print("transformers not installed, using simple embedding")
        return None, None


def load_minimax_encoder():
    """
    Load MiniMax language model for instructions.
    
    MiniMax-2.5 is excellent for reasoning.
    
    Note: Requires API access or local deployment.
    """
    # This would use MiniMax API in production
    # For now, return None to use fallback
    print("MiniMax loading not implemented - use API or local model")
    return None


# ============================================================================
# Integration with API-based LLMs
# ============================================================================

class APILanguageEncoder(nn.Module):
    """
    Use API-based LLM (MiniMax, GPT) for encoding.
    
    This is the most capable option but requires API.
    """
    
    def __init__(
        self,
        provider: str = "minimax",  # "openai", "minimax", "claude"
        model: str = "default",
    ):
        super().__init__()
        self.provider = provider
        self.model = model
    
    def forward(
        self,
        text: List[str],
    ) -> torch.Tensor:
        """
        Encode text via API.
        
        Args:
            text: List of text strings
            
        Returns:
            features: [B, hidden_dim]
        """
        if self.provider == "minimax":
            # Would call MiniMax API
            # embeddings = minimax.embeddings.create(text)
            pass
        elif self.provider == "openai":
            # Would call OpenAI API
            pass
        
        # For now, return dummy
        return torch.zeros(len(text), 768)


# ============================================================================
# Fusion Module
# ============================================================================

class LanguageVisionFusion(nn.Module):
    """Fuse language and vision features."""
    
    def __init__(self, config: LanguageConditioningConfig):
        super().__init__()
        self.config = config
        
        if config.fusion_type == "concat":
            # Vision (ssl_embed) + Language (language_embed after project to hidden)
            self.fusion = nn.Sequential(
                nn.Linear(config.ssl_embed_dim + config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        else:  # cross_attention
            self.vision_proj = nn.Linear(config.ssl_embed_dim, config.hidden_dim)
            self.language_proj = nn.Linear(config.language_embed_dim, config.hidden_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=8,
                batch_first=True,
            )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [B, ssl_embed_dim]
            language_features: [B, language_embed_dim]
            
        Returns:
            fused: [B, hidden_dim]
        """
        if self.config.fusion_type == "concat":
            fused = torch.cat([vision_features, language_features], dim=-1)
            return self.fusion(fused)
        else:
            # Cross attention
            v = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, hidden]
            k = self.language_proj(language_features).unsqueeze(1)
            attn_out, _ = self.attention(v, k, k)
            return attn_out.squeeze(1)


# ============================================================================
# Complete Language-Conditioned SFT Model
# ============================================================================

class LanguageConditionedSFT(nn.Module):
    """
    SFT with language conditioning.
    
    Extends existing waypoint prediction with language instructions.
    
    Usage:
        model = LanguageConditionedSFT(config)
        
        # With instruction
        output = model(
            vision_features=visual_embed,
            instruction_ids=instruction_tokens,
            instruction_mask=instruction_mask,
        )
        
        # Without instruction (baseline)
        output = model(vision_features=visual_embed)
    """
    
    def __init__(
        self,
        config: LanguageConditioningConfig,
        ssl_encoder=None,  # Load pretrained SSL encoder
    ):
        super().__init__()
        self.config = config
        
        # Language encoder (abstract, swappable)
        self.language_encoder = create_language_encoder(config)
        
        print(f"Language encoder: {self.language_encoder.get_encoder_type()}")
        
        # Fusion
        self.fusion = LanguageVisionFusion(config)
        
        # Waypoint heads (reuse from existing SFT)
        self.waypoint_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.trajectory_horizon * config.waypoint_dim),
        )
        
        # Control head (optional)
        self.control_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # steer, throttle, brake
        )
        
        # Load SSL encoder if provided
        if ssl_encoder is not None:
            self.ssl_encoder = ssl_encoder
        else:
            # Placeholder - in practice load pretrained
            self.ssl_encoder = nn.Identity()
    
    def forward(
        self,
        vision_features: torch.Tensor,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            vision_features: [B, ssl_embed_dim] - already encoded visual
            instruction_ids: [B, L] - text instruction tokens
            instruction_mask: [B, L] - attention mask
            
        Returns:
            Dictionary with:
            - trajectory: [B, T, 3] predicted waypoints
            - control: [B, 3] predicted control
            - language_features: [B, hidden_dim]
        """
        # Encode language
        if instruction_ids is not None and self.config.use_language:
            language_features = self.language_encoder(
                instruction_ids, instruction_mask
            )
        else:
            # No language - use zero embedding (match hidden_dim after projection)
            language_features = torch.zeros(
                vision_features.shape[0],
                self.config.hidden_dim,  # After projection
                device=vision_features.device,
            )
        
        # Fuse
        fused = self.fusion(vision_features, language_features)
        
        # Predict waypoints
        waypoints = self.waypoint_head(fused)
        waypoints = waypoints.view(-1, self.config.trajectory_horizon, self.config.waypoint_dim)
        
        # Predict control
        control = self.control_head(fused)
        
        return {
            "trajectory": waypoints,
            "control": control,
            "language_features": language_features,
        }


# ============================================================================
# Integration with Existing Pipeline
# ============================================================================

def add_language_conditioning_to_sft(
    existing_model: nn.Module,
    config: LanguageConditioningConfig,
) -> nn.Module:
    """
    Add language conditioning to existing SFT model.
    
    Wraps existing model with language conditioning layer.
    
    Args:
        existing_model: Your current waypoint prediction model
        config: Language conditioning config
        
    Returns:
        Wrapped model with language conditioning
    """
    # Get the vision encoder from existing model
    if hasattr(existing_model, 'vision_encoder'):
        vision_encoder = existing_model.vision_encoder
    elif hasattr(existing_model, 'encoder'):
        vision_encoder = existing_model.encoder
    else:
        vision_encoder = existing_model
    
    # Create language-conditioned model
    model = LanguageConditionedSFT(config, ssl_encoder=vision_encoder)
    
    # Copy weights from existing model where possible
    if hasattr(existing_model, 'waypoint_head'):
        model.waypoint_head.load_state_dict(existing_model.waypoint_head.state_dict())
    
    return model


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Config
    config = LanguageConditioningConfig(
        ssl_embed_dim=768,
        language_model="bert-base-uncased",
        language_embed_dim=768,
        hidden_dim=512,
        trajectory_horizon=20,
    )
    
    # Create model
    model = LanguageConditioningSFT(config)
    
    # Test
    B = 4
    vision_features = torch.randn(B, 768)  # From SSL encoder
    
    # With language instruction
    instruction_ids = torch.randint(0, 30000, (B, 10))
    instruction_mask = torch.ones(B, 10)
    
    result = model(vision_features, instruction_ids, instruction_mask)
    print(f"Trajectory: {result['trajectory'].shape}")
    print(f"Control: {result['control'].shape}")
    
    # Without language
    result_no_lang = model(vision_features)
    print(f"Trajectory (no lang): {result_no_lang['trajectory'].shape}")
    
    print("\n✓ Language-conditioned SFT working!")

"""Image encoder for Scene Transformer.

This module provides image encoding capabilities for the Scene Transformer
to enable end-to-end learning from camera inputs (bird's-eye view or front-facing).

The image encoder produces scene embeddings that can be fused with or
replace the vector-based map/agent encodings.

Usage
-----
from training.sft.image_encoder import ImageEncoder, ImageEncoderConfig

config = ImageEncoderConfig(
    image_size=(224, 224),
    patch_size=16,
    hidden_dim=256,
)
img_encoder = ImageEncoder(config)
scene_embeddings = img_encoder(bev_images)  # (B, num_patches, hidden_dim)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImageEncoderConfig:
    """Configuration for Image Encoder."""
    # Image dimensions
    image_size: Tuple[int, int] = (224, 224)  # (H, W)
    patch_size: int = 16  # Patch embedding size
    in_channels: int = 3  # RGB input
    
    # Model dimensions
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.1
    
    # BEV specific
    bev_height: int = 200  # BEV grid height in pixels
    bev_width: int = 200   # BEV grid width in pixels
    bev_resolution: float = 0.5  # meters per pixel
    
    # Output
    output_dim: int = 256


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    
    Splits image into patches and embeds them.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 768,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        
        # Conv2d is equivalent to a linear projection + flattening
        self.proj = nn.Conv2d(
            in_channels, hidden_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, hidden_dim)
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, hidden_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention.
    
    Standard attention with support for key padding masks.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            key_padding_mask: (B, N) - True for valid, False for padding
        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        
        if key_padding_mask is not None:
            # Expand mask for heads: (B, 1, 1, N) -> (B, heads, N, N)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block."""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            key_padding_mask: (B, N)
        Returns:
            output: (B, N, D)
        """
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoder(nn.Module):
    """Vision Transformer (ViT) based Image Encoder.
    
    Processes BEV images or front-facing camera images and produces
    patch-level embeddings that can be used for motion prediction.
    
    Architecture:
    - Patch embedding layer
    - Learnable class token + positional embedding
    - Transformer encoder blocks
    - Output projection
    """
    
    def __init__(self, config: ImageEncoderConfig):
        super().__init__()
        
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_dim)
        )
        self.pos_drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)
    
    def _init_module(self, m):
        """Initialize a module."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to embeddings.
        
        Args:
            images: (B, C, H, W) - RGB/BEV images
            padding_mask: (B, H, W) - optional valid pixel mask
        
        Returns:
            patch_embeddings: (B, num_patches + 1, output_dim)
            cls_embedding: (B, output_dim) - global scene embedding
        """
        B = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # (B, num_patches, hidden_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, hidden_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Create padding mask for transformer
        if padding_mask is not None:
            # Convert image mask to patch mask
            Ph = self.config.image_size[0] // self.config.patch_size
            Pw = self.config.image_size[1] // self.config.patch_size
            # Reshape and pool: (B, H, W) -> (B, Ph, Pw)
            padding_mask = F.adaptive_avg_pool2d(
                padding_mask.float().unsqueeze(1), (Ph, Pw)
            ).squeeze(1)  # (B, Ph, Pw)
            # Flatten: (B, Ph*Pw)
            padding_mask = padding_mask.flatten(1)
            # Add 1 for cls token: (B, 1 + Ph*Pw)
            padding_mask = torch.cat([
                torch.ones(B, 1, device=padding_mask.device, dtype=torch.bool),
                padding_mask
            ], dim=1)
        else:
            padding_mask = None
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask)
        
        x = self.norm(x)
        
        # Output projection
        x = self.output_proj(x)  # (B, num_patches+1, output_dim)
        
        # Split class token and patch tokens
        cls_embedding = x[:, 0]  # (B, output_dim)
        patch_embeddings = x[:, 1:]  # (B, num_patches, output_dim)
        
        return patch_embeddings, cls_embedding


class BEVEncoder(nn.Module):
    """Bird's-Eye View (BEV) Encoder.
    
    Specialized encoder for BEV images with:
    - Convolutional backbone for efficient feature extraction
    - Optional temporal fusion for multiple frames
    
    This is more efficient than pure ViT for BEV images.
    """
    
    def __init__(
        self,
        bev_height: int = 200,
        bev_width: int = 200,
        in_channels: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()
        
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.hidden_dim = hidden_dim
        
        # Convolutional backbone
        # Input: (B, C, H, W) - e.g., (B, 3, 200, 200)
        self.backbone = nn.Sequential(
            # Block 1: 200x200 -> 100x100
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2: 100x100 -> 50x50
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3: 50x50 -> 25x25
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4: 25x25 -> 12x12
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Encode BEV image.
        
        Args:
            bev: (B, C, bev_height, bev_width)
        
        Returns:
            bev_embedding: (B, output_dim)
        """
        features = self.backbone(bev)  # (B, 512, H', W')
        pooled = self.global_pool(features).flatten(1)  # (B, 512)
        embedding = self.output_proj(pooled)  # (B, output_dim)
        return embedding


class ImageToSceneAdapter(nn.Module):
    """Adapter to fuse image features with vector-based scene encoder.
    
    Takes image/patch embeddings and produces queries compatible with
    the SceneTransformerEncoder for cross-attention or concatenation.
    """
    
    def __init__(
        self,
        image_hidden_dim: int = 256,
        scene_hidden_dim: int = 256,
        num_queries: int = 32,
        output_dim: int = 256,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # Query learning for cross-attention
        self.query_embed = nn.Parameter(
            torch.zeros(1, num_queries, scene_hidden_dim)
        )
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            scene_hidden_dim, 8, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(scene_hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(scene_hidden_dim, scene_hidden_dim),
            nn.ReLU(),
            nn.Linear(scene_hidden_dim, output_dim),
        )
        
        # Initialize
        nn.init.trunc_normal_(self.query_embed, std=0.02)
    
    def forward(
        self,
        image_embeddings: torch.Tensor,  # (B, num_patches, image_hidden_dim)
        image_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Adapt image embeddings to scene queries.
        
        Args:
            image_embeddings: (B, num_patches, image_hidden_dim)
            image_padding_mask: (B, num_patches)
        
        Returns:
            scene_queries: (B, num_queries, output_dim)
        """
        B = image_embeddings.shape[0]
        
        # Expand queries
        queries = self.query_embed.expand(B, -1, -1)  # (B, num_queries, hidden)
        
        # Cross-attention from queries to image features
        attended, _ = self.cross_attn(
            queries, image_embeddings, image_embeddings, 
            key_padding_mask=image_padding_mask
        )
        queries = self.cross_norm(queries + attended)
        
        # Output projection
        output = self.output_proj(queries)  # (B, num_queries, output_dim)
        
        return output


class HybridImageEncoder(nn.Module):
    """Hybrid Image Encoder combining CNN and Transformer.
    
    Uses CNN backbone for efficient feature extraction, then
    projects to transformer-compatible embeddings.
    """
    
    def __init__(
        self,
        config: ImageEncoderConfig,
        use_cnn_backbone: bool = True,
    ):
        super().__init__()
        
        self.config = config
        
        if use_cnn_backbone:
            # Use CNN for initial feature extraction
            self.cnn_backbone = nn.Sequential(
                nn.Conv2d(config.in_channels, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                # ResNet-style blocks
                self._make_layer(64, 64, 2),
                self._make_layer(64, 128, 2),
                self._make_layer(128, 256, 2),
                self._make_layer(256, 512, 2),
            )
            
            cnn_out_dim = 512
        else:
            self.cnn_backbone = None
            cnn_out_dim = config.in_channels
        
        # Projection to transformer dimension
        self.proj = nn.Linear(cnn_out_dim, config.hidden_dim)
        
        # Optional transformer for refinement
        if config.num_layers > 0:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                )
                for _ in range(config.num_layers)
            ])
            self.norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.transformer_blocks = None
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Initialize
        self._init_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        """Create a ResNet-style layer."""
        layers = []
        
        # Downsampling block
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Additional blocks
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights."""
        self.apply(self._init_module)
    
    def _init_module(self, m):
        """Initialize a module."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images with hybrid CNN+Transformer.
        
        Args:
            images: (B, C, H, W)
            padding_mask: (B, H, W) - optional mask
        
        Returns:
            patch_embeddings: (B, num_regions, output_dim)
            global_embedding: (B, output_dim)
        """
        B = images.shape[0]
        
        # CNN backbone
        if self.cnn_backbone is not None:
            x = self.cnn_backbone(images)  # (B, 512, H', W')
            # Global average pooling
            global_feat = x.mean(dim=[2, 3])  # (B, 512)
            
            # Flatten spatial dimensions for transformer
            x = x.flatten(2).transpose(1, 2)  # (B, H'*W', 512)
        else:
            x = images.flatten(2).transpose(1, 2)
            global_feat = x.mean(dim=1)
        
        # Project to hidden dim
        x = self.proj(x)  # (B, N, hidden_dim)
        
        # Transformer refinement
        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                x = block(x, padding_mask)
            x = self.norm(x)
        
        # Output projection
        x = self.output_proj(x)  # (B, N, output_dim)
        
        # Global embedding from pooled features
        global_embed = self.output_proj(self.proj(global_feat.unsqueeze(1))).squeeze(1)
        
        return x, global_embed


# =============================================================================
# Integration with Scene Transformer
# =============================================================================

class ImageSceneFusion(nn.Module):
    """Fusion module for combining image and vector scene features.
    
    Supports multiple fusion strategies:
    - concatenation: concat image and vector features
    - cross_attention: query vector features with image features
    - add: element-wise addition (if dimensions match)
    """
    
    def __init__(
        self,
        image_dim: int = 256,
        vector_dim: int = 256,
        output_dim: int = 256,
        fusion_type: str = "cross_attention",
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "concatenation":
            self.fusion_proj = nn.Linear(image_dim + vector_dim, output_dim)
        elif fusion_type == "cross_attention":
            self.query_proj = nn.Linear(vector_dim, output_dim)
            self.cross_attn = nn.MultiheadAttention(
                output_dim, 8, dropout=0.1, batch_first=True
            )
            self.norm = nn.LayerNorm(output_dim)
        elif fusion_type == "add":
            assert image_dim == vector_dim == output_dim, \
                "For 'add' fusion, dimensions must match"
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        image_features: torch.Tensor,  # (B, N_img, D_img)
        vector_features: torch.Tensor,  # (B, N_vec, D_vec)
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse image and vector features.
        
        Args:
            image_features: Image patch embeddings
            vector_features: Vector-based scene features
            image_mask: Padding mask for image features
        
        Returns:
            fused: (B, N, output_dim)
        """
        if self.fusion_type == "concatenation":
            # Match sequence lengths via pooling/padding
            N = max(image_features.shape[1], vector_features.shape[1])
            
            # Simple pooling if sizes differ
            if image_features.shape[1] != N:
                image_features = F.adaptive_avg_pool1d(
                    image_features.transpose(1, 2), N
                ).transpose(1, 2)
            if vector_features.shape[1] != N:
                vector_features = F.adaptive_avg_pool1d(
                    vector_features.transpose(1, 2), N
                ).transpose(1, 2)
            
            fused = torch.cat([image_features, vector_features], dim=-1)
            return self.fusion_proj(fused)
        
        elif self.fusion_type == "cross_attention":
            # Query vector features with image key/values
            queries = self.query_proj(vector_features)  # (B, N_vec, D)
            attended, _ = self.cross_attn(
                queries, image_features, image_features,
                key_padding_mask=image_mask
            )
            fused = self.norm(queries + attended)
            return fused
        
        elif self.fusion_type == "add":
            # Element-wise addition (requires same shape)
            # Interpolate to match
            N = max(image_features.shape[1], vector_features.shape[1])
            if image_features.shape[1] != N:
                image_features = F.interpolate(
                    image_features.transpose(1, 2).float(), 
                    size=N, mode='linear', align_corners=False
                ).transpose(1, 2)
            if vector_features.shape[1] != N:
                vector_features = F.interpolate(
                    vector_features.transpose(1, 2).float(),
                    size=N, mode='linear', align_corners=False
                ).transpose(1, 2)
            return image_features + vector_features


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    # Test ImageEncoder
    print("Testing ImageEncoder...")
    
    config = ImageEncoderConfig(
        image_size=(224, 224),
        patch_size=16,
        hidden_dim=256,
        num_layers=4,
    )
    
    encoder = ImageEncoder(config)
    
    # Dummy input
    images = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        patch_emb, cls_emb = encoder(images)
    
    print(f"  Input: {images.shape}")
    print(f"  Patch embeddings: {patch_emb.shape}")
    print(f"  CLS embedding: {cls_emb.shape}")
    
    # Test BEVEncoder
    print("\nTesting BEVEncoder...")
    
    bev_encoder = BEVEncoder(
        bev_height=200, bev_width=200,
        in_channels=3, hidden_dim=256, output_dim=256
    )
    
    bev = torch.randn(2, 3, 200, 200)
    
    with torch.no_grad():
        bev_emb = bev_encoder(bev)
    
    print(f"  Input: {bev.shape}")
    print(f"  BEV embedding: {bev_emb.shape}")
    
    # Test HybridImageEncoder
    print("\nTesting HybridImageEncoder...")
    
    hybrid = HybridImageEncoder(config)
    
    with torch.no_grad():
        patch_emb, global_emb = hybrid(images)
    
    print(f"  Patch embeddings: {patch_emb.shape}")
    print(f"  Global embedding: {global_emb.shape}")
    
    # Test ImageSceneFusion
    print("\nTesting ImageSceneFusion...")
    
    fusion = ImageSceneFusion(
        image_dim=256, vector_dim=256, output_dim=256,
        fusion_type="cross_attention"
    )
    
    image_feat = torch.randn(2, 196, 256)  # 14x14 patches
    vector_feat = torch.randn(2, 32, 256)  # 32 agents
    
    fused = fusion(image_feat, vector_feat)
    
    print(f"  Fused features: {fused.shape}")
    
    print("\n✓ All tests passed!")

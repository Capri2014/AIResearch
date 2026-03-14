"""
SSL Pretrained Model Loader for Waypoint BC Transfer Learning.

Loads SSL pretrained checkpoints (JEPA, contrastive) and provides feature extraction
for downstream waypoint BC training.

Usage:
    from ssl_pretrained_loader import SSLFeatureExtractor, load_ssl_pretrained
    
    # Load SSL model
    ssl_model = load_ssl_pretrained("checkpoints/ssl_pretrain/model.pt")
    
    # Extract features
    extractor = SSLFeatureExtractor(ssl_model)
    features = extractor.extract(image)  # (B, C, H, W) features
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Optional imports
try:
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available, using simple CNN encoder")


@dataclass
class SSLConfig:
    """Configuration for SSL pretrained model."""
    model_type: str = "jepa"  # jepa, contrastive, temporal_contrastive
    feature_dim: int = 256
    backbone: str = "resnet50"
    pretrained_path: Optional[str] = None
    freeze_encoder: bool = True


class SSLEncoder(nn.Module):
    """SSL pretrained encoder wrapper."""
    
    def __init__(self, config: SSLConfig):
        super().__init__()
        self.config = config
        
        # Build encoder based on backbone type
        if not TORCHVISION_AVAILABLE:
            # Fallback simple CNN encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_dim = 256
        elif config.backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_dim = 2048
        elif config.backbone == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_dim = 512
        elif config.backbone == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(
                backbone.features,
                backbone.avgpool
            )
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")
        
        # Projection head to desired feature dim
        if self.feature_dim != config.feature_dim:
            self.projection = nn.Linear(self.feature_dim, config.feature_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze if specified
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            features: (B, feature_dim, H', W') spatial features
        """
        features = self.encoder(x)  # (B, C, 1, 1) for global pool
        features = features.flatten(2).transpose(1, 2)  # (B, C, 1) -> (B, 1, C)
        features = self.projection(features)  # (B, 1, feature_dim)
        return features


class JEPAEncoder(nn.Module):
    """JEPA (Joint Embedding Predictive Architecture) encoder."""
    
    def __init__(self, config: SSLConfig):
        super().__init__()
        self.config = config
        
        # Build encoder
        if not TORCHVISION_AVAILABLE:
            # Fallback simple CNN encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_dim = 256
        elif config.backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_dim = 2048
        else:
            from torchvision.models import resnet34, ResNet34_Weights
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_dim = 512
        
        # JEPA predictor (predicts target from context)
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Linear(512, config.feature_dim)
        )
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            embeddings: (B, feature_dim) context embeddings
            predictions: (B, feature_dim) predicted target embeddings
        """
        features = self.encoder(x)
        features = features.flatten(1)  # (B, C)
        
        embeddings = features
        predictions = self.predictor(features)
        
        return embeddings, predictions


class SSLFeatureExtractor:
    """Feature extractor using SSL pretrained models."""
    
    def __init__(self, model: nn.Module, model_type: str = "jepa"):
        self.model = model
        self.model_type = model_type
        self.model.eval()
    
    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            images: (B, 3, H, W) input images
        Returns:
            features: (B, feature_dim, H', W') or (B, feature_dim) depending on model
        """
        if isinstance(self.model, JEPAEncoder):
            embeddings, _ = self.model(images)
            return embeddings
        else:
            return self.model(images)
    
    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """Extract features in batches."""
        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            features = self.extract(batch)
            all_features.append(features)
        return torch.cat(all_features, dim=0)


def load_ssl_pretrained(
    checkpoint_path: str,
    model_type: str = "jepa",
    freeze: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """
    Load SSL pretrained checkpoint.
    
    Args:
        checkpoint_path: Path to SSL checkpoint
        model_type: Type of SSL model (jepa, contrastive, temporal_contrastive)
        freeze: Whether to freeze encoder weights
        device: Device to load model on
    
    Returns:
        Loaded SSL model
    """
    config = SSLConfig(
        model_type=model_type,
        pretrained_path=checkpoint_path,
        freeze_encoder=freeze
    )
    
    if model_type == "jepa":
        model = JEPAEncoder(config)
    else:
        model = SSLEncoder(config)
    
    # Load checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading SSL checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded SSL model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random init")
    
    model = model.to(device)
    return model


class BCWithSSLEncoder(nn.Module):
    """
    Waypoint BC model with SSL pretrained encoder initialization.
    
    This combines an SSL pretrained encoder with a waypoint prediction head,
    optionally loading pretrained weights for the encoder.
    """
    
    def __init__(
        self,
        ssl_model: Optional[nn.Module] = None,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
        hidden_dim: int = 256,
        feature_dim: int = 256
    ):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Use SSL encoder or create new one
        if ssl_model is not None:
            self.encoder = ssl_model
            if hasattr(ssl_model, 'feature_dim'):
                encoder_dim = ssl_model.feature_dim
            else:
                encoder_dim = feature_dim
        else:
            config = SSLConfig(feature_dim=feature_dim)
            self.encoder = SSLEncoder(config)
            encoder_dim = feature_dim
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * waypoint_dim)
        )
        
        # Initialize waypoint head
        self._init_weights()
    
    def _init_weights(self):
        for m in self.waypoint_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from images.
        
        Args:
            images: (B, 3, H, W) input images
        Returns:
            waypoints: (B, num_waypoints, waypoint_dim) predicted waypoints
        """
        # Extract features
        features = self.encoder.extract(images) if hasattr(self.encoder, 'extract') else self.encoder(images)
        
        # Handle tuple return from JEPA (embeddings, predictions)
        if isinstance(features, tuple):
            features = features[0]
        
        # If features are (B, C), add spatial dimension
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # (B, 1, C)
        
        # Predict waypoints
        output = self.waypoint_head(features)  # (B, 1, num_waypoints * waypoint_dim)
        output = output.reshape(-1, self.num_waypoints, self.waypoint_dim)
        
        return output


def create_bc_with_ssl_pretrained(
    checkpoint_path: Optional[str] = None,
    model_type: str = "jepa",
    num_waypoints: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> BCWithSSLEncoder:
    """
    Create BC model with SSL pretrained encoder.
    
    Args:
        checkpoint_path: Path to SSL checkpoint (optional)
        model_type: Type of SSL model
        num_waypoints: Number of waypoints to predict
        device: Device to load model on
    
    Returns:
        BC model with SSL pretrained encoder
    """
    ssl_model = None
    if checkpoint_path:
        ssl_model = load_ssl_pretrained(checkpoint_path, model_type=model_type, freeze=True, device=device)
    
    model = BCWithSSLEncoder(
        ssl_model=ssl_model,
        num_waypoints=num_waypoints
    )
    
    return model.to(device)


# Smoke test
if __name__ == "__main__":
    print("Testing SSL Pretrained Loader...")
    
    # Test config
    config = SSLConfig(model_type="jepa", feature_dim=256, backbone="resnet34")
    
    # Test JEPA encoder
    jepa = JEPAEncoder(config)
    dummy_input = torch.randn(2, 3, 224, 224)
    embeddings, predictions = jepa(dummy_input)
    print(f"JEPA embeddings shape: {embeddings.shape}")
    print(f"JEPA predictions shape: {predictions.shape}")
    
    # Test SSL encoder
    encoder = SSLEncoder(config)
    features = encoder(dummy_input)
    print(f"SSL encoder features shape: {features.shape}")
    
    # Test feature extractor
    extractor = SSLFeatureExtractor(jepa, model_type="jepa")
    extracted = extractor.extract(dummy_input)
    print(f"Extracted features shape: {extracted.shape}")
    
    # Test BC with SSL encoder
    bc_model = BCWithSSLEncoder(ssl_model=jepa, num_waypoints=8)
    waypoints = bc_model(dummy_input)
    print(f"BC waypoints shape: {waypoints.shape}")
    
    # Test factory function
    bc_model2 = create_bc_with_ssl_pretrained(num_waypoints=8)
    waypoints2 = bc_model2(dummy_input)
    print(f"BC model (no checkpoint) waypoints shape: {waypoints2.shape}")
    
    print("✓ All tests passed!")

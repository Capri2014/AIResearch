#!/usr/bin/env python3
"""Tests for BEV SSL Waypoint Predictor."""

import torch
import pytest
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.bc.bev_ssl_waypoint_predictor import (
    WaypointPredictionHead,
    WaypointHeadConfig,
    BEVSSLWaypointPredictor,
    WaypointBCLoss,
    WaypointPredictorTrainer,
    create_waypoint_predictor,
    load_bev_encoder_from_checkpoint,
)


class TestWaypointPredictionHead:
    """Test WaypointPredictionHead."""
    
    def test_creation(self):
        """Test head creation."""
        head = WaypointPredictionHead(
            encoder_dim=128,
            num_waypoints=8,
            waypoint_dim=2,
        )
        assert head.encoder_dim == 128
        assert head.num_waypoints == 8
    
    def test_forward(self):
        """Test forward pass."""
        head = WaypointPredictionHead(
            encoder_dim=128,
            num_waypoints=8,
        )
        
        # Single frame input
        x = torch.randn(4, 128)
        out = head(x)
        
        assert "waypoints" in out
        assert out["waypoints"].shape == (4, 8, 2)
    
    def test_forward_with_temporal(self):
        """Test forward with temporal input."""
        head = WaypointPredictionHead(
            encoder_dim=128,
            num_waypoints=8,
            use_temporal=True,
            temporal_history=3,
        )
        
        # Temporal input
        x = torch.randn(4, 3, 128)
        out = head(x)
        
        assert "waypoints" in out
        assert out["waypoints"].shape == (4, 8, 2)
    
    def test_forward_with_speed(self):
        """Test forward with speed prediction."""
        head = WaypointPredictionHead(
            encoder_dim=128,
            num_waypoints=8,
            predict_speed=True,
            speed_bins=10,
        )
        
        x = torch.randn(4, 128)
        out = head(x)
        
        assert "waypoints" in out
        assert "speed_logits" in out
        assert out["speed_logits"].shape == (4, 10)


class TestBEVSSLWaypointPredictor:
    """Test combined predictor."""
    
    def test_creation(self):
        """Test predictor creation."""
        from training.pretrain.bev_encoder import create_bev_encoder
        
        encoder = create_bev_encoder(in_channels=64, encoder_dim=128)
        head = WaypointPredictionHead(encoder_dim=128, num_waypoints=8)
        
        predictor = BEVSSLWaypointPredictor(
            bev_encoder=encoder,
            waypoint_head=head,
            freeze_encoder=True,
        )
        
        assert predictor.freeze_encoder is True
    
    def test_forward_with_images(self):
        """Test forward with images."""
        from training.pretrain.bev_encoder import create_bev_encoder
        
        encoder = create_bev_encoder(in_channels=64, encoder_dim=128)
        head = WaypointPredictionHead(encoder_dim=128, num_waypoints=8)
        
        predictor = BEVSSLWaypointPredictor(
            bev_encoder=encoder,
            waypoint_head=head,
            freeze_encoder=True,
        )
        
        # Mock images (B, C, H, W)
        images = torch.randn(2, 3, 256, 256)
        out = predictor(images=images)
        
        assert "waypoints" in out
        assert out["waypoints"].shape == (2, 8, 2)
    
    def test_freeze_encoder(self):
        """Test encoder freezing."""
        from training.pretrain.bev_encoder import create_bev_encoder
        
        encoder = create_bev_encoder(in_channels=64, encoder_dim=128)
        head = WaypointPredictionHead(encoder_dim=128, num_waypoints=8)
        
        predictor = BEVSSLWaypointPredictor(
            bev_encoder=encoder,
            waypoint_head=head,
            freeze_encoder=True,
        )
        
        # Check encoder is frozen
        for param in encoder.parameters():
            assert param.requires_grad is False
        
        # Check head is trainable
        for param in head.parameters():
            assert param.requires_grad is True


class TestWaypointBCLoss:
    """Test loss function."""
    
    def test_waypoint_loss(self):
        """Test waypoint loss computation."""
        criterion = WaypointBCLoss()
        
        pred = torch.randn(4, 8, 2)
        target = torch.randn(4, 8, 2)
        
        losses = criterion(pred_waypoints=pred, target_waypoints=target)
        
        assert "waypoint_loss" in losses
        assert "total_loss" in losses
        assert losses["waypoint_loss"].item() >= 0
    
    def test_waypoint_with_speed(self):
        """Test loss with speed prediction."""
        criterion = WaypointBCLoss(
            waypoint_loss_weight=1.0,
            speed_loss_weight=0.1,
        )
        
        pred = torch.randn(4, 8, 2)
        target = torch.randn(4, 8, 2)
        pred_speed = torch.randn(4, 10)
        target_speed = torch.randint(0, 10, (4,))
        
        losses = criterion(
            pred_waypoints=pred,
            target_waypoints=target,
            pred_speed=pred_speed,
            target_speed=target_speed,
        )
        
        assert "waypoint_loss" in losses
        assert "speed_loss" in losses
        assert "total_loss" in losses


class TestCreateWaypointPredictor:
    """Test predictor creation function."""
    
    def test_create_new(self):
        """Test creating predictor without checkpoint."""
        predictor, config = create_waypoint_predictor(
            ssl_checkpoint_path=None,
            num_waypoints=8,
            output_dir="/tmp/test_predictor",
            freeze_encoder=True,
        )
        
        assert predictor is not None
        assert config.num_waypoints == 8
    
    def test_create_with_frozen_encoder(self):
        """Test predictor with frozen encoder."""
        predictor, config = create_waypoint_predictor(
            num_waypoints=8,
            freeze_encoder=True,
        )
        
        assert predictor.freeze_encoder is True


def test_import():
    """Test module imports."""
    from training.bc.bev_ssl_waypoint_predictor import (
        WaypointPredictionHead,
        WaypointHeadConfig,
        BEVSSLWaypointPredictor,
        WaypointBCLoss,
        create_waypoint_predictor,
    )
    assert WaypointPredictionHead is not None


def test_config():
    """Test config defaults."""
    config = WaypointHeadConfig(
        num_waypoints=8,
        output_dir="test",
    )
    
    assert config.num_waypoints == 8
    assert config.encoder_dim == 128
    assert config.hidden_dims == [256, 128]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

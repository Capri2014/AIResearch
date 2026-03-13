"""
Unit tests for BEV encoder components.
"""

import torch
import pytest
from bev_encoder import (
    BEVEncoderConfig,
    LidarToBEV,
    CameraToBEV,
    BEVEncoder,
    create_bev_encoder,
    FusionType,
)


class TestLidarToBEV:
    """Tests for LiDAR to BEV conversion."""
    
    def test_initialization(self):
        """Test LidarToBEV initialization."""
        lidar_to_bev = LidarToBEV(
            bev_resolution=0.5,
            bev_size=(200, 200),
            num_height_bins=4,
            lidar_range=50.0,
            use_intensity=True,
        )
        assert lidar_to_bev.bev_resolution == 0.5
        assert lidar_to_bev.bev_height == 200
        assert lidar_to_bev.bev_width == 200
    
    def test_forward_empty_points(self):
        """Test with empty point cloud."""
        lidar_to_bev = LidarToBEV()
        points = torch.zeros(0, 3)
        bev = lidar_to_bev(points.unsqueeze(0))  # Add batch dim
        assert bev.shape == (1, 8, 200, 200)  # 4 height bins * 2 channels
    
    def test_forward_basic_points(self):
        """Test with basic point cloud."""
        lidar_to_bev = LidarToBEV(
            bev_resolution=1.0,
            bev_size=(100, 100),
            num_height_bins=2,
            lidar_range=50.0,
            use_intensity=False,
        )
        
        # Simple point cloud: (x, y, z)
        points = torch.tensor([
            [0, 0, 0],
            [1, 1, 0.5],
            [2, 2, 1.0],
            [-1, -1, 0.2],
        ])
        
        bev = lidar_to_bev(points.unsqueeze(0))
        assert bev.shape == (1, 2, 100, 100)  # 2 height bins * 1 channel
    
    def test_forward_with_intensity(self):
        """Test with intensity channel."""
        lidar_to_bev = LidarToBEV(use_intensity=True)
        
        # Point cloud with intensity: (x, y, z, intensity)
        points = torch.tensor([
            [0, 0, 0, 0.5],
            [1, 1, 0.5, 0.8],
        ])
        
        bev = lidar_to_bev(points.unsqueeze(0))
        assert bev.shape[1] == 8  # 4 height bins * 2 channels (density + intensity)


class TestCameraToBEV:
    """Tests for Camera to BEV conversion."""
    
    def test_initialization(self):
        """Test CameraToBEV initialization."""
        camera_to_bev = CameraToBEV(
            camera_channels=3,
            feature_dim=256,
            bev_size=(200, 200),
            bev_resolution=0.5,
            camera_fov=90.0,
        )
        assert camera_to_bev.feature_dim == 256
        assert camera_to_bev.bev_height == 200
    
    def test_forward(self):
        """Test camera to BEV forward pass."""
        camera_to_bev = CameraToBEV(
            camera_channels=3,
            feature_dim=128,
            bev_size=(100, 100),
        )
        
        # Random camera image
        images = torch.randn(2, 3, 224, 224)
        bev = camera_to_bev(images)
        
        assert bev.shape == (2, 128, 100, 100)


class TestBEVEncoder:
    """Tests for unified BEV encoder."""
    
    def test_initialization_camera_only(self):
        """Test initialization with camera only."""
        config = BEVEncoderConfig(
            input_types=["camera"],
            bev_resolution=0.5,
            bev_size=(100, 100),
            feature_dim=256,
        )
        encoder = BEVEncoder(config)
        assert "camera" in encoder.encoders
        assert "lidar" not in encoder.encoders
    
    def test_initialization_lidar_only(self):
        """Test initialization with LiDAR only."""
        config = BEVEncoderConfig(
            input_types=["lidar"],
            bev_resolution=0.5,
            bev_size=(100, 100),
            feature_dim=256,
        )
        encoder = BEVEncoder(config)
        assert "lidar" in encoder.encoders
        assert "camera" not in encoder.encoders
    
    def test_initialization_both(self):
        """Test initialization with both modalities."""
        config = BEVEncoderConfig(
            input_types=["camera", "lidar"],
            bev_resolution=0.5,
            bev_size=(100, 100),
            feature_dim=256,
            fusion_type=FusionType.CONCAT,
        )
        encoder = BEVEncoder(config)
        assert "camera" in encoder.encoders
        assert "lidar" in encoder.encoders
    
    def test_encode_camera_only(self):
        """Test encoding with camera only."""
        encoder = create_bev_encoder(
            input_types=["camera"],
            bev_size=(100, 100),
            feature_dim=128,
        )
        
        images = torch.randn(2, 3, 224, 224)
        bev = encoder.encode(cameras=images)
        
        assert bev.shape == (2, 128, 100, 100)
    
    def test_encode_lidar_only(self):
        """Test encoding with LiDAR only."""
        encoder = create_bev_encoder(
            input_types=["lidar"],
            bev_size=(100, 100),
            feature_dim=128,
        )
        
        points = torch.randn(2, 1000, 3)
        bev = encoder.encode(lidar_points=points)
        
        assert bev.shape == (2, 128, 100, 100)
    
    def test_encode_both_modalities(self):
        """Test encoding with both modalities."""
        encoder = create_bev_encoder(
            input_types=["camera", "lidar"],
            bev_size=(100, 100),
            feature_dim=128,
            fusion_type="concat",
        )
        
        images = torch.randn(2, 3, 224, 224)
        points = torch.randn(2, 1000, 3)
        
        bev = encoder.encode(cameras=images, lidar_points=points)
        
        assert bev.shape == (2, 128, 100, 100)
    
    def test_fusion_concat(self):
        """Test concat fusion."""
        encoder = create_bev_encoder(
            input_types=["camera", "lidar"],
            bev_size=(100, 100),
            feature_dim=128,
            fusion_type="concat",
        )
        
        images = torch.randn(2, 3, 224, 224)
        points = torch.randn(2, 1000, 3)
        
        bev = encoder.encode(cameras=images, lidar_points=points)
        assert bev.shape == (2, 128, 100, 100)
    
    def test_fusion_sum(self):
        """Test sum fusion."""
        encoder = create_bev_encoder(
            input_types=["camera", "lidar"],
            bev_size=(100, 100),
            feature_dim=128,
            fusion_type="sum",
        )
        
        images = torch.randn(2, 3, 224, 224)
        points = torch.randn(2, 1000, 3)
        
        bev = encoder.encode(cameras=images, lidar_points=points)
        assert bev.shape == (2, 128, 100, 100)
    
    def test_fusion_attention(self):
        """Test attention fusion."""
        encoder = create_bev_encoder(
            input_types=["camera", "lidar"],
            bev_size=(50, 50),  # Smaller for speed
            feature_dim=64,
            fusion_type="attention",
        )
        
        images = torch.randn(2, 3, 64, 64)
        points = torch.randn(2, 500, 3)
        
        bev = encoder.encode(cameras=images, lidar_points=points)
        assert bev.shape == (2, 64, 50, 50)
    
    def test_get_bev_image(self):
        """Test BEV visualization image generation."""
        encoder = create_bev_encoder(
            input_types=["camera", "lidar"],
            bev_size=(100, 100),
            feature_dim=256,
        )
        
        # Get some BEV features
        images = torch.randn(2, 3, 224, 224)
        points = torch.randn(2, 1000, 3)
        bev_features = encoder.encode(cameras=images, lidar_points=points)
        
        # Get visualization
        bev_image = encoder.get_bev_image(bev_features)
        
        assert bev_image.shape == (2, 3, 100, 100)
        assert bev_image.min() >= 0
        assert bev_image.max() <= 1


class TestCreateBEVEncoder:
    """Tests for factory function."""
    
    def test_defaults(self):
        """Test default configuration."""
        encoder = create_bev_encoder()
        assert encoder.config.feature_dim == 256
        assert encoder.config.bev_size == (200, 200)
        assert "camera" in encoder.config.input_types
        assert "lidar" in encoder.config.input_types
    
    def test_custom_config(self):
        """Test custom configuration."""
        encoder = create_bev_encoder(
            input_types=["camera"],
            bev_resolution=0.25,
            bev_size=(400, 400),
            feature_dim=512,
        )
        assert encoder.config.bev_resolution == 0.25
        assert encoder.config.bev_size == (400, 400)
        assert encoder.config.feature_dim == 512


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()

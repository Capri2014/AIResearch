"""Unit tests for BEV encoder module."""

import torch
import pytest
from sim.driving.carla_srunner.bev_encoder import (
    BEVEncoder,
    BEVEncoderConfig,
    create_bev_encoder,
    LidarToBEV,
    CameraToBEV,
)


class TestBEVEncoderConfig:
    """Tests for BEVEncoderConfig."""
    
    def test_default_config(self):
        config = BEVEncoderConfig()
        assert config.input_types == ["camera"]
        assert config.bev_resolution == 0.5
        assert config.bev_width == 200
        assert config.bev_height == 200
        assert config.feature_dim == 256
    
    def test_custom_config(self):
        config = BEVEncoderConfig(
            input_types=["camera", "lidar"],
            bev_resolution=0.3,
            bev_width=300,
            bev_height=300,
            feature_dim=128,
        )
        assert config.input_types == ["camera", "lidar"]
        assert config.bev_resolution == 0.3
        assert config.bev_width == 300
        assert config.bev_height == 300
        assert config.feature_dim == 128


class TestLidarToBEV:
    """Tests for LiDAR to BEV conversion."""
    
    def test_empty_points(self):
        config = BEVEncoderConfig(input_types=["lidar"])
        encoder = LidarToBEV(config)
        
        empty_points = torch.zeros(0, 3)
        with torch.no_grad():
            bev = encoder(empty_points)
        
        assert bev.shape == (1, config.feature_dim // 2, config.bev_height, config.bev_width)
    
    def test_points_to_bev(self):
        config = BEVEncoderConfig(
            input_types=["lidar"],
            bev_width=100,
            bev_height=100,
            bev_resolution=1.0,
            bev_range_x=(-50.0, 50.0),
            bev_range_y=(-50.0, 50.0),
        )
        encoder = LidarToBEV(config)
        
        # Simple point at origin
        points = torch.tensor([[0.0, 0.0, 0.0]])
        
        with torch.no_grad():
            bev = encoder(points)
        
        assert bev.shape == (1, 4, 100, 100)
    
    def test_lidar_with_intensity(self):
        config = BEVEncoderConfig(
            input_types=["lidar"],
            lidar_use_intensity=True,
        )
        encoder = LidarToBEV(config)
        
        # Point with intensity
        points = torch.tensor([[0.0, 0.0, 0.0, 0.8]])
        
        with torch.no_grad():
            bev = encoder(points)
        
        assert bev.shape[1] == 4  # height bins + intensity


class TestCameraToBEV:
    """Tests for Camera to BEV conversion."""
    
    def test_camera_encoding(self):
        config = BEVEncoderConfig(
            input_types=["camera"],
            bev_width=100,
            bev_height=100,
        )
        encoder = CameraToBEV(config)
        
        # Dummy camera image
        images = [torch.randn(1, 3, 224, 224)]
        
        with torch.no_grad():
            features = encoder(images)
        
        assert features.shape == (1, config.feature_dim // 2, 100, 100)
    
    def test_multi_camera(self):
        config = BEVEncoderConfig(
            input_types=["camera"],
            bev_width=100,
            bev_height=100,
        )
        encoder = CameraToBEV(config)
        
        # Multiple camera views
        images = [
            torch.randn(1, 3, 224, 224),
            torch.randn(1, 3, 224, 224),
            torch.randn(1, 3, 224, 224),
        ]
        
        with torch.no_grad():
            features = encoder(images)
        
        assert features.shape == (1, config.feature_dim // 2, 100, 100)


class TestBEVEncoder:
    """Tests for unified BEV encoder."""
    
    def test_camera_only(self):
        encoder = create_bev_encoder(
            input_types=["camera"],
            bev_size=(100, 100),
            feature_dim=256,
        )
        
        cameras = [torch.randn(1, 3, 224, 224)]
        
        with torch.no_grad():
            bev_features = encoder.encode(cameras=cameras)
        
        assert bev_features.shape == (1, 256, 100, 100)
    
    def test_lidar_only(self):
        encoder = create_bev_encoder(
            input_types=["lidar"],
            bev_size=(100, 100),
            feature_dim=256,
        )
        
        lidar_points = torch.randn(1000, 4)
        
        with torch.no_grad():
            bev_features = encoder.encode(lidar_points=lidar_points)
        
        assert bev_features.shape == (1, 256, 100, 100)
    
    def test_camera_lidar_fusion(self):
        encoder = create_bev_encoder(
            input_types=["camera", "lidar"],
            bev_size=(100, 100),
            feature_dim=256,
            fusion_type="concat",
        )
        
        cameras = [torch.randn(1, 3, 224, 224)]
        lidar_points = torch.randn(1000, 4)
        
        with torch.no_grad():
            bev_features = encoder.encode(
                cameras=cameras,
                lidar_points=lidar_points
            )
        
        # Concat fusion should produce feature_dim
        assert bev_features.shape == (1, 256, 100, 100)
    
    def test_bev_image_generation(self):
        encoder = create_bev_encoder(
            input_types=["camera"],
            bev_size=(100, 100),
            feature_dim=256,
        )
        
        bev_features = torch.randn(1, 256, 100, 100)
        
        with torch.no_grad():
            bev_img = encoder.get_bev_image(bev_features)
        
        assert bev_img.shape == (1, 3, 100, 100)
    
    def test_forward_pass(self):
        encoder = create_bev_encoder(
            input_types=["camera", "lidar"],
            bev_size=(100, 100),
            feature_dim=256,
        )
        
        cameras = [torch.randn(1, 3, 224, 224)]
        lidar_points = torch.randn(1000, 4)
        
        with torch.no_grad():
            bev_features = encoder(cameras=cameras, lidar_points=lidar_points)
        
        assert bev_features.shape == (1, 256, 100, 100)


class TestCreateBEVEncoder:
    """Tests for factory function."""
    
    def test_default_creation(self):
        encoder = create_bev_encoder()
        assert isinstance(encoder, BEVEncoder)
        assert "camera" in encoder.config.input_types
    
    def test_custom_creation(self):
        encoder = create_bev_encoder(
            input_types=["lidar"],
            bev_resolution=0.3,
            bev_size=(150, 150),
            feature_dim=128,
        )
        
        assert isinstance(encoder, BEVEncoder)
        assert encoder.config.input_types == ["lidar"]
        assert encoder.config.bev_resolution == 0.3
        assert encoder.config.bev_width == 150
        assert encoder.config.bev_height == 150
        assert encoder.config.feature_dim == 128
    
    def test_device_override(self):
        encoder = create_bev_encoder(device="cpu")
        assert encoder.config.device == "cpu"


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()

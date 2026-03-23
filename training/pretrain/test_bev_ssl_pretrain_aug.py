"""
Test script for BEV SSL pretraining with augmentations.
"""

import torch
import sys

def test_import():
    """Test that the module imports correctly."""
    print("Test 1: Import verification")
    try:
        from training.pretrain.bev_ssl_pretrain_aug import (
            BEVSSLConfig,
            AugmentationPipeline,
            BEVSSLTrainer,
            bev_ssl_training_loop,
            test_augmentation_pipeline,
        )
        print("  PASS: All imports successful")
        return True
    except ImportError as e:
        print(f"  FAIL: Import error - {e}")
        return False


def test_config():
    """Test configuration creation."""
    print("\nTest 2: Configuration creation")
    try:
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLConfig
        
        config = BEVSSLConfig(
            encoder_dim=128,
            batch_size=8,
            num_epochs=1,
            use_image_augmentations=True,
            use_bev_augmentations=True,
        )
        
        assert config.encoder_dim == 128
        assert config.batch_size == 8
        assert config.use_image_augmentations == True
        assert config.use_bev_augmentations == True
        
        print(f"  PASS: Config created (encoder_dim={config.encoder_dim})")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_augmentation_pipeline():
    """Test the combined augmentation pipeline."""
    print("\nTest 3: Augmentation pipeline creation")
    try:
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLConfig, AugmentationPipeline
        
        config = BEVSSLConfig(
            use_image_augmentations=True,
            use_bev_augmentations=True,
        )
        
        pipeline = AugmentationPipeline(config, is_training=True)
        print(f"  Image aug: {'enabled' if pipeline.image_aug else 'disabled'}")
        print(f"  BEV aug: {'enabled' if pipeline.bev_aug else 'disabled'}")
        
        print("  PASS: Pipeline created")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_image_augmentation():
    """Test image augmentation."""
    print("\nTest 4: Image augmentation")
    try:
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLConfig, AugmentationPipeline
        
        config = BEVSSLConfig(use_image_augmentations=True)
        pipeline = AugmentationPipeline(config, is_training=True)
        
        test_images = torch.randn(4, 3, 224, 224)
        aug_images, _ = pipeline.augment_images(test_images)
        
        assert aug_images.shape == test_images.shape
        print(f"  Input shape: {test_images.shape}")
        print(f"  Output shape: {aug_images.shape}")
        print("  PASS: Image augmentation works")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_bev_augmentation():
    """Test BEV augmentation."""
    print("\nTest 5: BEV augmentation")
    try:
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLConfig, AugmentationPipeline
        
        config = BEVSSLConfig(use_bev_augmentations=True)
        pipeline = AugmentationPipeline(config, is_training=True)
        
        test_bev = torch.randn(64, 200, 200)
        aug_bev, _ = pipeline.augment_bev(test_bev)
        
        assert aug_bev.shape == test_bev.shape
        print(f"  Input shape: {test_bev.shape}")
        print(f"  Output shape: {aug_bev.shape}")
        print("  PASS: BEV augmentation works")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_full_pipeline():
    """Test full augmentation pipeline."""
    print("\nTest 6: Full augmentation pipeline")
    try:
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLConfig, AugmentationPipeline
        
        config = BEVSSLConfig(
            use_image_augmentations=True,
            use_bev_augmentations=True,
        )
        pipeline = AugmentationPipeline(config, is_training=True)
        
        test_images = torch.randn(4, 6, 3, 224, 224)
        test_lidar = torch.randn(4, 1000, 2) * 50
        
        aug_i, aug_l, aug_i_pos, aug_l_pos = pipeline(
            test_images, test_lidar, test_images, test_lidar
        )
        
        assert aug_i.shape == test_images.shape
        print(f"  Images shape: {aug_i.shape}")
        print("  PASS: Full pipeline works")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_trainer_creation():
    """Test trainer creation."""
    print("\nTest 7: Trainer creation")
    try:
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLConfig, BEVSSLTrainer
        
        config = BEVSSLConfig(
            encoder_dim=128,
            bev_channels=64,
            batch_size=4,
            num_epochs=1,
            output_dir="out/test_bev_ssl_aug",
            episode_dir="data/waymo_episodes",
        )
        
        trainer = BEVSSLTrainer(config)
        
        num_params = sum(p.numel() for p in trainer.query_encoder.parameters())
        print(f"  Query encoder params: {num_params:,}")
        print(f"  Device: {trainer.device}")
        print("  PASS: Trainer created")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_training_step():
    """Test a single training step."""
    print("\nTest 8: Training step")
    try:
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLConfig, BEVSSLTrainer
        
        config = BEVSSLConfig(
            encoder_dim=128,
            bev_channels=64,
            batch_size=4,
            num_epochs=1,
            output_dir="out/test_bev_ssl_aug",
            episode_dir="data/waymo_episodes",
            use_image_augmentations=True,
            use_bev_augmentations=True,
        )
        
        trainer = BEVSSLTrainer(config)
        
        # Create dummy batch
        batch = {
            "images": torch.randn(4, 6, 3, 224, 224),
            "lidar": torch.randn(4, 1000, 2) * 50,
            "images_pos": torch.randn(4, 6, 3, 224, 224),
            "lidar_pos": torch.randn(4, 1000, 2) * 50,
        }
        
        metrics = trainer.train_step(batch)
        
        print(f"  Loss: {metrics['total_loss']:.4f}")
        print(f"  Pos sim: {metrics['pos_sim']:.4f}")
        print(f"  Neg sim: {metrics['neg_sim']:.4f}")
        print("  PASS: Training step successful")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BEV SSL Pretraining with Augmentations - Test Suite")
    print("=" * 60)
    
    tests = [
        test_import,
        test_config,
        test_augmentation_pipeline,
        test_image_augmentation,
        test_bev_augmentation,
        test_full_pipeline,
        test_trainer_creation,
        test_training_step,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

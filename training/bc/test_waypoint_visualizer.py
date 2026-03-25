"""Tests for waypoint visualizer."""

import torch
import sys

def test_import():
    """Test module imports."""
    from training.bc.waypoint_visualizer import (
        WaypointVisConfig,
        waypoints_to_image,
        visualize_prediction,
        compute_metrics,
        visualize_bev_features,
        visualize_trajectory,
        create_waypoint_summary,
        test_visualizer
    )
    print("✓ Import test passed")
    return True

def test_config():
    """Test config creation."""
    from training.bc.waypoint_visualizer import WaypointVisConfig
    
    config = WaypointVisConfig(
        image_size=512,
        waypoint_scale=20.0,
        num_waypoints=8
    )
    assert config.image_size == 512
    assert config.waypoint_scale == 20.0
    assert config.num_waypoints == 8
    print("✓ Config test passed")
    return True

def test_metrics():
    """Test metric computation."""
    from training.bc.waypoint_visualizer import compute_metrics
    
    pred = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    target = torch.tensor([[1.0, 0.0], [2.1, 0.9], [3.1, 2.1]])
    
    metrics = compute_metrics(pred, target)
    assert 'ade' in metrics
    assert 'fde' in metrics
    assert metrics['ade'] > 0
    print(f"  ADE: {metrics['ade']:.4f}")
    print(f"  FDE: {metrics['fde']:.4f}")
    print("✓ Metrics test passed")
    return True

def test_waypoints_to_image():
    """Test waypoint image rendering."""
    from training.bc.waypoint_visualizer import waypoints_to_image, WaypointVisConfig
    
    waypoints = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    config = WaypointVisConfig()
    
    img = waypoints_to_image(waypoints, config)
    assert img.shape == (config.image_size, config.image_size, 3)
    print("✓ Waypoints to image test passed")
    return True

def test_visualize_prediction():
    """Test prediction visualization."""
    from training.bc.waypoint_visualizer import visualize_prediction, WaypointVisConfig
    
    pred = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    target = torch.tensor([[1.0, 0.0], [2.1, 0.9], [3.1, 2.1]])
    config = WaypointVisConfig()
    
    img = visualize_prediction(pred, target, config)
    assert img.shape[2] == 3  # RGB
    print("✓ Visualize prediction test passed")
    return True

def test_visualize_trajectory():
    """Test trajectory visualization."""
    from training.bc.waypoint_visualizer import visualize_trajectory, WaypointVisConfig
    
    positions = torch.randn(20, 2) * 3
    waypoints = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    config = WaypointVisConfig()
    
    img = visualize_trajectory(positions, waypoints, config)
    assert img.shape == (config.image_size, config.image_size, 3)
    print("✓ Visualize trajectory test passed")
    return True

def test_summary():
    """Test text summary creation."""
    from training.bc.waypoint_visualizer import create_waypoint_summary
    
    pred = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    target = torch.tensor([[1.0, 0.0], [2.1, 0.9], [3.1, 2.1]])
    
    summary = create_waypoint_summary(pred, target, prefix="> ")
    assert "ADE:" in summary
    assert "FDE:" in summary
    print("✓ Summary test passed")
    return True

def run_all_tests():
    """Run all tests."""
    print("\n=== Waypoint Visualizer Tests ===\n")
    
    tests = [
        test_import,
        test_config,
        test_metrics,
        test_waypoints_to_image,
        test_visualize_prediction,
        test_visualize_trajectory,
        test_summary,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed}/{passed+failed} passed ===\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

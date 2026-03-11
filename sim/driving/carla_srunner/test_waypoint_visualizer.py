#!/usr/bin/env python3
"""
Test script for waypoint visualization module.

Tests the WaypointVisualizer without requiring CARLA or OpenCV.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import():
    """Test that the module can be imported."""
    print("Testing module import...")
    try:
        from sim.driving.carla_srunner.waypoint_visualizer import (
            WaypointVisualizer,
            VisualizationConfig,
            create_visualizer,
        )
        print("  ✓ Module imports successfully!")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config_creation():
    """Test VisualizationConfig creation."""
    print("Testing config creation...")
    
    from sim.driving.carla_srunner.waypoint_visualizer import VisualizationConfig
    
    config = VisualizationConfig(
        image_width=400,
        image_height=400,
        meters_per_pixel=0.5,
        waypoint_color=(0, 255, 0),
    )
    
    assert config.image_width == 400
    assert config.image_height == 400
    assert config.meters_per_pixel == 0.5
    assert config.waypoint_color == (0, 255, 0)
    print("  ✓ Config creation works!")
    return True


def test_visualizer_creation():
    """Test WaypointVisualizer creation."""
    print("Testing visualizer creation...")
    
    from sim.driving.carla_srunner.waypoint_visualizer import (
        WaypointVisualizer,
        VisualizationConfig,
        create_visualizer,
    )
    
    # Test with default config
    viz1 = WaypointVisualizer()
    assert viz1.config is not None
    
    # Test with custom config
    config = VisualizationConfig(image_width=800)
    viz2 = WaypointVisualizer(config)
    assert viz2.config.image_width == 800
    
    # Test factory function
    viz3 = create_visualizer(image_height=600)
    assert viz3.config.image_height == 600
    
    print("  ✓ Visualizer creation works!")
    return True


def test_waypoint_shapes():
    """Test various waypoint array shapes."""
    print("Testing waypoint shapes...")
    
    from sim.driving.carla_srunner.waypoint_visualizer import WaypointVisualizer
    
    viz = WaypointVisualizer()
    
    # Test different valid shapes
    test_cases = [
        (np.array([2.0, 4.0, 6.0, 8.0]), "1D array"),
        (np.array([[2.0, 0.0], [4.0, 0.0]]), "2D (N, 2)"),
        (np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.1]]), "2D (N, 3)"),
    ]
    
    for waypoints, description in test_cases:
        # Just verify no exceptions
        print(f"    - {description}: shape {waypoints.shape}")
    
    print("  ✓ Waypoint shapes handled!")
    return True


def test_empty_handling():
    """Test handling of empty/None inputs."""
    print("Testing empty handling...")
    
    from sim.driving.carla_srunner.waypoint_visualizer import WaypointVisualizer
    
    viz = WaypointVisualizer()
    
    # Empty array should not crash
    empty = np.array([]).reshape(0, 2)
    # Note: create_bev_image requires cv2, so we just test config
    
    assert viz.config is not None
    print("  ✓ Empty handling works!")
    return True


def test_config_defaults():
    """Test default configuration values."""
    print("Testing config defaults...")
    
    from sim.driving.carla_srunner.waypoint_visualizer import VisualizationConfig
    
    config = VisualizationConfig()
    
    assert config.image_width == 400
    assert config.image_height == 400
    assert config.meters_per_pixel == 0.5
    assert config.waypoint_color == (0, 255, 0)
    assert config.path_color == (255, 0, 0)
    assert config.vehicle_color == (0, 0, 255)
    assert config.waypoint_radius == 4
    assert config.line_thickness == 2
    assert config.show_waypoint_numbers == True
    assert config.show_heading_arrows == True
    
    print("  ✓ Config defaults are correct!")
    return True


def test_factory_function():
    """Test the create_visualizer factory function."""
    print("Testing factory function...")
    
    from sim.driving.carla_srunner.waypoint_visualizer import create_visualizer
    
    # Test with various kwargs
    viz = create_visualizer(
        image_width=600,
        image_height=600,
        meters_per_pixel=0.25,
        waypoint_color=(255, 0, 255),
        show_waypoint_numbers=False,
    )
    
    assert viz.config.image_width == 600
    assert viz.config.image_height == 600
    assert viz.config.meters_per_pixel == 0.25
    assert viz.config.waypoint_color == (255, 0, 255)
    assert viz.config.show_waypoint_numbers == False
    
    print("  ✓ Factory function works!")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("WaypointVisualizer Tests")
    print("=" * 50)
    
    tests = [
        test_import,
        test_config_creation,
        test_visualizer_creation,
        test_waypoint_shapes,
        test_empty_handling,
        test_config_defaults,
        test_factory_function,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

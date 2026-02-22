#!/usr/bin/env python3
"""
Smoke tests for waypoint smoothing module.

Run: python -m training.rl.test_waypoint_smoothing
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.waypoint_smoothing import (
    WaypointSmootherConfig,
    WaypointSmoother,
    exponential_smooth,
    moving_average_smooth,
    savgol_smooth,
    smooth_waypoints,
    ensure_kinematic_feasibility,
    compute_waypoint_speeds,
    compute_waypoint_headings,
)


def test_exponential_smooth():
    """Test exponential smoothing."""
    waypoints = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
    smoothed = exponential_smooth(waypoints, alpha=0.3)
    
    assert smoothed.shape == waypoints.shape
    assert np.allclose(smoothed[0], waypoints[0])  # First point unchanged
    print("✓ exponential_smooth works")


def test_moving_average_smooth():
    """Test moving average smoothing."""
    waypoints = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float32)
    smoothed = moving_average_smooth(waypoints, window_size=3)
    
    assert smoothed.shape == waypoints.shape
    assert np.allclose(smoothed[0], waypoints[0])  # Endpoints unchanged
    assert np.allclose(smoothed[-1], waypoints[-1])  # Endpoints unchanged
    print("✓ moving_average_smooth works")


def test_savgol_smooth():
    """Test Savitzky-Golay smoothing."""
    waypoints = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
    smoothed = savgol_smooth(waypoints, window_size=5, poly_order=2)
    
    assert smoothed.shape == waypoints.shape
    print("✓ savgol_smooth works")


def test_kinematic_feasibility():
    """Test kinematic feasibility enforcement."""
    # Create waypoints with impossible speed
    waypoints = np.array([[0, 0], [100, 100], [200, 200]], dtype=np.float32)
    feasible = ensure_kinematic_feasibility(
        waypoints,
        max_speed=15.0,
        max_acceleration=3.0,
        dt=0.1
    )
    
    # Check speeds are limited
    speeds = compute_waypoint_speeds(feasible, dt=0.1)
    assert np.all(speeds <= 15.0 + 1e-3), f"Speeds exceed max: {speeds}"
    print("✓ ensure_kinematic_feasibility works")


def test_compute_speeds_headings():
    """Test speed and heading computation."""
    waypoints = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    
    speeds = compute_waypoint_speeds(waypoints, dt=1.0)
    assert len(speeds) == 3
    assert np.isclose(speeds[0], 1.0)  # 1 m/s between (0,0) and (1,0)
    
    headings = compute_waypoint_headings(waypoints)
    assert len(headings) == 4
    assert np.isclose(headings[1], 0.0)  # Heading 0 for (1,0) - (0,0)
    
    print("✓ compute_waypoint_speeds and compute_waypoint_headings work")


def test_smooth_waypoints():
    """Test main smooth_waypoints function."""
    config = WaypointSmootherConfig(
        smoothing_window=5,
        apply_speed_profile=False,  # Disable for this test
        ensure_start_match=True,
        ensure_end_match=True
    )
    
    waypoints = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float32)
    smoothed = smooth_waypoints(waypoints, config, method="savgol")
    
    assert smoothed.shape == waypoints.shape
    assert np.allclose(smoothed[0], waypoints[0])
    assert np.allclose(smoothed[-1], waypoints[-1])
    print("✓ smooth_waypoints works")


def test_pytorch_smoother():
    """Test PyTorch WaypointSmoother module."""
    config = WaypointSmootherConfig()
    smoother = WaypointSmoother(config)
    
    # Test forward pass
    batch_waypoints = torch.randn(4, 20, 2)
    smoothed = smoother(batch_waypoints)
    
    assert smoothed.shape == batch_waypoints.shape
    
    # Test smoothness loss
    loss = smoother.compute_smoothness_loss(batch_waypoints)
    assert loss.item() >= 0
    
    print("✓ WaypointSmoother module works")


def test_smoothness_improves_ade():
    """Test that smoothing reduces ADE error."""
    np.random.seed(42)
    
    # Generate ground truth trajectory
    t = np.linspace(0, 2 * np.pi, 20)
    true_x = 10 * t + 5 * np.sin(t)
    true_y = 10 * t + 3 * np.cos(t)
    true_waypoints = np.column_stack([true_x, true_y])
    
    # Add moderate noise
    noisy_waypoints = true_waypoints + np.random.randn(*true_waypoints.shape) * 0.5
    
    # Compute ADE
    def ade_error(pred, true):
        return np.mean(np.sqrt(np.sum((pred - true)**2, axis=1)))
    
    noisy_ade = ade_error(noisy_waypoints, true_waypoints)
    
    # Apply smoothing (without kinematic constraints for this test)
    config = WaypointSmootherConfig(smoothing_window=5, apply_speed_profile=False)
    smoothed = smooth_waypoints(noisy_waypoints, config, method="savgol")
    smoothed_ade = ade_error(smoothed, true_waypoints)
    
    assert smoothed_ade < noisy_ade, f"Smoothing didn't help: {smoothed_ade} >= {noisy_ade}"
    print(f"✓ Smoothing improves ADE: {noisy_ade:.3f}m → {smoothed_ade:.3f}m")


def main():
    print("Running waypoint smoothing smoke tests...\n")
    
    test_exponential_smooth()
    test_moving_average_smooth()
    test_savgol_smooth()
    test_kinematic_feasibility()
    test_compute_speeds_headings()
    test_smooth_waypoints()
    test_pytorch_smoother()
    test_smoothness_improves_ade()
    
    print("\n✅ All smoke tests passed!")


if __name__ == "__main__":
    main()

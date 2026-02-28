"""
Tests for contingency planning module.
"""

import torch
from planning.contingency_planner import (
    create_contingency_planner,
    create_safe_driving_tree,
    FailureMode,
    ContingencyConfig,
)
from planning.failure_detector import create_failure_detector


def test_contingency_planner():
    """Test contingency planner forward pass."""
    planner = create_contingency_planner()
    
    batch_size = 4
    state = torch.randn(batch_size, 256)
    primary_waypoints = torch.randn(batch_size, 10, 3)
    sensor_health = torch.rand(batch_size, 4)
    
    final_waypoints, info = planner(state, primary_waypoints, sensor_health)
    
    assert final_waypoints.shape == primary_waypoints.shape
    assert "failure_mode" in info
    assert "degradation_params" in info
    print("✓ Contingency planner test passed")


def test_behavior_tree():
    """Test behavior tree execution."""
    tree = create_safe_driving_tree()
    
    # Test normal driving
    state = {
        "failure_mode": FailureMode.NONE,
        "executed_action": "none",
        "speed_reduction": 0,
        "used_fallback": False,
    }
    result = tree.execute(state)
    assert result == True
    print("✓ Behavior tree (normal) test passed")
    
    # Test emergency
    state["failure_mode"] = FailureMode.COLLISION_IMMINENT
    result = tree.execute(state)
    assert result == True  # Should select emergency action
    print("✓ Behavior tree (emergency) test passed")


def test_failure_detector():
    """Test failure detection."""
    detector = create_failure_detector()
    
    batch_size = 4
    state = torch.randn(batch_size, 256)
    position = torch.randn(batch_size, 3)
    velocity = torch.randn(batch_size, 3) * 10
    waypoints = torch.randn(batch_size, 10, 3)
    
    # With nearby obstacles
    obstacles = torch.randn(batch_size, 5, 3)
    obstacles[:, :2, :2] = position[:, :2] + torch.randn(batch_size, 2, 2) * 2
    
    failure_mode, uncertainty, details = detector(
        state, position, velocity, waypoints, 
        obstacle_positions=obstacles
    )
    
    assert isinstance(failure_mode, FailureMode)
    assert uncertainty.shape[0] == batch_size
    print("✓ Failure detector test passed")


if __name__ == "__main__":
    test_contingency_planner()
    test_behavior_tree()
    test_failure_detector()
    print("\n✅ All tests passed!")

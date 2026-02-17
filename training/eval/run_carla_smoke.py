"""
Smoke test for CARLA ScenarioRunner evaluation module.

Quick validation that the module imports and basic structures are correct.
"""

import sys
sys.path.insert(0, '/data/.openclaw/workspace/AIResearch-repo')

from training.eval.carla_scenariorunner_eval import (
    CARLAEvalConfig,
    CARLAScenarioRunner,
    EvalResult
)


def test_config():
    """Test CARLAEvalConfig dataclass."""
    config = CARLAEvalConfig(
        host="localhost",
        port=2000,
        fps=20,
        map_name="Town01"
    )
    assert config.host == "localhost"
    assert config.port == 2000
    assert config.fps == 20
    print("✓ CARLAEvalConfig works")


def test_eval_result():
    """Test EvalResult dataclass."""
    result = EvalResult(
        route_completion=0.85,
        collision_count=0,
        offroad_count=1,
        route_deviation_avg=1.5,
        waypoint_accuracy_avg=0.92,
        episode_length=45.0,
        success=True
    )
    summary = result.summary()
    assert "✓ SUCCESS" in summary
    assert "85.0%" in summary
    print("✓ EvalResult works")
    print(f"  Summary: {summary}")


def test_scenario_runner_interface():
    """Test CARLAScenarioRunner interface."""
    runner = CARLAScenarioRunner()
    assert hasattr(runner, 'connect')
    assert hasattr(runner, 'disconnect')
    assert hasattr(runner, 'spawn_ego_vehicle')
    assert hasattr(runner, 'apply_waypoint_control')
    print("✓ CARLAScenarioRunner interface complete")


def main():
    print("CARLA ScenarioRunner Eval Smoke Test")
    print("=" * 50)
    
    test_config()
    test_eval_result()
    test_scenario_runner_interface()
    
    print()
    print("All smoke tests passed!")
    print()
    print("Next steps:")
    print("  - Run with CARLA server: python -m training.eval.carla_scenariorunner_eval")
    print("  - Integrate with unified_eval.py for full pipeline")
    print("  - Add checkpoint selection by best FDE from waypoint BC")


if __name__ == "__main__":
    main()

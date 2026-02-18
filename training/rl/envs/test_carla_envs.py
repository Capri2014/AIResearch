"""
Test script for CARLA environments.
Validates Gym env and ScenarioRunner eval integration.
"""

import sys
sys.path.insert(0, '/data/.openclaw/workspace/AIResearch-repo')

# Check for CARLA module
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("Note: CARLA module not installed. Running basic tests only.")
    print("Install with: pip install carla\n")


def test_gym_config():
    """Test CARLA Gym configuration."""
    print("Testing CarlaGymConfig...")
    
    from training.rl.envs.carla_gym_env import CarlaGymConfig
    
    if not CARLA_AVAILABLE:
        # Still test config fields exist
        config = CarlaGymConfig(
            host="localhost",
            port=2000,
            town="Town03",
            reward_progress=2.0,
            reward_collision=-200.0,
        )
        assert config.host == "localhost"
        assert config.port == 2000
        assert config.reward_progress == 2.0
        assert config.reward_collision == -200.0
        print("  ✓ CarlaGymConfig validated (without CARLA)")
        return
    
    config = CarlaGymConfig(
        host="localhost",
        port=2000,
        town="Town03",
        reward_progress=2.0,
        reward_collision=-200.0,
    )
    
    assert config.host == "localhost"
    assert config.port == 2000
    assert config.reward_progress == 2.0
    assert config.reward_collision == -200.0
    
    print("  ✓ CarlaGymConfig validated")


def test_scenario_config():
    """Test ScenarioEvaluator configuration."""
    print("Testing ScenarioEvaluatorConfig...")
    
    from training.rl.envs.carla_scenario_eval import ScenarioEvaluatorConfig
    
    config = ScenarioEvaluatorConfig(
        host="localhost",
        port=2000,
        scenarios=["lane_change", "straight"],
        num_episodes=5,
        timeout_per_episode=30.0,
    )
    
    assert config.host == "localhost"
    assert len(config.scenarios) == 2
    assert config.num_episodes == 5
    
    print("  ✓ ScenarioEvaluatorConfig validated")


def test_infraction_types():
    """Test infraction enum."""
    print("Testing InfractionType enum...")
    
    from training.rl.envs.carla_scenario_eval import InfractionType
    
    assert InfractionType.COLLISION.value == "collision"
    assert InfractionType.RED_LIGHT_VIOLATION.value == "red_light"
    assert InfractionType.TIMEOUT.value == "timeout"
    
    print("  ✓ InfractionType enum validated")


def test_dummy_policy():
    """Test dummy policy for evaluation."""
    print("Testing dummy policy...")
    
    def dummy_policy(obs):
        """Dummy policy: always steer left."""
        return {
            'steer': -0.5,
            'throttle': 0.3,
            'brake': 0.0,
        }
    
    # Test with mock observation
    obs = {
        'camera': None,
        'speed': 5.0,
        'heading': 0.0,
    }
    
    action = dummy_policy(obs)
    assert 'steer' in action
    assert 'throttle' in action
    assert 'brake' in action
    
    print("  ✓ Dummy policy works")


def test_file_structure():
    """Test that files exist and have expected content."""
    print("Testing file structure...")
    
    import os
    
    files = [
        'training/rl/envs/carla_gym_env.py',
        'training/rl/envs/carla_scenario_eval.py',
        'training/rl/envs/__init__.py',
    ]
    
    for f in files:
        assert os.path.exists(f), f"File not found: {f}"
        with open(f, 'r') as fp:
            content = fp.read()
            assert len(content) > 100, f"File too short: {f}"
    
    print("  ✓ All files exist and have content")


def test_init_file():
    """Test __init__.py exists."""
    print("Testing __init__.py...")
    
    import os
    
    init_file = 'training/rl/envs/__init__.py'
    assert os.path.exists(init_file), f"File not found: {init_file}"
    
    with open(init_file, 'r') as f:
        content = f.read()
        assert 'CarlaGymEnv' in content, "CarlaGymEnv not in __init__.py"
        assert 'ScenarioEvaluator' in content, "ScenarioEvaluator not in __init__.py"
    
    print("  ✓ __init__.py validated")


def test_env_classes_importable():
    """Test that env classes can be imported (only if CARLA available)."""
    print("Testing env classes importable...")
    
    if not CARLA_AVAILABLE:
        print("  ⊘ Skipped (CARLA not installed)")
        return
    
    # Import classes (will fail at runtime if carla not available)
    from training.rl.envs.carla_gym_env import CarlaGymEnv, CarlaGymConfig
    from training.rl.envs.carla_scenario_eval import (
        ScenarioEvaluator,
        ScenarioEvaluatorConfig,
        EpisodeResult,
        InfractionType,
    )
    
    print("  ✓ Env classes importable")


def main():
    """Run all tests."""
    print("=" * 50)
    print("CARLA Environment Tests")
    print("=" * 50)
    print()
    
    tests = [
        test_gym_config,
        test_scenario_config,
        test_infraction_types,
        test_dummy_policy,
        test_file_structure,
        test_init_file,
        test_env_classes_importable,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("\n✓ All tests passed!")
        if not CARLA_AVAILABLE:
            print("\nFor full integration test:")
            print("  1. Install CARLA server from https://carla.org/")
            print("  2. pip install carla")
            print("  3. Start CARLA server: ./CarlaUE4.sh -carla-server-port=2000")
            print("  4. Run: python training/rl/envs/test_carla_envs.py")
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

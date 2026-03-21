"""
Tests for CARLA ScenarioRunner Scenario Factory Module.
"""

import pytest
from sim.driving.carla_srunner.scenario_factory import (
    ScenarioType,
    ScenarioDifficulty,
    ScenarioConfig,
    ScenarioResult,
    ScenarioFactory,
    MultiScenarioConfig,
    MultiScenarioResult,
    MultiScenarioRunner,
    create_standard_scenario_suite,
)


class TestScenarioConfig:
    """Tests for ScenarioConfig dataclass."""
    
    def test_default_config(self):
        """Test default scenario configuration."""
        config = ScenarioConfig(scenario_type=ScenarioType.CUT_IN)
        
        assert config.scenario_type == ScenarioType.CUT_IN
        assert config.town == "Town01"
        assert config.num_vehicles == 10
        assert config.weather == "ClearNoon"
        assert config.difficulty == ScenarioDifficulty.MEDIUM
    
    def test_custom_config(self):
        """Test custom scenario configuration."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.LANE_CHANGE,
            town="Town03",
            num_vehicles=20,
            actor_speed=15.0,
            difficulty=ScenarioDifficulty.HARD,
        )
        
        assert config.scenario_type == ScenarioType.LANE_CHANGE
        assert config.town == "Town03"
        assert config.num_vehicles == 20
        assert config.actor_speed == 15.0
        assert config.difficulty == ScenarioDifficulty.HARD
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.FOLLOW,
            difficulty=ScenarioDifficulty.EASY,
        )
        
        d = config.to_dict()
        assert d["scenario_type"] == "follow"
        assert d["town"] == "Town01"
        assert d["difficulty"] == 1


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""
    
    def test_default_result(self):
        """Test default scenario result."""
        result = ScenarioResult(
            scenario_id="test_001",
            scenario_type=ScenarioType.CUT_IN,
            success=False,
            completed=False,
        )
        
        assert result.scenario_id == "test_001"
        assert result.scenario_type == ScenarioType.CUT_IN
        assert result.collisions == 0
        assert result.waypoint_ade == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ScenarioResult(
            scenario_id="test_001",
            scenario_type=ScenarioType.CUT_IN,
            success=True,
            completed=True,
            distance_traveled=100.0,
            average_speed=5.0,
            collisions=0,
        )
        
        d = result.to_dict()
        assert d["scenario_id"] == "test_001"
        assert d["success"] is True
        assert d["metrics"]["distance_traveled"] == 100.0


class TestScenarioFactory:
    """Tests for ScenarioFactory class."""
    
    def test_factory_creation(self):
        """Test factory instantiation."""
        factory = ScenarioFactory()
        assert factory is not None
    
    def test_create_scenario(self):
        """Test basic scenario creation."""
        factory = ScenarioFactory()
        config = factory.create_scenario(
            ScenarioType.CUT_IN,
            town="Town02",
            spawn_point=1,
        )
        
        assert config.scenario_type == ScenarioType.CUT_IN
        assert config.town == "Town02"
        assert config.spawn_point == 1
    
    def test_create_cut_in_scenario(self):
        """Test cut-in scenario creation."""
        factory = ScenarioFactory()
        config = factory.create_cut_in_scenario(
            town="Town03",
            actor_distance=25.0,
            actor_speed=12.0,
            difficulty=ScenarioDifficulty.HARD,
        )
        
        assert config.scenario_type == ScenarioType.CUT_IN
        assert config.town == "Town03"
        assert config.actor_distance == 25.0
        assert config.actor_speed == 12.0
        assert config.difficulty == ScenarioDifficulty.HARD
    
    def test_create_follow_scenario(self):
        """Test follow scenario creation."""
        factory = ScenarioFactory()
        config = factory.create_follow_scenario(
            following_distance=20.0,
            target_speed=10.0,
        )
        
        assert config.scenario_type == ScenarioType.FOLLOW
        assert config.actor_distance == 20.0
        assert config.actor_speed == 10.0
    
    def test_create_lane_change_scenario(self):
        """Test lane change scenario creation."""
        factory = ScenarioFactory()
        config = factory.create_lane_change_scenario(
            target_lane=2,
            difficulty=ScenarioDifficulty.HARD,
        )
        
        assert config.scenario_type == ScenarioType.LANE_CHANGE
        assert config.actor_lane_offset == 7.0  # 2 * 3.5
        assert config.difficulty == ScenarioDifficulty.HARD
    
    def test_create_pedestrian_scenario(self):
        """Test pedestrian scenario creation."""
        factory = ScenarioFactory()
        config = factory.create_pedestrian_scenario(
            walker_speed=2.0,
            crossing_distance=15.0,
            difficulty=ScenarioDifficulty.EASY,
        )
        
        assert config.scenario_type == ScenarioType.PEDESTRIAN
        assert config.actor_speed == 2.0
        assert config.actor_distance == 15.0
        assert config.difficulty == ScenarioDifficulty.EASY
    
    def test_create_emergency_brake_scenario(self):
        """Test emergency brake scenario creation."""
        factory = ScenarioFactory()
        config = factory.create_emergency_brake_scenario(
            obstacle_distance=40.0,
            difficulty=ScenarioDifficulty.HARD,
        )
        
        assert config.scenario_type == ScenarioType.EMERGENCY_BRAKE
        assert config.actor_distance == 40.0
        assert config.actor_speed == 0.0
        assert config.difficulty == ScenarioDifficulty.HARD
    
    def test_get_spawn_point(self):
        """Test spawn point retrieval."""
        factory = ScenarioFactory()
        point = factory.get_spawn_point("Town01", 0)
        
        assert "x" in point
        assert "y" in point
        assert "z" in point
        assert "yaw" in point
    
    def test_get_weather_preset(self):
        """Test weather preset retrieval."""
        factory = ScenarioFactory()
        weather = factory.get_weather_preset("ClearNoon")
        
        assert "sun_altitude_angle" in weather
        assert weather["cloudiness"] == 0.0
    
    def test_register_custom_scenario(self):
        """Test custom scenario registration."""
        factory = ScenarioFactory()
        
        def custom_factory(config: ScenarioConfig):
            return config
        
        factory.register_custom_scenario(ScenarioType.MERGE, custom_factory)
        
        assert ScenarioType.MERGE in factory._custom_scenarios


class TestMultiScenarioConfig:
    """Tests for MultiScenarioConfig dataclass."""
    
    def test_default_config(self):
        """Test default multi-scenario configuration."""
        config = MultiScenarioConfig(scenarios=[])
        
        assert config.scenarios == []
        assert config.output_dir == "out/scenario_results"
        assert config.save_trajectories is True
        assert config.carla_host == "localhost"
        assert config.carla_port == 2000
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = MultiScenarioConfig(
            scenarios=[ScenarioConfig(scenario_type=ScenarioType.CUT_IN)],
            output_dir="test_output",
        )
        
        d = config.to_dict()
        assert len(d["scenarios"]) == 1
        assert d["output_dir"] == "test_output"


class TestMultiScenarioResult:
    """Tests for MultiScenarioResult dataclass."""
    
    def test_default_result(self):
        """Test default multi-scenario result."""
        result = MultiScenarioResult(
            total_scenarios=5,
            successful_scenarios=3,
            failed_scenarios=2,
            completed_scenarios=4,
        )
        
        assert result.total_scenarios == 5
        assert result.successful_scenarios == 3
        assert result.failed_scenarios == 2
    
    def test_compute_aggregates(self):
        """Test aggregate computation."""
        result = MultiScenarioResult(
            total_scenarios=3,
            successful_scenarios=0,
            failed_scenarios=0,
            completed_scenarios=0,
            scenario_results=[
                ScenarioResult(
                    scenario_id="s1", scenario_type=ScenarioType.CUT_IN,
                    success=True, completed=True,
                    distance_traveled=100.0, average_speed=5.0,
                    waypoint_ade=0.5, waypoint_fde=1.0, collisions=0,
                ),
                ScenarioResult(
                    scenario_id="s2", scenario_type=ScenarioType.FOLLOW,
                    success=True, completed=True,
                    distance_traveled=150.0, average_speed=6.0,
                    waypoint_ade=0.3, waypoint_fde=0.8, collisions=0,
                ),
                ScenarioResult(
                    scenario_id="s3", scenario_type=ScenarioType.LANE_CHANGE,
                    success=False, completed=True,
                    distance_traveled=80.0, average_speed=4.0,
                    waypoint_ade=1.0, waypoint_fde=2.0, collisions=1,
                ),
            ],
        )
        
        result.compute_aggregates()
        
        assert result.success_rate == pytest.approx(2/3)
        assert result.total_distance == 330.0
        assert result.average_speed == pytest.approx(5.0)
        assert result.average_ade == pytest.approx(0.6)
        assert result.total_collisions == 1


class TestMultiScenarioRunner:
    """Tests for MultiScenarioRunner class."""
    
    def test_runner_creation(self):
        """Test runner instantiation."""
        config = MultiScenarioConfig(scenarios=[])
        runner = MultiScenarioRunner(config)
        
        assert runner is not None
        assert runner.config == config
    
    def test_run_scenario_mock(self):
        """Test running a scenario with mock (no CARLA)."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.CUT_IN,
            town="Town01",
        )
        
        runner_config = MultiScenarioConfig(
            scenarios=[config],
            output_dir="/tmp/test_scenarios",
        )
        runner = MultiScenarioRunner(runner_config)
        
        result = runner.run_scenario(config, "test_001")
        
        assert result.scenario_id == "test_001"
        assert result.completed is True  # Mock execution
        assert result.duration > 0


class TestCreateStandardScenarioSuite:
    """Tests for create_standard_scenario_suite function."""
    
    def test_create_suite_with_all_types(self):
        """Test creating standard scenario suite."""
        scenarios = create_standard_scenario_suite(
            town="Town02",
            include_all_types=True,
        )
        
        assert len(scenarios) > 0
        assert all(s.town == "Town02" for s in scenarios)
    
    def test_suite_contains_multiple_types(self):
        """Test that suite contains different scenario types."""
        scenarios = create_standard_scenario_suite(include_all_types=True)
        
        types = set(s.scenario_type for s in scenarios)
        assert ScenarioType.CUT_IN in types
        assert ScenarioType.FOLLOW in types
        assert ScenarioType.PEDESTRIAN in types
        assert ScenarioType.EMERGENCY_BRAKE in types


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

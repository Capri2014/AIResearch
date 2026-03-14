"""Tests for scenario configuration module."""

import pytest
from sim.driving.carla_srunner.scenario_config import (
    ScenarioConfig,
    WeatherPreset,
    WeatherConfig,
    ScenarioType,
    MapName,
    TimeOfDay,
    get_scenario,
    get_scenario_suite,
    list_available_scenarios,
    get_scenarios_by_tag,
    get_scenarios_by_difficulty,
    to_dict,
    SCENARIO_DEFINITIONS,
)


class TestWeatherConfig:
    """Tests for WeatherConfig."""
    
    def test_weather_preset_clear(self):
        config = WeatherConfig.from_preset(WeatherPreset.CLEAR)
        assert config.preset == WeatherPreset.CLEAR
        assert config.cloudiness == 0.0
        assert config.precipitation == 0.0
    
    def test_weather_preset_rain(self):
        config = WeatherConfig.from_preset(WeatherPreset.RAIN)
        assert config.preset == WeatherPreset.RAIN
        assert config.precipitation > 0
        assert config.wetness > 0
    
    def test_weather_preset_night(self):
        config = WeatherConfig.from_preset(WeatherPreset.NIGHT)
        assert config.preset == WeatherPreset.NIGHT
        assert config.sun_altitude_angle < 0
    
    def test_weather_preset_fog(self):
        config = WeatherConfig.from_preset(WeatherPreset.FOG)
        assert config.preset == WeatherPreset.FOG
        assert config.fog_density > 0


class TestScenarioDefinitions:
    """Tests for scenario definitions."""
    
    def test_straight_clear_exists(self):
        scenario = get_scenario("straight_clear")
        assert scenario is not None
        assert scenario.type == ScenarioType.STRAIGHT
        assert scenario.map == MapName.TOWN01
    
    def test_straight_cloudy_exists(self):
        scenario = get_scenario("straight_cloudy")
        assert scenario is not None
        assert scenario.weather.preset == WeatherPreset.CLOUDY
    
    def test_straight_night_exists(self):
        scenario = get_scenario("straight_night")
        assert scenario is not None
        assert scenario.time_of_day == TimeOfDay.NIGHT
    
    def test_straight_rain_exists(self):
        scenario = get_scenario("straight_rain")
        assert scenario is not None
        assert scenario.weather.preset == WeatherPreset.RAIN
    
    def test_turn_scenarios_exist(self):
        assert get_scenario("turn_left_clear") is not None
        assert get_scenario("turn_right_clear") is not None
    
    def test_all_scenarios_have_routes(self):
        for sid, scenario in SCENARIO_DEFINITIONS.items():
            assert scenario.route is not None, f"{sid} missing route"


class TestScenarioSuites:
    """Tests for scenario suites."""
    
    def test_smoke_suite(self):
        scenarios = get_scenario_suite("smoke")
        assert len(scenarios) == 2
        assert "straight_clear" in [s.id for s in scenarios]
    
    def test_quick_suite(self):
        scenarios = get_scenario_suite("quick")
        assert len(scenarios) == 3
    
    def test_full_suite(self):
        scenarios = get_scenario_suite("full")
        assert len(scenarios) == 8
    
    def test_adverse_suite(self):
        scenarios = get_scenario_suite("adverse")
        assert len(scenarios) == 3
    
    def test_invalid_suite_returns_smoke(self):
        scenarios = get_scenario_suite("invalid_suite")
        assert scenarios == get_scenario_suite("smoke")


class TestScenarioFiltering:
    """Tests for scenario filtering."""
    
    def test_get_by_tag_straight(self):
        scenarios = get_scenarios_by_tag("straight")
        assert len(scenarios) >= 4
    
    def test_get_by_tag_night(self):
        scenarios = get_scenarios_by_tag("night")
        assert len(scenarios) >= 1
    
    def test_get_by_tag_rain(self):
        scenarios = get_scenarios_by_tag("rain")
        assert len(scenarios) >= 1
    
    def test_get_by_difficulty_easy(self):
        scenarios = get_scenarios_by_difficulty("easy")
        assert len(scenarios) >= 2
    
    def test_get_by_difficulty_hard(self):
        scenarios = get_scenarios_by_difficulty("hard")
        assert len(scenarios) >= 1


class TestSerialization:
    """Tests for scenario serialization."""
    
    def test_to_dict_structure(self):
        scenario = get_scenario("straight_clear")
        assert scenario is not None
        
        data = to_dict(scenario)
        
        assert "id" in data
        assert "name" in data
        assert "type" in data
        assert "map" in data
        assert "weather" in data
        assert "difficulty" in data
        assert "tags" in data
    
    def test_to_dict_weather_data(self):
        scenario = get_scenario("straight_clear")
        assert scenario is not None
        
        data = to_dict(scenario)
        weather = data["weather"]
        
        assert "preset" in weather
        assert "cloudiness" in weather
        assert "precipitation" in weather


class TestListScenarios:
    """Tests for listing scenarios."""
    
    def test_list_available_scenarios(self):
        scenarios = list_available_scenarios()
        assert len(scenarios) >= 8
        assert "straight_clear" in scenarios
        assert "straight_cloudy" in scenarios


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

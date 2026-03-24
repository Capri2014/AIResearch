"""
Tests for CARLA ScenarioRunner RL Checkpoint Evaluation Module.
"""

import pytest
from sim.driving.carla_srunner.rl_srunner_eval import (
    RLEvalConfig,
    RLScenarioMetrics,
    RLCheckpointResult,
    RLScenarioEvaluator,
    MultiCheckpointComparison,
    EvalSuite,
    run_evaluation,
)


class TestRLEvalConfig:
    """Tests for RLEvalConfig dataclass."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = RLEvalConfig()
        
        assert config.suite == "smoke"
        assert config.carla_host == "127.0.0.1"
        assert config.carla_port == 2000
        assert config.num_runs_per_scenario == 3
        assert config.timeout_s == 60 * 30
        assert config.dry_run is False
    
    def test_config_custom(self):
        """Test custom configuration values."""
        config = RLEvalConfig(
            checkpoint="out/ppo/model.pt",
            output_dir=Path("out/eval/custom"),
            suite="standard",
            num_runs_per_scenario=5,
            dry_run=True,
        )
        
        assert config.checkpoint == "out/ppo/model.pt"
        assert str(config.output_dir) == "out/eval/custom"
        assert config.suite == "standard"
        assert config.num_runs_per_scenario == 5
        assert config.dry_run is True
    
    def test_checkpoint_list_conversion(self):
        """Test automatic checkpoint list conversion."""
        config = RLEvalConfig(checkpoint="out/ppo/model.pt")
        
        assert config.checkpoints == ["out/ppo/model.pt"]
        
        config2 = RLEvalConfig(checkpoints=["a.pt", "b.pt"])
        assert config2.checkpoints == ["a.pt", "b.pt"]


class TestRLScenarioMetrics:
    """Tests for RLScenarioMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test scenario metrics creation."""
        metrics = RLScenarioMetrics(
            scenario_id="test_001",
            scenario_type="CutIn",
            success=True,
            completed=True,
            ade=1.5,
            fde=3.2,
            collisions=0,
        )
        
        assert metrics.scenario_id == "test_001"
        assert metrics.scenario_type == "CutIn"
        assert metrics.success is True
        assert metrics.ade == 1.5
        assert metrics.collisions == 0
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = RLScenarioMetrics(
            scenario_id="test_001",
            scenario_type="FollowVehicle",
            success=True,
            completed=True,
            ade=2.0,
            fde=4.5,
            collisions=1,
            max_acceleration=2.5,
            max_deceleration=5.0,
            max_lateral_accel=3.0,
            jerk_avg=2.0,
            distance_traveled=100.0,
            average_speed=8.0,
            travel_time=12.5,
            efficiency_score=0.8,
        )
        
        d = metrics.to_dict()
        
        assert d["scenario_id"] == "test_001"
        assert d["success"] is True
        assert d["metrics"]["ade"] == 2.0
        assert d["metrics"]["collisions"] == 1


class TestRLCheckpointResult:
    """Tests for RLCheckpointResult dataclass."""
    
    def test_result_creation(self):
        """Test checkpoint result creation."""
        result = RLCheckpointResult(
            checkpoint_path="out/ppo/model.pt",
            checkpoint_step=1000,
            success_rate=85.0,
            completion_rate=90.0,
            avg_ade=1.8,
            avg_fde=4.0,
            avg_collisions=0.3,
            avg_comfort_score=0.75,
            avg_efficiency=0.8,
            overall_score=78.5,
            num_scenarios=5,
        )
        
        assert result.checkpoint_path == "out/ppo/model.pt"
        assert result.checkpoint_step == 1000
        assert result.success_rate == 85.0
        assert result.overall_score == 78.5
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = RLCheckpointResult(
            checkpoint_path="out/ppo/model.pt",
            checkpoint_step=500,
            success_rate=80.0,
            avg_ade=2.0,
            num_scenarios=3,
        )
        
        d = result.to_dict()
        
        assert d["checkpoint_path"] == "out/ppo/model.pt"
        assert d["checkpoint_step"] == 500
        assert d["success_rate"] == 80.0


class TestRLScenarioEvaluator:
    """Tests for RLScenarioEvaluator class."""
    
    def test_smoke_suite(self):
        """Test smoke evaluation suite."""
        config = RLEvalConfig(suite="smoke")
        evaluator = RLScenarioEvaluator(config)
        
        assert len(evaluator._scenarios) >= 2
        assert "FollowVehicle" in evaluator._scenarios
    
    def test_standard_suite(self):
        """Test standard evaluation suite."""
        config = RLEvalConfig(suite="standard")
        evaluator = RLScenarioEvaluator(config)
        
        assert len(evaluator._scenarios) >= 5
        assert "CutIn" in evaluator._scenarios
        assert "LaneChange" in evaluator._scenarios
    
    def test_comprehensive_suite(self):
        """Test comprehensive evaluation suite."""
        config = RLEvalConfig(suite="full")
        evaluator = RLScenarioEvaluator(config)
        
        assert len(evaluator._scenarios) >= 10
    
    def test_checkpoint_step_parsing(self):
        """Test checkpoint step extraction from filename."""
        config = RLEvalConfig()
        evaluator = RLScenarioEvaluator(config)
        
        assert evaluator._parse_checkpoint_step("checkpoint_050.pt") == 50
        assert evaluator._parse_checkpoint_step("model_1000.pt") == 1000
        assert evaluator._parse_checkpoint_step("step_500.pt") == 500
        assert evaluator._parse_checkpoint_step("final.pt") is None
    
    def test_evaluate_single_checkpoint(self):
        """Test single checkpoint evaluation."""
        config = RLEvalConfig(
            checkpoint="out/bev_ssl_ppo_refine/checkpoint_050.pt",
            suite="smoke",
            num_runs_per_scenario=2,
            dry_run=True,
        )
        evaluator = RLScenarioEvaluator(config)
        
        result = evaluator.evaluate_checkpoint(config.checkpoint)
        
        assert result.checkpoint_path == config.checkpoint
        assert result.checkpoint_step == 50
        assert len(result.scenario_results) >= 2
        assert 0 <= result.success_rate <= 100
        assert result.avg_ade >= 0
    
    def test_aggregate_results(self):
        """Test results aggregation."""
        config = RLEvalConfig()
        evaluator = RLScenarioEvaluator(config)
        
        results = [
            RLScenarioMetrics("s1", "CutIn", True, True, ade=1.0, fde=2.0, collisions=0),
            RLScenarioMetrics("s2", "CutIn", True, True, ade=1.5, fde=3.0, collisions=1),
            RLScenarioMetrics("s3", "Follow", False, True, ade=3.0, fde=6.0, collisions=0),
        ]
        
        aggregated = evaluator._aggregate_results(results)
        
        assert aggregated.num_scenarios == 3
        assert aggregated.avg_ade == pytest.approx(1.83, rel=0.1)
        assert aggregated.success_rate == pytest.approx(66.67, rel=0.1)


class TestMultiCheckpointComparison:
    """Tests for MultiCheckpointComparison class."""
    
    def test_comparison_creation(self):
        """Test comparison creation with rankings."""
        results = [
            RLCheckpointResult("a.pt", 100, success_rate=90.0, avg_ade=1.5, overall_score=85.0),
            RLCheckpointResult("b.pt", 200, success_rate=80.0, avg_ade=2.0, overall_score=75.0),
            RLCheckpointResult("c.pt", 150, success_rate=85.0, avg_ade=1.8, overall_score=80.0),
        ]
        
        comparison = MultiCheckpointComparison(
            checkpoints=["a.pt", "b.pt", "c.pt"],
            results=results,
        )
        
        assert len(comparison.results) == 3
        assert comparison.best_by_success == "a.pt"
        assert comparison.best_by_ade == "a.pt"
        assert comparison.best_by_overall == "a.pt"


class TestRunEvaluation:
    """Tests for run_evaluation function."""
    
    def test_single_checkpoint_dry_run(self):
        """Test single checkpoint evaluation in dry-run mode."""
        result, comparison = run_evaluation(
            checkpoint="out/ppo/model.pt",
            suite="smoke",
            num_runs=2,
            dry_run=True,
        )
        
        assert result is not None
        assert comparison is None
        assert result.checkpoint_path == "out/ppo/model.pt"
        assert result.num_scenarios >= 2
    
    def test_multiple_checkpoints_dry_run(self):
        """Test multiple checkpoint comparison in dry-run mode."""
        result, comparison = run_evaluation(
            checkpoints=["out/ppo/model_100.pt", "out/ppo/model_200.pt"],
            suite="smoke",
            num_runs=1,
            dry_run=True,
        )
        
        assert result is None
        assert comparison is not None
        assert len(comparison.results) == 2


# Helper to test Path import
from pathlib import Path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

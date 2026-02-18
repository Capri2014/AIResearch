"""
CARLA Environments Package
=========================

CARLA integration for RL training and evaluation.

Modules:
- carla_gym_env: Gym-like environment for fast training
- carla_scenario_eval: ScenarioRunner evaluation wrapper

Usage:
    from training.rl.envs import CarlaGymEnv, ScenarioEvaluator
    
    # Training
    env = CarlaGymEnv()
    obs = env.reset()
    
    # Evaluation
    evaluator = ScenarioEvaluator(scenarios=["lane_change", "straight"])
    results = evaluator.evaluate(policy)
"""

from .carla_gym_env import (
    CarlaGymEnv,
    CarlaGymConfig,
    make_carla_env,
)

from .carla_scenario_eval import (
    ScenarioEvaluator,
    ScenarioEvaluatorConfig,
    EpisodeResult,
    InfractionType,
    evaluate_rl_policy,
)

__all__ = [
    # Gym env
    "CarlaGymEnv",
    "CarlaGymConfig",
    "make_carla_env",
    # Scenario eval
    "ScenarioEvaluator",
    "ScenarioEvaluatorConfig",
    "EpisodeResult",
    "InfractionType",
    "evaluate_rl_policy",
]

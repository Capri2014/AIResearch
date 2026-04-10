"""CARLA ScenarioRunner Integration Package.

Driving-first plan: Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval

This package provides:
- scenarios.py: Scenario/route definitions for CARLA evaluation
- runner.py: Execution layer that runs scenarios via ScenarioRunner
"""

from sim.driving.carla_srunner.scenarios import (
    SCENARIO_CATALOG,
    ROUTE_CATALOG,
    SCENARIO_SUITES,
    ScenarioDef,
    RouteDef,
    get_scenario,
    get_route,
    get_suite,
    list_scenarios,
    list_routes,
    list_suites,
    generate_srunner_xml,
    generate_metrics_config,
)

from sim.driving.carla_srunner.runner import (
    RunnerConfig,
    ScenarioRunner,
    build_srunner_command,
    build_srunner_command_for_route,
    check_carla_connection,
)

from sim.driving.carla_srunner.evaluate import (
    EvalRunConfig,
    EvalMetrics,
    ScenarioResult,
    run_suite_evaluation,
    run_single_scenario,
    parse_srunner_output,
)

from sim.driving.carla_srunner.policy_wrapper import (
    PolicyConfig,
    WaypointPolicyWrapper,
    StubPolicyWrapper,
    load_policy,
)

__all__ = [
    # Scenarios
    "SCENARIO_CATALOG",
    "ROUTE_CATALOG", 
    "SCENARIO_SUITES",
    "ScenarioDef",
    "RouteDef",
    "get_scenario",
    "get_route",
    "get_suite",
    "list_scenarios",
    "list_routes",
    "list_suites",
    "generate_srunner_xml",
    "generate_metrics_config",
    # Runner
    "RunnerConfig",
    "ScenarioRunner",
    "build_srunner_command",
    "build_srunner_command_for_route",
    "check_carla_connection",
    # Evaluate
    "EvalRunConfig",
    "EvalMetrics",
    "ScenarioResult",
    "run_suite_evaluation",
    "run_single_scenario",
    "parse_srunner_output",
    # Policy
    "PolicyConfig",
    "WaypointPolicyWrapper",
    "StubPolicyWrapper",
    "load_policy",
]

__version__ = "0.1.0"
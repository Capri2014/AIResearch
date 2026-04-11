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

from sim.driving.carla_srunner.visualize import (
    plot_single_run,
    plot_comparison,
    load_metrics,
    load_all_runs,
    generate_markdown_table,
)

from sim.driving.carla_srunner.test_harness import (
    run_tests,
    TestPolicyWrapper,
    TestScenarios,
    TestRunner,
    TestEvaluate,
    TestIntegration,
    TestVisualize,
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
    # Visualize
    "plot_single_run",
    "plot_comparison",
    "load_metrics",
    "load_all_runs",
    "generate_markdown_table",
    # Test harness
    "run_tests",
    "TestPolicyWrapper",
    "TestScenarios",
    "TestRunner",
    "TestEvaluate",
    "TestIntegration",
    "TestVisualize",
]

__version__ = "0.2.0"
"""
Contingency Planning Package

Implements and compares tree-based and model-based contingency planning
for safety-critical autonomous driving.
"""

__version__ = "0.1.0"

from .evaluation.metrics import ContingencyMetrics, ComparisonResults, compute_metrics
from .simulation.scenarios import (
    ContingencyScenario,
    ContingencyType,
    DiscreteUncertainty,
    SCENARIOS,
    get_scenario,
    get_all_scenarios,
)

__all__ = [
    "ContingencyMetrics",
    "ComparisonResults", 
    "compute_metrics",
    "ContingencyScenario",
    "ContingencyType",
    "DiscreteUncertainty",
    "SCENARIOS",
    "get_scenario",
    "get_all_scenarios",
]

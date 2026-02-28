"""
Contingency Scenarios for Testing

Defines discrete uncertainties for each scenario type.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any


class ContingencyType(Enum):
    """Categories of contingencies."""
    # External/Interactive
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    VEHICLE_CUT_IN = "vehicle_cut_in"
    OCCLUDED_INTERSECTION = "occluded_intersection"
    
    # External/Environmental
    ADVERSE_WEATHER = "adverse_weather"
    ROAD_CONSTRUCTION = "road_construction"
    
    # Internal/System
    SENSOR_DEGRADATION = "sensor_degradation"
    ACTUATOR_FAILURE = "actuator_failure"


@dataclass
class DiscreteUncertainty:
    """Represents a discrete uncertainty (contingency hypothesis)."""
    name: str
    probability: float  # Prior probability
    description: str


@dataclass
class ContingencyScenario:
    """A scenario with discrete uncertainties."""
    name: str
    contingency_type: ContingencyType
    hypotheses: List[DiscreteUncertainty]
    initial_state: Dict[str, Any]
    goal_state: Dict[str, Any]
    obstacles: List[Dict[str, Any]]


# Define standard scenarios
SCENARIOS = {
    "pedestrian_crossing": ContingencyScenario(
        name="Pedestrian Crossing",
        contingency_type=ContingencyType.PEDESTRIAN_CROSSING,
        hypotheses=[
            DiscreteUncertainty("pedestrian_yields", 0.7, "Pedestrian waits for ego"),
            DiscreteUncertainty("pedestrian_crosses", 0.3, "Pedestrian crosses the road"),
        ],
        initial_state={"x": 0, "y": 0, "v": 10.0, "heading": 0},
        goal_state={"x": 100, "y": 0},
        obstacles=[]
    ),
    
    "highway_cut_in": ContingencyScenario(
        name="Highway Cut-in",
        contingency_type=ContingencyType.VEHICLE_CUT_IN,
        hypotheses=[
            DiscreteUncertainty("vehicle_maintains_lane", 0.8, "Other vehicle stays in lane"),
            DiscreteUncertainty("vehicle_cuts_in", 0.2, "Other vehicle cuts into ego lane"),
        ],
        initial_state={"x": 0, "y": 0, "v": 30.0, "heading": 0},
        goal_state={"x": 500, "y": 0},
        obstacles=[{"x": 50, "y": 3.5, "type": "vehicle"}]  # Adjacent lane
    ),
    
    "occluded_intersection": ContingencyScenario(
        name="Occluded Intersection",
        contingency_type=ContingencyType.OCCLUDED_INTERSECTION,
        hypotheses=[
            DiscreteUncertainty("clear", 0.6, "No crossing traffic"),
            DiscreteUncertainty("blocked", 0.4, "Hidden vehicle/pedestrian"),
        ],
        initial_state={"x": 0, "y": 0, "v": 5.0, "heading": 0},
        goal_state={"x": 50, "y": 0},
        obstacles=[{"x": 30, "y": 0, "occluded": True}]  # Occlusion point
    ),
    
    "sensor_degradation": ContingencyScenario(
        name="Sensor Degradation",
        contingency_type=ContingencyType.SENSOR_DEGRADATION,
        hypotheses=[
            DiscreteUncertainty("nominal", 0.95, "All sensors working"),
            DiscreteUncertainty("degraded", 0.05, "Camera/lidar degradation"),
        ],
        initial_state={"x": 0, "y": 0, "v": 10.0, "heading": 0},
        goal_state={"x": 100, "y": 0},
        obstacles=[]
    ),
}


def get_scenario(name: str) -> ContingencyScenario:
    """Get scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def get_all_scenarios() -> Dict[str, ContingencyScenario]:
    """Get all defined scenarios."""
    return SCENARIOS.copy()

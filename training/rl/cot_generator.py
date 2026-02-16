"""
CoT (Chain of Thought) Trace Generator for Autonomous Driving

Generates structured reasoning traces from driving episodes using:
1. Rule-based heuristic planner
2. LLM-augmented traces (optional)

Usage:
    python -m training.rl.cot_generator --input /data/waymo --output /data/cot_traces.jsonl --strategy rule_based
"""

import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class PerceptionState:
    """Perception output from ego vehicle."""
    vehicles: List[Dict] = field(default_factory=list)
    pedestrians: List[Dict] = field(default_factory=list)
    lane_lines: Dict = field(default_factory=dict)
    traffic_signs: List[Dict] = field(default_factory=list)
    traffic_lights: Dict = field(default_factory=dict)
    ego_speed: float = 0.0
    ego_heading: float = 0.0


@dataclass
class CoTTrace:
    """Chain of Thought reasoning trace for driving decisions."""
    episode_id: str = ""
    timestamp: float = 0.0
    perception: str = ""
    situation_understanding: str = ""
    behavior_prediction: str = ""
    trajectory_planning: str = ""
    confidence: str = ""
    expert_waypoints: List[Dict] = field(default_factory=list)
    control_command: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DrivingCoTGenerator:
    """Generate reasoning traces from driving scenes using rule-based heuristics."""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def generate_trace(
        self, state: PerceptionState, episode_id: str = "", 
        timestamp: float = 0.0, waypoints=None, control=None
    ) -> CoTTrace:
        trace = CoTTrace(
            episode_id=episode_id, timestamp=timestamp,
            expert_waypoints=waypoints or [], control_command=control or {}
        )
        trace.perception = self._summarize_perception(state)
        trace.situation_understanding = self._understand_scene(state)
        trace.behavior_prediction = self._predict_behaviors(state)
        trace.trajectory_planning = self._plan_trajectory(state)
        trace.confidence = self._evaluate_confidence(state)
        return trace
    
    def _summarize_perception(self, state: PerceptionState) -> str:
        n_vehicles = len(state.vehicles)
        n_pedestrians = len(state.pedestrians)
        lane_conf = state.lane_lines.get('confidence', 0.0)
        lane_desc = "high" if lane_conf > 0.8 else "moderate" if lane_conf > 0.5 else "low"
        return f"{n_vehicles} vehicles, {n_pedestrians} pedestrians. Lane detection {lane_desc}. Ego speed: {state.ego_speed:.1f} m/s."
    
    def _understand_scene(self, state: PerceptionState) -> str:
        parts = []
        if self._detect_intersection(state):
            parts.append("Approaching intersection")
        if self._detect_highway(state):
            parts.append("On highway")
        leading = self._get_leading_vehicle(state)
        if leading:
            parts.append(f"Leading vehicle {leading.get('distance', 0):.1f}m ahead")
        return ". ".join(parts) + "." if parts else "Open road."
    
    def _predict_behaviors(self, state: PerceptionState) -> str:
        predictions = []
        for v in sorted(state.vehicles, key=lambda x: x.get('distance', float('inf')))[:3]:
            if v.get('distance', float('inf')) > 50:
                continue
            rel_vel = v.get('relative_velocity', 0)
            if rel_vel > 1:
                predictions.append(f"Vehicle ahead accelerating (+{rel_vel:.1f})")
            elif rel_vel < -1:
                predictions.append(f"Vehicle ahead slowing ({rel_vel:.1f})")
            else:
                predictions.append(f"Vehicle ahead stable")
        return " | ".join(predictions) if predictions else "No relevant behaviors."
    
    def _plan_trajectory(self, state: PerceptionState) -> str:
        parts = []
        leading = self._get_leading_vehicle(state)
        if leading and leading.get('distance', float('inf')) < 20:
            parts.append("Slowing for safe distance")
        elif state.ego_speed < 3:
            parts.append("Accelerating to cruising speed")
        else:
            parts.append("Maintaining speed")
        parts.append("Center lane positioning")
        return ". ".join(parts) + "."
    
    def _evaluate_confidence(self, state: PerceptionState) -> str:
        lane_conf = state.lane_lines.get('confidence', 0.0)
        n_agents = len(state.vehicles) + len(state.pedestrians)
        traffic = "low" if n_agents < 2 else "moderate" if n_agents < 5 else "complex"
        return f"Confidence: lane {lane_conf:.2f}, traffic {traffic}."
    
    def _detect_intersection(self, state: PerceptionState) -> bool:
        return any('stop' in s.get('type', '').lower() for s in state.traffic_signs)
    
    def _detect_highway(self, state: PerceptionState) -> bool:
        return state.ego_speed > 20 or any('highway' in s.get('type', '').lower() for s in state.traffic_signs)
    
    def _get_leading_vehicle(self, state: PerceptionState):
        ahead = [v for v in state.vehicles if 0 < v.get('distance', float('inf')) < 100]
        return min(ahead, key=lambda v: v.get('distance', float('inf')), default=None)


def main():
    parser = argparse.ArgumentParser(description="Generate CoT traces for driving data")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--strategy", default="rule_based")
    args = parser.parse_args()
    
    generator = DrivingCoTGenerator()
    print(f"CoT Generator ready. Input: {args.input}, Output: {args.output}")


if __name__ == "__main__":
    main()

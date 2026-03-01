"""
Behavior Tree + Rollout Planner

A hierarchical planning approach:
1. High-level behavior selection via behavior tree
2. Trajectory rollout for each behavior
3. Best behavior selected based on rollout cost

Best for: Urban negotiation, unprotected turns, merges
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import heapq


class Behavior(Enum):
    """High-level driving behaviors"""
    LANE_KEEP = "lane_keep"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    FOLLOW_LEAD = "follow_lead"
    YIELD = "yield"
    STOP = "stop"
    CREEP = "creep"
    MERGE = "merge"
    MERGE_LATE = "merge_late"
    MERGE_EARLY = "merge_early"


@dataclass
class BehaviorConfig:
    """Configuration for behavior tree planner"""
    # Behavior parameters
    lane_change_duration: float = 3.0  # seconds
    creep_speed: float = 2.0  # m/s
    yield_duration: float = 2.0  # seconds
    
    # Cost weights
    cost_progress: float = -1.0  # Negative = reward
    cost_lateral_motion: float = 2.0
    cost_acceleration: float = 1.0
    cost_jerk: float = 3.0
    cost_collision: float = 1000.0
    cost_violation: float = 500.0  # Rule violations
    

class BehaviorNode:
    """Base node for behavior tree"""
    
    def __init__(self, name: str):
        self.name = name
        self.parent: Optional[BehaviorNode] = None
        self.children: List[BehaviorNode] = []
    
    def execute(self, state: Dict) -> Tuple[bool, Dict]:
        """Execute behavior, return (success, result)"""
        raise NotImplementedError
    
    def add_child(self, child: 'BehaviorNode'):
        child.parent = self
        self.children.append(child)


class SequenceNode(BehaviorNode):
    """Sequence node: succeeds if ALL children succeed"""
    
    def __init__(self, name: str, children: List[BehaviorNode] = None):
        super().__init__(name)
        if children:
            for child in children:
                self.add_child(child)
    
    def execute(self, state: Dict) -> Tuple[bool, Dict]:
        for child in self.children:
            success, result = child.execute(state)
            if not success:
                return False, result
        return True, state


class SelectorNode(BehaviorNode):
    """Selector node: succeeds if ANY child succeeds"""
    
    def __init__(self, name: str, children: List[BehaviorNode] = None):
        super().__init__(name)
        if children:
            for child in children:
                self.add_child(child)
    
    def execute(self, state: Dict) -> Tuple[bool, Dict]:
        for child in self.children:
            success, result = child.execute(state)
            if success:
                return True, result
        return False, state


class LaneKeepNode(BehaviorNode):
    """Lane keep behavior"""
    
    def __init__(self, config: BehaviorConfig):
        super().__init__("lane_keep")
        self.config = config
    
    def execute(self, state: Dict) -> Tuple[bool, Dict]:
        # Simple: always executable if in lane
        trajectory = self._generate_trajectory(state)
        return True, {'behavior': Behavior.LANE_KEEP, 'trajectory': trajectory}
    
    def _generate_trajectory(self, state: Dict) -> List[Tuple[float, float, float]]:
        """Generate lane keep trajectory"""
        s = state.get('s', 0)
        l = state.get('l', 0)
        v = state.get('speed', 10)
        
        trajectory = []
        dt = 0.1
        for i in range(60):  # 6 seconds
            t = i * dt
            trajectory.append((s + v * t, l, t))
        
        return trajectory


class LaneChangeNode(BehaviorNode):
    """Lane change behavior"""
    
    def __init__(self, config: BehaviorConfig, direction: str = "left"):
        super().__init__(f"lane_change_{direction}")
        self.config = config
        self.direction = 1 if direction == "right" else -1
    
    def execute(self, state: Dict) -> Tuple[bool, Dict]:
        # Check if lane change is feasible
        target_l = state.get('l', 0) + self.direction * 3.5
        
        # Check gaps
        gaps = state.get('gaps', [])
        valid_gap = any(abs(gap['l'] - target_l) < 1.0 for gap in gaps)
        
        if not valid_gap:
            return False, {'reason': 'no_gap'}
        
        trajectory = self._generate_trajectory(state)
        return True, {
            'behavior': Behavior.LANE_CHANGE_RIGHT if self.direction > 0 else Behavior.LANE_CHANGE_LEFT,
            'trajectory': trajectory,
            'target_lane': target_l
        }
    
    def _generate_trajectory(self, state: Dict) -> List[Tuple[float, float, float]]:
        """Generate lane change trajectory"""
        s = state.get('s', 0)
        l = state.get('l', 0)
        v = state.get('speed', 10)
        
        target_l = l + self.direction * 3.5
        duration = self.config.lane_change_duration
        
        trajectory = []
        dt = 0.1
        n_steps = int(duration / dt)
        
        for i in range(n_steps):
            t = i * dt
            progress = t / duration
            
            # Smooth lateral motion (sigmoid-like)
            lat_progress = progress ** 2 * (3 - 2 * progress)  # Smoothstep
            current_l = l + (target_l - l) * lat_progress
            
            trajectory.append((s + v * t, current_l, t))
        
        # Fill remaining horizon
        for i in range(n_steps, 60):
            t = i * dt
            trajectory.append((s + v * t, target_l, t))
        
        return trajectory


class FollowLeadNode(BehaviorNode):
    """Follow lead vehicle behavior"""
    
    def __init__(self, config: BehaviorConfig):
        super().__init__("follow_lead")
        self.config = config
    
    def execute(self, state: Dict) -> Tuple[bool, Dict]:
        # Check if lead vehicle detected
        lead = state.get('lead_vehicle')
        if lead is None:
            return False, {'reason': 'no_lead'}
        
        trajectory = self._generate_trajectory(state, lead)
        return True, {'behavior': Behavior.FOLLOW_LEAD, 'trajectory': trajectory}
    
    def _generate_trajectory(self, state: Dict, lead: Dict) -> List[Tuple[float, float, float]]:
        """Generate follow trajectory"""
        s = state.get('s', 0)
        l = state.get('l', 0)
        lead_s = lead.get('s', s + 20)
        
        # Target: follow at safe distance
        target_gap = 20  # meters
        target_s = lead_s - target_gap
        
        trajectory = []
        dt = 0.1
        current_s = s
        
        for i in range(60):
            t = i * dt
            # Gradually approach target gap
            current_s = current_s + (target_s - current_s) * 0.1
            trajectory.append((current_s, l, t))
        
        return trajectory


class YieldNode(BehaviorNode):
    """Yield to other agents"""
    
    def __init__(self, config: BehaviorConfig):
        super().__init__("yield")
        self.config = config
    
    def execute(self, state: Dict) -> Tuple[bool, Dict]:
        # Check if there are crossing agents
        crossing = state.get('crossing_agents', [])
        if not crossing:
            return False, {'reason': 'no_crossing'}
        
        trajectory = self._generate_trajectory(state)
        return True, {'behavior': Behavior.YIELD, 'trajectory': trajectory}
    
    def _generate_trajectory(self, state: Dict) -> List[Tuple[float, float, float]]:
        """Generate yield trajectory (slow down)"""
        s = state.get('s', 0)
        l = state.get('l', 0)
        v = state.get('speed', 10)
        
        trajectory = []
        dt = 0.1
        
        # Decelerate
        for i in range(20):
            t = i * dt
            v_current = max(v * (1 - i/20), 1.0)
            s += v_current * dt
            trajectory.append((s, l, t))
        
        # Creep
        for i in range(40):
            t = i * dt + 2.0
            s += self.config.creep_speed * dt
            trajectory.append((s, l, t))
        
        return trajectory


class BehaviorTreePlanner:
    """
    Behavior Tree + Rollout Planner
    
    Architecture:
    - Selector at root (try behaviors in priority order)
    - Sequence nodes for complex behaviors
    - Leaf nodes are executable behaviors
    """
    
    def __init__(self, config: Optional[BehaviorConfig] = None):
        self.config = config or BehaviorConfig()
        self.tree = self._build_tree()
        
    def _build_tree(self) -> BehaviorNode:
        """Build the behavior tree"""
        
        # Leaf behaviors
        lane_keep = LaneKeepNode(self.config)
        follow_lead = FollowLeadNode(self.config)
        change_left = LaneChangeNode(self.config, "left")
        change_right = LaneChangeNode(self.config, "right")
        yield_behavior = YieldNode(self.config)
        
        # Priority: follow > change > yield > keep
        root = SelectorNode("root", [
            follow_lead,
            change_left,
            change_right,
            yield_behavior,
            lane_keep
        ])
        
        return root
    
    def plan(self, 
             state: Dict,
             obstacles: Optional[List[Dict]] = None) -> Dict:
        """
        Plan using behavior tree.
        
        Args:
            state: Current state {'s', 'l', 'speed', 'gaps', 'lead_vehicle', 'crossing_agents'}
            obstacles: Optional obstacle list
            
        Returns:
            {'behavior', 'trajectory', 'cost'}
        """
        obstacles = obstacles or []
        
        # Execute tree
        success, result = self.tree.execute(state)
        
        if not success:
            # Fallback to lane keep
            result = {'behavior': Behavior.LANE_KEEP, 
                     'trajectory': self._fallback_trajectory(state)}
        
        # Compute cost
        cost = self._compute_cost(result['trajectory'], obstacles)
        result['cost'] = cost
        
        return result
    
    def _compute_cost(self, 
                      trajectory: List[Tuple[float, float, float]],
                      obstacles: List[Dict]) -> float:
        """Compute trajectory cost"""
        cost = 0.0
        
        # Progress reward
        final_s = trajectory[-1][0]
        cost += self.config.cost_progress * final_s
        
        # Lateral motion cost
        l_values = [p[1] for p in trajectory]
        lateral_motion = max(l_values) - min(l_values)
        cost += self.config.cost_lateral_motion * lateral_motion
        
        # Collision cost (simplified)
        for obs in obstacles:
            obs_s = obs.get('s', 0)
            obs_l = obs.get('l', 0)
            for s, l, t in trajectory:
                dist = np.sqrt((s - obs_s)**2 + (l - obs_l)**2)
                if dist < 3.0:
                    cost += self.config.cost_collision
        
        return cost
    
    def _fallback_trajectory(self, state: Dict) -> List[Tuple[float, float, float]]:
        """Generate simple fallback"""
        s = state.get('s', 0)
        l = state.get('l', 0)
        v = state.get('speed', 10)
        
        return [(s + v * t, l, t * 0.1) for t in range(60)]
    
    def get_all_behaviors(self, state: Dict) -> List[Dict]:
        """
        Get rollout for ALL behaviors (for evaluation).
        
        This enables the "rollout" part of Behavior Tree + Rollout.
        """
        behaviors = []
        
        # Try each behavior
        all_behavior_nodes = [
            LaneKeepNode(self.config),
            LaneChangeNode(self.config, "left"),
            LaneChangeNode(self.config, "right"),
            FollowLeadNode(self.config),
            YieldNode(self.config)
        ]
        
        for node in all_behavior_nodes:
            success, result = node.execute(state)
            if success:
                cost = self._compute_cost(result['trajectory'], [])
                behaviors.append({
                    'behavior': result['behavior'],
                    'trajectory': result['trajectory'],
                    'cost': cost,
                    'is_feasible': True
                })
            else:
                behaviors.append({
                    'behavior': node.name,
                    'trajectory': [],
                    'cost': float('inf'),
                    'is_feasible': False,
                    'failure_reason': result.get('reason', 'unknown')
                })
        
        # Sort by cost
        behaviors.sort(key=lambda x: x['cost'])
        return behaviors


def create_behavior_tree_planner() -> BehaviorTreePlanner:
    """Factory function"""
    config = BehaviorConfig(
        lane_change_duration=3.0,
        creep_speed=2.0,
        yield_duration=2.0,
        cost_progress=-1.0,
        cost_lateral_motion=2.0,
        cost_acceleration=1.0,
        cost_jerk=3.0,
        cost_collision=1000.0,
        cost_violation=500.0
    )
    return BehaviorTreePlanner(config)


if __name__ == "__main__":
    # Test behavior tree planner
    planner = create_behavior_tree_planner()
    
    # Test state
    state = {
        's': 0,
        'l': 0,
        'speed': 10,
        'gaps': [{'l': 3.5, 'available': True}],
        'lead_vehicle': {'s': 30, 'speed': 8},
        'crossing_agents': []
    }
    
    # Get best behavior
    result = planner.plan(state)
    print(f"Best behavior: {result['behavior'].value}")
    print(f"Cost: {result['cost']:.2f}")
    
    # Get all behavior rollouts
    all_behaviors = planner.get_all_behaviors(state)
    print(f"\nAll behaviors:")
    for b in all_behaviors:
        print(f"  {b['behavior'].value if hasattr(b['behavior'], 'value') else b['behavior']}: "
              f"cost={b['cost']:.2f}, feasible={b['is_feasible']}")

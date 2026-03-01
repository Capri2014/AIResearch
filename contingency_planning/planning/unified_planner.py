"""
Unified Planner - Integrates all planning approaches

This is the top-level planner that coordinates:
1. Corridor Manager - generates corridor hypotheses
2. Candidate Generators (Lattice, Behavior Tree)
3. SDF Collision Checker - evaluates candidates
4. Selector - picks best feasible trajectory

Target: 20 Hz, 50 ms budget
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .lattice_planner import create_lattice_planner, LatticePlanner
from .corridor_manager import create_corridor_manager, CorridorManager, Corridor
from .sdf_collision import create_collision_checker, SDFCollisionChecker
from .behavior_tree import create_behavior_tree_planner, BehaviorTreePlanner


@dataclass
class UnifiedPlannerConfig:
    """Configuration for unified planner"""
    # Rate
    rate_hz: int = 20
    
    # Corridor
    n_corridors: int = 4
    
    # Candidates
    k_candidates: int = 128
    
    # Evaluation
    coarse_steps: int = 60
    fine_steps: int = 12
    
    # Risk
    risk_alpha: float = 0.2  # CVaR


class UnifiedPlanner:
    """
    Unified planner integrating multiple planning approaches.
    
    Architecture:
    1. Corridor Manager → N corridor hypotheses
    2. Candidate Generator → K trajectory candidates
    3. Evaluator → Score all candidates
    4. Selector → Pick best, apply MRM if none feasible
    """
    
    def __init__(self, config: Optional[UnifiedPlannerConfig] = None):
        self.config = config or UnifiedPlannerConfig()
        
        # Initialize components
        self.corridor_manager = create_corridor_manager(self.config.n_corridors)
        self.lattice_planner = create_lattice_planner()
        self.behavior_tree = create_behavior_tree_planner()
        self.collision_checker = create_collision_checker()
        
        # State
        self.current_corridors: List[Corridor] = []
        self.previous_trajectory: Optional[List[Tuple[float, float]]] = None
        
    def update(self, 
               ego_state: Dict,
               map_data: Optional[Dict] = None,
               perception_data: Optional[Dict] = None) -> Dict:
        """
        Main planning update.
        
        Args:
            ego_state: {'x', 'y', 'heading', 'speed', 's', 'l'}
            map_data: Optional map information
            perception_data: {'obstacles': [...]}
            
        Returns:
            {'trajectory': [(x, y), ...], 
             'behavior': str,
             'is_feasible': bool,
             'candidates': int}
        """
        # 1. Update corridors
        self.current_corridors = self.corridor_manager.update(
            ego_state, map_data, perception_data
        )
        
        # 2. Generate candidates
        candidates = self._generate_candidates(ego_state)
        
        # 3. Evaluate candidates
        evaluations = self._evaluate_candidates(candidates)
        
        # 4. Select best
        best = self._select(evaluations)
        
        if best is None:
            # Fallback to MRM
            return self._generate_mrm(ego_state)
        
        return best
    
    def _generate_candidates(self, ego_state: Dict) -> List[Dict]:
        """Generate K candidates from all approaches"""
        candidates = []
        
        # Get start state
        s = ego_state.get('s', 0)
        l = ego_state.get('l', 0)
        heading = ego_state.get('heading', 0)
        start_state = (s, l, heading)
        
        # Goal
        goal_s = ego_state.get('goal_s', 100)
        
        # Per-corridor candidates
        for corridor in self.current_corridors:
            # Lattice candidates for this corridor
            self.lattice_planner.config.l_lanes = 3
            lattice_candidates = self.lattice_planner.get_candidate_trajectories(
                start_state, goal_s, n_candidates=32
            )
            
            for traj in lattice_candidates:
                candidates.append({
                    'trajectory': traj,
                    'corridor_id': corridor.corridor_id,
                    'corridor_confidence': corridor.confidence,
                    'approach': 'lattice'
                })
        
        # Behavior tree rollouts
        bt_state = {
            's': ego_state.get('s', 0),
            'l': ego_state.get('l', 0),
            'speed': ego_state.get('speed', 10),
            'gaps': ego_state.get('gaps', []),
            'lead_vehicle': ego_state.get('lead_vehicle'),
            'crossing_agents': ego_state.get('crossing_agents', [])
        }
        
        bt_behaviors = self.behavior_tree.get_all_behaviors(bt_state)
        
        for behavior in bt_behaviors:
            if behavior['is_feasible']:
                # Convert (s, l) to (x, y)
                traj_xy = [(s, l) for s, l, t in behavior['trajectory']]
                candidates.append({
                    'trajectory': traj_xy,
                    'corridor_id': 0,
                    'corridor_confidence': 0.9,
                    'approach': 'behavior_tree',
                    'behavior': behavior['behavior']
                })
        
        # Limit to K
        return candidates[:self.config.k_candidates]
    
    def _evaluate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Evaluate all candidates"""
        if not candidates:
            return []
        
        # Update collision checker with current corridor
        if self.current_corridors:
            corridor = self.current_corridors[0]
            self.collision_checker.update_static_environment(
                corridor.left_bound,
                corridor.right_bound,
                corridor.keepout_regions
            )
        
        evaluations = []
        
        for i, cand in enumerate(candidates):
            traj = cand['trajectory']
            
            # Convert to (s, l, t) format
            s_l_t = []
            for j, (x, y) in enumerate(traj):
                t = j * 0.1
                s_l_t.append((x, y, t))  # Simplified: using x as s, y as l
            
            # Collision check
            collision, min_dist, step = self.collision_checker.check_collision(s_l_t)
            
            # Compute cost
            cost = self._compute_cost(cand)
            
            # Risk score (CVaR-style: penalize low-confidence corridors)
            if not collision:
                risk_penalty = (1 - cand['corridor_confidence']) * 10
            else:
                risk_penalty = 100
            
            final_cost = cost + risk_penalty
            
            evaluations.append({
                **cand,
                'collision': collision,
                'min_distance': min_dist,
                'cost': final_cost,
                'is_feasible': not collision
            })
        
        return evaluations
    
    def _compute_cost(self, candidate: Dict) -> float:
        """Compute cost for a candidate"""
        traj = candidate['trajectory']
        
        # Distance traveled
        total_dist = 0
        for i in range(1, len(traj)):
            dx = traj[i][0] - traj[i-1][0]
            dy = traj[i][1] - traj[i-1][1]
            total_dist += np.sqrt(dx**2 + dy**2)
        
        # Lateral motion
        l_values = [p[1] for p in traj]
        lateral_motion = max(l_values) - min(l_values)
        
        # Combined cost
        cost = -total_dist * 0.1 + lateral_motion * 2.0
        
        return cost
    
    def _select(self, evaluations: List[Dict]) -> Optional[Dict]:
        """Select best feasible candidate"""
        # Filter feasible
        feasible = [e for e in evaluations if e['is_feasible']]
        
        if not feasible:
            return None
        
        # Sort by cost
        feasible.sort(key=lambda x: x['cost'])
        
        best = feasible[0]
        
        # Convert trajectory to output format
        return {
            'trajectory': best['trajectory'],
            'behavior': best.get('behavior', 'lane_keep'),
            'approach': best['approach'],
            'is_feasible': True,
            'candidates': len(evaluations),
            'min_distance': best['min_distance']
        }
    
    def _generate_mrm(self, ego_state: Dict) -> Dict:
        """Generate Minimal Risk Maneuver (fallback)"""
        # Simple: stop in place
        s = ego_state.get('s', 0)
        l = ego_state.get('l', 0)
        
        trajectory = []
        for i in range(60):
            trajectory.append((s, l))
        
        return {
            'trajectory': trajectory,
            'behavior': 'stop',
            'approach': 'mrm',
            'is_feasible': True,
            'candidates': 0,
            'min_distance': float('inf')
        }


def create_unified_planner() -> UnifiedPlanner:
    """Factory function"""
    config = UnifiedPlannerConfig(
        rate_hz=20,
        n_corridors=4,
        k_candidates=128,
        coarse_steps=60,
        fine_steps=12,
        risk_alpha=0.2
    )
    return UnifiedPlanner(config)


if __name__ == "__main__":
    # Test unified planner
    planner = create_unified_planner()
    
    # Test state
    ego_state = {
        'x': 0, 'y': 0,
        'heading': 0,
        'speed': 15,
        's': 0, 'l': 0,
        'goal_s': 100,
        'gaps': [{'l': 3.5, 'available': True}],
        'lead_vehicle': None,
        'crossing_agents': []
    }
    
    result = planner.update(ego_state)
    print(f"Result: {result['behavior']}")
    print(f"Feasible: {result['is_feasible']}")
    print(f"Candidates evaluated: {result['candidates']}")
    print(f"Trajectory length: {len(result['trajectory'])}")

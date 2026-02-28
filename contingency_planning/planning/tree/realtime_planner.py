"""
Real-Time Async Planner

Combines extended belief tracking with fast QP optimization
for real-time contingency planning with asynchronous updates.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import time
from threading import Thread, Lock

from extended_belief_tracker import ExtendedBeliefTracker, MultiObstacleTracker
from fast_optimizer import FastTreeQPOptimizer, AdaptiveHorizonOptimizer


@dataclass
class PlannerConfig:
    """Configuration for real-time planner."""
    # Timing
    replan_hz: float = 10.0
    perception_hz: float = 20.0
    
    # Safety
    ttc_threshold: float = 1.5  # seconds - trigger contingency
    safety_margin: float = 1.5  # meters
    velocity_limit: float = 15.0  # m/s
    
    # Optimization
    horizon: int = 15
    q_weight: float = 1.0
    r_weight: float = 0.1
    
    # Branching
    branch_threshold: float = 0.7  # belief threshold for branching
    n_contingencies: int = 2


@dataclass
class PlanningResult:
    """Result from planning step."""
    control: np.ndarray
    trajectory: np.ndarray
    ttc: float
    risk_score: float
    belief_state: Dict[str, float]
    is_contingency: bool
    timestamp: float


class RealTimeTreePlanner:
    """
    Real-time async planner with belief tracking and fast optimization.
    
    Features:
    - Parallel belief updates and planning
    - Contingency triggers based on TTC
    - Warm-starting for faster solves
    """
    
    def __init__(self, config: PlannerConfig = None):
        self.config = config or PlannerConfig()
        
        # Obstacle tracker
        self.obstacle_tracker = MultiObstacleTracker(
            process_noise=0.1,
            observation_noise=0.3,
        )
        
        # Belief over contingencies
        self.contingency_belief: Dict[str, float] = {
            'nominal': 1.0,  # No obstacle action
            'evasive': 0.0,   # Evasive action needed
        }
        
        # QP optimizer
        self.optimizer = AdaptiveHorizonOptimizer(
            min_horizon=8,
            max_horizon=self.config.horizon,
            q_weight=self.config.q_weight,
            r_weight=self.config.r_weight,
        )
        
        # Set dynamics
        A, B = self.optimizer.default_dynamics(velocity=10.0)
        self.optimizer.set_dynamics(A, B)
        
        # State
        self.current_state = None
        self.current_goal = None
        self.current_control = np.zeros(2)
        self.last_plan_time = 0.0
        self.is_planning = False
        
        # Cached trajectories
        self.nominal_trajectory = None
        self.contingency_trajectory = None
        
        # Threading
        self.lock = Lock()
        
        # Performance metrics
        self.metrics = {
            'solve_time': [],
            'ttc': [],
            'risk_score': [],
        }
    
    def initialize(
        self,
        initial_state: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict] = None,
    ):
        """Initialize planner with state and goal."""
        self.current_state = initial_state.copy()
        self.current_goal = goal.copy()
        
        # Initialize obstacle tracking
        if obstacles:
            for i, obs in enumerate(obstacles):
                pos = np.array(obs['position'][:2])
                vel = obs.get('velocity', np.zeros(2))
                self.obstacle_tracker.add_obstacle(i, pos, vel, timestamp=0.0)
    
    def update_belief(
        self,
        observations: Dict[int, np.ndarray],
        timestamp: float,
    ) -> Dict[int, float]:
        """
        Update obstacle beliefs with new observations.
        
        Args:
            observations: Dict of {obstacle_id: position}
            timestamp: Current time
            
        Returns:
            TTC values for each obstacle
        """
        # Update each observed obstacle
        for obs_id, position in observations.items():
            self.obstacle_tracker.update(obs_id, position, timestamp)
        
        # Predict all obstacles
        if self.current_state is not None:
            dt = timestamp - self.last_plan_time
            if dt > 0:
                self.obstacle_tracker.predict_all(dt)
        
        # Compute TTC for all obstacles
        if self.current_state is not None:
            ego_pos = self.current_state[:2]
            ego_vel = self.current_state[2:4] if len(self.current_state) >= 4 else np.zeros(2)
            
            ttcs = self.obstacle_tracker.get_all_ttc(ego_pos, ego_vel)
            
            # Update contingency belief
            self._update_contingency_belief(ttcs)
            
            return ttcs
        
        return {}
    
    def _update_contingency_belief(self, ttcs: Dict[int, float]):
        """Update belief over nominal vs evasive contingency."""
        min_ttc = min(ttcs.values()) if ttcs else float('inf')
        
        if min_ttc < self.config.ttc_threshold:
            # High risk - increase evasive belief
            risk_factor = 1.0 - (min_ttc / self.config.ttc_threshold)
            self.contingency_belief['evasive'] = risk_factor
            self.contingency_belief['nominal'] = 1.0 - risk_factor
        else:
            # Low risk - nominal
            self.contingency_belief['nominal'] = 0.9
            self.contingency_belief['evasive'] = 0.1
    
    def plan(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict] = None,
    ) -> PlanningResult:
        """
        Main planning function.
        
        Args:
            state: Current ego state [x, y, v, heading]
            goal: Goal position [x, y]
            obstacles: List of obstacle dicts
            
        Returns:
            PlanningResult with control and trajectory
        """
        self.current_state = state.copy()
        
        # Update belief/tracking
        obs_dict = {}
        if obstacles:
            for i, obs in enumerate(obstacles):
                obs_dict[i] = np.array(obs['position'][:2])
        
        timestamp = time.time()
        ttcs = self.update_belief(obs_dict, timestamp)
        
        # Get closest obstacle TTC
        min_ttc = min(ttcs.values()) if ttcs else float('inf')
        
        # Compute risk
        risk = 0.0
        for tracker in self.obstacle_tracker.trackers.values():
            risk = max(risk, tracker.get_risk_score(state[:2], state[2:4] if len(state) >= 4 else np.zeros(2)))
        
        # Determine if contingency needed
        is_contingency = (
            self.contingency_belief['evasive'] > self.config.branch_threshold or
            min_ttc < self.config.ttc_threshold
        )
        
        # Solve QP
        start_time = time.time()
        
        if is_contingency:
            # Plan evasive trajectory
            trajectory, controls = self._plan_evasive(state, goal, obstacles)
        else:
            # Plan nominal trajectory
            trajectory, controls = self._plan_nominal(state, goal)
        
        solve_time = time.time() - start_time
        self.metrics['solve_time'].append(solve_time)
        self.metrics['ttc'].append(min_ttc)
        self.metrics['risk_score'].append(risk)
        
        # Get control
        if len(controls) > 0:
            control = controls[0]
        else:
            control = np.zeros(2)
        
        self.current_control = control
        
        return PlanningResult(
            control=control,
            trajectory=trajectory,
            ttc=min_ttc,
            risk_score=risk,
            belief_state=self.contingency_belief.copy(),
            is_contingency=is_contingency,
            timestamp=timestamp,
        )
    
    def _plan_nominal(
        self,
        state: np.ndarray,
        goal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plan nominal trajectory to goal."""
        trajectory, controls = self.optimizer.solve_warm(
            state,
            goal=goal,
            obstacles=None,
            velocity_limit=self.config.velocity_limit,
        )
        
        self.nominal_trajectory = trajectory
        return trajectory, controls
    
    def _plan_evasive(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plan evasive trajectory avoiding obstacles."""
        # Find evasive goal (offset from original goal)
        direction_to_goal = goal[:2] - state[:2]
        if np.linalg.norm(direction_to_goal) > 0:
            direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
        
        # Lateral offset for evasion
        lateral = np.array([-direction_to_goal[1], direction_to_goal[0]])
        evasive_goal = goal[:2] + lateral * 5.0  # 5m lateral offset
        
        # Solve with obstacle avoidance
        trajectory, controls = self.optimizer.solve_warm(
            state,
            goal=evasive_goal,
            obstacles=obstacles,
            velocity_limit=self.config.velocity_limit * 0.7,  # Slow down
        )
        
        self.contingency_trajectory = trajectory
        return trajectory, controls
    
    def step(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        observations: Dict[int, np.ndarray],
        obstacles: List[Dict] = None,
    ) -> PlanningResult:
        """
        Single planning step with observations.
        
        Args:
            state: Current ego state
            goal: Goal position
            observations: New obstacle observations
            obstacles: Obstacle info (positions, velocities)
            
        Returns:
            PlanningResult
        """
        timestamp = time.time()
        
        # Update belief with new observations
        ttcs = self.update_belief(observations, timestamp)
        
        # Plan
        result = self.plan(state, goal, obstacles)
        
        return result
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        if self.metrics['solve_time']:
            return {
                'avg_solve_time': np.mean(self.metrics['solve_time']),
                'max_solve_time': np.max(self.metrics['solve_time']),
                'min_ttc': np.min(self.metrics['ttc']) if self.metrics['ttc'] else float('inf'),
                'avg_risk': np.mean(self.metrics['risk_score']),
            }
        return {}


class HierarchicalTreePlanner(RealTimeTreePlanner):
    """
    Hierarchical planner that builds a tree of contingencies.
    """
    
    def __init__(self, config: PlannerConfig = None, max_depth: int = 3):
        super().__init__(config)
        self.max_depth = max_depth
        self.tree_nodes = []
    
    def build_contingency_tree(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict],
        depth: int = 0,
    ) -> List[Dict]:
        """
        Build contingency tree with multiple branches.
        
        Args:
            state: Current state
            goal: Goal position
            obstacles: List of obstacles
            depth: Current tree depth
            
        Returns:
            List of tree nodes
        """
        if depth >= self.max_depth:
            return []
        
        # Plan at this node
        result = self.plan(state, goal, obstacles)
        
        node = {
            'depth': depth,
            'state': state.copy(),
            'control': result.control.copy(),
            'trajectory': result.trajectory.copy() if result.trajectory is not None else None,
            'ttc': result.ttc,
            'risk': result.risk_score,
            'is_contingency': result.is_contingency,
            'children': [],
        }
        
        # If contingency triggered, branch
        if result.is_contingency and depth < self.max_depth - 1:
            # Sample different contingency actions
            for offset in [-3, 3]:  # Lateral offsets
                child_goal = goal.copy()
                child_goal[1] += offset
                
                child_state = result.trajectory[5] if len(result.trajectory) > 5 else state
                
                children = self.build_contingency_tree(
                    child_state, child_goal, obstacles, depth + 1
                )
                node['children'].extend(children)
        
        self.tree_nodes.append(node)
        return [node]
    
    def select_best_branch(self) -> np.ndarray:
        """
        Select best branch based on risk and progress.
        
        Returns:
            Best control input
        """
        if not self.tree_nodes:
            return np.zeros(2)
        
        # Score each node
        best_node = None
        best_score = -float('inf')
        
        for node in self.tree_nodes:
            if node['trajectory'] is None:
                continue
            
            # Score: low risk + progress toward goal
            progress = np.linalg.norm(node['trajectory'][-1][:2] - self.current_goal[:2])
            score = progress - node['risk'] * 10
            
            if score > best_score:
                best_score = score
                best_node = node
        
        if best_node is not None:
            return best_node['control']
        
        return np.zeros(2)


def create_realtime_planner(
    replan_hz: float = 10.0,
    ttc_threshold: float = 1.5,
    horizon: int = 15,
) -> RealTimeTreePlanner:
    """Create configured real-time planner."""
    config = PlannerConfig(
        replan_hz=replan_hz,
        ttc_threshold=ttc_threshold,
        horizon=horizon,
    )
    
    return RealTimeTreePlanner(config)

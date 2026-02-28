"""
Tree-Based Planner

High-level planner that combines Control-Tree with optimization and belief tracking.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import yaml

from .contingency_tree import ControlTree, create_tree_from_scenario
from .optimization import TreeQPOptimizer
from .belief_tracker import BeliefTracker, create_belief_tracker_for_scenario


class TreeBasedPlanner:
    """
    Classical Tree-Based Contingency Planner.
    
    Combines:
    - Control-Tree structure for branching
    - QP optimization for trajectory generation
    - Belief tracking for uncertainty resolution
    
    Based on Control-Tree Optimization (ICRA 2021).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize planner.
        
        Args:
            config: Configuration dictionary
        """
        tree_config = config.get("tree", {})
        
        # Tree structure
        self.max_depth = tree_config.get("max_depth", 4)
        self.n_branches = tree_config.get("n_branches", 6)
        self.steps_per_phase = tree_config.get("steps_per_phase", 4)
        self.horizon = tree_config.get("horizon", 20)
        
        # Optimization
        self.qp_optimizer = TreeQPOptimizer(
            state_dim=4,
            action_dim=2,
            horizon=self.horizon,
            q_weight=tree_config.get("q_weight", 1.0),
            r_weight=tree_config.get("r_weight", 0.1),
            terminal_weight=tree_config.get("terminal_weight", 2.0),
        )
        
        # Safety
        self.safety_margin = tree_config.get("safety_margin", 2.0)
        
        # Current state
        self.tree: Optional[ControlTree] = None
        self.belief_tracker: Optional[BeliefTracker] = None
        self.current_state: Optional[np.ndarray] = None
        self.goal_state: np.ndarray = None
        
        # Execution state
        self.execution_step: int = 0
        self.current_hypothesis: str = "shared"
    
    def initialize(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        scenario_name: str,
        prior_beliefs: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize planner for a scenario.
        
        Args:
            initial_state: [x, y, v, heading]
            goal_state: [x, y, v, heading]
            scenario_name: Name of scenario
            prior_beliefs: Optional prior beliefs
        """
        self.current_state = initial_state.copy()
        self.goal_state = goal_state.copy()
        self.execution_step = 0
        self.current_hypothesis = "shared"
        
        # Create belief tracker
        self.belief_tracker = create_belief_tracker_for_scenario(scenario_name)
        
        # Set prior beliefs if provided
        if prior_beliefs:
            self.belief_tracker.belief = prior_beliefs.copy()
        
        # Create tree
        from ...simulation.scenarios import get_scenario
        scenario = get_scenario(scenario_name)
        
        tree_config = {
            "max_depth": self.max_depth,
            "n_branches": self.n_branches,
            "steps_per_phase": self.steps_per_phase,
            "horizon": self.horizon,
        }
        
        self.tree = create_tree_from_scenario(scenario, tree_config)
        
        # Optimize tree
        constraints = {
            "obstacles": scenario.obstacles,
            "safety_margin": self.safety_margin,
        }
        
        self.qp_optimizer.optimize_tree(self.tree, goal_state, constraints)
    
    def plan(self, observation: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Plan action given current state and optional observation.
        
        Args:
            observation: Optional observation for belief update
            
        Returns:
            action: [acceleration, steering]
            info: Planning info (belief, hypothesis, etc.)
        """
        # Update belief if observation provided
        if observation is not None and self.belief_tracker is not None:
            self._update_belief(observation)
        
        # Select action based on belief
        action, hypothesis = self.tree.select_action(self.current_state)
        
        # Check for MRC trigger
        mrc_triggered = self._check_mrc_trigger()
        
        if mrc_triggered:
            action = self._get_mrc_action()
            hypothesis = "mrc"
        
        info = {
            "belief": self.belief_tracker.get_belief() if self.belief_tracker else {},
            "hypothesis": hypothesis,
            "mrc_triggered": mrc_triggered,
            "execution_step": self.execution_step,
        }
        
        return action, info
    
    def execute(self, action: np.ndarray) -> np.ndarray:
        """
        Execute action and return next state.
        
        Args:
            action: [acceleration, steering]
            
        Returns:
            next_state: [x, y, v, heading]
        """
        # Simple dynamics
        dt = 0.1
        v = self.current_state[2] + action[0] * dt
        heading = self.current_state[3] + action[1] * dt
        
        x = self.current_state[0] + v * np.cos(heading) * dt
        y = self.current_state[1] + v * np.sin(heading) * dt
        
        # Update state
        self.current_state = np.array([x, y, v, heading])
        self.execution_step += 1
        
        return self.current_state.copy()
    
    def _update_belief(self, observation: Dict):
        """Update belief based on observation."""
        if "observation_type" in observation and "observation_value" in observation:
            from .belief_tracker import Observation
            obs = Observation(
                type=observation["observation_type"],
                value=observation["observation_value"],
                timestamp=observation.get("timestamp", 0),
            )
            self.belief_tracker.update(obs)
    
    def _check_mrc_trigger(self) -> bool:
        """
        Check if MRC should be triggered.
        
        MRC triggered when:
        - Belief is uncertain AND close to hazard
        - Safety constraint violated
        """
        if self.belief_tracker is None:
            return False
        
        # Check if belief is unresolved (uncertain)
        uncertain = not self.belief_tracker.is_resolved(threshold=0.85)
        
        # Check if close to goal but still uncertain
        if uncertain:
            dist_to_goal = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])
            if dist_to_goal < 20 and self.execution_step > 10:
                return True
        
        return False
    
    def _get_mrc_action(self) -> np.ndarray:
        """Get MRC (Minimal Risk Condition) action - emergency stop."""
        return np.array([-6.0, 0.0])  # Maximum braking
    
    def get_tree_visualization(self) -> Dict:
        """Get tree structure for visualization."""
        if self.tree is None:
            return {}
        
        return {
            "hypotheses": self.tree.hypotheses,
            "beliefs": self.belief_tracker.get_belief() if self.belief_tracker else {},
            "shared_trunk_length": self.tree.shared_trunk_length,
            "max_depth": self.max_depth,
            "nodes": [
                {
                    "id": n.id,
                    "hypothesis": n.hypothesis,
                    "belief": n.belief,
                    "is_shared": n.is_shared,
                    "depth": n.depth,
                    "cost": n.cost,
                }
                for n in self.tree.nodes.values()
            ],
        }
    
    def reset(self):
        """Reset planner state."""
        self.tree = None
        self.belief_tracker = None
        self.current_state = None
        self.goal_state = None
        self.execution_step = 0


def create_planner(config_path: str) -> TreeBasedPlanner:
    """Create planner from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return TreeBasedPlanner(config)

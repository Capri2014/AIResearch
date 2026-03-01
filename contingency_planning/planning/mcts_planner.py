"""
MCTS (Monte Carlo Tree Search) Planner for Interactive Scenarios

Handles multi-agent uncertainty through game-theoretic planning:
- Models other agents' intentions
- Uses Upper Confidence Bound (UCT/PUCT) for exploration
- Natural handling of interactive scenarios (merges, cut-ins, nudging)

Best for: Dense interactive scenes, ambiguous right-of-way
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import math
import random


class AgentMode(Enum):
    """Possible modes for other agents"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"  
    NORMAL = "normal"
    YIELD = "yield"


@dataclass
class MCTSConfig:
    """Configuration for MCTS planner"""
    # Search budget
    max_iterations: int = 1000
    max_depth: int = 10
    expand_threshold: int = 5
    
    # Exploration vs exploitation
    exploration_constant: float = 1.41  # sqrt(2) for UCT
    
    # Simulation
    simulation_horizon: int = 20
    num_simulations: int = 50
    
    # Agent modeling
    agent_modes: List[AgentMode] = None
    mode_prior: Dict[AgentMode, float] = None
    
    # Cost weights
    cost_collision: float = 1000.0
    cost_progress: float = -1.0
    cost_lateral: float = 2.0
    cost_comfort: float = 1.0


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, state: Dict, parent: Optional['MCTSNode'] = None, action: str = None):
        self.state = state  # {'ego': (s, l, v), 'agents': [(s, l, v, mode), ...]}
        self.parent = parent
        self.action = action  # Action taken to reach this node
        
        self.children: Dict[str, MCTSNode] = {}
        self.visit_count: int = 0
        self.value: float = 0.0
        
        # For expansion
        self.is_expanded: bool = False
    
    def uct_value(self, parent_visits: int) -> float:
        """Upper Confidence Bound for Trees (UCT) value"""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.value / self.visit_count
        exploration = self.exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        return exploitation + exploration
    
    def __lt__(self, other):
        return self.visit_count < other.visit_count


class AgentModel:
    """Model of other agents' behavior"""
    
    def __init__(self, config: MCTSConfig):
        self.config = config
        self.mode_prior = config.mode_prior or {
            AgentMode.NORMAL: 0.6,
            AgentMode.CONSERVATIVE: 0.25,
            AgentMode.AGGRESSIVE: 0.1,
            AgentMode.YIELD: 0.05
        }
    
    def sample_mode(self) -> AgentMode:
        """Sample an agent mode based on prior"""
        r = random.random()
        cumulative = 0
        for mode, prob in self.mode_prior.items():
            cumulative += prob
            if r < cumulative:
                return mode
        return AgentMode.NORMAL
    
    def predict_trajectory(self, agent_state: Dict, mode: AgentMode, 
                          ego_action: str, horizon: int) -> List[Tuple[float, float]]:
        """
        Predict agent trajectory given ego action and agent mode.
        
        This is a simplified model - in production would use learned predictors.
        """
        s, l, v = agent_state['s'], agent_state['l'], agent_state['v']
        
        trajectory = []
        dt = 0.1
        
        for _ in range(horizon):
            # Simple kinematic model based on mode
            if mode == AgentMode.AGGRESSIVE:
                # Cut in, maintain speed
                l += 0.1 if l < 1.5 else 0
                s += v * dt
            elif mode == AgentMode.CONSERVATIVE:
                # Slow down, stay in lane
                v = max(v * 0.99, 5)
                s += v * dt
            elif mode == AgentMode.YIELD:
                # Slow down significantly
                v = max(v * 0.95, 2)
                s += v * dt
            else:  # NORMAL
                # Maintain with slight adaptation
                s += v * dt
            
            trajectory.append((s, l))
        
        return trajectory


class MCSS:
    """
    Monte Carlo Tree Search for motion planning.
    
    Uses UCT (Upper Confidence Trees) for balancing exploration/exploitation.
    """
    
    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()
        self.agent_model = AgentModel(self.config)
        
        # Actions available to ego
        self.actions = [
            'lane_keep',
            'lane_change_left',
            'lane_change_right', 
            'slow_down',
            'accelerate'
        ]
        
    def plan(self, 
             ego_state: Dict,
             agents: List[Dict],
             goal_s: float = 100) -> Tuple[List[Tuple[float, float]], str]:
        """
        Plan using MCTS.
        
        Args:
            ego_state: {'s', 'l', 'v', 'heading'}
            agents: List of agent states
            goal_s: Goal longitudinal position
            
        Returns:
            (trajectory, chosen_action)
        """
        # Initialize root
        root_state = {
            'ego': (ego_state.get('s', 0), ego_state.get('l', 0), ego_state.get('v', 10)),
            'agents': [(a.get('s', 0), a.get('l', 0), a.get('v', 10)) for a in agents],
            'goal_s': goal_s,
            'step': 0
        }
        
        root = MCTSNode(root_state)
        
        # Run MCTS iterations
        for i in range(self.config.max_iterations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_expanded and node.visit_count >= self.config.expand_threshold:
                self._expand(node)
            
            # Simulation
            value = self._simulate(node)
            
            # Backpropagation
            self._backpropagate(node, value)
            
            # Early termination if confident
            if i > 100 and root.visit_count > 50:
                best_child = max(root.children.values(), key=lambda n: n.visit_count)
                if best_child.visit_count / root.visit_count > 0.9:
                    break
        
        # Select best action
        best_action, best_child = self._get_best_action(root)
        
        # Extract trajectory
        trajectory = self._extract_trajectory(best_child, root_state)
        
        return trajectory, best_action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select child node using UCT (selection phase)"""
        while node.children and node.is_expanded:
            best_child = max(
                node.children.values(),
                key=lambda n: n.uct_value(node.visit_count)
            )
            node = best_child
        return node
    
    def _expand(self, node: MCTSNode):
        """Expand node with available actions (expansion phase)"""
        for action in self.actions:
            # Compute resulting state
            new_state = self._apply_action(node.state, action)
            child = MCTSNode(new_state, parent=node, action=action)
            node.children[action] = child
        
        node.is_expanded = True
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate from node to horizon (simulation phase)"""
        state = node.state.copy()
        
        for step in range(self.config.simulation_horizon):
            # Check terminal condition
            if self._is_terminal(state):
                return self._compute_reward(state, terminal=True)
            
            # Sample agent modes
            agent_modes = [self.agent_model.sample_mode() for _ in state['agents']]
            
            # Choose random ego action for simulation
            action = random.choice(self.actions)
            
            # Update state
            state = self._apply_action(state, action, agent_modes)
        
        return self._compute_reward(state, terminal=False)
    
    def _backpropagate(self, node: Optional[MCTSNode], value: float):
        """Backpropagate value to root (backpropagation phase)"""
        while node is not None:
            node.visit_count += 1
            node.value += value
            node = node.parent
    
    def _get_best_action(self, root: MCTSNode) -> Tuple[str, MCTSNode]:
        """Get best action based on visit count"""
        if not root.children:
            return 'lane_keep', root
        
        best_action = max(
            root.children.keys(),
            key=lambda a: root.children[a].visit_count
        )
        return best_action, root.children[best_action]
    
    def _apply_action(self, state: Dict, ego_action: str, 
                       agent_modes: List[AgentMode] = None) -> Dict:
        """Apply action to state"""
        ego_s, ego_l, ego_v = state['ego']
        dt = 0.1
        
        # Apply ego action
        if ego_action == 'lane_change_left':
            ego_l += 0.35  # 3.5m over 10 steps
        elif ego_action == 'lane_change_right':
            ego_l -= 0.35
        elif ego_action == 'slow_down':
            ego_v = max(ego_v * 0.98, 3)
        elif ego_action == 'accelerate':
            ego_v = min(ego_v * 1.02, 30)
        
        # Progress
        ego_s += ego_v * dt
        
        # Update agents (simplified)
        agents = []
        for i, (a_s, a_l, a_v) in enumerate(state['agents']):
            mode = agent_modes[i] if agent_modes else AgentMode.NORMAL
            
            if mode == AgentMode.AGGRESSIVE:
                a_l += 0.05 if a_l < 1 else 0
            
            a_s += a_v * dt
            agents.append((a_s, a_l, a_v))
        
        return {
            'ego': (ego_s, ego_l, ego_v),
            'agents': agents,
            'goal_s': state.get('goal_s', 100),
            'step': state.get('step', 0) + 1
        }
    
    def _is_terminal(self, state: Dict) -> bool:
        """Check if state is terminal (collision or goal reached)"""
        ego_s, ego_l, _ = state['ego']
        
        # Check goal
        if ego_s >= state.get('goal_s', 100):
            return True
        
        # Check collision with agents
        for a_s, a_l, _ in state['agents']:
            if abs(ego_s - a_s) < 5 and abs(ego_l - a_l) < 3:
                return True
        
        return False
    
    def _compute_reward(self, state: Dict, terminal: bool = False) -> float:
        """Compute reward for state"""
        ego_s, ego_l, ego_v = state['ego']
        
        reward = 0.0
        
        # Progress reward
        reward += self.config.cost_progress * ego_s
        
        # Terminal collision penalty
        if terminal:
            for a_s, a_l, _ in state['agents']:
                if abs(ego_s - a_s) < 5 and abs(ego_l - a_l) < 3:
                    reward -= self.config.cost_collision
        
        # Comfort (lateral motion)
        lateral_motion = abs(ego_l)
        reward -= self.config.cost_lateral * lateral_motion
        
        return reward
    
    def _extract_trajectory(self, node: MCTSNode, root_state: Dict) -> List[Tuple[float, float]]:
        """Extract full trajectory from root to leaf"""
        trajectory = []
        state = root_state
        
        # Start from root state
        trajectory.append((state['ego'][0], state['ego'][1]))
        
        # Follow best path
        current = node
        while current.children and current.action:
            state = current.state
            trajectory.append((state['ego'][0], state['ego'][1]))
            current = max(current.children.values(), 
                        key=lambda n: n.visit_count)
        
        return trajectory


def create_mcts_planner() -> MCSS:
    """Factory function"""
    config = MCTSConfig(
        max_iterations=500,
        max_depth=10,
        exploration_constant=1.41,
        cost_collision=1000.0,
        cost_progress=-1.0,
        cost_lateral=2.0
    )
    return MCSS(config)


if __name__ == "__main__":
    # Test MCTS planner
    planner = create_mcts_planner()
    
    ego_state = {'s': 0, 'l': 0, 'v': 15}
    agents = [
        {'s': 30, 'l': 0, 'v': 12},
        {'s': -10, 'l': 3.5, 'v': 10}
    ]
    
    trajectory, action = planner.plan(ego_state, agents, goal_s=100)
    print(f"Best action: {action}")
    print(f"Trajectory: {len(trajectory)} points")
    print(f"First 5: {trajectory[:5]}")

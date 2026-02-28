"""
Control-Tree Data Structures

Implements the tree structure for classical contingency planning
based on Control-Tree Optimization (Phiquepal & Toussaint, ICRA 2021).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import uuid


@dataclass
class TreeNode:
    """A node in the contingency tree."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    hypothesis: str = ""              # e.g., "pedestrian_cross", "cut_in"
    belief: float = 1.0              # P(hypothesis | observation)
    state: np.ndarray = None         # State at this node [state_dim]
    control: np.ndarray = None        # Optimized control [horizon, action_dim]
    cost: float = 0.0                # Stage + terminal cost
    is_shared: bool = True           # Part of common trunk
    depth: int = 0                  # Depth in tree
    observation_time: int = 0        # When observation resolves
    
    def __post_init__(self):
        if self.state is None:
            self.state = np.zeros(4)  # [x, y, v, heading]
        if self.control is None:
            self.control = np.zeros((20, 2))  # [horizon, action_dim]


class ControlTree:
    """
    Control-Tree for contingency planning.
    
    Constructs a tree where:
    - Root represents initial state
    - Shared trunk: common trajectory prefix until uncertainty resolves
    - Branches: different hypotheses about discrete uncertainties
    
    Key concept: "Delayed decision" - wait until observation resolves
    before committing to a branch.
    """
    
    def __init__(
        self,
        max_depth: int = 4,
        n_branches: int = 6,
        steps_per_phase: int = 4,
        horizon: int = 20,
    ):
        self.max_depth = max_depth
        self.n_branches = n_branches
        self.steps_per_phase = steps_per_phase  # Steps before observation
        self.horizon = horizon
        
        # Tree structure
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: Optional[str] = None
        
        # Shared trunk
        self.shared_trunk_length: int = steps_per_phase
        
        # Discrete uncertainties
        self.hypotheses: List[str] = []
        self.current_belief: Dict[str, float] = {}
        
    def initialize(
        self, 
        initial_state: np.ndarray, 
        hypotheses: List[str],
        prior_beliefs: Optional[Dict[str, float]] = None
    ):
        """Initialize tree with initial state and hypotheses."""
        # Create root
        root = TreeNode(
            hypothesis="root",
            state=initial_state.copy(),
            is_shared=True,
            depth=0,
        )
        self.root_id = root.id
        self.nodes[root.id] = root
        
        # Set hypotheses
        self.hypotheses = hypotheses
        
        # Initialize beliefs (uniform if not provided)
        if prior_beliefs is None:
            self.current_belief = {h: 1.0 / len(hypotheses) for h in hypotheses}
        else:
            self.current_belief = prior_beliefs.copy()
        
    def build(self) -> None:
        """
        Build tree structure.
        
        Tree structure:
        - Depth 0: root
        - Depth 1 to shared_trunk_length: shared trunk (same for all hypotheses)
        - Depth > shared_trunk_length: branches for each hypothesis
        """
        # Build shared trunk first
        for d in range(1, self.shared_trunk_length + 1):
            parent_id = self.root_id if d == 1 else f"shared_{d-1}"
            node_id = f"shared_{d}"
            
            node = TreeNode(
                id=node_id,
                parent_id=parent_id,
                hypothesis="shared",
                is_shared=True,
                depth=d,
                observation_time=0,
            )
            self.nodes[node_id] = node
        
        # Build branches for each hypothesis
        for h in self.hypotheses:
            self._build_branch(hypothesis=h, depth=self.shared_trunk_length)
    
    def _build_branch(self, hypothesis: str, depth: int) -> None:
        """Build a branch for a specific hypothesis."""
        parent_id = f"shared_{depth}" if depth > 0 else self.root_id
        
        for d in range(depth + 1, self.max_depth + 1):
            node_id = f"{hypothesis}_{d}"
            
            node = TreeNode(
                id=node_id,
                parent_id=parent_id,
                hypothesis=hypothesis,
                belief=self.current_belief.get(hypothesis, 1.0),
                is_shared=False,
                depth=d,
                observation_time=depth,
            )
            self.nodes[node_id] = node
            parent_id = node_id
    
    def get_shared_trunk(self) -> List[TreeNode]:
        """Get all nodes in the shared trunk."""
        shared = []
        for d in range(1, self.shared_trunk_length + 1):
            node_id = f"shared_{d}"
            if node_id in self.nodes:
                shared.append(self.nodes[node_id])
        return shared
    
    def get_branches(self, hypothesis: str) -> List[TreeNode]:
        """Get all nodes for a specific hypothesis branch."""
        branch = []
        for node in self.nodes.values():
            if node.hypothesis == hypothesis and not node.is_shared:
                branch.append(node)
        return sorted(branch, key=lambda n: n.depth)
    
    def get_all_branches(self) -> Dict[str, List[TreeNode]]:
        """Get all branches grouped by hypothesis."""
        branches = {h: self.get_branches(h) for h in self.hypotheses}
        return branches
    
    def update_belief(self, observation: str, likelihoods: Dict[str, float]) -> None:
        """
        Update belief state given observation.
        
        Uses Bayesian update: P(h | o) ∝ P(o | h) * P(h)
        """
        # Compute posterior
        posterior = {}
        for h in self.hypotheses:
            prior = self.current_belief.get(h, 1.0 / len(self.hypotheses))
            likelihood = likelihoods.get(h, 0.01)  # Small default
            posterior[h] = likelihood * prior
        
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            self.current_belief = {h: p / total for h, p in posterior.items()}
        
        # Update node beliefs
        for node in self.nodes.values():
            if node.hypothesis in self.current_belief:
                node.belief = self.current_belief[node.hypothesis]
    
    def select_action(self, current_state: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Select action based on current belief and state.
        
        Strategy:
        1. If uncertainty resolved (one belief close to 1): commit to branch
        2. If uncertain: follow shared trunk
        3. If safety critical: trigger MRC
        """
        # Check if belief is resolved
        resolved_hypothesis = None
        for h, belief in self.current_belief.items():
            if belief > 0.9:
                resolved_hypothesis = h
                break
        
        if resolved_hypothesis is not None:
            # Commit to branch
            branch = self.get_branches(resolved_hypothesis)
            if branch:
                return branch[0].control[0], resolved_hypothesis
            else:
                # Fallback to nominal
                return self._get_nominal_control(), "nominal"
        else:
            # Follow shared trunk
            trunk = self.get_shared_trunk()
            if trunk:
                return trunk[0].control[0], "shared"
            else:
                return self._get_nominal_control(), "nominal"
    
    def _get_nominal_control(self) -> np.ndarray:
        """Get nominal control (emergency brake)."""
        return np.array([-3.0, 0.0])  # [acceleration, steering]
    
    def __repr__(self):
        return (
            f"ControlTree(\n"
            f"  hypotheses: {self.hypotheses}\n"
            f"  beliefs: {self.current_belief}\n"
            f"  shared_trunk_length: {self.shared_trunk_length}\n"
            f"  max_depth: {self.max_depth}\n"
            f"  n_nodes: {len(self.nodes)}\n"
            f")"
        )


def create_tree_from_scenario(scenario, config) -> ControlTree:
    """Create a ControlTree from a ContingencyScenario."""
    hypotheses = [h.name for h in scenario.hypotheses]
    prior = {h.name: h.probability for h in scenario.hypotheses}
    
    # Initial state [x, y, v, heading]
    initial_state = np.array([
        scenario.initial_state.get("x", 0),
        scenario.initial_state.get("y", 0),
        scenario.initial_state.get("v", 0),
        scenario.initial_state.get("heading", 0),
    ])
    
    tree = ControlTree(
        max_depth=config.get("max_depth", 4),
        n_branches=config.get("n_branches", 6),
        steps_per_phase=config.get("steps_per_phase", 4),
        horizon=config.get("horizon", 20),
    )
    
    tree.initialize(initial_state, hypotheses, prior)
    tree.build()
    
    return tree

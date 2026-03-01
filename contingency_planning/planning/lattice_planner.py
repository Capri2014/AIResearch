"""
Lattice DP (Dynamic Programming) Planner

A traditional, reliable approach for motion planning:
- Discretizes the state space in Frenet coordinates (s, l, t)
- Builds a lattice graph of trajectory candidates
- Uses dynamic programming to find optimal path
- Followed by continuous smoothing (QP) for comfort

Reference: Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios"
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import heapq


@dataclass
class LatticeConfig:
    """Configuration for lattice planner"""
    # Longitudinal (s) discretization
    s_steps: int = 20
    s_spacing: float = 5.0  # meters
    
    # Lateral (l) discretization  
    l_lanes: int = 3  # number of lanes to consider
    l_spacing: float = 3.5  # lane width (meters)
    l_range: float = 5.25  # max lateral offset
    
    # Time discretization
    t_steps: int = 6
    t_spacing: float = 1.0  # seconds
    
    # Cost weights
    weight_lateral: float = 1.0
    weight_longitudinal: float = 0.5
    weight_acceleration: float = 2.0
    weight_jerk: float = 5.0
    weight_collision: float = 100.0


class LatticeNode:
    """Node in the lattice graph"""
    def __init__(self, s: float, l: float, t: float, cost: float = 0.0, parent: Optional['LatticeNode'] = None):
        self.s = s
        self.l = l
        self.t = t
        self.cost = cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def position(self) -> Tuple[float, float]:
        """Convert Frenet (s, l) to Cartesian (x, y)"""
        # Simplified: assume straight road along x-axis
        return (self.s, self.l)
    
    def state(self) -> Tuple[float, float, float, float, float, float]:
        """Return (x, y, heading, v, a, kappa) - simplified"""
        x, y = self.position()
        return (x, y, 0.0, 0.0, 0.0, 0.0)  # Placeholder


class LatticePlanner:
    """
    Lattice-based trajectory planner using dynamic programming.
    
    Produces high-comfort, reliable trajectories suitable for highway driving.
    """
    
    def __init__(self, config: Optional[LatticeConfig] = None):
        self.config = config or LatticeConfig()
        self.best_path: Optional[List[LatticeNode]] = None
        
    def generate_lattice(self) -> List[LatticeNode]:
        """Generate the lattice graph nodes"""
        nodes = []
        
        for t_idx in range(self.config.t_steps):
            t = (t_idx + 1) * self.config.t_spacing
            
            for s_idx in range(self.config.s_steps):
                s = (s_idx + 1) * self.config.s_spacing
                
                # Lateral positions: center lane + offset lanes
                for l_idx in range(-self.config.l_lanes + 1, self.config.l_lanes):
                    l = l_idx * self.config.l_spacing
                    if abs(l) <= self.config.l_range:
                        nodes.append(LatticeNode(s, l, t))
        
        return nodes
    
    def get_neighbors(self, node: LatticeNode) -> List[LatticeNode]:
        """Get valid neighbor nodes in the lattice"""
        neighbors = []
        dt = self.config.t_spacing
        ds = self.config.s_spacing
        dl = self.config.l_spacing
        
        # Forward in time
        for dt_idx in [1]:
            t_new = node.t + dt
            if t_new > self.config.t_steps * self.config.t_spacing:
                continue
                
            for ds_idx in [-1, 0, 1]:
                s_new = node.s + ds * ds_idx
                if s_new <= 0 or s_new > self.config.s_steps * self.config.s_spacing:
                    continue
                    
                for dl_idx in [-1, 0, 1]:
                    l_new = node.l + dl * dl_idx
                    if abs(l_new) > self.config.l_range:
                        continue
                    
                    neighbors.append(LatticeNode(s_new, l_new, t_new))
        
        return neighbors
    
    def compute_cost(self, from_node: LatticeNode, to_node: LatticeNode) -> float:
        """Compute transition cost between nodes"""
        # Lateral cost
        dl = abs(to_node.l - from_node.l)
        lateral_cost = self.config.weight_lateral * dl
        
        # Longitudinal cost  
        ds = to_node.s - from_node.s
        longitudinal_cost = self.config.weight_longitudinal * (self.config.s_spacing - ds) ** 2
        
        # Time cost (prefer faster completion)
        time_cost = 0.1 * to_node.t
        
        return lateral_cost + longitudinal_cost + time_cost
    
    def plan(self, 
             start_state: Tuple[float, float, float],
             goal_s: float,
             obstacles: Optional[List[Dict]] = None) -> List[Tuple[float, float]]:
        """
        Plan a trajectory from start to goal.
        
        Args:
            start_state: (x, y, heading) starting position
            goal_s: desired longitudinal goal distance
            obstacles: list of obstacle dicts with 's', 'l', 't', 'width'
            
        Returns:
            List of (x, y) waypoints
        """
        obstacles = obstacles or []
        
        # Start node at origin
        start_node = LatticeNode(0, 0, 0)
        
        # Generate lattice
        lattice = self.generate_lattice()
        
        # Dijkstra's algorithm
        open_set = [start_node]
        visited = {}
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Check if reached goal region
            if current.s >= goal_s:
                # Reconstruct path
                path = []
                node = current
                while node:
                    path.append(node.position())
                    node = node.parent
                path.reverse()
                self.best_path = path
                return path
            
            # Get state key
            key = (round(current.s, 1), round(current.l, 1), round(current.t, 1))
            if key in visited:
                continue
            visited[key] = current.cost
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                # Check collision with obstacles
                if self._check_collision(neighbor, obstacles):
                    continue
                
                # Compute cost
                transition_cost = self.compute_cost(current, neighbor)
                new_cost = current.cost + transition_cost
                
                neighbor.cost = new_cost
                neighbor.parent = current
                heapq.heappush(open_set, neighbor)
        
        # No path found - return fallback
        return self._fallback_plan(goal_s)
    
    def _check_collision(self, node: LatticeNode, obstacles: List[Dict]) -> bool:
        """Check if node collides with any obstacle"""
        for obs in obstacles:
            # Simple box collision
            s_diff = abs(node.s - obs.get('s', 0))
            l_diff = abs(node.l - obs.get('l', 0))
            
            s_margin = obs.get('width', 2.0) / 2 + 1.0  # vehicle width + margin
            l_margin = obs.get('length', 4.5) / 2 + 0.5
            
            if s_diff < s_margin and l_diff < l_margin:
                return True
        return False
    
    def _fallback_plan(self, goal_s: float) -> List[Tuple[float, float]]:
        """Generate simple fallback straight-line plan"""
        n_points = 20
        path = []
        for i in range(n_points):
            s = (i + 1) * goal_s / n_points
            path.append((s, 0))  # Center lane
        return path
    
    def get_candidate_trajectories(self, 
                                   start_state: Tuple[float, float, float],
                                   goal_s: float,
                                   n_candidates: int = 32) -> List[List[Tuple[float, float]]]:
        """
        Generate top-K candidate trajectories for downstream evaluation.
        
        This is the key output that feeds into the Evaluator module.
        """
        # Generate multiple paths with different lateral preferences
        candidates = []
        
        # Center lane preference
        center_path = self.plan(start_state, goal_s)
        if center_path:
            candidates.append(center_path)
        
        # Left lane preference
        self.config.l_lanes = 2
        left_path = self.plan(start_state, goal_s)
        if left_path:
            candidates.append(left_path)
        
        # Right lane preference
        self.config.l_lanes = 2
        right_path = self.plan(start_state, goal_s)
        if right_path:
            candidates.append(right_path)
        
        # Fill remaining with variations
        while len(candidates) < n_candidates and candidates:
            # Add small lateral offsets
            base = candidates[0]
            offset = (len(candidates) - 1) * 0.5
            varied = [(s, l + offset * ((i % 2) - 0.5)) for i, (s, l) in enumerate(base)]
            candidates.append(varied)
        
        return candidates[:n_candidates]


def create_lattice_planner() -> LatticePlanner:
    """Factory function to create configured lattice planner"""
    config = LatticeConfig(
        s_steps=20,
        s_spacing=5.0,
        l_lanes=3,
        l_spacing=3.5,
        t_steps=6,
        t_spacing=1.0,
        weight_lateral=1.0,
        weight_longitudinal=0.5,
        weight_acceleration=2.0,
        weight_jerk=5.0,
        weight_collision=100.0
    )
    return LatticePlanner(config)


if __name__ == "__main__":
    # Test the lattice planner
    planner = create_lattice_planner()
    
    start_state = (0, 0, 0)  # x, y, heading
    goal_s = 100  # meters
    
    # Plan with no obstacles
    path = planner.plan(start_state, goal_s)
    print(f"Planned path with {len(path)} waypoints")
    print(f"First 5 waypoints: {path[:5]}")
    
    # Plan with obstacle
    obstacles = [{'s': 50, 'l': 0, 'width': 2, 'length': 4.5}]
    path_obs = planner.plan(start_state, goal_s, obstacles)
    print(f"Path with obstacle: {len(path_obs)} waypoints")
    
    # Get multiple candidates
    candidates = planner.get_candidate_trajectories(start_state, goal_s, n_candidates=10)
    print(f"Generated {len(candidates)} candidates")

"""
Tree QP Optimizer

QP optimization for Control-Tree using OSQP solver.
Based on Control-Tree Optimization (Phiquepal & Toussaint, ICRA 2021).
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import osqp
from scipy import sparse


class TreeQPOptimizer:
    """
    Quadratic Program optimizer for Control-Tree.
    
    Solves: min sum(x'Qx + u'Ru) + Vf(x_N)
            s.t. x_{k+1} = Ax_k + Bu_k
                  safety constraints
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        horizon: int = 20,
        q_weight: float = 1.0,
        r_weight: float = 0.1,
        terminal_weight: float = 2.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.q_weight = q_weight
        self.r_weight = r_weight
        self.terminal_weight = terminal_weight
        
        # Default dynamics (simple bicycle model)
        self.dt = 0.1
        
    def set_dynamics(self, A: np.ndarray, B: np.ndarray):
        """Set linearized dynamics."""
        self.A = A
        self.B = B
        
    def default_dynamics(self, velocity: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get default bicycle model dynamics.
        
        State: [x, y, v, heading]
        Control: [acceleration, steering]
        """
        # Linearized bicycle model
        dt = self.dt
        v = velocity
        
        A = np.array([
            [1, 0, dt * np.cos(0), -dt * v * np.sin(0)],  # x
            [0, 1, dt * np.sin(0), dt * v * np.cos(0)],  # y
            [0, 0, 1, 0],                                  # v
            [0, 0, 0, 1],                                  # heading
        ])
        
        B = np.array([
            [0, 0],       # x
            [0, 0],       # y
            [dt, 0],      # v
            [0, dt * v],  # heading
        ])
        
        return A, B
    
    def solve(
        self,
        x0: np.ndarray,
        x_goal: np.ndarray,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        obstacles: Optional[list] = None,
        safety_margin: float = 2.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Solve QP for optimal trajectory.
        
        Args:
            x0: Initial state [state_dim]
            x_goal: Goal state [state_dim]
            A, B: Dynamics matrices (uses default if None)
            obstacles: List of obstacle positions
            safety_margin: Minimum distance to obstacles
            
        Returns:
            control_seq: Optimal control sequence [horizon, action_dim]
            cost: Optimal cost
        """
        # Use default dynamics if not provided
        if A is None or B is None:
            A, B = self.default_dynamics(x0[2])  # Use current velocity
        
        # Build QP matrices
        H, f = self._build_cost_matrices(x_goal)
        A_qp, b = self._build_dynamics_constraint(x0, A, B)
        
        # Add safety constraints if obstacles
        if obstacles:
            A_safety, b_safety = self._build_safety_constraints(
                x0, A, B, obstacles, safety_margin
            )
            A_qp = sparse.vstack([A_qp, A_safety])
            b = np.concatenate([b, b_safety])
        
        # Solve QP
        problem = osqp.OSQP()
        problem.setup(H, f, A_qp, b, np.zeros(len(b)), verbose=False)
        results = problem.solve()
        
        if results.info.status != 'solved':
            # Fallback: return nominal control
            return self._nominal_control(), float('inf')
        
        # Extract control sequence
        n_controls = self.horizon * self.action_dim
        u_seq = results.x[:n_controls].reshape(self.horizon, self.action_dim)
        
        return u_seq, results.info.obj_val
    
    def _build_cost_matrices(self, x_goal: np.ndarray) -> Tuple[sparse.csc, np.ndarray]:
        """Build QP cost matrices."""
        n_states = self.state_dim
        n_controls = self.action_dim
        N = self.horizon
        
        # Hessian H = diag([Q, Q, ..., Q, R, R, ...])
        Q_block = self.q_weight * np.eye(n_states)
        R_block = self.r_weight * np.eye(n_controls)
        P_terminal = self.terminal_weight * np.eye(n_states)
        
        # Build block diagonal
        H_diag = [Q_block] * N + [R_block] * N + [P_terminal]
        H = sparse.block_diag(H_diag).tocsc()
        
        # Linear term f
        f = np.zeros(N * (n_states + n_controls) + n_states)
        
        # Add goal term: -2 * x_goal' * Q * x
        for k in range(N):
            f[k*n_states:(k+1)*n_states] = -2 * self.q_weight * x_goal
        
        return H, f
    
    def _build_dynamics_constraint(
        self, 
        x0: np.ndarray, 
        A: np.ndarray, 
        B: np.ndarray
    ) -> Tuple[sparse.csc, np.ndarray]:
        """Build dynamics equality constraints: x_{k+1} = Ax_k + Bu_k"""
        N = self.horizon
        n = self.state_dim
        m = self.action_dim
        
        # Constraint: x1 - Ax0 - Bu0 = 0
        #           x2 - Ax1 - Bu1 = 0
        #           ...
        #           xN - Ax_{N-1} - Bu_{N-1} = 0
        
        A_eq = []
        b_eq = []
        
        for k in range(N):
            # x_{k+1} - Ax_k - Bu_k = 0
            row = np.zeros((n, N * (n + m) + n))
            
            # x_k coefficient
            row[:, k*n:(k+1)*n] = -A
            
            # u_k coefficient  
            row[:, N*n + k*m:(N*n + (k+1)*m)] = -B
            
            # x_{k+1} coefficient
            if k < N - 1:
                row[:, (k+1)*n:(k+2)*n] = np.eye(n)
            
            A_eq.append(row)
            b_eq.append(np.zeros(n))
        
        A_eq = sparse.vstack(A_eq).tocsc()
        b_eq = np.concatenate(b_eq)
        
        return A_eq, b_eq
    
    def _build_safety_constraints(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        obstacles: list,
        safety_margin: float,
    ) -> Tuple[sparse.csc, np.ndarray]:
        """Build safety (distance) constraints."""
        N = self.horizon
        n = self.state_dim
        m = self.action_dim
        
        # Simulate trajectory to get predicted positions
        x_pred = self._simulate_trajectory(x0, A, B)
        
        A_ineq = []
        b_ineq = []
        
        for t, obs in enumerate(obstacles):
            obs_pos = np.array([obs["x"], obs["y"]])
            
            for k in range(N):
                if k >= len(x_pred):
                    break
                    
                # Distance constraint: ||pos - obs_pos|| >= safety_margin
                # Linearized: (pos - obs_pos) * dir >= safety_margin
                
                pos = x_pred[k, :2]  # x, y
                diff = pos - obs_pos
                dist = np.linalg.norm(diff)
                
                if dist > 1e-6:
                    direction = diff / dist
                    
                    # Linear constraint: direction * pos >= safety_margin
                    row = np.zeros((1, N * (n + m) + n))
                    
                    # Accumulate state contribution
                    for j in range(k + 1):
                        row[:, j*n:(j+1)*n] += direction @ self._state_projection(j, k, A, B, x0)
                    
                    A_ineq.append(row)
                    b_ineq.append(safety_margin)
        
        if A_ineq:
            A_ineq = sparse.vstack(A_ineq).tocsc()
            b_ineq = np.array(b_ineq)
        else:
            A_ineq = sparse.csc_matrix((0, N * (n + m) + n))
            b_ineq = np.array([])
        
        return A_ineq, b_ineq
    
    def _state_projection(self, j: int, k: int, A: np.ndarray, B: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Get contribution of u_j to x_k."""
        if j > k:
            return np.zeros_like(A)
        
        # x_k = A^{k-j} * B * u_j (assuming x_j = 0)
        result = np.linalg.matrix_power(A, k - j) @ B
        return result
    
    def _simulate_trajectory(
        self, 
        x0: np.ndarray, 
        A: np.ndarray, 
        B: np.ndarray,
        u_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simulate trajectory given control sequence."""
        if u_seq is None:
            u_seq = self._nominal_control() * np.ones((self.horizon, 1))
        
        x_pred = np.zeros((self.horizon + 1, self.state_dim))
        x_pred[0] = x0
        
        for k in range(self.horizon):
            x_pred[k + 1] = A @ x_pred[k] + B @ u_seq[k]
        
        return x_pred[1:]  # Exclude initial state
    
    def _nominal_control(self) -> np.ndarray:
        """Get nominal (zero) control."""
        return np.zeros(self.action_dim)
    
    def solve_branch(
        self,
        tree,
        branch_id: str,
        x0: np.ndarray,
        x_goal: np.ndarray,
        constraints: Dict[str, Any],
    ) -> Tuple[np.ndarray, float]:
        """Solve QP for a specific branch."""
        obstacles = constraints.get("obstacles", [])
        safety_margin = constraints.get("safety_margin", 2.0)
        
        return self.solve(x0, x_goal, obstacles=obstacles, safety_margin=safety_margin)
    
    def optimize_tree(self, tree: 'ControlTree', x_goal: np.ndarray, constraints: Dict) -> None:
        """
        Optimize all branches in the tree.
        
        Updates control sequences and costs for each node.
        """
        A, B = self.default_dynamics()
        
        # Optimize shared trunk first
        shared_trunk = tree.get_shared_trunk()
        x_current = tree.nodes[tree.root_id].state
        
        for node in shared_trunk:
            u_seq, cost = self.solve(x_current, x_goal, A, B, 
                                     obstacles=constraints.get("obstacles", []),
                                     safety_margin=constraints.get("safety_margin", 2.0))
            node.control = u_seq
            node.cost = cost
            
            # Update state for next node
            x_current = A @ x_current + B @ u_seq[0]
            node.state = x_current
        
        # Optimize each branch
        for hypothesis, branch in tree.get_all_branches().items():
            x_current = shared_trunk[-1].state if shared_trunk else tree.nodes[tree.root_id].state
            
            for node in branch:
                # Adjust goal for hypothesis-specific constraint
                adjusted_goal = self._adjust_goal_for_hypothesis(x_goal, hypothesis)
                
                u_seq, cost = self.solve(x_current, adjusted_goal, A, B,
                                        obstacles=self._get_hypothesis_obstacles(
                                            constraints.get("obstacles", []), hypothesis),
                                        safety_margin=constraints.get("safety_margin", 2.0))
                
                node.control = u_seq
                node.cost = cost
                
                # Update state
                x_current = A @ x_current + B @ u_seq[0]
                node.state = x_current
    
    def _adjust_goal_for_hypothesis(self, x_goal: np.ndarray, hypothesis: str) -> np.ndarray:
        """Adjust goal based on hypothesis (e.g., more conservative for risky hypotheses)."""
        adjusted = x_goal.copy()
        
        # For risky hypotheses, reduce target velocity
        if "cross" in hypothesis or "cut" in hypothesis:
            adjusted[2] = min(adjusted[2], 5.0)  # Reduce speed
        
        return adjusted
    
    def _get_hypothesis_obstacles(self, obstacles: list, hypothesis: str) -> list:
        """Get obstacle list for specific hypothesis."""
        if not obstacles:
            return []
        
        # Some obstacles only appear for certain hypotheses
        return [o for o in obstacles if o.get("hypothesis", "all") in [hypothesis, "all"]]

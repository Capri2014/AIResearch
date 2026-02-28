"""
Fast QP Optimizer with Warm Start

Optimized OSQP solver for real-time control with warm starting
and sparse matrix support for faster computation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import osqp
from scipy import sparse


class FastTreeQPOptimizer:
    """
    Fast QP optimizer with warm start and sparse matrices.
    
    Optimizations:
    - Warm starting from previous solution
    - Sparse matrix formulation
    - Early termination on convergence
    - Reduced horizon for real-time
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        control_dim: int = 2,
        horizon: int = 15,  # Reduced for speed
        q_weight: float = 1.0,
        r_weight: float = 0.1,
        terminal_weight: float = 5.0,
        max_iter: int = 100,
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.q_weight = q_weight
        self.r_weight = r_weight
        self.terminal_weight = terminal_weight
        self.max_iter = max_iter
        
        # Time step
        self.dt = 0.1
        
        # Previous solution for warm start
        self.prev_solution = None
        self.previous_states = None
        
        # Cached problem
        self.problem_cache = None
        
        # Initialize OSQP solver
        self._init_solver()
    
    def _init_solver(self):
        """Initialize OSQP solver."""
        self.osqp = osqp.OSQP()
    
    def set_dynamics(self, A: np.ndarray, B: np.ndarray):
        """Set linearized dynamics."""
        self.A = A
        self.B = B
    
    def default_dynamics(self, velocity: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Get default bicycle model dynamics."""
        dt = self.dt
        v = velocity
        
        A = np.array([
            [1, 0, dt * np.cos(0), -dt * v * np.sin(0)],
            [0, 1, dt * np.sin(0), dt * v * np.cos(0)],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        
        B = np.array([
            [0, 0],
            [0, 0],
            [dt, 0],
            [0, dt * v],
        ])
        
        return A, B
    
    def setup_problem(
        self,
        x0: np.ndarray,
        goal: np.ndarray = None,
        obstacles: list = None,
        velocity_limit: float = 15.0,
    ):
        """
        Setup QP problem with sparse matrices.
        
        Minimize: sum(x'Qx + u'Ru) + x_N'Q_f*x_N
        s.t.:    x_{k+1} = A*x_k + B*u_k
                 velocity constraints
                 obstacle avoidance
        """
        n_states = self.state_dim
        n_controls = self.control_dim
        N = self.horizon
        
        # Goal (default to origin)
        if goal is None:
            goal = np.zeros(2)
        
        # Weight matrices
        Q = np.eye(n_states) * self.q_weight
        Q[2, 2] = self.q_weight * 0.5  # velocity weight
        R = np.eye(n_controls) * self.r_weight
        Qf = np.eye(n_states) * self.terminal_weight
        
        # Build Hessian (sparse)
        # H = block_diag(Q, Q, ..., Qf, R, R, ...)
        H_diag = []
        for _ in range(N):
            H_diag.append(Q)
        H_diag.append(Qf)  # terminal
        for _ in range(N):
            H_diag.append(R)
        
        H = sparse.block_diag(H_diag, format='csc')
        
        # Linear term (gradient) - toward goal
        # For state variables: -Q @ goal
        # For control: 0
        q = np.zeros(N * n_states + N * n_controls)
        for i in range(N):
            q[i * n_states:(i + 1) * n_states] = -Q @ np.append(goal, [0, 0])
        
        # Equality constraints: x_{k+1} = A*x_k + B*u_k
        # N constraints, each: -A*x_k + x_{k+1} - B*u_k = 0
        n_eq = N * n_states
        
        # Build A_eq and b_eq
        A_eq_rows = []
        b_eq = np.zeros(n_eq)
        
        for k in range(N):
            # Constraint: x_{k+1} - A*x_k - B*u_k = 0
            row = np.zeros((n_states, N * n_states + N * n_controls))
            
            # x_k terms (with minus sign)
            row[:, k * n_states:(k + 1) * n_states] = -np.eye(n_states) @ self.A
            
            # x_{k+1} terms
            row[:, (k + 1) * n_states:(k + 2) * n_states] = np.eye(n_states)
            
            # u_k terms
            row[:, N * n_states + k * n_controls:N * n_states + (k + 1) * n_controls] = -np.eye(n_states) @ self.B
            
            A_eq_rows.append(row)
        
        A_eq = sparse.vstack(A_eq_rows, format='csc')
        
        # Initial condition: x_0 = x0
        # Add to equality constraints
        A_eq_initial = np.zeros((n_states, N * n_states + N * n_controls))
        A_eq_initial[:, :n_states] = np.eye(n_states)
        b_eq_initial = x0
        
        A_eq = sparse.vstack([A_eq_initial, A_eq], format='csc')
        b_eq = np.concatenate([b_eq_initial, b_eq])
        
        # Inequality constraints (velocity + obstacles)
        n_ineq = N * 2  # velocity limits per step
        if obstacles:
            n_ineq += N * len(obstacles) * 2  # obstacle avoidance
        
        A_ineq = np.zeros((n_ineq, N * n_states + N * n_controls))
        b_ineq = np.zeros(n_ineq)
        
        # Velocity constraints: -v <= vel <= v_max
        vel_idx = 0
        for k in range(N):
            # Extract velocity from state [x, y, v, heading]
            # v >= 0: -state[2] <= 0
            A_ineq[vel_idx, k * n_states + 2] = -1
            b_ineq[vel_idx] = 0
            vel_idx += 1
            
            # v <= v_max: state[2] <= v_max
            A_ineq[vel_idx, k * n_states + 2] = 1
            b_ineq[vel_idx] = velocity_limit
            vel_idx += 1
        
        # Obstacle constraints (simple circular)
        if obstacles:
            for obs in obstacles:
                obs_pos = np.array(obs['position'][:2])
                obs_radius = obs['radius'] + 1.0  # safety margin
                
                for k in range(N):
                    # (x - ox)^2 + (y - oy)^2 >= r^2
                    # Linearized: (x-ox)*ox + (y-oy)*oy >= r*||o||
                    # We'll use a simpler box constraint around obstacle
                    pass  # Simplified for now
        
        A_ineq_sparse = sparse.csc_matrix(A_ineq) if n_ineq > 0 else None
        
        # Setup OSQP
        self.osqp.setup(H, q, A_eq, b_eq, A_ineq_sparse, b_ineq, max_iter=self.max_iter)
        
        # Cache for warm start
        self.problem_cache = {
            'H': H, 'q': q, 'A_eq': A_eq, 'b_eq': b_eq,
            'A_ineq': A_ineq_sparse, 'b_ineq': b_ineq,
        }
    
    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray = None,
        obstacles: list = None,
        prev_solution: np.ndarray = None,
        velocity_limit: float = 15.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve QP with optional warm start.
        
        Args:
            x0: Initial state
            goal: Goal position
            obstacles: List of obstacle dicts
            prev_solution: Previous solution for warm start
            velocity_limit: Maximum velocity
            
        Returns:
            (states, controls) trajectories
        """
        # Setup problem
        self.setup_problem(x0, goal, obstacles, velocity_limit)
        
        # Warm start if available
        if prev_solution is not None:
            self.osqp.warm_start({'x': prev_solution})
        
        # Solve
        result = self.osqp.solve()
        
        if result.info.status_val != 1:  # OSQP_MAX_ITER_REACHED or error
            # Try with more iterations
            self.osqp.update_settings(max_iter=200)
            result = self.osqp.solve()
        
        # Extract solution
        n_states = self.state_dim
        n_controls = self.control_dim
        N = self.horizon
        
        solution = result.x
        
        # Reshape to states and controls
        states = solution[:N * n_states].reshape(N, n_states)
        controls = solution[N * n_states:].reshape(N, n_controls)
        
        # Store for next warm start
        self.prev_solution = solution
        
        return states, controls
    
    def solve_warm(
        self,
        x0: np.ndarray,
        goal: np.ndarray = None,
        obstacles: list = None,
        velocity_limit: float = 15.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve with automatic warm starting.
        """
        return self.solve(x0, goal, obstacles, self.prev_solution, velocity_limit)
    
    def solve_mpc(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        obstacles: list,
        max_steps: int = 100,
        replan_hz: float = 10.0,
    ) -> Tuple[np.ndarray, list]:
        """
        Run MPC loop.
        
        Args:
            x0: Initial state
            goal: Goal position
            obstacles: List of obstacles
            max_steps: Maximum simulation steps
            replan_hz: Replanning frequency
            
        Returns:
            (trajectory, control_history)
        """
        states = [x0.copy()]
        controls = []
        
        current_state = x0.copy()
        
        for step in range(max_steps):
            # Solve
            state_traj, control_traj = self.solve_warm(
                current_state, goal, obstacles
            )
            
            # Apply first control
            if len(control_traj) > 0:
                u = control_traj[0]
                controls.append(u.copy())
                
                # Simulate forward
                A, B = self.default_dynamics(current_state[2])
                next_state = A @ current_state + B @ u
                current_state = next_state
                
                states.append(current_state.copy())
                
                # Check goal
                if np.linalg.norm(current_state[:2] - goal) < 1.0:
                    break
        
        return np.array(states), controls


class AdaptiveHorizonOptimizer(FastTreeQPOptimizer):
    """
    Adaptive horizon optimizer that adjusts horizon based on situation.
    """
    
    def __init__(self, min_horizon: int = 8, max_horizon: int = 20, **kwargs):
        super().__init__(horizon=max_horizon, **kwargs)
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
    
    def compute_optimal_horizon(
        self,
        distance_to_goal: float,
        ttc: float,
        uncertainty: float,
    ) -> int:
        """
        Compute optimal horizon based on situation.
        
        Args:
            distance_to_goal: Distance to goal
            ttc: Time to collision
            uncertainty: State uncertainty
        """
        # Base horizon on distance
        horizon = int(distance_to_goal / 2.0)
        
        # Reduce horizon if TTC is low
        if ttc < 2.0:
            horizon = min(horizon, int(ttc * 3))
        
        # Reduce horizon if uncertainty is high
        if uncertainty > 2.0:
            horizon = min(horizon, 10)
        
        # Clamp
        return max(self.min_horizon, min(self.max_horizon, horizon))
    
    def solve_adaptive(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        obstacles: list,
        ttc: float = float('inf'),
        uncertainty: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve with adaptive horizon."""
        distance = np.linalg.norm(x0[:2] - goal)
        
        self.horizon = self.compute_optimal_horizon(distance, ttc, uncertainty)
        
        return self.solve_warm(x0, goal, obstacles)


def create_fast_optimizer(
    state_dim: int = 4,
    control_dim: int = 2,
    horizon: int = 15,
) -> FastTreeQPOptimizer:
    """Create configured fast optimizer."""
    optimizer = FastTreeQPOptimizer(
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
    )
    
    # Set default dynamics
    A, B = optimizer.default_dynamics(velocity=10.0)
    optimizer.set_dynamics(A, B)
    
    return optimizer

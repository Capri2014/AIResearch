# Contingency Planning Implementation Plan

## Overview
Implement and compare three approaches:
1. **Classical Tree-Based** (Control-Tree Optimization)
2. **Model-Based** (Learning-based: neural networks + safety filters)
3. **Simulation & Visualization** (CARLA + comparison metrics)

---

## Phase 1: Environment Setup (Week 1)

### 1.1 CARLA Integration
```python
# requirements.txt
carla>=0.9.14
numpy>=1.21
torch>=2.0
pyyaml
osqp  # For tree-based QP solver
```

### 1.2 Define Contingency Scenarios
| Scenario | Type | Discrete Uncertainty |
|----------|------|---------------------|
| Pedestrian crossing | External/Interactive | Cross vs Yield |
| Highway cut-in | External/Interactive | Cut-in vs Pass |
| Occluded intersection | External/Env | Clear vs Blocked |
| Sensor degradation | Internal | Nominal vs Degraded |

### 1.3 Metrics Definition
```python
@dataclass
class ContingencyMetrics:
    # Safety
    collision_rate: float          # % of episodes with collision
    mrc_trigger_rate: float        # % of episodes requiring MRC
    
    # Efficiency
    avg_completion_time: float     # Time to reach goal
    avg_speed: float               # Average speed
    deviation_from_optimal: float  # Deviation from nominal plan
    
    # Computation
    avg_planning_time: float       # ms
    tree_depth_used: int           # Actual branching depth
```

---

## Phase 2: Classical Tree-Based (Week 2-3)

### 2.1 Core Data Structures
```python
# planning/tree/contingency_tree.py

@dataclass
class TreeNode:
    id: str
    parent_id: Optional[str]
    hypothesis: str              # e.g., "pedestrian_cross"
    belief: float               # P(hypothesis | observation)
    state: np.ndarray           # ego state at this node
    control: np.ndarray         # optimized control
    cost: float                 # stage + terminal cost
    is_shared: bool             # part of common trunk

class ControlTree:
    def __init__(self, config):
        self.max_depth = config.n_branches
        self.observation_times = config.steps_per_phase
        self.shared_trunk_length = 0
    
    def build(self, initial_state, discrete_uncertainties):
        """Build tree over discrete hypotheses"""
        # Root → shared trunk → branching at observation points
        pass
    
    def optimize(self, dynamics, cost_fn, constraints):
        """QP optimization per branch (OSQP)"""
        pass
    
    def select_action(self, current_belief):
        """Follow shared trunk or commit to branch"""
        pass
```

### 2.2 QP Solver Integration
```python
# planning/tree/optimization.py
import osqp
import numpy as np

class TreeQPOptimizer:
    def solve_branch(self, A, B, Q, R, x0, x_goal, constraints):
        """
        Minimize: sum(x'Qx + u'Ru) + Vf(x_N)
        s.t.: x_{k+1} = Ax_k + Bu_k
              safety constraints per branch
        """
        # Standard MPC QP formulation
        pass
```

### 2.3 Discrete Uncertainty Handling
```python
# planning/tree/belief_tracker.py

class BeliefTracker:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses
        self.belief = {h: 1.0/len(hypotheses) for h in hypotheses}
    
    def update(self, observation):
        """Bayesian update of belief given observation"""
        # P(h | o) ∝ P(o | h) * P(h)
        pass
```

---

## Phase 3: Model-Based (Week 4-5)

### 3.1 Architecture Overview
```
Input: [ego_state, route, perception]
           ↓
    ┌──────┴──────┐
    ↓             ↓
Encoder    UncertaintyHead
    ↓             ↓
    └──────┬──────┘
           ↓
   BranchingLayer → [plan_1, plan_2, ..., plan_n]
           ↓
    SafetyFilter (CBF) → Safe output
```

### 3.2 Neural Contingency Network
```python
# planning/models/contingency_network.py
import torch
import torch.nn as nn

class ContingencyNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        # Branching heads (one per contingency)
        self.branch_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, config.horizon * config.action_dim)
            )
            for _ in range(config.n_contingencies)
        ])
        
        # Uncertainty predictor
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, len(config.contingencies)),
            nn.Softmax(dim=-1)
        )
        
        # Safety critic (for training)
        self.safety_critic = nn.Sequential(
            nn.Linear(256 + config.horizon * config.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Safety score
        )
    
    def forward(self, state, context):
        """
        Args:
            state: [B, state_dim]
            context: [B, n_contingencies] - context for each contingency
        Returns:
            plans: List of [B, horizon, action_dim] trajectories
            uncertainty: [B, n_contingencies] probability
        """
        enc = self.encoder(state)
        
        plans = [head(enc).view(-1, self.horizon, self.action_dim) 
                 for head in self.branch_heads]
        
        uncertainty = self.uncertainty_head(enc)
        
        return plans, uncertainty
```

### 3.3 Safety Filter (CBF)
```python
# planning/models/safety_filter.py

class ControlBarrierFilter:
    def __init__(self, config):
        self.safety_margin = config.safety_margin
        self.barrier_gain = config.barrier_gain
    
    def filter(self, nominal_plan, ego_state, obstacles):
        """
        CBF-QP: minimally invasive safety correction
        min ||u - u_nom||^2
        s.t. h(x) >= 0 (barrier constraint)
        """
        safe_plan = []
        for u_nom in nominal_plan:
            u_safe = self._solve_cbf_qp(u_nom, ego_state, obstacles)
            safe_plan.append(u_safe)
            ego_state = self._dynamics(ego_state, u_safe)
        
        return safe_plan
    
    def _solve_cbf_qp(self, u_nom, state, obstacles):
        # Build QP: min (u - u_nom)^2 subject to CBF constraint
        # ∂h/∂x * f(x,u) >= -α(h)
        pass
```

### 3.4 Training Loop
```python
# training/train_contingency.py

def train_contingency_network(dataset, config):
    model = ContingencyNetwork(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # Load pretrained waypoint model
    waypoint_model = load_pretrained(config.waypoint_model_path)
    freeze(waypoint_model)  # Keep frozen
    
    for epoch in range(config.epochs):
        for batch in dataset:
            # 1. Get nominal plans from waypoint model
            nominal_plans = waypoint_model(batch.state)
            
            # 2. Generate contingency plans
            plans, uncertainty = model(batch.state, batch.context)
            
            # 3. Apply safety filter
            safe_plans = [safety_filter(plan, batch.state, batch.obstacles) 
                         for plan in plans]
            
            # 4. Compute loss
            loss = (
                config.lambda_plan * plan_mse(safe_plans, batch.expert_plans) +
                config.lambda_uncertainty * uncertainty_loss(uncertainty, batch.labels) +
                config.lambda_safety * safety_violation_loss(safe_plans, batch.obstacles)
            )
            
            # 5. Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Phase 4: Simulation & Comparison (Week 6)

### 4.1 CARLA Scenario Runner
```python
# simulation/carla_comparison.py
import carla

class ContingencyBenchmark:
    def __init__(self, config):
        self.client = carla.Client(config.host, config.port)
        self.world = self.client.load_world(config.map)
        
        # Metrics
        self.metrics = defaultdict(list)
    
    def run_comparison(self, scenarios, tree_model, model_model):
        """Run same scenarios with both approaches"""
        results = {"tree": [], "model": [], "baseline": []}
        
        for scenario in scenarios:
            # Reset
            self.reset_scenario(scenario)
            
            # Run tree-based
            tree_metrics = self.run_episode(tree_model, scenario)
            results["tree"].append(tree_metrics)
            
            # Run model-based  
            model_metrics = self.run_episode(model_model, scenario)
            results["model"].append(model_metrics)
            
            # Run baseline (no contingency)
            baseline_metrics = self.run_episode(baseline, scenario)
            results["baseline"].append(baseline_metrics)
        
        return self.compute_statistics(results)
    
    def run_episode(self, model, scenario):
        """Run single episode"""
        episode_data = {
            "collisions": 0,
            "mrc_triggers": 0,
            "completion_time": 0,
            "planning_times": [],
            "trajectory": []
        }
        
        while not scenario.is_complete():
            # Get observation
            obs = self.get_observation()
            
            # Plan
            start_time = time.time()
            action = model.plan(obs)
            plan_time = time.time() - start_time
            episode_data["planning_times"].append(plan_time)
            
            # Execute
            self.apply_action(action)
            
            # Check safety
            if self.check_collision():
                episode_data["collisions"] += 1
            
            if self.check_mrc_trigger():
                episode_data["mrc_triggers"] += 1
            
            episode_data["trajectory"].append(self.ego.get_state())
        
        return episode_data
```

### 4.2 Visualization
```python
# visualization/dashboard.py

class ComparisonDashboard:
    def __init__(self):
        self.fig = go.Figure()
    
    def plot_trajectories(self, tree_traj, model_traj, scenario):
        """Plot trajectories in CARLA top-down view"""
        # Tree: blue
        # Model: red  
        # Baseline: gray dashed
        pass
    
    def plot_metrics(self, results):
        """Bar chart comparison"""
        # Collision rate
        # MRC rate
        # Completion time
        # Planning time
        pass
    
    def animate_episode(self, episode_data):
        """Animation of contingency decisions"""
        # Show belief updates
        # Show branch selection
        # Show safety filter activation
        pass
```

### 4.3 Results Table
| Metric | Baseline | Tree-Based | Model-Based |
|--------|----------|------------|-------------|
| Collision Rate | ? | ? | ? |
| MRC Trigger Rate | ? | ? | ? |
| Avg Completion Time | ? | ? | ? |
| Avg Speed | ? | ? | ? |
| Planning Time (ms) | N/A | ? | ? |

---

## File Structure
```
contingency_planning/
├── __init__.py
├── configs/
│   ├── tree_config.yaml
│   └── model_config.yaml
├── planning/
│   ├── __init__.py
│   ├── tree/
│   │   ├── __init__.py
│   │   ├── contingency_tree.py    # Core tree structure
│   │   ├── optimization.py        # QP solver
│   │   └── belief_tracker.py      # Belief updates
│   └── models/
│       ├── __init__.py
│       ├── contingency_network.py # Neural branching
│       └── safety_filter.py       # CBF
├── training/
│   ├── __init__.py
│   └── train_contingency.py
├── simulation/
│   ├── __init__.py
│   ├── carla_comparison.py        # Benchmark runner
│   └── scenarios.py               # Test scenarios
├── visualization/
│   ├── __init__.py
│   └── dashboard.py               # Comparison plots
└── evaluation/
    ├── __init__.py
    └── metrics.py                 # Metric computation
```

---

## Implementation Order

1. **Week 1**: Setup + define scenarios + metrics
2. **Week 2-3**: Classical tree-based (Control-Tree)
3. **Week 4-5**: Model-based (neural + CBF)
4. **Week 6**: Integration + CARLA benchmark + visualization

**Total: ~6 weeks**

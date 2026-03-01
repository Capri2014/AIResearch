# Contingency Planning for Safety-Critical Autonomous Vehicles: Survey & Comparison

**Survey Date:** 2026-02-27  
**Base Paper:** [arXiv:2601.14880](https://arxiv.org/abs/2601.14880) - Lei Zheng et al.  
**Authors:** Lei Zheng (CMU), Luyao Zhang (TU Delft), Peiqi Yu (CMU), Yifan Sun (CMU), Sergio Grammatico (TU Delft), Jun Ma (HKUST), Changliu Liu (CMU)

---

## 1. Abstract & Core Definition

**Contingency Planning** = the architectural capability for AVs to anticipate and mitigate discrete, low-frequency, high-impact hazards (sensor outages, adversarial interactions).

### Key Distinction
- **Contingency** = "known unknown" — structural form is known at design time, but occurrence time/likelihood/trajectory are uncertain
- Different from continuous uncertainties (sensor noise)

### Safety-Efficiency Trade-off
- Goal: ensure safety under contingencies while avoiding overly conservative behavior
- Critical for SAE Level 4+ AVs (no human fallback)

---

## 2. Unified Framework: Logic-Conditioned Hybrid Control

### Hybrid System Model
```
x_{k+1} = f_{σ_k}(x_k, u_k)        # Continuous dynamics
σ_{k+1} = T_k(x_k, σ_k, v_k)       # Discrete mode transition (contingency)
y_k = h_{σ_k}(x_k)                  # Observation model
```

Where:
- **σ_k ∈ Σ** = logical mode (e.g., nominal → fault, clear → blocked)
- **v_k** = stochastic contingency realization (binary sensor fault, categorical agent intent)

### How Modes Affect System
1. **Dynamics** (f_σ): e.g., tire blowout changes vehicle response
2. **Safety spec** (X_safe,σ): e.g., occlusion shrinks safe set
3. **Observation** (h_σ): e.g., sensor fault degrades localization

---

## 3. Two Core Paradigms

### Paradigm 1: Reactive Safety
**Assumption:** Logical uncertainty is RESOLVED (contingency detected/occurred)

```
u_k = π^ℓ(x_k, π^nom(I_k), σ_k)
```
- Nominal controller + supervisory filter
- Maintains state in **control-invariant set** X_inv,σ ⊆ X_safe,σ

**Key Methods:**
- Control Barrier Functions (CBF)
- Hamilton-Jacobi (HJ) Reachability
- Fail-safe supervision (Minimal Risk Condition)

### Paradigm 2: Proactive Safety  
**Assumption:** Logical uncertainty is UNRESOLVED

- Constructs **scenario tree** of plausible mode sequences
- Optimizes over all branches with **non-anticipativity constraint**
- Maintains "common nominal trunk" + branching decisions

**Key Methods:**
- Contingency MPC
- Game-theoretic planning
- Learning-based approaches

---

## 4. Detailed Comparison

### Reactive vs Proactive

| Aspect | Reactive | Proactive |
|--------|----------|-----------|
| **When triggers** | After contingency occurs | Before contingency, anticipates |
| **Computational cost** | Low (online filtering) | High (scenario tree optimization) |
| **Conservatism** | High (must react to worst-case) | Lower (adapts to actual scenario) |
| **Safety guarantee** | Strong (invariant sets) | Depends on scenario coverage |
| **Example** | CBF-based collision avoidance | Branching MPC at intersections |

### Key Insight from Paper
> "Internal faults (system failures) → Reactive mechanisms; External interactions (other agents) → Proactive strategies"

This is because:
- Internal faults are **deterministic once detected** → reactive is sufficient
- External interactions are **inherently uncertain** → need proactive branching

---

## 5. Taxonomy: External vs Internal Contingencies

### Internal Contingencies (System Faults)
- Actuator malfunction
- Sensor outages (camera, lidar, GPS)
- Software failures
- Loss of communication

**Typical Approach:** Reactive fail-safe → Minimal Risk Condition (MRC)

### External Contingencies (Environment + Interaction)
**Environmental:**
- Occlusions (hidden pedestrians)
- Adverse weather
- Road infrastructure changes

**Interactive:**
- Aggressive cut-in
- Unexpected pedestrian behavior
- "Deception games" (human hidden intent)

**Typical Approach:** Proactive branching → adapt to resolved uncertainty

---

## 6. Key Methods Deep Dive

### 6.1 Control Barrier Functions (CBF)

**Formulation:**
```
u_safe = argmin_{u∈U} ||u - π_nom(x)||²
         s.t.  ḣ_σ(x,u) ≥ -α(h_σ(x))
```

- **Pros:** Interpretable, low computational cost, formal guarantees
- **Cons:** Conservative safe sets, manual design complexity
- **Paper Citation:** ames2019control

**Variants:**
- **FT-CBF:** Fault-tolerant for actuator failures
- **ATOM-CBF:** Adaptive based on epistemic uncertainty

### 6.2 Hamilton-Jacobi Reachability

- Solves HJ PDE for maximal control-invariant set
- **Pros:** Least conservative, maximal safe set
- **Cons:** Curse of dimensionality (intractable for >5D)
- **Paper Citation:** bansal2017hamilton, fisac2019general

**Variant:** Control Barrier-Value Function (CBVF) - less conservative

### 6.3 Fail-Safe Supervision

**Objective:** Achieve Minimal Risk Condition (MRC)
- Safely stop on road shoulder
- Terminal safe set X_MRC ⊂ X_safe

**Methods:**
- Online invariant set approximation (Pek2018, Pek2021)
- Safe stochastic MPC
- Formal methods (LTL synthesis)

### 6.4 Branching Model Predictive Control

**Formulation:** Risk-Constrained Branching Decision Process
- Constructs scenario tree S_tree
- Non-anticipativity: u_k,s = u_k,s' for k < τ(s,s')
- Hard safety constraints for ALL scenarios

**Example:** Highway cut-in → optimize "cut-in" and "no cut-in" branches

### 6.5 Game-Theoretic & Learning-Based

- Treats multi-agent interaction as game
- Robust to adversarial agents
- Learning-based safety certificates (Neural CBFs, DeepReach)

---

## 7. Synergy: Reactive + Proactive

The paper emphasizes these paradigms are **complementary**, not competing:

### Proactive → Reactive
- Proactive tools (HJ reachability) compute invariant sets that Reactive filter relies on

### Reactive → Proactive  
- Reactive invariant sets as **terminal constraints** in Proactive optimization
- Guarantees recursive feasibility after horizon N

**Hybrid Architecture (Paper's Recommendation):**
```
Proactive planning (long-horizon) 
    ↓
Terminal anchor in invariant set
    ↓
Reactive filter (short-horizon safety)
```

---

## 8. Critical Challenges Identified

### 8.1 Dynamic Branching
- **Heuristic:** Hand-crafted scenario selection
- **Reachability-based:** Exhaustive but expensive
- **Deferred optimization:** Wait until uncertainty resolves

### 8.2 Verification Gap
| Hazard Type | Typical Approach | Guarantee Level |
|-------------|-----------------|-----------------|
| Physical (collision) | Formal methods | Strong |
| Semantic (intent) | Empirical validation | Weak |
| OOD anomalies | Detection + fallback | Very weak |

### 8.3 Practical Challenges
- Perception uncertainty composition
- Real-time hardware constraints
- Model mismatch between shield and actual controller

---

## 9. Comparison with Our Prior Implementation

Our implementation (`feature/contingency-planning-impl`) aligns with the **Reactive** paradigm:

| Our Implementation | Paper Equivalent |
|-------------------|------------------|
| ContingencyNetwork | Proactive branching (partial) |
| BehaviorTree | Reactive supervisory logic |
| GracefulDegradationController | Task-preserving filter |
| FailureDetector | Contingency detection |

### What's Missing (from paper)
1. **HJ Reachability** for maximal invariant sets
2. **Formal verification** of safety certificates
3. **Minimal Risk Condition (MRC)** explicit fail-safe
4. **Game-theoretic** multi-agent reasoning
5. **Hybrid architecture** with terminal invariant constraints

---

## 10. Action Items

### Short-term
- [ ] Add MRC (Minimal Risk Condition) fallback to behavior tree
- [ ] Integrate HJ reachability for better invariant set computation
- [ ] Add formal verification tests

### Medium-term  
- [ ] Implement branching MPC for proactive planning
- [ ] Add game-theoretic agent modeling
- [ ] Build hybrid architecture with terminal constraints

### Research Directions
- [ ] Physics-data fusion for safety certificates
- [ ] Standardized benchmarks for contingency evaluation
- [ ] OOD detection + graceful degradation

---

## 11. Key Papers Cited

| Citation | Topic |
|----------|-------|
| alsterda2021contingency | Contingency planning definition |
| li2023marc | MARC framework |
| ames2019control | Control Barrier Functions |
| bansal2017hamilton | HJ Reachability |
| fisac2019general | General HJ framework |
| choi2021robust | Control Barrier-Value Function |
| bansal2021deepreach | DeepReach (Neural HJ) |
| dawson2023safe | Safe learning survey |
| wabersich2023data | Data-driven safety |
| liu2016enabling | Intention tracking + CBF |
| leung2020infusing | HJ for traffic weaving |
| hu2023deception | Deception games |
| yun2025atom | ATOM-CBF (adaptive) |
| Pek2018SafeStates | Online invariant sets |
| stolte2021taxonomy | MRC taxonomy |

---

## 13. Open Source Implementations

### 1. Control-Tree Optimization (Primary Reference)

**Paper:** Phiquepal & Toussaint, "Control-Tree Optimization: an approach to MPC under discrete Partial Observability", ICRA 2021
- **arXiv:** https://arxiv.org/abs/2302.00116
- **Original Code:** https://github.com/ControlTrees/icra2021
- **Application Code:** https://github.com/PuYuuu/dive-into-contingency-planning

**Key Concepts:**
- **Control-Tree**: Tree structure where each branch assumes a different discrete state hypothesis
- **Shared Trunk**: Common trajectory prefix until uncertainty resolves
- **Delayed Decision**: Branching point when discrete variables become observable
- **Belief-Aware**: Optimizes using probability distribution over discrete states

**Algorithm:**
```
1. Construct tree: root → shared trunk → branching at observation points
2. Each branch: assumes specific hypothesis (e.g., "pedestrian crosses" vs "yields")
3. Optimize: QP solver (OSQP) for each branch in parallel
4. Execute: Follow shared trunk until observation resolves, then commit to branch
```

**Key Features:**
- Guarantees constraint satisfaction for ALL hypotheses
- Balances risk (optimizes for likely states) vs robustness (safe for unlikely)
- Parallel optimization for scalability
- Applied to: adaptive cruise control, pedestrian crossing, obstacle avoidance

**Configuration (from dive-into-contingency-planning):**
```yaml
planning:
  steps_per_phase: 4    # Phases before observation
  n_branches: 6         # Number of contingencies
  desired_speed: 13.89  # 50 km/h
  u_max: 2.0            # Max acceleration
  u_min: -6.0           # Min acceleration (braking)
  solver_type: "osqp"    # QP solver
```

### 2. Other Related Resources

| Resource | Type | Description |
|----------|------|-------------|
| ControlTrees/solver | Standalone | ROS-free solver (fewer dependencies) |
| pycbf | Python | Control Barrier Functions library |
| drake | C++ | MIT robotics toolbox with CBF/JBJ reachability |
| reachability-analyzer | Python | UCLA ACT lab HJ reachability |

### 3. How Control-Tree Relates to Paper Framework

| Control-Tree Concept | Paper Paradigm | Description |
|---------------------|----------------|-------------|
| Branch over hypotheses | Proactive | Anticipates multiple futures |
| Shared trunk | Common nominal | Reduces conservatism |
| Delayed decision | Deferred optimization | Wait until observation |
| Belief-weighted cost | Risk-constrained | Optimize for likely states |

This is exactly the **Proactive** paradigm from the Zheng et al. survey!
## 12. Summary

This survey establishes contingency planning as a formal discipline with:
1. **Rigorous mathematical foundation** (stochastic hybrid systems)
2. **Two complementary paradigms** (Reactive + Proactive)
3. **Clear taxonomy** (Internal faults vs External interactions)
4. **Identified gaps** (semantic/OOD verification, hybrid architectures)

The field is moving toward **hybrid architectures** that combine reactive safety filters with proactive planning — exactly what our implementation should evolve toward.

---

## 13. Iterative Planner Comparison (2026-03-01 Update)

### Comprehensive Comparison Table

| Approach | What the "Tree" Expands | Strengths | Weaknesses / Failure Modes | Best Fit Scenarios | Typical Stack Pattern |
|----------|-------------------------|-----------|---------------------------|-------------------|----------------------|
| **Lattice DP / Graph Search** | Discrete trajectory samples in s-l-t (or x-y-t) | Very reliable, deterministic, easy to debug; great pruning; strong for comfort if sampling is good | Can miss solutions if lattice is too coarse; struggles with interactive negotiation unless you add modes | Highway, structured roads, lane changes, merges with clear rules | Lattice for proposal → continuous smoother/QP |
| **Semantic / Behavior Tree + Rollout** | High-level maneuvers (keep, change left, yield, creep, commit…) then rollout | Human-interpretable decisions; stable; low branching; easy to add "abort/creep" | Needs careful design to avoid deadlocks/hesitation; limited optimality | Urban negotiation (unprotected turns, merges), policy-heavy domains | Maneuver tree → trajectory optimizer per maneuver |
| **MCTS (UCT / PUCT)** | Actions over time (often maneuver + control) | Anytime; can handle interaction/multi-agent uncertainty well; naturally explores alternatives | Hard to make deterministic; compute heavy; needs good priors + value estimate; can be noisy | Dense interactive scenes, merges, cut-ins, nudging, ambiguous right-of-way | MCTS over maneuvers + short-horizon controls; learned value/priors help a lot |
| **Beam Search over Modes** | Mode sequences (lane/gap/speed profile) | Simple, fast, very "production-friendly"; good tradeoff of exploration vs stability | Can get stuck if K small or scoring biased; needs good pruning & diversity | L2++ highway + ramp merges; cases where you want "top-K" options fast | Enumerate mode sequences → optimize each → pick best |
| **Scenario Tree / Multi-Hypothesis** | Branches on prediction hypotheses (agent intentions) | Robust to prediction uncertainty; explicit risk handling | Explosion in branches; requires strong pruning / risk aggregation | VRU-heavy urban, uncertain oncoming vehicles | Candidate trajs × prediction modes → risk aggregation |
| **RRT / Kinodynamic Sampling** | Continuous state samples (randomized) | Great for complex geometry/off-road; finds feasible paths in clutter | Usually too random/noisy for on-road comfort; hard to guarantee smoothness | Low-speed parking, pull-over, tight maneuvers | RRT* / kinodynamic → smoothing |
| **MPPI / CEM Sampling** | Many sampled control sequences (iterative, not a classic tree) | Works with nonlinear dynamics; good anytime; easy GPU scaling | Can be jittery; needs good cost shaping; safety constraints tricky | High-speed control-ish problems, aggressive avoidance, race-like | Sample controls → weighted average → safety filter |

### Key Insights

1. **Production Reality**: Lattice DP + Behavior Tree combinations dominate actual deployments
2. **Interactive Scenes**: MCTS or Scenario Tree approaches are needed for multi-agent uncertainty
3. **Comfort vs Capability**: RRT/MPPI are great for capability but need smoothing for comfort
4. **Hybrid is Common**: Most production systems combine approaches (e.g., lattice for proposal → QP smooth)

---

## 14. Production Planner Implementation Plan (2026-03-01)

### Core Loop Targets
- **Planner rate**: 20 Hz
- **Budget**: ≤ 50 ms wall time
- **Output**: 1 committed trajectory + MRM/fallback + top-K debug

### Suggested Parameters (Starter Configuration)

#### Corridor Hypotheses
- **N = 4** (min 2, max 6)
  - 1 map-aligned nominal
  - 1-2 perception-shifted (cones/barrels)
  - 1 conservative "tight boundary + lower speed"
- **Corridor horizon**: 120-200 m (or 8-10 s, whichever smaller)

#### Candidate Count
- **K = 128** total candidates per cycle
- **Per corridor**: Kc = 32
- **Modes per corridor**:
  - lane keep / follow lead (10)
  - lane change/merge left/right variants (10)
  - merge-early/merge-late around taper (8)
  - yield / abort variants (4)

#### Rollout Discretization
- **Horizon**: 6.0 s
- **Coarse eval**: T = 60 steps (Δt = 0.1 s) for all K candidates
- **Fine check**: top Kf = 12 candidates at Δt = 0.05 s (T = 120)

#### Prediction Hypotheses
- **Option A (Actor-wise)**: A = 16 relevant actors, Mi = 3 modes each
- **Option B (Joint samples, recommended)**: M = 16 joint scenarios per cycle
- **Total evaluations**: K × M × T = 128 × 16 × 60 = 122,880 state-steps

#### Compute Budget Split (50 ms)
| Stage | Time | Notes |
|-------|------|-------|
| CPU (corridors + candidates) | 5-10 ms | Corridor manager + candidate generator |
| GPU coarse eval | 15-25 ms | All K candidates |
| GPU fine check | 5-10 ms | Top Kf candidates |
| CPU selection + arbitration | 5-10 ms | Final selection + smoothing |

### Module Breakdown

1. **Corridor Manager** (CPU, deterministic)
2. **Candidate Generator** (CPU)
3. **Trajectory Synthesizer** (GPU)
4. **Evaluator** (GPU, workhorse)
5. **Selector + Arbitration** (CPU)
6. **Finalizer / Smoother** (CPU/GPU)
7. **Safety Supervisor** (separate, production-critical)

### Minimal Prototype Timeline
| Week | Tasks |
|------|-------|
| 1-2 | Corridor manager, K=64, CPU eval |
| 3-4 | SDF boundaries + top-K logging |
| 5-8 | GPU port, N=4, K=128, CVaR risk |

### Concrete Configuration Defaults
```yaml
planner:
  rate_hz: 20
  N_corridors: 4
  K_candidates: 128
  T_coarse: 60
  dt_coarse: 0.1
  K_fine: 12
  M_scenarios: 16
  risk_metric: "CVaR(alpha=0.2)"
```

---

## 15. What We've Implemented vs What's Needed

### Already Implemented
| Component | Location |
|-----------|----------|
| Contingency survey | `docs/surveys/2026-02-27-contingency-planning-arxiv.md` |
| Tree-based planner | `contingency_planning/planning/tree/` |
| Model-based planner | `contingency_planning/planning/models/` |
| Simulation benchmark | `contingency_planning/simulation/` |
| Visualizations | `out/contingency_animation/` |

### Not Yet Implemented (Future Work)
| Component | Priority |
|-----------|----------|
| Lattice DP baseline | Medium |
| Behavior tree + rollout | High |
| MCTS for multi-agent | High |
| SDF collision (GPU) | High |
| CVaR risk aggregation | High |
| Corridor manager | High |
| MRM/fallback system | Critical |
| Safety Supervisor | Critical |

---

## 16. Updated Action Items

### Short-Term
- [ ] Implement Lattice DP baseline
- [ ] Add Behavior Tree + Rollout approach
- [ ] Implement corridor manager module
- [ ] Add SDF collision checking

### Medium-Term
- [ ] Implement MCTS for interactive scenarios
- [ ] Add beam search top-K
- [ ] GPU-accelerated evaluation
- [ ] CVaR risk aggregation
- [ ] Full MRM/fallback system
- [ ] Safety Supervisor module

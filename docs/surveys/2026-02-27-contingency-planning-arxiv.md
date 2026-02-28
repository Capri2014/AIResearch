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

## 12. Summary

This survey establishes contingency planning as a formal discipline with:
1. **Rigorous mathematical foundation** (stochastic hybrid systems)
2. **Two complementary paradigms** (Reactive + Proactive)
3. **Clear taxonomy** (Internal faults vs External interactions)
4. **Identified gaps** (semantic/OOD verification, hybrid architectures)

The field is moving toward **hybrid architectures** that combine reactive safety filters with proactive planning — exactly what our implementation should evolve toward.

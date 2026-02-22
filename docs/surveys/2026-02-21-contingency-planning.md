# Contingency Planning in Autonomous Driving Survey

**Date:** 2026-02-21  
**Surveyed by:** Agent (pipeline)  
**Source:** WeChat article (https://mp.weixin.qq.com/s/IHUT_ry_VMn8IoZmKt53TA) - title: "搞定了99%的自动驾驶，为何卡在最后1%？——应急规划Contingency Planning解析"

## TL;DR

The "last 1%" problem in autonomous driving refers to long-tail edge cases that are hard to cover with pure perception-planning-control pipelines. **Contingency Planning** is emerging as a key solution - maintaining fallback behaviors when the primary policy fails.

## Key Insights

### 1. The Last 1% Problem
- AV systems can handle 99% of driving scenarios
- The remaining 1% consists of rare, complex edge cases:
  - Emergency vehicles
  - Accident scenes
  - Unusual road markings
  - Construction zones
  - Adverse weather with sensor degradation
  - Unexpected pedestrian behavior

### 2. What is Contingency Planning?
- **Definition:** Maintain multiple contingency plans (fallback behaviors) that can be activated when the primary plan becomes invalid
- **Key difference from traditional planning:** Not just "what's the best path" but "what if the best path fails"

### 3. Approaches

#### A. Behavior Trees
- Hierarchical state machines for fail-safe behaviors
- Each node has success/failure conditions
- Easy to verify and debug

#### B. Contingency Networks
- Learn multiple futures/plans simultaneously
- Select best contingency based on runtime conditions
- Example: "if lane change fails, abort to original lane"

#### C. Graceful Degradation
- Reduce capabilities safely when degradation detected
- Example: "if vision degraded, switch to map-based routing"

### 4. Key Papers & Concepts
- **Contingency-aware Planning**: Maintain backup trajectories
- **Safe Reinforcement Learning**: Learn fail-safe policies
- **Monte Carlo Tree Search (MCTS)**: Explore contingency outcomes
- **Failsafe Systems**: Hardware-level safety guarantees

## Architecture Pattern

```
Primary Policy → [Monitor Conditions] → 
  ├── Normal: Execute primary plan
  ├── Degraded: Switch to conservative fallback
  └── Failed: Emergency contingency (minimal risk condition)
```

## Relevance to Our Work

### Potential Applications
1. **Waypoint BC with contingency** - predict fallback waypoints
2. **RL refinement for safety** - reward for safe fallback behaviors
3. **Graceful degradation** - handle sensor failures gracefully

### Implementation Considerations
- Need to define "failure modes" explicitly
- Trade-off: conservative vs. efficient driving
- Evaluation: need edge case datasets

## Action Items

- [ ] Survey contingency planning papers (2023-2026)
- [ ] Explore behavior tree implementations
- [ ] Consider: adding fallback waypoint prediction to our pipeline
- [ ] Study: how other AV companies handle the "last 1%"

## Related Reading

- Behavior Trees for Robotics (Ianov et al.)
- Contingency Planning in Robotics (Kress-Gazit et al.)
- Failsafe Systems for Autonomous Vehicles

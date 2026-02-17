# ResAD & GoalFlow Survey: Normalized Residual & Goal-Driven Flow Matching

**Date:** 2026-02-17  
**Focus:** Two latest Horizon/HUST papers on E2E driving trajectory generation  
**Papers:**
- ResAD: Normalized Residual Trajectory Modeling (Oct 2025)
- GoalFlow: Goal-Driven Flow Matching (Mar 2025, updated Oct 2025)

---

## TL;DR

Both papers address **multimodal trajectory generation** with different approaches:

| Paper | Key Innovation | Performance |
|-------|---------------|-------------|
| **ResAD** | Normalized residual + inertial reference | 88.8 PDMS (NAVSIM v1) |
| **GoalFlow** | Goal-driven flow matching + scoring | 90.3 PDMS (NAVSIM v1) |

**Key Insight:** Both aim to solve "trajectory divergence" and "uncertainty in long horizons" problems.

---

## ResAD: Normalized Residual Trajectory Modeling

**Paper:** https://arxiv.org/abs/2510.08562  
**Authors:** Zhiyu Zheng, Shaoyu Chen, Haoran Yin, et al. (HUST + Horizon)  
**Date:** Oct 2025 (v1), Nov 2025 (v2)  
**Code:** Coming soon

### Problem

```
End-to-end AD faces spatio-temporal imbalance:
├── Short-term: Certain, safety-critical
├── Long-term: Uncertain, diverse outcomes
└── Result: Optimization prioritizes distant (uncertain) predictions
```

### Solution: Residual + Normalization

```
┌─────────────────────────────────────────────────────────────────┐
│                    ResAD Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Sensor Data (camera, LiDAR)                              │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │   Perception Encoder    │                                    │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Inertial Reference      │  ← Physical prior (straight line) │
│  │ (deterministic)         │                                    │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Residual Predictor      │  ← Deviation from inertial path   │
│  │ + Point-wise Norm       │  ← Re-weight optimization        │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │   Final Trajectory      │  = Inertial + Residual            │
│  └─────────────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Inertial Reference

**What:** A deterministic "default path" based on physics
- Assumes vehicle continues straight at current velocity
- Strong physical prior

**Why:** 
- Compels model to learn context-driven deviations
- Avoids spurious correlations

#### 2. Normalized Residual Prediction

```
Traditional: Predict trajectory directly
             ↓
             Problem: Optimization imbalance

ResAD: Predict residual (deviation) + normalize
             ↓
             Solution: Re-weight by uncertainty
```

#### 3. Point-wise Normalization

**Technique:** Weight residuals by horizon uncertainty
- Short-term: Higher weight (certain, safety-critical)
- Long-term: Lower weight (uncertain)

### Results

| Benchmark | ResAD | Previous Best |
|-----------|-------|---------------|
| NAVSIM v1 PDMS | **88.8** | - |
| NAVSIM v2 EPDMS | **85.5** | - |
| Denoising steps | **2** | 10+ (DiffusionDrive) |

### Why It Matters for Us

```python
# We can apply ResAD to our trajectory prediction:

class ResADWaypointPredictor(nn.Module):
    """
    ResAD-style residual trajectory prediction.
    
    Instead of predicting waypoints directly,
    predict deviation from inertial reference.
    """
    def __init__(self, config):
        self.encoder = BEVEncoder()
        
        # Inertial reference: straight line at current velocity
        self.inertial_ref = InertialReference(
            current_velocity,
            horizon=config.horizon,
        )
        
        # Residual predictor with normalization
        self.residual_predictor = ResidualPredictor(
            embed_dim=config.embed_dim,
            horizon=config.horizon,
        )
        
        # Point-wise normalization for optimization balance
        self.pointwise_norm = PointWiseNormalization()
    
    def forward(self, bev, velocity):
        # Encode
        z = self.encoder(bev)
        
        # Inertial reference (physical prior)
        inertial = self.inertial_ref(velocity)  # [B, T, 3]
        
        # Predict residual
        residual = self.residual_predictor(z)  # [B, T, 3]
        
        # Normalize by uncertainty
        normalized_residual = self.pointwise_norm(residual)
        
        # Final: inertial + residual
        trajectory = inertial + normalized_residual
        
        return trajectory
```

---

## GoalFlow: Goal-Driven Flow Matching

**Paper:** https://arxiv.org/abs/2503.05689  
**Authors:** Zebin Xing, Xingyu Zhang, Yang Hu, et al. (HUST + Horizon)  
**Date:** Mar 2025 (v1), Oct 2025 (v6)  
**Code:** https://github.com/YvanYin/GoalFlow

### Problem

```
Multimodal trajectory generation challenges:
├── High trajectory divergence
├── Inconsistencies between guidance and scene
├── Trajectory selection complexity
└── Quality degradation in diffusion methods
```

### Solution: Goal-Driven Flow Matching

```
┌─────────────────────────────────────────────────────────────────┐
│                    GoalFlow Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Sensor Data + Goal Candidates                             │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │   Perception Encoder    │                                    │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Goal Scoring Network   │  ← Select best goal point          │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Goal-Conditioned       │  ← Constrain generation            │
│  │ Flow Matching          │                                    │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Trajectory Scoring      │  ← Select best trajectory          │
│  └─────────────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Goal Point Selection

```
Problem: Which goal point to aim for?
Solution: Learn to score goal candidates

Goal Selection Network:
├── Input: Scene features + goal candidates
├── Output: Confidence score per goal
└── Selection: argmax or weighted sum
```

#### 2. Flow Matching

**What:** Efficient generative method (alternative to diffusion)
- Simpler than diffusion
- Faster convergence
- Single denoising step possible

**Flow Matching vs Diffusion:**
| Aspect | Flow Matching | Diffusion |
|--------|---------------|----------|
| Steps | 1-2 | 10-1000 |
| Speed | Fast | Slow |
| Quality | Excellent | Excellent |

#### 3. Trajectory Scoring

```
After generation, score each trajectory:
├── Consistency with goal
├── Safety score
├── Comfort metrics
└── Select best
```

### Results

| Benchmark | GoalFlow | Previous Best |
|-----------|----------|---------------|
| NAVSIM v1 PDMS | **90.3** | 88.8 (ResAD) |
| Denoising steps | **1** | 2+ |

### Why It Matters for Us

```python
# We can apply GoalFlow to our planning:

class GoalFlowPlanner(nn.Module):
    """
    GoalFlow-style multimodal trajectory planner.
    
    Key: Goal-driven generation + flow matching.
    """
    def __init__(self, config):
        self.encoder = BEVEncoder()
        
        # Goal candidates from HD map
        self.goal_selector = GoalScoringNetwork(
            embed_dim=config.embed_dim,
        )
        
        # Flow matching for trajectory generation
        self.flow_matcher = FlowMatching(
            embed_dim=config.embed_dim,
            trajectory_dim=config.trajectory_dim,
        )
        
        # Score and select best trajectory
        self.trajectory_scorer = TrajectoryScoringNetwork(
            embed_dim=config.embed_dim,
        )
    
    def forward(self, bev, goal_candidates):
        # Encode scene
        z = self.encoder(bev)
        
        # Select best goal
        goal_scores = self.goal_selector(z, goal_candidates)
        selected_goal = goal_candidates[goal_scores.argmax()]
        
        # Generate trajectory conditioned on goal
        trajectory = self.flow_matcher(z, selected_goal)
        
        # Score trajectory
        trajectory_score = self.trajectory_scorer(z, trajectory)
        
        return {
            'trajectory': trajectory,
            'goal': selected_goal,
            'score': trajectory_score,
        }
```

---

## Comparison: ResAD vs GoalFlow

| Aspect | ResAD | GoalFlow |
|--------|-------|----------|
| **Core Idea** | Residual from inertial | Goal-driven generation |
| **Generation** | Normalized residual | Flow matching |
| **Multimodal** | Implicit (residual diversity) | Explicit (goal selection) |
| **Steps** | 2 | 1 |
| **NAVSIM PDMS** | 88.8 | 90.3 |
| **Strength** | Physical prior | Goal conditioning |

---

## Combined Approach: ResAD + GoalFlow

Can we combine both?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Combined Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Sensor Data + Goal Candidates                            │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │   Perception Encoder    │                                    │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Goal Scoring Network   │                                    │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Inertial Reference      │  ← ResAD component                 │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Residual Flow Matching  │  ← Combined!                       │
│  │ (+ Goal Conditioning)  │                                    │
│  └─────────────────────────┘                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────┐                                    │
│  │ Trajectory Scoring      │                                    │
│  └─────────────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Application to Our Pipeline

### 1. ResAD for Waypoint Prediction

Current: Direct regression
Proposed: Residual prediction with inertial prior

```python
# Current
trajectory = waypoint_head(z)

# Proposed (ResAD)
inertial = compute_inertial_reference(velocity, horizon)
residual = residual_head(z)
trajectory = inertial + normalized(residual)
```

### 2. GoalFlow for Multimodal Planning

Current: Single trajectory
Proposed: Multimodal with goal selection

```python
# Current
trajectory = waypoint_head(z)

# Proposed (GoalFlow)
goals = sample_goal_candidates(HD_map)
goal_scores = goal_selector(z, goals)
trajectory = flow_matcher(z, goals[goal_scores.argmax()])
```

### 3. Combined: ResAD + GoalFlow

```python
class CombinedPlanner(nn.Module):
    """
    Combine ResAD and GoalFlow for robust planning.
    """
    def __init__(self, config):
        self.encoder = BEVEncoder()
        self.goal_selector = GoalScoringNetwork()
        self.inertial_ref = InertialReference()
        self.residual_flow = ResidualFlowMatching()
        self.scorer = TrajectoryScoringNetwork()
    
    def forward(self, bev, velocity, goal_candidates):
        z = self.encoder(bev)
        
        # Goal selection
        goal_score = self.goal_selector(z, goal_candidates)
        goal = goal_candidates[goal_score.argmax()]
        
        # Inertial reference
        inertial = self.inertial_ref(velocity)
        
        # Residual flow matching conditioned on goal
        residual = self.residual_flow(z, goal)
        
        # Final trajectory
        trajectory = inertial + residual
        
        # Score
        score = self.scorer(z, trajectory)
        
        return trajectory, score
```

---

## Questions for Implementation

1. **Data:** Do we have goal candidates for planning?
2. **Velocity:** Can we extract inertial reference from state?
3. **Evaluation:** How to evaluate multimodal on our test set?
4. **Speed:** Flow matching vs diffusion for real-time?

---

## References

- ResAD: https://arxiv.org/abs/2510.08562
- GoalFlow: https://arxiv.org/abs/2503.05689
- Code: https://github.com/YvanYin/GoalFlow

---

## Related Papers in Our Repo

- `docs/surveys/2026-02-16-horizon-robotics.md` - Full Horizon Robotics survey
- `docs/surveys/2026-02-16-vla-world-model-2025-survey.md` - VLA+World Model survey

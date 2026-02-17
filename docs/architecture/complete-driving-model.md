# Complete Driving Model Architecture & Feature Requirements

**Date:** 2026-02-16  
**Purpose:** Identify gaps and requirements for a production autonomous driving model

---

## TL;DR

**Current State vs Production Requirements:**

| Component | Current | Needed | Priority |
|-----------|----------|---------|----------|
| Perception | Basic | Full stack | High |
| Planning | Waypoints | Trajectory optimization | High |
| Control | Simple | MPC + safety | Medium |
| World Model | None | Required | High |
| VLA Integration | None | Required | High |
| Evaluation | Basic | Comprehensive | Medium |
| Safety | None | Required | Critical |

---

## Complete Driving Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Production Autonomous Driving Stack                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                   Perception Stack                          │       │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │       │
│  │  │ Bird's-Eye │ │ Object     │ │ Lane       │         │       │
│  │  │ View (BEV) │ │ Detection  │ │ Detection  │         │       │
│  │  └─────────────┘ └─────────────┘ └─────────────┘         │       │
│  │       │               │              │                    │       │
│  │       └───────────────┴──────────────┘                    │       │
│  │                       │                                   │       │
│  │              ┌─────────┴─────────┐                       │       │
│  │              │   Sensor Fusion  │                       │       │
│  │              │   (Camera+LiDAR)│                       │       │
│  │              └─────────┬─────────┘                       │       │
│  └───────────────────────┼─────────────────────────────────┘       │
│                          │                                            │
│                          ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                 World Model                                   │  │
│  │  ┌─────────────────────────────────────────────────────┐   │  │
│  │  │  Video Prediction + Action-Conditioned Future      │   │  │
│  │  │  "Given current state, predict future scenes"     │   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  │                        │                                    │  │
│  │                        ▼                                    │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Scenario Prediction                                 │  │  │
│  │  │  • Traffic prediction                               │  │  │
│  │  │  • Risk assessment                                  │  │  │
│  │  │  • Behavior prediction                              │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────┬───────────────────────────────┘  │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 Planning Stack                            │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Behavior Planner                                 │   │  │
│  │  │  • Lane change left/right                        │   │  │
│  │  │  • Follow lane                                   │   │  │
│  │  │  • Stop at intersection                          │   │  │
│  │  │  • Emergency brake                               │   │  │
│  │  └───────────────────────┬──────────────────────────┘   │  │
│  │                          │                                │  │
│  │                          ▼                                │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Trajectory Optimizer                           │   │  │
│  │  │  • Generate multiple trajectories               │   │  │
│  │  │  • Optimize for comfort + safety + efficiency  │   │  │
│  │  │  • Handle constraints                          │   │  │
│  │  │  • MPC (Model Predictive Control)             │   │  │
│  │  └───────────────────────┬──────────────────────────┘   │  │
│  │                          │                                │  │
│  │                          ▼                                │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Trajectory Selector                            │   │  │
│  │  │  • Safety check                                │   │  │
│  │  │  • Rule-based override                        │   │  │
│  │  │  • Fallback behavior                         │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────┬───────────────────────────────┘  │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 Control Stack                            │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  Low-Level Controller                           │   │  │
│  │  │  • PID controller                              │   │  │
│  │  │  • Feedforward + feedback                      │   │  │
│  │  │  • Safety limits                              │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                            │                                      │
│                            ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 Safety & Monitoring                     │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │  │ Collision   │ │ Monitoring  │ │ Fallback   │     │  │
│  │  │ Avoidance   │ │             │ │ Manager    │     │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Missing Components Analysis

### 1. Perception Stack (High Priority)

```
Current:  ❌ Nothing implemented
Needed:
```

| Component | Description | Complexity |
|-----------|-------------|------------|
| **BEV Encoder** | Bird's-eye view from multi-camera | Medium |
| **Object Detection** | 3D bounding box detection | Medium |
| **Lane Detection** | Lane line + boundary detection | Low |
| **Depth Estimation** | Monocular depth or LiDAR | Medium |
| **Sensor Fusion** | Camera + LiDAR + Radar fusion | High |
| **Occlusion Handling** | Predict hidden objects | High |

**Implementation Options:**

```python
# Option 1: BEVFormer-style (Transformer)
class BEVEncoder(nn.Module):
    """
    Bird's-eye view encoder from multi-camera.
    
    Uses transformer to fuse camera features into BEV.
    """
    def __init__(self, config):
        self.camera_backbone = ResNetBackbone()
        self.bev_transformer = BEVFormerTransformer()
        self.neck = FPNNeck()
    
    def forward(self, images):
        # Extract camera features
        camera_features = self.camera_backbone(images)
        
        # Transform to BEV
        bev_features = self.bev_transformer(camera_features)
        
        return self.neck(bev_features)


# Option 2: LiDAR-based (PointPillars)
class LiDAREncoder(nn.Module):
    """
    LiDAR point cloud encoder.
    
    Uses PointPillars or PointPillar-style encoding.
    """
    def __init__(self, config):
        self.pillar_encoder = PointPillarEncoder()
        self.backbone = SECONDBackbone()
        self.neck = AnchorNeck()
    
    def forward(self, points):
        # Encode pillars
        pillars = self.pillar_encoder(points)
        
        # 2D backbone
        features = self.backbone(pillars)
        
        # Detection neck
        detections = self.neck(features)
        
        return detections
```

### 2. World Model (Critical Missing)

```
Current:  ❌ Not implemented
Needed:   Action-conditioned video prediction
```

**Why World Models Matter:**

```
┌─────────────────────────────────────────┐
│         Without World Model               │
├─────────────────────────────────────────┤
│  Input: Current observation             │
│  Output: Action                        │
│                                         │
│  Problem: "What if I brake now?"        │
│  Answer: Don't know until it happens   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│          With World Model                │
├─────────────────────────────────────────┐
│  Input: Current observation + action     │
│  Output: Predicted future observation   │
│                                         │
│  Problem: "What if I brake now?"        │
│  Answer: World model predicts result   │
└─────────────────────────────────────────┘
```

**Implementation:**

```python
class WorldModel(nn.Module):
    """
    Action-conditioned world model.
    
    Predicts future observations given current state + actions.
    """
    def __init__(self, config):
        self.encoder = VAEEncoder()      # Observation → latent
        self dynamics = RSSM()          # Latent dynamics model
        self.decoder = Decoder()         # Latent → observation
        
        # For driving specifically
        self.traffic_predictor = TrafficPredictor()
        self.ego_dynamics = VehicleDynamics()
    
    def forward(self, obs, actions, horizon=10):
        """
        Predict future observations.
        
        Args:
            obs: [B, C, H, W] current observation
            actions: [B, T, action_dim] planned actions
            horizon: Number of steps to predict
            
        Returns:
            predictions: [B, T, C, H, W] predicted future obs
        """
        # Encode current observation
        h = self.encoder(obs)
        z = h  # Latent state
        
        predictions = []
        
        for t in range(horizon):
            # Predict next latent given action
            z = self.dynamics(z, actions[:, t])
            
            # Decode to observation
            pred_obs = self.decoder(z)
            predictions.append(pred_obs)
        
        return torch.stack(predictions, dim=1)
    
    def predict_traffic(self, obs, ego_action, horizon=10):
        """Predict traffic behavior given ego vehicle action."""
        return self.traffic_predictor(obs, ego_action, horizon)
```

### 3. VLA Integration (Critical)

```
Current:  ❌ Not implemented
Needed:   Vision-Language-Action model for interpretable planning
```

**Why VLA Matters:**

```python
class VLADrivingModel(nn.Module):
    """
    Vision-Language-Action model for driving.
    
    Can explain decisions in natural language.
    """
    def __init__(self, config):
        self.vision_encoder = VisionEncoder()      # Images → features
        self.language_encoder = LangEncoder()       # Text → features
        self.fusion = CrossAttention()             # Fuse vision + language
        self.planner = TrajectoryPlanner()         # Generate plan
        self.explainer = ExplainerHead()           # Generate explanations
    
    def forward(self, images, query=None):
        """
        Plan driving action with explanation.
        
        Returns:
            trajectory: Planned trajectory
            explanation: Natural language explanation
        """
        # Encode vision
        visual_features = self.vision_encoder(images)
        
        # Encode language query (optional)
        if query is not None:
            lang_features = self.language_encoder(query)
            features = self.fusion(visual_features, lang_features)
        else:
            features = visual_features
        
        # Plan trajectory
        trajectory = self.planner(features)
        
        # Generate explanation
        explanation = self.explainer(features, trajectory)
        
        return trajectory, explanation


# Usage:
model = VLADrivingModel(config)

# Plan and explain
trajectory, explanation = model(
    images=front_camera,
    query="Plan a safe trajectory considering the pedestrians ahead"
)

print(explanation)
# "I see 2 pedestrians at the crosswalk, one looking at their phone.
# The light is green but I should slow down to yield..."
```

### 4. Trajectory Optimization (High Priority)

```
Current:  ❌ Only basic waypoint prediction
Needed:   Full trajectory optimization with constraints
```

**Implementation:**

```python
class TrajectoryOptimizer(nn.Module):
    """
    Optimal trajectory generation with constraints.
    
    Uses convex optimization or differentiable MPC.
    """
    def __init__(self, config):
        self.vehicle_model = BicycleModel()
        self.cost_weights = {
            'progress': 1.0,
            'comfort': 0.5,
            'safety': 2.0,
            'efficiency': 0.3,
        }
    
    def optimize(
        self,
        initial_state,
        goal_state,
        obstacles,
        road_boundaries,
        horizon=20,
        dt=0.1,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Find optimal trajectory.
        
        Uses trajectory optimization (e.g., Frenet optimal planning).
        """
        # Generate candidate trajectories
        candidates = self._generate_candidates(
            initial_state, goal_state, horizon, dt
        )
        
        # Evaluate costs
        best_trajectory = None
        best_cost = float('inf')
        
        for traj in candidates:
            cost = self._compute_cost(
                traj, initial_state, obstacles, 
                road_boundaries
            )
            
            if cost < best_cost:
                best_cost = cost
                best_trajectory = traj
        
        return best_trajectory, {'cost': best_cost}
    
    def _generate_candidates(
        self,
        initial_state,
        goal_state,
        horizon,
        dt,
    ) -> List[np.ndarray]:
        """Generate candidate trajectories."""
        candidates = []
        
        # Different speed profiles
        for target_speed in [5, 10, 15, 20]:
            traj = self._generate_speed_profile(
                initial_state, target_speed, horizon, dt
            )
            candidates.append(traj)
        
        # Different lateral offsets
        for offset in [-1, 0, 1]:
            traj = self._generate_lateral_offset(
                initial_state, offset, horizon, dt
            )
            candidates.append(traj)
        
        return candidates
    
    def _compute_cost(
        self,
        trajectory,
        initial_state,
        obstacles,
        boundaries,
    ) -> float:
        """Compute trajectory cost."""
        total_cost = 0.0
        
        # Progress cost (how close to goal)
        total_cost += self.cost_weights['progress'] * (
            self._progress_cost(trajectory, initial_state)
        )
        
        # Comfort cost (smoothness)
        total_cost += self.cost_weights['comfort'] * (
            self._comfort_cost(trajectory)
        )
        
        # Safety cost (obstacle distance)
        total_cost += self.cost_weights['safety'] * (
            self._safety_cost(trajectory, obstacles)
        )
        
        # Efficiency cost (speed profile)
        total_cost += self.cost_weights['efficiency'] * (
            self._efficiency_cost(trajectory)
        )
        
        return total_cost
```

### 5. Safety Layer (Critical)

```
Current:  ❌ Not implemented
Needed:   Collision avoidance + fallback behavior
```

**Implementation:**

```python
class SafetyLayer(nn.Module):
    """
    Safety validation and fallback layer.
    
    Ensures all trajectories are safe before execution.
    """
    def __init__(self, config):
        self.collision_checker = CollisionChecker()
        self.lane_validator = LaneValidator()
        self.fallback_planner = FallbackPlanner()
    
    def validate_and_fix(
        self,
        trajectory,
        obstacles,
        road_boundaries,
    ) -> np.ndarray:
        """
        Validate trajectory and fix if unsafe.
        
        Returns:
            Safe trajectory (possibly modified)
        """
        # Check collision
        if self.collision_checker.has_collision(trajectory, obstacles):
            # Emergency maneuver
            trajectory = self._emergency_brake(trajectory)
        
        # Check lane boundaries
        if not self.lane_validator.is_valid(trajectory, road_boundaries):
            # Return to lane center
            trajectory = self._return_to_lane(trajectory, road_boundaries)
        
        # Check feasibility
        if not self._is_feasible(trajectory):
            # Use fallback
            trajectory = self.fallback_planner.plan(
                current_state=trajectory[0],
                obstacles=obstacles,
            )
        
        return trajectory
    
    def _emergency_brake(self, trajectory):
        """Generate emergency braking trajectory."""
        # Maximum deceleration
        max_decel = -8.0  # m/s²
        
        # Create stopping trajectory
        current_speed = self._get_speed(trajectory[0])
        
        stop_trajectory = []
        t = 0
        x = trajectory[0][:2]
        
        while current_speed > 0.1:
            # Update position
            x = x + current_speed * 0.1
            current_speed = max(0, current_speed + max_decel * 0.1)
            
            stop_trajectory.append([x[0], x[1], current_speed])
            t += 0.1
        
        return np.array(stop_trajectory)


class CollisionChecker:
    """Check for potential collisions."""
    
    def __init__(self, config):
        self.ego_radius = config.get('ego_radius', 2.0)  # meters
        self.safe_distance = config.get('safe_distance', 2.0)  # meters
    
    def has_collision(
        self,
        trajectory: np.ndarray,
        obstacles: List[Dict],
    ) -> bool:
        """Check if trajectory collides with any obstacle."""
        for t in range(len(trajectory)):
            ego_pos = trajectory[t][:2]
            
            for obs in obstacles:
                obs_pos = np.array([obs['x'], obs['y']])
                distance = np.linalg.norm(ego_pos - obs_pos)
                
                # Check collision (accounting for sizes)
                min_dist = self.ego_radius + obs.get('radius', 1.0)
                if distance < min_dist:
                    return True
        
        return False
    
    def compute_ttc(
        self,
        ego_trajectory: np.ndarray,
        obstacle_trajectory: np.ndarray,
    ) -> float:
        """
        Compute time-to-collision.
        
        Returns minimum TTC across trajectory.
        """
        min_ttc = float('inf')
        
        for t in range(len(ego_trajectory)):
            ego_pos = ego_trajectory[t][:2]
            obs_pos = obstacle_trajectory[t][:2]
            
            # Relative velocity
            ego_vel = self._estimate_velocity(ego_trajectory, t)
            obs_vel = self._estimate_velocity(obstacle_trajectory, t)
            
            rel_vel = ego_vel - obs_vel
            rel_pos = obs_pos - ego_pos
            
            # Distance to collision point
            dist = np.linalg.norm(rel_pos)
            
            if dist < 1e-6:
                return 0.0  # Already collided
            
            # Project onto relative velocity
            ttc = -np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_vel)**2 + 1e-6)
            
            if ttc > 0 and ttc < min_ttc:
                min_ttc = ttc
        
        return min_ttc
```

### 6. Model Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│              Complete Model Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Perception:                                               │
│  ┌─────────────────────────────────────────────┐           │
│  │  Camera Backbone (ResNet/ViT)           │           │
│  │  LiDAR Backbone (PointPillars/PointNet) │           │
│  │  BEV Fusion (BEVFormer/Lift-Splat)    │           │
│  │  Object Detection (CenterPoint)         │           │
│  │  Lane Detection (HDMapNet)              │           │
│  └─────────────────────────────────────────────┘           │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │  World Model (GAIA-2 style)            │           │
│  │  • Latent dynamics                    │           │
│  │  • Traffic prediction                 │           │
│  │  • Scenario forecasting               │           │
│  └─────────────────────────────────────────────┘           │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │  VLA Planner                            │           │
│  │  • Vision encoder                      │           │
│  │  • Language encoder (optional)          │           │
│  │  • Trajectory head                    │           │
│  │  • Explanation head                   │           │
│  └─────────────────────────────────────────────┘           │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │  Trajectory Optimizer                   │           │
│  │  • Candidate generation                │           │
│  │  • Cost evaluation                     │           │
│  │  • Constraint handling                 │           │
│  └─────────────────────────────────────────────┘           │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │  Safety Layer                           │           │
│  │  • Collision checker                   │           │
│  │  • Fallback planner                    │           │
│  │  • Emergency brake                     │           │
│  └─────────────────────────────────────────────┘           │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │  Control                               │           │
│  │  • PID controller                      │           │
│  │  • Feedforward                        │           │
│  │  • Safety limits                      │           │
│  └─────────────────────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature Gap Summary

| Category | Current | Needed | Effort |
|----------|---------|--------|--------|
| **Perception** | ❌ | BEV + Detection + Fusion | High |
| **World Model** | ❌ | GAIA-2 style | High |
| **VLA Integration** | ❌ | Vision-Language-Action | High |
| **Trajectory Optimization** | Basic | MPC + Convex | Medium |
| **Safety Layer** | ❌ | Collision + Fallback | High |
| **Control** | Basic | PID + Feedforward | Low |
| **Evaluation** | Basic | Comprehensive | Medium |
| **Data Pipeline** | Synthetic | Real + Synthetic | Medium |

---

## Implementation Roadmap

```
Phase 1 (Week 1-2): Foundation
├── Perception stack (BEV encoder)
├── Basic world model
└── Trajectory optimizer

Phase 2 (Week 3-4): Intelligence
├── VLA integration
├── Advanced world model
└── Safety layer

Phase 3 (Week 5-6): Hardening
├── Comprehensive evaluation
├── Safety validation
└── Edge case handling
```

---

## Files Created

- `docs/architecture/complete-driving-model.md` - This document

---

## Questions for Next Steps

1. **Which component should we prioritize?**
2. **Do we have access to perception data (BEV, detection)?**
3. **Should we start with a simpler end-to-end approach?**
4. **What's the evaluation metric priority (safety vs comfort)?**

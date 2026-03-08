# Space-Time Hybrid A* Algorithm for Dynamic Environment Path Planning

**Date:** 2026-03-06  
**Status:** Survey Complete

---

## 1. Introduction

Traditional path planning algorithms (e.g., A*, Dijkstra) only consider spatial dimension with static obstacle avoidance, and cannot handle space-time conflicts introduced by moving obstacles. **Space-Time Hybrid A*** combines Hybrid A*'s kinematic feasibility with space-time reasoning, ensuring paths satisfy vehicle dynamics while avoiding dynamic obstacles through time-dimension expansion.

---

## 2. Algorithm Principles

### 2.1 State Space Modeling

Traditional Hybrid A* uses only spatial coordinates, while Space-Time Hybrid A* introduces time dimension, constructing a 4D state space:

```
State: (x, y, yaw, t)
  ├── Spatial: x, y (vehicle rear axle center), yaw (heading)
  └── Temporal: t (current time, for matching dynamic obstacle positions)
```

**Discretization:**
| Dimension | Resolution | Purpose |
|-----------|------------|---------|
| Spatial | 0.5m | Balance search efficiency and precision |
| Yaw | 15° | Match vehicle steering steps |
| Time | 0.05s | Match obstacle/vehicle update steps |

### 2.2 Kinematic Constraints

Using bicycle model:

```
x' = x + v * cos(yaw) * dt
y' = y + v * sin(yaw) * dt
yaw' = yaw + v / L * tan(δ) * dt
```

Where:
- v: vehicle velocity
- δ: front wheel steering angle
- L: wheelbase

### 2.3 Space-Time Cost Function

```
Total Cost = Path Length + Safety + Smoothness + Time + Reference Line
```

| Cost Term | Formula | Purpose |
|-----------|---------|---------|
| **Path Length** | ∫√(dx²+dy²) | Economic path |
| **Safety** | f(space-time distance) | Avoid dynamic obstacles |
| **Smoothness** | f(steer, Δsteer) | Smooth steering |
| **Time** | f(t - t_max) | Constrain planning time |
| **Reference Line** | f(distance to line) | Guide along lane |

### 2.4 Heuristic Design

```
h(n) = w_h * (spatial_distance + temporal_distance)
```

Key design:
- Combine with dynamic obstacle predicted positions
- Precompute collision-free shortest distance
- Avoid underestimated heuristics causing ineffective search

### 2.5 Dynamic Obstacle Collision Detection

**Obstacle motion model** (constant velocity):

```
obs(t) = obs_0 + v * t * [cos(θ), sin(θ)]
```

**Collision criterion:**
1. Compute obstacle position at each trajectory time point
2. Perform 2D plane collision detection (SAT)
3. If collision, node is invalid

---

## 3. Algorithm Flow

```
1. Environment modeling: Build reference line, boundaries, dynamic obstacles
2. State initialization: Define start S and goal G
3. Node expansion: Generate motion primitives via kinematic model
4. Space-time collision detection: Filter colliding nodes
5. Priority queue: Sort by cost ascending
6. Goal check: spatial distance < tolerance AND heading error < tolerance
7. Path reconstruction: Backtrack parent nodes
```

---

## 4. Related Work Comparison

### 4.1 Path Planning Methods

| Method |特点 |适用场景 |
|--------|------|----------|
| **A*** | Static obstacles, fast | Simple environments |
| **Dijkstra** | Global optimal | Static shortest path |
| **Hybrid A*** | Kinematic constraints | Vehicle motion planning |
| **RRT*** | Probabilistically complete | Complex/high-DOF |
| **Space-Time Hybrid A*** | + temporal dimension | Dynamic obstacle avoidance |
| **Lattice Planner** | State lattice | Structured roads |
| **EM Planner** | Expectation maximization | Multi-modal decisions |

### 4.2 Dynamic Obstacle Handling

| Method | Core Idea | Pros/Cons |
|--------|-----------|------------|
| **Velocity Obstacle (VO)** | Velocity space avoidance | Real-time but simplified |
| **Reciprocal VO (RVO)** | Bidirectional VO | Multi-agent |
| **Games** | Game theory | Elegant but complex |
| **Space-Time A*** | Space-time search | Accurate but expensive |
| **MPDM** | Prediction + planning | Considers prediction uncertainty |

### 4.3 Algorithm Characteristics

| Aspect | Space-Time Hybrid A* |
|--------|---------------------|
| **Advantages** | Accurate dynamic obstacle handling, kinematic feasible, space-time optimization |
| **Challenges** | High computational complexity, state space explosion |
| **Suitable for** | Low-medium speed scenarios (parking, campus), structured roads |

---

## 5. Code Implementation

### 5.1 config.py - Configuration

```python
import numpy as np

class VehicleParam:
    WIDTH = 2.0
    LF = 3.8  # Rear axle to front bumper
    LB = 0.8  # Rear axle to rear bumper
    LENGTH = LF + LB
    WHEELBASE = 3.0
    MAX_STEER = np.deg2rad(30)
    MAX_SPEED = 3.0
    MAX_ACCEL = 2.0

class SpaceTimeHybridAStarConfig:
    XY_RESOLUTION = 0.5  # meters
    YAW_RESOLUTION = np.deg2rad(15.0)
    TIME_RESOLUTION = 0.05  # seconds
    TIME_STEP = 1.0  # seconds
    MAX_PLAN_TIME = 60.0
    SPEED_RESOLUTION = 0.6
    STEER_NUM = 12
    MAX_ITERATIONS = 50000
    GOAL_TOLERANCE_XY = 2.0
    GOAL_TOLERANCE_YAW = np.deg2rad(20.0)
    HEURISTIC_WEIGHT = 5.0
    SAFE_WEIGHT = 5.0
    # ... more parameters
```

### 5.2 kinematic_model.py - Kinematic Model

```python
import math

class KinematicModel:
    def motion_prediction(self, x, y, yaw, velocity, steer, dt=0.5):
        """Bicycle model state prediction"""
        steer = max(-self.max_steer, min(self.max_steer, steer))
        velocity = max(-self.max_speed, min(self.max_speed, velocity))
        
        new_x = x + velocity * math.cos(yaw) * dt
        new_y = y + velocity * math.sin(yaw) * dt
        new_yaw = yaw + velocity / self.wheelbase * math.tan(steer) * dt
        
        return new_x, new_y, new_yaw
```

### 5.3 obstacle.py - Dynamic Obstacles

```python
class Obstacle:
    def __init__(self, center, length=3, width=2.0, theta=0, velocity=0.0):
        self.center = center
        self.velocity = velocity
        self.theta = theta
    
    def get_position(self, time):
        """Get obstacle position at time t"""
        x = self.center[0] + self.velocity * time * math.cos(self.theta)
        y = self.center[1] + self.velocity * time * math.sin(self.theta)
        return x, y
    
    def has_overlap(self, other_x, other_y, other_length, other_width, 
                   other_theta, time=0.0, safe_distance=0.2):
        """Space-time collision detection"""
        cur_x, cur_y = self.get_position(time)
        # SAT-based collision detection
        return has_collision
    
    def get_min_distance(self, other_x, other_y, other_length, other_width,
                       other_theta, time=0.0, safe_distance=0.5):
        """Minimum distance to obstacle"""
        if self.has_overlap(...):
            return 0.0
        return min_distance
```

### 5.4 env.py - Environment

```python
class Env:
    def __init__(self):
        self.ref_line, self.bound1, self.bound2 = self.get_refline_info()
    
    def get_refline_info(self):
        """Build reference line and boundaries"""
        refline, bound1, bound2 = [], [], []
        for i in np.arange(0, 60, 10):
            refline.append((i, 0))
            bound1.append((i, -2.5))
            bound2.append((i, 7.5))
        return refline, bound1, bound2
```

### 5.5 space_time_hybrid_a_star.py - Main Algorithm

```python
class HybridAStarNode:
    def __init__(self, x_ind, y_ind, yaw_ind, time_ind, x_list, y_list, 
                 yaw_list, velocity_list, steer_list, time_list, 
                 parent_index=None, cost=0.0):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.time_index = time_ind
        # ... trajectory and control sequences

class SpaceTimeHybridAStar:
    def plan(self, start_x, start_y, start_yaw, start_velocity,
             goal_x, goal_y, goal_yaw, goal_velocity, 
             obstacles, env, animate=False):
        """
        Main planning:
        1. Initialize start and goal nodes
        2. Iteratively expand nodes (A* search)
        3. Space-time collision detection
        4. Path reconstruction
        """
        start_node = HybridAStarNode(...)
        
        # Priority queue (cost, node_id)
        open_list, closed_list = {}, {}
        pq = []
        heapq.heappush(pq, (start_node.cost, calc_index(start_node)))
        
        while iteration < self.max_iterations:
            # Pop minimum cost node
            # Check goal arrival
            # Expand neighbors (get_motion_primitives)
            # Space-time collision detection
            # Update cost and add to queue
        
        return path  # Return Path object
```

### 5.6 main.py - Usage Example

```python
from env import Env
from obstacle import Obstacle
from space_time_hybrid_a_star import SpaceTimeHybridAStar
from kinematic_model import KinematicModel

def main():
    env = Env()
    model = KinematicModel()
    planner = SpaceTimeHybridAStar(model)

    # Define start and goal
    start_x, start_y, start_yaw, start_velocity = 5, 0, 0, 0
    goal_x, goal_y, goal_yaw, goal_velocity = 45, 0, 0, 0

    # Define dynamic obstacles
    obstacles = [
        Obstacle(center=(15, 0), length=3.8, width=2, theta=0.0, velocity=0.7),
        Obstacle(center=(5, 5), length=3.8, width=2, theta=0.0, velocity=0.0),
        Obstacle(center=(40, 5), length=3.8, width=2, theta=np.pi, velocity=0.5)
    ]

    # Run planning
    path = planner.plan(start_x, start_y, start_yaw, start_velocity,
                      goal_x, goal_y, goal_yaw, goal_velocity,
                      obstacles=obstacles, env=env)

    if path:
        print(f"Path found: {len(path.x_list)} points")
        # Generate animation...
    else:
        print("No path found")

if __name__ == "__main__":
    main()
```

---

## 6. Execution Result

```
Path found in X iterations
2D animation saved as 'vehicle_animation.gif'
```

Generated visualizations:
- 2D animation of vehicle following path
- 3D space-time corridor (x, y, t)

---

## 7. Extension Directions

### 7.1 Short-term Improvements
- **Prediction-aware**: Combine with obstacle prediction trajectories
- **Uncertainty**: Consider prediction uncertainty weighting
- **Smoothing**: B-spline or polynomial fitting

### 7.2 Mid-term Directions
- **Hierarchical planning**: Top-level decision + bottom-level trajectory
- **Parallel search**: Multi-start / multi-goal parallel
- **Learning acceleration**: Neural network approximation of heuristic

### 7.3 Long-term Directions
- **End-to-end learning**: Learn planner from data
- **Reinforcement learning**: RL-based path optimization
- **Transformer**: Use Transformer for space-time modeling

---

## 8. Integration with Prediction System

This algorithm can integrate with the prediction system as the **planning module**:

```
Perception → Prediction → Planning → Control
                            ↑
              Space-Time Hybrid A*
              (handles dynamic obstacles)
```

**Integration:**
1. Prediction module outputs obstacle future trajectories
2. Space-Time Hybrid A* uses predicted trajectories for collision detection
3. Output safe, feasible trajectory
4. Send to control module

---

## 9. References

1. Montemerlo, M., et al. (2008). Junior: The Stanford entry in the Urban Challenge.
2. Dolgov, D., et al. (2010). Path Planning for Autonomous Driving in Unknown Environments.
3. Ziegler, J., et al. (2014). Making Bertha drive - An autonomous journey on a historic route.
4. Werling, M., et al. (2010). Optimal trajectory generation for dynamic street scenarios.

---

## Survey Status

- [x] Space-Time Hybrid A* algorithm principle
- [x] Code implementation analysis
- [x] Related work comparison
- [x] Extension directions

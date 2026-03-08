"""
Space-Time Hybrid A* Path Planning for Dynamic Environments

This module implements a path planning algorithm that combines Hybrid A* with
space-time reasoning to handle dynamic obstacles.

Author: Qi
Date: 2026-03-06
"""

import numpy as np
import heapq
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class VehicleParam:
    """Vehicle physical parameters"""
    WIDTH: float = 2.0
    LF: float = 3.8  # Distance from rear axle to front bumper
    LB: float = 0.8  # Distance from rear axle to rear bumper
    LENGTH: float = LF + LB
    WHEELBASE: float = 3.0
    MAX_STEER: float = np.deg2rad(30)
    MAX_SPEED: float = 3.0
    MAX_ACCEL: float = 2.0
    TR: float = 0.5  # Tire radius
    TW: float = 0.5  # Tire width
    WD: float = WIDTH  # Wheel distance


@dataclass
class STHybridAStarConfig:
    """Configuration for Space-Time Hybrid A*"""
    XY_RESOLUTION: float = 0.5  # meters
    YAW_RESOLUTION: float = np.deg2rad(15.0)  # radians
    TIME_RESOLUTION: float = 0.05  # seconds
    TIME_STEP: float = 1.0  # seconds
    MAX_PLAN_TIME: float = 60.0  # seconds
    SPEED_RESOLUTION: float = 0.6  # m/s
    STEER_NUM: int = 12
    EXPAND_DISTANCE: float = 10.0  # meters
    MAX_ITERATIONS: int = 50000
    GOAL_TOLERANCE_XY: float = 2.0  # meters
    GOAL_TOLERANCE_YAW: float = np.deg2rad(20.0)  # radians
    HEURISTIC_WEIGHT: float = 5.0
    STEER_CHANGE_WEIGHT: float = 0.2
    STEER_WEIGHT: float = 0.3
    GOAL_TIME_WEIGHT: float = 0.1
    SAFE_WEIGHT: float = 5.0
    SB_COST: float = 100.0  # Switch backward cost
    REF_LINE_WEIGHT: float = 0.01
    SAFE_BUFFER: float = 0.5  # meters
    ACTIVATE_DISTANCE: float = 3.0  # meters


class KinematicModel:
    """Bicycle model for vehicle kinematics"""
    
    def __init__(self, param: VehicleParam = None):
        self.param = param or VehicleParam()
        self.wheelbase = self.param.WHEELBASE
        self.max_steer = self.param.MAX_STEER
        self.max_speed = self.param.MAX_SPEED
        self.max_accel = self.param.MAX_ACCEL
    
    def motion_prediction(self, x: float, y: float, yaw: float, 
                         velocity: float, steer: float, dt: float = 0.5):
        """Predict next state using bicycle model"""
        steer = max(-self.max_steer, min(self.max_steer, steer))
        velocity = max(-self.max_speed, min(self.max_speed, velocity))
        
        new_x = x + velocity * math.cos(yaw) * dt
        new_y = y + velocity * math.sin(yaw) * dt
        new_yaw = yaw + velocity / self.wheelbase * math.tan(steer) * dt
        new_yaw = self.normalize_angle(new_yaw)
        
        return new_x, new_y, new_yaw
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


class Obstacle:
    """Dynamic obstacle with constant velocity motion"""
    
    def __init__(self, center: Tuple[float, float], length: float = 3.0, 
                 width: float = 2.0, theta: float = 0.0, velocity: float = 0.0):
        self.center = center  # (x, y)
        self.length = length
        self.width = width
        self.theta = theta  # heading angle
        self.velocity = velocity
        self.radius = math.hypot(length / 2, width / 2)
    
    def get_position(self, time: float) -> Tuple[float, float]:
        """Get obstacle position at time t"""
        x = self.center[0] + self.velocity * time * math.cos(self.theta)
        y = self.center[1] + self.velocity * time * math.sin(self.theta)
        return x, y
    
    def get_vertices(self, time: float) -> List[Tuple[float, float]]:
        """Get obstacle vertices at time t"""
        x, y = self.get_position(time)
        l, w = self.length, self.width
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)
        half_l, half_w = l / 2.0, w / 2.0
        
        return [
            (x + half_l * cos_theta - half_w * sin_theta,
             y + half_l * sin_theta + half_w * cos_theta),
            (x + half_l * cos_theta + half_w * sin_theta,
             y + half_l * sin_theta - half_w * cos_theta),
            (x - half_l * cos_theta + half_w * sin_theta,
             y - half_l * sin_theta - half_w * cos_theta),
            (x - half_l * cos_theta - half_w * sin_theta,
             y - half_l * sin_theta + half_w * cos_theta),
        ]
    
    def has_overlap(self, other_x: float, other_y: float, 
                    other_length: float, other_width: float, 
                    other_theta: float, time: float = 0.0, 
                    safe_distance: float = 0.2) -> bool:
        """Check collision at time t using circle approximation"""
        cur_x, cur_y = self.get_position(time)
        dist = math.hypot(cur_x - other_x, cur_y - other_y)
        other_radius = math.hypot(other_length / 2, other_width / 2)
        
        if dist > self.radius + other_radius + safe_distance:
            return False
        
        # Simplified: just use circle collision for now
        return dist < (self.radius + other_radius + safe_distance)
    
    def get_min_distance(self, other_x: float, other_y: float,
                        other_length: float, other_width: float,
                        other_theta: float, time: float = 0.0,
                        safe_distance: float = 0.5) -> float:
        """Get minimum distance to obstacle"""
        if self.has_overlap(other_x, other_y, other_length, other_width,
                          other_theta, time, safe_distance):
            return 0.0
        
        cur_x, cur_y = self.get_position(time)
        dist = math.hypot(cur_x - other_x, cur_y - other_y)
        other_radius = math.hypot(other_length / 2, other_width / 2)
        
        return max(0.0, dist - self.radius - other_radius - safe_distance)


class Environment:
    """Planning environment with reference line and boundaries"""
    
    def __init__(self):
        self.ref_line, self.bound1, self.bound2, self.other_line = self._get_refline_info()
    
    def _get_refline_info(self):
        """Initialize reference line and boundaries"""
        refline, bound1, bound2, other_line = [], [], [], []
        
        for i in np.arange(0, 60, 10):
            refline.append((i, 0))
            other_line.append((i, 5))
            bound1.append((i, -2.5))
            bound2.append((i, 7.5))
        
        return refline, bound1, bound2, other_line
    
    @staticmethod
    def point_to_line_distance(point: Tuple[float, float], 
                             line_start: Tuple[float, float],
                             line_end: Tuple[float, float]) -> float:
        """Distance from point to line segment"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        
        if length_sq < 1e-10:
            return math.hypot(x - x1, y - y1)
        
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
        proj_x, proj_y = x1 + t * dx, y1 + t * dy
        
        return math.hypot(x - proj_x, y - proj_y)
    
    def get_distance_to_refline(self, x: float, y: float) -> float:
        """Distance to reference line"""
        min_dist = float('inf')
        for i in range(len(self.ref_line) - 1):
            dist = self.point_to_line_distance(
                (x, y), self.ref_line[i], self.ref_line[i + 1]
            )
            min_dist = min(min_dist, dist)
        return min_dist


class HybridAStarNode:
    """Node in Hybrid A* search"""
    
    def __init__(self, x_ind: int, y_ind: int, yaw_ind: int, time_ind: int,
                 x_list: List[float], y_list: List[float], yaw_list: List[float],
                 velocity_list: List[float], steer_list: List[float],
                 time_list: List[float], parent_index: Optional[str] = None,
                 cost: float = 0.0):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.time_index = time_ind
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.velocity_list = velocity_list
        self.steer_list = steer_list
        self.time_list = time_list
        self.parent_index = parent_index
        self.cost = cost
    
    def __repr__(self):
        return f"Node(x={self.x_index}, y={self.y_index}, yaw={self.yaw_index}, t={self.time_index}, cost={self.cost:.2f})"


class Path:
    """Planning result path"""
    
    def __init__(self, x_list: List[float], y_list: List[float], 
                 yaw_list: List[float], velocity_list: List[float],
                 steer_list: List[float], time_list: List[float]):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.velocity_list = velocity_list
        self.steer_list = steer_list
        self.time_list = time_list
    
    def __len__(self):
        return len(self.x_list)
    
    @property
    def duration(self) -> float:
        """Total planning time"""
        return self.time_list[-1] - self.time_list[0] if self.time_list else 0.0
    
    @property
    def length(self) -> float:
        """Total path length"""
        if len(self.x_list) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.x_list)):
            total += math.hypot(
                self.x_list[i] - self.x_list[i-1],
                self.y_list[i] - self.y_list[i-1]
            )
        return total


class SpaceTimeHybridAStar:
    """
    Space-Time Hybrid A* Path Planner
    
    Features:
    - Bicycle model kinematics
    - Space-time collision checking with dynamic obstacles
    - Multi-objective cost function
    - Heuristic-guided search
    """
    
    def __init__(self, kinematic_model: KinematicModel = None,
                 config: STHybridAStarConfig = None):
        self.model = kinematic_model or KinematicModel()
        self.config = config or STHybridAStarConfig()
        
        # Resolution parameters
        self.xy_res = self.config.XY_RESOLUTION
        self.yaw_res = self.config.YAW_RESOLUTION
        self.time_res = self.config.TIME_RESOLUTION
        self.speed_res = self.config.SPEED_RESOLUTION
        self.steer_num = self.config.STEER_NUM
        self.time_step = self.config.TIME_STEP
        self.max_plan_time = self.config.MAX_PLAN_TIME
        
        # Cost weights
        self.safe_buffer = self.config.SAFE_BUFFER
        self.activate_distance = self.config.ACTIVATE_DISTANCE
        self.heuristic_weight = self.config.HEURISTIC_WEIGHT
        self.steer_change_weight = self.config.STEER_CHANGE_WEIGHT
        self.steer_weight = self.config.STEER_WEIGHT
        self.goal_time_weight = self.config.GOAL_TIME_WEIGHT
        self.safe_weight = self.config.SAFE_WEIGHT
        self.ref_line_weight = self.config.REF_LINE_WEIGHT
        self.sb_cost = self.config.SB_COST
        
        # Search limits
        self.max_iterations = self.config.MAX_ITERATIONS
        self.goal_tolerance_xy = self.config.GOAL_TOLERANCE_XY
        self.goal_tolerance_yaw = self.config.GOAL_TOLERANCE_YAW
        
        # Runtime state
        self.min_x = self.max_x = self.min_y = self.max_y = None
        self.env = None
    
    def _calc_bounds(self, start_x: float, start_y: float, 
                     goal_x: float, goal_y: float):
        """Calculate search bounds"""
        self.min_x = min(start_x, goal_x) - self.config.EXPAND_DISTANCE
        self.max_x = max(start_x, goal_x) + self.config.EXPAND_DISTANCE
        self.min_y = min(start_y, goal_y) - self.config.EXPAND_DISTANCE
        self.max_y = max(start_y, goal_y) + self.config.EXPAND_DISTANCE
    
    def _is_within_bounds(self, x: float, y: float) -> bool:
        """Check if point is within bounds"""
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
    
    def _calc_index(self, node: HybridAStarNode) -> str:
        """Calculate node index for hashing"""
        return f"{node.x_index}_{node.y_index}_{node.yaw_index}_{node.time_index}"
    
    def _get_motion_primitives(self, current_velocity: float) -> List[Tuple[float, float]]:
        """Generate velocity-steering motion primitives"""
        primitives = []
        
        # Generate velocity options
        velocities = np.arange(
            self.speed_res, 
            self.model.max_speed + self.speed_res, 
            self.speed_res
        )
        
        # Generate steering options
        steers = []
        for i in range(self.steer_num + 1):
            angle = i * self.model.max_steer / self.steer_num
            steers.append(angle)
            if angle != 0:
                steers.append(-angle)
        
        # Create all combinations
        for velocity in velocities:
            for steer in steers:
                primitives.append((velocity, steer))
        
        return primitives
    
    def _check_collision(self, x: float, y: float, yaw: float,
                        obstacles: List[Obstacle], time: float) -> Tuple[bool, float]:
        """
        Check collision and get minimum distance
        
        Returns: (is_collision, min_distance)
        """
        if not self._is_within_bounds(x, y):
            return True, 0.0
        
        # Vehicle dimensions
        length = self.model.param.LENGTH
        width = self.model.param.WIDTH
        rear_to_center = length / 2.0 - self.model.param.LB
        cx = x + rear_to_center * math.cos(yaw)
        cy = y + rear_to_center * math.sin(yaw)
        
        # Check dynamic obstacles
        min_distance = float('inf')
        for obs in obstacles:
            dist = obs.get_min_distance(cx, cy, length, width, yaw, 
                                       time, self.safe_buffer)
            if dist < 1e-2:  # Collision
                return True, 0.0
            min_distance = min(min_distance, dist)
        
        return False, min_distance
    
    def _calc_next_node(self, current: HybridAStarNode, velocity: float, steer: float,
                        obstacles: List[Obstacle]) -> Optional[HybridAStarNode]:
        """Generate next node from current state"""
        x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]
        
        x_list, y_list, yaw_list = [], [], []
        velocity_list, steer_list, time_list = [], [], []
        
        # Simulate forward
        for t in np.arange(self.time_res, self.time_step, self.time_res):
            x, y, yaw = self.model.motion_prediction(x, y, yaw, velocity, steer, 
                                                     self.time_res)
            
            # Check collision
            is_collision, _ = self._check_collision(x, y, yaw, obstacles,
                                                     current.time_list[-1] + t)
            if is_collision:
                return None
            
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)
            velocity_list.append(velocity)
            steer_list.append(steer)
            time_list.append(current.time_list[-1] + t)
        
        if not x_list:
            return None
        
        # Calculate cost
        x_ind = round(x_list[-1] / self.xy_res)
        y_ind = round(y_list[-1] / self.yaw_res)
        yaw_ind = round(yaw_list[-1] / self.yaw_res)
        time_ind = round(time_list[-1] / self.time_res)
        
        # Cost components
        added_cost = 0.0
        
        # Switch backward cost
        if current.velocity_list[-1] * velocity < 0:
            added_cost += self.sb_cost
        
        # Steering cost
        added_cost += self.steer_weight * abs(steer)
        added_cost += self.steer_change_weight * abs(current.steer_list[-1] - steer)
        
        # Path length cost
        path_length = math.hypot(x_list[-1] - current.x_list[-1],
                                y_list[-1] - current.y_list[-1])
        
        # Reference line cost (if available)
        refline_cost = 0.0
        if self.env:
            for px, py in zip(x_list, y_list):
                refline_cost += self.ref_line_weight * self.env.get_distance_to_refline(px, py)
        
        total_cost = current.cost + added_cost + path_length + refline_cost
        
        return HybridAStarNode(
            x_ind=x_ind, y_ind=y_ind, yaw_ind=yaw_ind, time_ind=time_ind,
            x_list=current.x_list + x_list,
            y_list=current.y_list + y_list,
            yaw_list=current.yaw_list + yaw_list,
            velocity_list=current.velocity_list + velocity_list,
            steer_list=current.steer_list + steer_list,
            time_list=current.time_list + time_list,
            parent_index=self._calc_index(current),
            cost=total_cost
        )
    
    def _get_neighbors(self, current: HybridAStarNode, 
                      obstacles: List[Obstacle]) -> List[HybridAStarNode]:
        """Get valid neighbor nodes"""
        neighbors = []
        primitives = self._get_motion_primitives(current.velocity_list[-1])
        
        for velocity, steer in primitives:
            next_node = self._calc_next_node(current, velocity, steer, obstacles)
            if next_node:
                neighbors.append(next_node)
        
        return neighbors
    
    def _reconstruct_path(self, closed_list: Dict[str, HybridAStarNode],
                         goal_node: HybridAStarNode) -> Path:
        """Reconstruct path from goal to start"""
        reversed_x = list(reversed(goal_node.x_list))
        reversed_y = list(reversed(goal_node.y_list))
        reversed_yaw = list(reversed(goal_node.yaw_list))
        reversed_vel = list(reversed(goal_node.velocity_list))
        reversed_steer = list(reversed(goal_node.steer_list))
        reversed_time = list(reversed(goal_node.time_list))
        
        nid = goal_node.parent_index
        while nid and nid in closed_list:
            n = closed_list[nid]
            reversed_x.extend(list(reversed(n.x_list)))
            reversed_y.extend(list(reversed(n.y_list)))
            reversed_yaw.extend(list(reversed(n.yaw_list)))
            reversed_vel.extend(list(reversed(n.velocity_list)))
            reversed_steer.extend(list(reversed(n.steer_list)))
            reversed_time.extend(list(reversed(n.time_list)))
            nid = n.parent_index
        
        return Path(
            x_list=list(reversed(reversed_x)),
            y_list=list(reversed(reversed_y)),
            yaw_list=list(reversed(reversed_yaw)),
            velocity_list=list(reversed(reversed_vel)),
            steer_list=list(reversed(reversed_steer)),
            time_list=list(reversed(reversed_time))
        )
    
    def plan(self, start_x: float, start_y: float, start_yaw: float,
            start_velocity: float, goal_x: float, goal_y: float, 
            goal_yaw: float, goal_velocity: float,
            obstacles: List[Obstacle], env: Environment,
            animate: bool = False) -> Optional[Path]:
        """
        Main planning function
        
        Args:
            start_x, start_y, start_yaw, start_velocity: Start state
            goal_x, goal_y, goal_yaw, goal_velocity: Goal state
            obstacles: List of dynamic obstacles
            env: Planning environment
            animate: Whether to generate animation (not implemented)
        
        Returns:
            Path object if found, None otherwise
        """
        self.env = env
        self._calc_bounds(start_x, start_y, goal_x, goal_y)
        
        # Create start node
        start_node = HybridAStarNode(
            x_ind=round(start_x / self.xy_res),
            y_ind=round(start_y / self.xy_res),
            yaw_ind=round(start_yaw / self.yaw_res),
            time_ind=0,
            x_list=[start_x], y_list=[start_y], yaw_list=[start_yaw],
            velocity_list=[start_velocity], steer_list=[0.0],
            time_list=[0.0], cost=0.0
        )
        
        # A* search
        open_list = {}
        closed_list = {}
        pq = []
        
        start_idx = self._calc_index(start_node)
        open_list[start_idx] = start_node
        heapq.heappush(pq, (start_node.cost, start_idx))
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if not open_list:
                print(f"Planning failed: No open nodes after {iteration} iterations")
                return None
            
            # Get node with lowest cost
            cost, node_id = heapq.heappop(pq)
            
            if node_id in open_list:
                current = open_list.pop(node_id)
                closed_list[node_id] = current
            else:
                continue
            
            # Check if goal reached
            dist_to_goal = math.hypot(
                current.x_list[-1] - goal_x,
                current.y_list[-1] - goal_y
            )
            yaw_error = abs(self.model.normalize_angle(
                current.yaw_list[-1] - goal_yaw
            ))
            
            if dist_to_goal < self.goal_tolerance_xy and yaw_error < self.goal_tolerance_yaw:
                print(f"Path found in {iteration} iterations!")
                return self._reconstruct_path(closed_list, current)
            
            # Skip if exceeded max time
            if current.time_list[-1] > self.max_plan_time:
                continue
            
            # Expand neighbors
            for neighbor in self._get_neighbors(current, obstacles):
                neighbor_id = self._calc_index(neighbor)
                
                if neighbor_id in closed_list:
                    continue
                
                if neighbor_id not in open_list or neighbor.cost < open_list[neighbor_id].cost:
                    heapq.heappush(pq, (neighbor.cost, neighbor_id))
                    open_list[neighbor_id] = neighbor
        
        print(f"Planning failed: Max iterations ({self.max_iterations}) reached")
        return None


def plan_with_st_hybrid_astar(start: Tuple[float, float, float, float],
                              goal: Tuple[float, float, float, float],
                              obstacles: List[Obstacle],
                              env: Environment = None,
                              verbose: bool = True) -> Optional[Path]:
    """
    Convenience function for Space-Time Hybrid A* planning
    
    Args:
        start: (x, y, yaw, velocity)
        goal: (x, y, yaw, velocity)
        obstacles: List of dynamic obstacles
        env: Environment (creates default if None)
        verbose: Print progress
    
    Returns:
        Path object or None
    """
    env = env or Environment()
    planner = SpaceTimeHybridAStar()
    
    return planner.plan(
        start_x=start[0], start_y=start[1], start_yaw=start[2], start_velocity=start[3],
        goal_x=goal[0], goal_y=goal[1], goal_yaw=goal[2], goal_velocity=goal[3],
        obstacles=obstacles, env=env
    )


# Example usage
if __name__ == "__main__":
    # Create environment
    env = Environment()
    
    # Define start and goal
    start = (5.0, 0.0, 0.0, 0.0)  # x, y, yaw, velocity
    goal = (45.0, 0.0, 0.0, 0.0)
    
    # Define dynamic obstacles
    obstacles = [
        Obstacle(center=(15, 0), length=3.8, width=2, theta=0.0, velocity=0.7),
        Obstacle(center=(5, 5), length=3.8, width=2, theta=0.0, velocity=0.0),
        Obstacle(center=(40, 5), length=3.8, width=2, theta=np.pi, velocity=0.5),
    ]
    
    # Plan
    print("Planning...")
    path = plan_with_st_hybrid_astar(start, goal, obstacles, env)
    
    if path:
        print(f"Path found!")
        print(f"  Duration: {path.duration:.2f}s")
        print(f"  Length: {path.length:.2f}m")
        print(f"  Waypoints: {len(path)}")
    else:
        print("No path found")

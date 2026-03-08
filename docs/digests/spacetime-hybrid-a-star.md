# 时空联合 Hybrid A* 算法调研报告

**日期:** 2026-03-06  
**状态:** 调研完成

---

## 一、引言

传统路径规划算法（如A*、Dijkstra）仅考虑空间维度的静态障碍物规避，无法应对移动障碍物带来的时空冲突。**时空联合的混合A*算法（Space-Time Hybrid A*）** 融合了混合A*的运动学约束适配能力与时空联合建模思想，既保证路径满足车辆的动力学特性，又通过时间维度的扩展实现动态障碍物的提前规避。

---

## 二、算法原理

### 2.1 状态空间建模

传统混合A*的状态仅包含空间坐标，而时空联合的混合A*引入时间维度，构建四维状态空间：

```
状态: (x, y, yaw, t)
  ├── 空间维度: x, y (车辆后轴中心坐标), yaw (航向角)
  └── 时间维度: t (当前时刻，用于匹配动态障碍物位置)
```

**离散化原则：**
| 维度 | 分辨率 | 作用 |
|------|--------|------|
| 空间 | 0.5m | 平衡搜索效率与精度 |
| 航向角 | 15° | 适配车辆转向步长 |
| 时间 | 0.05s | 匹配障碍物运动更新步长 |

### 2.2 运动学约束建模

使用车辆自行车模型（Bicycle Model）：

```
x' = x + v * cos(yaw) * dt
y' = y + v * sin(yaw) * dt
yaw' = yaw + v / L * tan(δ) * dt
```

其中：
- v: 车辆速度
- δ: 前轮转向角
- L: 轴距

### 2.3 时空联合代价函数

```
总代价 = 路径长度 + 安全代价 + 运动平滑代价 + 时间代价 + 参考线代价
```

| 代价项 | 公式 | 作用 |
|--------|------|------|
| **路径长度** | ∫√(dx²+dy²) | 保证路径经济性 |
| **安全代价** | f(时空间距) | 规避动态障碍物 |
| **运动平滑** | f(steer, Δsteer) | 保证转向平稳 |
| **时间代价** | f(t - t_max) | 约束规划时间 |
| **参考线** | f(到参考线距离) | 引导沿车道行驶 |

### 2.4 启发式函数设计

```
h(n) = w_h * (空间距离 + 时间距离)
```

关键设计：
- 结合动态障碍物的实时预测位置
- 预计算无碰撞最短距离
- 避免启发式估值过低导致无效搜索

### 2.5 动态障碍物时空碰撞检测

**障碍物运动模型**（匀速直线运动）：
```
obs(t) = obs_0 + v * t * [cos(θ), sin(θ)]
```

**碰撞判据**：
1. 计算轨迹上每点对应时刻的障碍物位置
2. 进行二维平面碰撞检测（SAT分离轴定理）
3. 若碰撞则该节点无效

---

## 三、算法流程

```
1. 环境建模: 构建参考线、边界、动态障碍物参数
2. 状态初始化: 定义起始状态 S 和目标状态 G
3. 节点扩展: 基于运动学模型生成运动基元
4. 时空碰撞检测: 过滤碰撞节点
5. 优先级队列: 按代价升序排列
6. 目标判定: 空间距离 < 容差 且 航向角误差 < 容差
7. 路径重构: 反向追溯父节点
```

---

## 四、相关工作对比

### 4.1 路径规划方法谱系

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| **A*** | 静态障碍物，快速 | 简单环境 |
| **Dijkstra** | 全局最优 | 静态最短路径 |
| **Hybrid A*** | 运动学约束 | 车辆运动规划 |
| **RRT*** | 概率完备 | 复杂/高维空间 |
| **时空 Hybrid A*** | +时间维度 | 动态障碍物规避 |
| **Lattice Planner** | 状态格 | 结构化道路 |
| **EM Planner** | 预期最大化 | 多模态决策 |

### 4.2 动态障碍物处理方法

| 方法 | 核心思想 | 优缺点 |
|------|----------|--------|
| **Velocity Obstacle (VO)** | 速度空间避障 | 实时性好，但简化 |
| **Reciprocal Velocity Obstacle (RVO)** | 双向VO | 多智能体 |
| **Games** | 博弈论 | 理论优雅，计算复杂 |
| **Space-Time A*** | 时空搜索 | 精确但计算量大 |
| **MPDM** | 预测+规划 | 考虑预测不确定性 |

### 4.3 本算法特点

| 方面 | 时空 Hybrid A* |
|------|----------------|
| **优势** | 精确处理动态障碍物、运动学可行、时空联合优化 |
| **挑战** | 计算复杂度高、状态空间维度灾难 |
| **适用** | 中低速场景（停车场、园区）、结构化道路 |

---

## 五、代码实现要点

### 5.1 核心数据结构

```python
class HybridAStarNode:
    x_index, y_index, yaw_index, time_index  # 状态坐标
    x_list, y_list, yaw_list, time_list      # 轨迹
    velocity_list, steer_list                  # 控制序列
    cost                                      # 总代价
    parent_index                               # 父节点
```

### 5.2 运动基元生成

```python
def get_motion_primitives(velocity):
    primitives = []
    for v in velocities:
        for steer in steers:
            primitives.append((v, steer))  # 速度-转向角组合
    return primitives
```

### 5.3 时空碰撞检测

```python
def check_space_time_collision(trajectory, obstacles):
    for (x, y, yaw), t in zip(trajectory, times):
        obs_pos = obs.position(t)  # 障碍物在t时刻的位置
        if polygon_overlap(vehicle(x,y,yaw), obs(obs_pos)):
            return True  # 碰撞
    return False
```

---

## 六、扩展方向

### 6.1 短期改进

- **预测感知**: 结合障碍物预测轨迹
- **不确定性**: 考虑预测不确定性加权
- **平滑后处理**: B样条/多项式拟合

### 6.2 中期方向

- **分层规划**: 顶层决策 + 底层轨迹规划
- **并行搜索**: 多起点/多目标并行
- **学习加速**: 用神经网络近似启发式

### 6.3 长期方向

- **端到端学习**: 从数据学习时空规划器
- **强化学习**: RL-based 路径优化
- **Transformer**: 用Transformer建模时空关系

---

## 七、参考文献

1. Montemerlo, M., et al. (2008). Junior: The Stanford entry in the Urban Challenge.
2. Dolgov, D., et al. (2010). Path Planning for Autonomous Driving in Unknown Environments.
3. Ziegler, J., et al. (2014). Making Bertha drive - An autonomous journey on a historic route.
4. Werling, M., et al. (2010). Optimal trajectory generation for dynamic street scenarios.
5. 知乎 - 自动驾驶: 时空联合规划

---

## 八、与预测系统集成

该算法可作为**规划模块**与预测系统集成：

```
感知 → 预测 → 规划 → 控制
              ↑
         时空 Hybrid A*
         (处理动态障碍物)
```

**集成方式：**
1. 预测模块输出障碍物未来轨迹
2. 时空 Hybrid A* 使用预测轨迹进行时空碰撞检测
3. 输出安全、可行的轨迹
4. 下发控制指令

---

## 调研状态

- [x] 时空 Hybrid A* 算法原理
- [x] 代码实现分析
- [x] 相关工作对比
- [x] 扩展方向

---

## 九、完整代码实现

以下是Qi提供的**完整实现代码**：

### 9.1 config.py - 配置参数

```python
import numpy as np

class VehicleParam:
    WIDTH = 2.0
    LF = 3.8  # 后轴中心到车头距离
    LB = 0.8  # 后轴中心到车尾距离
    LENGTH = LF + LB
    WHEELBASE = 3.0
    MAX_STEER = np.deg2rad(30)
    MAX_SPEED = 3.0
    MAX_ACCEL = 2.0
    TR = 0.5
    TW = 0.5
    WD = WIDTH

class SpaceTimeHybridAStarConfig:
    XY_RESOLUTION = 0.5
    YAW_RESOLUTION = np.deg2rad(15.0)
    TIME_RESOLUTION = 0.05
    TIME_STEP = 1.0
    MAX_PLAN_TIME = 60.0
    SPEED_RESOLUTION = 0.6
    STEER_NUM = 12
    EXPAND_DISTANCE = 10.0
    MAX_ITERATIONS = 50000
    GOAL_TOLERANCE_XY = 2.0
    GOAL_TOLERANCE_YAW = np.deg2rad(20.0)
    HEURISTIC_WEIGHT = 5.0
    STEER_CHANGE_WEIGHT = 0.2
    STEER_WEIGHT = 0.3
    GOAL_TIME_WEIGHT = 0.1
    SAFE_WEIGHT = 5.0
    SB_COST = 100.0
    REF_LINE_WEIGHT = 0.01
    SAFE_BUFFER = 0.5
    ACTIVATE_DISTANCE = 3.0
```

### 9.2 kinematic_model.py - 自行车模型

```python
import numpy as np
import math

class KinematicModel:
    def __init__(self, param=VehicleParam()):
        self.param = param
        self.wheelbase = param.WHEELBASE
        self.max_steer = param.MAX_STEER
        self.max_speed = param.MAX_SPEED

    def motion_prediction(self, x, y, yaw, velocity, steer, dt=0.5):
        steer = max(-self.max_steer, min(self.max_steer, steer))
        velocity = max(-self.max_speed, min(self.max_speed, velocity))
        new_x = x + velocity * math.cos(yaw) * dt
        new_y = y + velocity * math.sin(yaw) * dt
        new_yaw = yaw + velocity / self.wheelbase * math.tan(steer) * dt
        new_yaw = self.normalize_angle(new_yaw)
        return new_x, new_y, new_yaw

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
```

### 9.3 obstacle.py - 动态障碍物

```python
import as np

class math
import numpy Obstacle:
    def __init__(self, center, length=3, width=2.0, theta=0, velocity=0.0):
        self.center = center
        self.length = length
        self.width = width
        self.theta = theta
        self.velocity = velocity
        self.radius = math.hypot(length/2, width/2)

    def get_vertices(self, time=0):
        """获取时刻t的障碍物顶点位置"""
        x = self.center[0] + self.velocity * time * math.cos(self.theta)
        y = self.center[1] + self.velocity * time * math.sin(self.theta)
        # 计算四个顶点 (略)
        return vertices

    def has_overlap(self, other_center_x, other_center_y, other_length,
                    other_width, other_theta, time=0.0, safe_distance=0.2):
        """时空碰撞检测 - 返回是否碰撞"""
        cur_x = self.center[0] + self.velocity * time * math.cos(self.theta)
        cur_y = self.center[1] + self.velocity * time * math.sin(self.theta)
        # 基于SAT分离轴定理的碰撞检测
        return has_collision

    def get_min_distance(self, other_center_x, other_center_y, other_length,
                        other_width, other_theta, time=0.0, safe_distance=0.5):
        """计算到障碍物的最小距离"""
        if self.has_overlap(...):
            return 0.0
        return min_distance
```

### 9.4 env.py - 环境

```python
import numpy as np

class Env:
    def __init__(self):
        self.ref_line, self.bound1, self.bound2, self.other_line = self.get_refline_info()

    def get_refline_info(self):
        refline, bound1, bound2, other_line = [], [], [], []
        for i in np.arange(0, 60, 10):
            refline.append((i, 0))
            other_line.append((i, 5))
            bound1.append((i, -2.5))
            bound2.append((i, 7.5))
        return refline, bound1, bound2, other_line
```

### 9.5 space_time_hybrid_a_star.py - 主算法

```python
import numpy as np
import heapq
import math

class HybridAStarNode:
    def __init__(self, x_ind, y_ind, yaw_ind, time_ind, x_list, y_list, yaw_list,
                 velocity_list, steer_list, time_list=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.time_index = time_ind
        # ... 轨迹和控制序列

class SpaceTimeHybridAStar:
    def __init__(self, kinematic_model, config):
        self.model = kinematic_model
        self.config = config

    def plan(self, start_x, start_y, start_yaw, start_velocity,
             goal_x, goal_y, goal_yaw, goal_velocity, 
             obstacles, env, animate=False):
        # 1. 初始化
        start_node = HybridAStarNode(...)
        
        # 2. A*搜索循环
        while iteration < self.max_iterations:
            # 弹出代价最小节点
            # 检查目标是否到达
            # 扩展邻居节点
            # 时空碰撞检测
            
        return path

    def get_motion_primitives(self, current_velocity):
        """生成速度-转向角运动基元"""
        primitives = []
        velocities = np.arange(self.speed_resolution, self.model.max_speed, 
                              self.speed_resolution)
        for velocity in velocities:
            for i in range(self.steer_num + 1):
                angle = i * self.model.max_steer / self.steer_num
                if angle != 0:
                    primitives.append((velocity, angle))
                    primitives.append((velocity, -angle))
        return primitives

    def check_car_collision_and_get_min_distance(self, x_list, y_list, yaw_list, 
                                               obstacles, time):
        """时空碰撞检测"""
        for x, y, yaw in zip(x_list, y_list, yaw_list):
            if self.is_point_out_of_bounds(x, y):
                return False, 0.0
            for obstacle in obstacles:
                distance = obstacle.get_min_distance(...)
                if distance < 1e-2:
                    return False, 0.0
        return True, min_distance
```

### 9.6 main.py - 运行示例

```python
from env import Env
from obstacle import Obstacle
from space_time_hybrid_a_star import SpaceTimeHybridAStar
from kinematic_model import KinematicModel

def main():
    env = Env()
    model = KinematicModel()
    planner = SpaceTimeHybridAStar(model)

    start_x, start_y, start_yaw, start_velocity = 5, 0, 0, 0
    goal_x, goal_y, goal_yaw, goal_velocity = 45, 0, 0, 0

    obstacles = [
        Obstacle(center=(15, 0), length=3.8, width=2, theta=0.0, velocity=0.7),
        Obstacle(center=(5, 5), length=3.8, width=2, theta=0.0, velocity=0.0),
        Obstacle(center=(40, 5), length=3.8, width=2, theta=np.pi, velocity=0.5)
    ]

    path = planner.plan(start_x, start_y, start_yaw, start_velocity,
                       goal_x, goal_y, goal_yaw, goal_velocity,
                       obstacles=obstacles, env=env)
    
    if path:
        print(f"Path found: {len(path.x_list)} points")
        # 生成动画...
```

---

## 运行结果

```
Path found in X iterations
2D animation saved as 'vehicle_animation.gif'
```

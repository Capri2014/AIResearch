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

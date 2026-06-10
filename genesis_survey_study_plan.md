# Genesis World: Comprehensive Survey & Study Plan

**Document Version:** 1.1  
**Date:** June 2026  
**Target Audience:** Researchers and engineers seeking to learn and use Genesis for embodied AI, robotics, and simulation research

---

## Table of Contents

1. [Project Overview & Intuition](#1-project-overview--intuition)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How It Works — Architecture Deep Dive](#3-how-it-works--architecture-deep-dive)
4. [Actual Code Examples](#4-actual-code-examples)
5. [Cross-Comparison with Alternative Simulators](#5-cross-comparison-with-alternative-simulators)
6. [When to Use Genesis](#6-when-to-use-genesis)
7. [Quick Reference Decision Table](#7-quick-reference-decision-table)
8. [Study Plan](#8-study-plan)
9. [References & Resources](#9-references--resources)

---

## 1. Project Overview & Intuition

### 1.1 What is Genesis World?

**Genesis World** is a unified simulation platform for physical AI development, designed to accelerate robotics research and embodied AI. It was previously named "Genesis" and started as an academic project in December 2024, with development now officially supported by **Genesis AI** (genesis.ai).

Think of Genesis as a "one-stop physics engine" that combines:

- **Multi-physics simulation** (rigid bodies, deformable objects, fluids, cloth, granular materials)
- **Photo-realistic rendering** (the Nyx renderer)
- **Cross-platform compilation** (the Quadrants compiler)
- **Pythonic API** that's easy to read, extend, and embed in research code

### 1.2 The Intuition — Why Genesis Matters

Before Genesis, researchers had to piece together multiple tools:

- **MuJoCo** for rigid body physics
- **PyBullet** or **ODE** for simpler simulations
- **FLIP/FEM** solvers for fluids and deformables
- **Blender** or **Isaac Sim** for rendering
- Various wrappers and glue code to make them work together

Genesis wraps all of this into **one unified framework** with:

- A single Python API
- Shared scene state across all physics types
- 10-80x faster simulation speeds than existing GPU-accelerated simulators
- Built-in sensors, controllers, and parallel environment support

### 1.3 Mission & Positioning

Genesis positions itself as:

> *"A simulation platform for physical AI developments that combines a unified multi-physics engine, a photo-realistic renderer, and a cross-platform compiler behind a Pythonic simulation interface."*

It's designed to scale from a single laptop kernel to datacenter-grade GPUs, making it suitable for:

- Academic research (fast prototyping, education)
- Industry R&D (large-scale simulation, data generation)
- Foundation model training (sim-to-real transfer, evaluation)

### 1.4 Core Statistics & Community Signals

| Metric | Value (as of June 2026) |
|--------|------------------------|
| Initial Release | December 2024 |
| Current Version | Genesis World 1.0 |
| GitHub Stars | Active and growing |
| License | Apache 2.0 |
| Python Version | 3.10 - 3.13 |
| Primary Language | Python (with Quadrants compiler backend in C++/CUDA) |
| Support | Discord, GitHub Issues, Discussions |

---

## 2. The Problem It Solves

### 2.1 The Fragmentation Problem

Prior to Genesis, robotics simulation suffered from **tool fragmentation**:

| Use Case | Typical Toolchain |
|---------|-----------------|
| Rigid body manipulation | MuJoCo + MJX |
| Deformable objects | FEM solvers (separate) |
| Fluids/cloth | FLIP/PBD solvers (separate) |
| Photo-realistic rendering | Blender, Isaac Sim, NVIDIA Omniverse |
| GPU acceleration | Isaac Gym, Brax (each with limited scope) |

**The problem:** Each tool has its own API, physics assumptions, and limitations. Combining them requires extensive glue code, and data must be converted between formats.

### 2.2 The Performance Problem

Traditional simulators are **too slow** for modern AI training needs:

- Isaac Gym/Sim: Fast for rigid bodies, but limited to simpler physics
- MuJoCo: Accurate but single-threaded by default
- PyBullet: Easy but slow for complex scenes

**Genesis delivers 10-80x speedup** over these tools while maintaining accuracy.

### 2.3 The Differentiation Problem

Most simulators don't support differentiable physics, which is critical for:

- End-to-end RL (backprop through simulation)
- Sim-to-real transfer learning
- Model-based RL with learned simulators

Genesis provides **built-in autodiff** via the Quadrants compiler.

---

## 3. How It Works — Architecture Deep Dive

### 3.1 Four-Layer Stack

```
┌─────────────────────────────────────┐
│  Simulation Interface (Python API)   │  ← User-facing: asset parsing, sensors, controllers, GUI
├─────────────────────────────────────┤
│  Physics Engine                    │  ← Unified: Rigid, FEM, MPM, PBD/SPH, IPC, SAP, Coupler
├─────────────────────────────────────┤
│  Render                          │  ← Nyx (ray-trace), Luisa (DSL ray-tracer), Pyrender
├─────────────────────────────────────┤
│  Compiler (Quadrants)            │  ← CUDA/ROCm/Metal/Vulkan/x86/ARM64 + autodiff
└─────────────────────────────────────┘
```

### 3.2 Physics Solvers Explained

| Solver | What It Simulates | Use Case |
|--------|------------------|----------|
| **Rigid** | Solid objects with mass/inertia | Robot manipulation, locomotion |
| **FEM** (Finite Element) | Deformable soft bodies | Soft robotics, tissue interaction |
| **MPM** (Material Point Method) | Granular materials, snow, soil | Sand/soil manipulation |
| **PBD** (Position-Based Dynamics) | Cloth, rope, liquids | Cloth folding, fluid pouring |
| **SPH** (Smoothed Particle Hydrodynamics) | Water, fluids | Fluid simulation |
| **IPC** (Incremental Potential Contact) | Accurate contact for cloth/soft | Cloth teleoperation |
| **SAP** (Spatial Hashing) | Fast broad-phase collision | Grasp planning |
| **Coupler** | Multi-physics coupling | Cloth on rigid, rigid+MPM |

### 3.3 Rendering Options

| Renderer | Type | Best For |
|----------|------|----------|
| **Nyx** | Ray-tracing (photo-realistic) | Training vision-based policies |
| **Luisa** | DSL ray-tracer | Custom rendering pipelines |
| **Pyrender** | Rasterization | Fast visualization |

---

## 4. Actual Code Examples

### 4.1 Installation

```bash
# PyPI (stable)
pip install genesis-world

# Latest from git
pip install git+https://github.com/Genesis-Embodied-AI/genesis-world.git

# Development mode
git clone https://github.com/Genesis-Embodied-AI/genesis-world.git
cd genesis-world
pip install -e ".[dev]"
```

### 4.2 Basic Scene Setup — Franka Cube Manipulation

Real code from `examples/rigid/franka_cube.py`:

```python
import argparse
import numpy as np
import genesis as gs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu, precision="32")
    
    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
    )
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
    )
    
    ########################## build ##########################
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )

    franka.control_dofs_position(qpos[:-2], motors_dof)

    # hold
    for i in range(100):
        print("hold", i)
        scene.step()

    # grasp
    finder_pos = -0.0
    for i in range(100):
        print("grasp", i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(200):
        print("lift", i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()

if __name__ == "__main__":
    main()
```

### 4.3 Robot Control — PD Control Example

Real code from `examples/tutorials/control_your_robot.py`:

```python
import os
import numpy as np
import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build()

joints_name = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2"
)
motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]

############ Optional: set control gains ############
# set positional gains
franka.set_dofs_kp(
    kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local=motors_dof_idx,
)
# set velocity gains
franka.set_dofs_kv(
    kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local=motors_dof_idx,
)
# set force range for safety
franka.set_dofs_force_range(
    lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    dofs_idx_local=motors_dof_idx,
)

# Hard reset
for i in range(150):
    if i < 50:
        franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), motors_dof_idx)
    elif i < 100:
        franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), motors_dof_idx)
    else:
        franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), motors_dof_idx)
    scene.step()

# PD control loop
for i in range(horizon):
    if i == 0:
        franka.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), motors_dof_idx
        )
    elif i == 250:
        franka.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), motors_dof_idx
        )
    elif i == 500:
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), motors_dof_idx
        )
    elif i == 750:
        # velocity control mode
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:], motors_dof_idx[1:]
        )
        franka.control_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1], motors_dof_idx[:1]
        )
    elif i == 1000:
        franka.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), motors_dof_idx
        )
    
    print("control force:", franka.get_dofs_control_force(motors_dof_idx))
    print("internal force:", franka.get_dofs_force(motors_dof_idx))
    scene.step()
```

### 4.4 Key API Patterns

| Pattern | Code |
|---------|------|
| **Initialize** | `gs.init(backend=gs.gpu)` |
| **Create Scene** | `scene = gs.Scene(...)` |
| **Add Entity** | `scene.add_entity(gs.morphs.MJCF(file="robot.xml"))` |
| **Build** | `scene.build()` |
| **Step** | `scene.step()` |
| **Position Control** | `robot.control_dofs_position(qpos, dofs_idx)` |
| **Velocity Control** | `robot.control_dofs_velocity(qvel, dofs_idx)` |
| **Force Control** | `robot.control_dofs_force(force, dofs_idx)` |
| **Inverse Kinematics** | `robot.inverse_kinematics(link, pos, quat)` |
| **Get DOF** | `robot.get_dofs_position(dofs_idx)` |

---

## 5. Cross-Comparison with Alternative Simulators

| Feature | Genesis World | Isaac Gym | MuJoCo | PyBullet |
|---------|--------------|----------|-------|---------|
| **Speed** | 10-80x faster | Fast | 1x | 1x |
| **Multi-physics** | ✅ (all-in-one) | ❌ | ❌ | ❌ |
| **Differentiable** | ✅ | ❌ | Partial | ❌ |
| **Cross-platform** | ✅ | NVIDIA only | ✅ | ✅ |
| **Sensors** | Built-in | Limited | Limited | Limited |
| **Python-only** | ✅ | ✅ | ❌ | ✅ |
| **Open Source** | ✅ | Proprietary | ✅ | ✅ |

---

## 6. When to Use Genesis

### 6.1 Great for:

- **Learning-based manipulation** (grasp, push, cloth folding)
- **Sim-to-real transfer research** (differentiable physics)
- **Large-scale data generation** (1000s of parallel envs)
- **Fluid/granular manipulation tasks**
- **Differentiable RL / sim-to-real**

### 6.2 Consider alternatives if:

- You need MuJoCo-specific features (native ROS integration)
- You already have Isaac Gym workflows (lock-in to NVIDIA)
- You need physics verification (Genesis prioritizes speed)

---

## 7. Quick Reference Decision Table

| Scenario | Recommendation |
|----------|--------------|
| Robot manipulation learning | Genesis ✅ |
| Cloth/fluid simulation | Genesis ✅ |
| Fast rigid body sim (NVIDIA) | Isaac Gym |
| Physics verification | MuJoCo |
| Easy prototyping | PyBullet |
| Differentiable RL | Genesis ✅ |
| Cross-platform (AMD/Apple) | Genesis ✅ |

---

## 8. Study Plan

### Phase 1: Setup & Basics (Week 1)

| Day | Topic | Activity |
|-----|------|---------|
| 1-2 | Installation | Install Genesis, verify with `examples/rigid/single_franka.py` |
| 3-4 | Core API | Read docs: scene, entities, stepping |
| 5-7 | Simple examples | Run & modify: cube manipulation, joint control |

### Phase 2: Physics & Sensors (Week 2)

| Day | Topic | Activity |
|-----|------|---------|
| 8-9 | Rigid body dynamics | Explore collision, constraints |
| 10-11 | Multi-physics intro | Run cloth, MPM examples |
| 12-14 | Sensors | LiDAR, tactile, IMU — read sensor API |

### Phase 3: Control & RL (Week 3)

| Differentiable IK | Implement diff-IK controller |
| Domain randomization | Run `domain_randomization.py` |
| RL integration | Try training with PyTorch (actor-critic) |

### Phase 4: Advanced (Week 4+)

| Custom environments | Build your own manipulation task |
| Nyx rendering | Explore photo-realistic sensing |
| Differentiable simulation | Backprop through physics |
| Multi-physics coupling | Combine rigid + fluid + cloth |

### Recommended Resources

- **Docs**: https://genesis-world.readthedocs.io/
- **Examples**: `/examples/` in repo
- **Nyx**: https://github.com/Genesis-Embodied-AI/genesis-nyx
- **Quadrants**: https://github.com/Genesis-Embodied-AI/quadrants
- **Discord**: https://discord.gg/nukCuhB47p

---

## 9. References & Resources

- **GitHub**: https://github.com/Genesis-Embodied-AI/genesis-world
- **PyPI**: https://pypi.org/project/genesis-world/
- **Docs**: https://genesis-world.readthedocs.io/
- **Discord**: https://discord.gg/nukCuhB47p

---

*Survey compiled: 2026-06-09*
*Updated with real code examples: 2026-06-10*
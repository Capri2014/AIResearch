# Genesis World: Comprehensive Survey & Study Plan

**Document Version:** 1.0  
**Date:** June 2026  
**Target Audience:** Researchers and engineers seeking to learn and use Genesis for embodied AI, robotics, and simulation research

---

## Table of Contents

1. [Project Overview & Intuition](#1-project-overview--intuition)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How It Works — Architecture Deep Dive](#3-how-it-works--architecture-deep-dive)
4. [The Math & Technical Details](#4-the-math--technical-details)
5. [Code Examples](#5-code-examples)
6. [Cross-Comparison with Alternative Simulators](#6-cross-comparison-with-alternative-simulators)
7. [When to Use Genesis](#7-when-to-use-genesis)
8. [Quick Reference Decision Table](#8-quick-reference-decision-table)
9. [Study Plan](#9-study-plan)
10. [References & Resources](#10-references--resources)

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
| GitHub Stars | Active and growing (check: github.com/Genesis-Embodied-AI/genesis-world) |
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
- MuJoCo MJX: GPU-accelerated, but still 10x slower than Genesis in benchmarks
- Habitat: Fast visual simulation, but limited physics fidelity

The Genesis team reports **10-80x speedup** over existing GPU-accelerated simulators for equivalent fidelity.

### 2.3 The Sim-to-Real Gap

Simulation is only useful if it **generalizes to the real world**. The blog post states:

> *"After this work, simulation evaluation correlates with on-hardware rollouts at 89%, and our reality gap is 45% smaller, measured by FID score on our dataset, than the next-best alternative simulator."*

This correlation (Pearson = 0.8996) means Genesis can be trusted for meaningful evaluation.

### 2.4 The Differentiability Problem

Many research applications require **gradient-based optimization** (RL, optimal control, imitation learning). Traditional simulators either:

- Don't support autodiff at all
- Require manual gradient computation
- Are extremely slow when computing gradients

Genesis provides **built-in differentiable simulation** through the Quadrants compiler's autodiff system.

---

## 3. How It Works — Architecture Deep Dive

### 3.1 The Four-Layer Architecture

Genesis World occupies **four layers**, each building on the one below:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER CODE / APPLICATION                        │
│  (robotics environments, ML pipelines, data generation, agents)   │
├─────────────────────────────────────────────────────────────────┤
│              SIMULATION INTERFACE (Python API)                    │
│  • Asset parsing (URDF, MJCF, OBJ, GLB, USD, …)                │
│  • Entity accessors, controllers, sensors                        │
│  • Parallel & heterogeneous environments                         │
│  • Built-in GUI                                             │
├─────────────────────────────────────────────────────────────────┤
│              PHYSICS LAYER (Unified Multi-Physics)               │
│  • Rigid body dynamics                                        │
│  • FEM (Finite Element Method) for elastic deformables       │
│  • MPM (Material Point Method) for granular materials           │
│  • PBD (Position Based Dynamics) for cloth/fast liquids      │
│  • SPH (Smoothed Particle Hydrodynamics) for fluids            │
│  • IPC (Incremental Potential Contact) for intersection-free │
│  • SAP (Semi-Analytic Primal) coupler                        │
│  • Explicit couplers for multi-physics coupling              │
├─────────────────────────────────────────────────────────────────┤
│              RENDER LAYER (Camera Sensors)                     │
│  • Nyx (in-house path tracer for robotics)                    │
│  • Luisa (DSL ray tracer)                                    │
│  • Pyrender (rasterizer)                                     │
├──────���─��────────────────────────────────────────────────────────┤
│              COMPILER LAYER (Quadrants)                        │
│  • Python → CUDA / AMD ROCm / Apple Metal / Vulkan / x86 / ARM64 │
│  • Autodiff & backpropagation                                 │
│  • GPU graphs for optimized kernel sequences                 │
│  • Fastcache for reduced warm-load times                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 The Physics Engine — Multi-Solver Integration

Genesis integrates **multiple physics solvers** that can run in the same scene:

| Solver | Acronym | Use Case | Key Feature |
|--------|---------|----------|------------|
| **Rigid** | - | Articulated robots, rigid objects | MJCF/URDF/USD support |
| **FEM** | Finite Element Method | Elastic deformables, soft tissues | Hard & soft constraints |
| **MPM** | Material Point Method | Granular materials, snow, sand | Coupling with rigid |
| **PBD** | Position Based Dynamics | Cloth, fast liquids | Stability, speed |
| **SPH** | Smoothed Particle Hydrodynamics | Water, fluids | Navier-Stokes approximation |
| **IPC** | Incremental Potential Contact | Delicate deformables | Intersection-free contact |
| **SAP** | Semi-Analytic Primal | Rigid + deformable coupling | Hydroelastic contact |

### 3.3 The Coupling System — Why It Matters

Real-world manipulation involves **multiple physics modes simultaneously**:

- A robot picking up a **cloth** (rigid + PBD)
- A humanoid walking on **sand** (rigid + MPM)
- A robot pushing through **granular material** (rigid + SPH)

Genesis provides **three interchangeable couplers**:

1. **Fast general-purpose coupler** — Quick coupling for most scenarios
2. **Drake-style SAP coupler** — Semi-analytic primal with hydroelastic contact
3. **IPC coupler** — Incremental Potential Contact with intersection-free guarantee

**One-line switching** between couplers without changing assets, sensors, or policy interface.

### 3.4 The Rendering Engine — Nyx

Nyx is Genesis's **in-house path tracer** designed specifically for robotics:

| Feature | Implementation |
|---------|---------------|
| Rendering method | Path tracing (baseline), with rasterization shortcuts |
| Performance target | 4ms per 1080p frame on high-end consumer GPU |
| Materials | PBR (Physically Based Rendering) |
| Lighting | HDRI + analytic lights |
| 3D Gaussian Splat | Supported for mesh reconstruction |
| Multi-camera | Batch rendering for parallel environments |
| Object picking | Per-pixel picking for interaction |

**Why not use existing renderers?**

- Game engines optimize for visual appeal with baking
- Offline renderers are accurate but minutes per frame
- Neither fits "millions of frames for policy evaluation at scale"

### 3.5 The Compiler — Quadrants

Quadrants is the **high-performance compiler** that makes Genesis fast:

- **Forked from Taichi** in June 2025
- **Multi-platform:** CUDA, AMD ROCm, Apple Metal, Vulkan, x86, ARM64
- **Autodiff:** Built-in gradient computation
- **GPU Graphs:** Captures kernel sequences for hardware-level optimization
- **Fastcache:** Reduces kernel load from 7.2s → 0.3s

This is what enables Genesis to run **10-80x faster** than competitors.

### 3.6 The Sensor System — Out of the Box

Genesis includes comprehensive sensors:

| Sensor | Use Case |
|--------|----------|
| **Camera** | RGB, depth, segmentation |
| **Tactile** | Physically accurate, differentiable |
| **IMU** | Accelerometer, gyroscope |
| **LiDAR** | 3D depth scanning |
| **Contact Force** | Force/torque at contact points |
| **Surface Distance** | Proximity sensing |
| **Temperature Grid** | Thermal sensing |

All sensors support **parallel and heterogeneous environments**.

---

## 4. The Math & Technical Details

### 4.1 External Articulation Constraint (IPC)

The blog post describes a key innovation: **External Articulation Constraint** that couples IPC with articulated robots:

For an articulated system with $m$ joints:

1. The rigid solver predicts joint displacements: $\tilde{\delta\boldsymbol{\theta}}$
2. Computes joint-space effective mass matrix: $\mathbf{M}^t$
3. Injected into IPC as external articulation kinetic energy:

$$K = \frac{1}{2}\left(\delta\boldsymbol{\theta}(\mathbf{q}, \mathbf{q}^t) - \tilde{\delta\boldsymbol{\theta}}\right)^T \mathbf{M}^t \left(\delta\boldsymbol{\theta}(\mathbf{q}, \mathbf{q}^t) - \tilde{\delta\boldsymbol{\theta}}\right)$$

This allows joint-space forces and contact forces to resolve **simultaneously** rather than staggered between separate solvers.

### 4.2 Performance Benchmarks

From the documentation (claims 10-80x speedup):

| Metric | Genesis | Isaac Gym/MuJoCo MJX | Speedup |
|--------|---------|---------------------|---------|
| Rigid body parallel envs | ~10M steps/sec | ~100K-1M steps/sec | 10-80x |
| Deformable (IPC) | Faster than real-time | Not supported | N/A |
| Multi-physics coupling | Single scene | Requires separate tools | Unified |

**Note:** These are vendor-reported benchmarks. Independent verification recommended.

### 4.3 Sim-to-Real Correlation

From the blog post (zero-shot real-to-sim evaluation):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pearson Correlation | **0.8996** (95% CI: [0.7439, 0.9314]) | Strong correlation between sim and real |
| MMRV (Mean Maximum Rank Violation) | **0.0166** (95% CI: [0.0102, 0.0474]) | Performance rankings preserved |
| Reality gap reduction | **45% smaller** than next-best alternative | Significant improvement |

---

## 5. Code Examples

### 5.1 Basic Setup — Hello Genesis

```python
import genesis as gs

# Initialize Genesis with your backend
gs.init(backend=gs.cuda)  # Options: cuda, cpu, amdgpu, metal, vulkan

# Create a scene
scene = gs.Scene()

# Create a simple rigid cube
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.1, 0.1, 0.1),
        pos=(0, 0, 0.05),
    ),
    gs.materials.Rigid(),
)

# Create a ground plane
ground = scene.add_entity(
    gs.morphs.Plane(),
    gs.materials.Rigid(friction=0.5),
)

# Build the scene
scene.build()

# Run simulation loop
for _ in range(1000):
    scene.step()
```

### 5.2 Loading a Robot (Franka Panda)

```python
import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene()

# Load a robot from URDF
franka = scene.add_entity(
    gs.morphs.URDF(
        file="franka_panda.urdf",
        pos=(0, 0, 0),
    ),
    gs.materials.Rigid(),
    fix_base=True,
)

# Add a cube to manipulate
cube = scene.add_entity(
    gs.morphs.Box(size=(0.05, 0.05, 0.05)),
    gs.materials.Rigid(),
)

scene.build()

# Control the robot
for i in range(1000):
    # Move to target position
    franka.set_joint_pos(target_pos)
    scene.step()
```

### 5.3 Multi-Physics Coupling — Cloth on Rigid

```python
import genesis as gs

gs.init(gs.cuda)
scene = gs.Scene()

# Rigid robot
robot = scene.add_entity(
    gs.morphs.URDF("franka_panda.urdf"),
    gs.materials.Rigid(),
)

# PBD cloth
cloth = scene.add_entity(
    gs.morphs.Mesh(
        file="cloth.obj",
        scale=(0.5, 0.5, 0.5),
    ),
    gs.materials.PBD(
        density=0.1,
        stiffness=0.9,
    ),
)

# The coupler handles physics automatically
scene.build()

for _ in range(1000):
    scene.step()
```

### 5.4 Using Sensors — Camera + Tactile

```python
import genesis as gs

gs.init(gs.cuda)
scene = gs.Scene()

# Add robot
robot = scene.add_entity(gs.morphs.URDF("robot.urdf"))
cube = scene.add_entity(gs.morphs.Box())

# Add camera sensor
camera = scene.add_camera(
    res=(640, 480),
    pos=(0.5, 0.5, 0.5),
    lookat=(0, 0, 0),
)

# Add tactile sensor to gripper
tactile = scene.add_sensor(
    sensor_type="tactile",
    entity=robot,
    link_name="gripper_link",
    res=(32, 32),
)

scene.build()

for _ in range(1000):
    scene.step()
    
    # Get RGB image
    rgb = camera.get_rgb()
    
    # Get tactile reading
    tactile_data = tactile.read()
```

### 5.5 Parallel Environments

```python
import genesis as gs

gs.init(backend=gs.cuda)

# Create batched environments
scene = gs.Scene(
    n_envs=1024,  # 1024 parallel environments
    env_spacing=1.0,
)

# Each environment gets the same objects
# but with different random initializations
for i in range(1024):
    cube = scene.add_entity(
        gs.morphs.Box(),
        gs.materials.Rigid(),
        env_id=i,  # Specify which env
    )

scene.build()

# Run all environments in parallel
for _ in range(100):
    scene.step()
```

### 5.6 Differentiable Simulation

```python
import genesis as gs
import torch

gs.init(backend=gs.cuda)

# Enable differentiation
scene = gs.Scene(differentiate=True)

cube = scene.add_entity(
    gs.morphs.Box(),
    gs.materials.Rigid(),
)

scene.build()

# Forward pass
scene.step()

# Access differentiated tensor
diff_tensor = scene.get_differentiated_tensor()

# Backward pass
diff_tensor.backward()
```

---

## 6. Cross-Comparison with Alternative Simulators

### 6.1 Comprehensive Comparison Table

| Feature | **Genesis World** | **Isaac Sim (NVIDIA)** | **MuJoCo + MJX** | **PyBullet** | **Brax (Google)** | **Habitat** |
|---------|-------------------|------------------------|------------------|--------------|-------------------|-------------|
| **Rigid body** | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | Limited |
| **FEM/Deformable** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **MPM/Granular** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Fluids (SPH)** | ✅ | ✅ (limited) | ❌ | ❌ | ✅ | ❌ |
| **Cloth (PBD)** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Multi-physics coupling** | ✅ (3 couplers) | ❌ | ❌ | ❌ | ❌ | ❌ |
| **GPU acceleration** | ✅ (10-80x faster) | ✅ | ✅ (MJX) | ❌ | ✅ | ❌ |
| **CPU fallback** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Photo-realistic render** | ✅ (Nyx) | ✅ (Omniverse) | ❌ | ❌ | ❌ | ✅ |
| **Python API** | ✅ (Pythonic) | ✅ (Python) | ✅ | ✅ | ✅ | ✅ |
| **Differentiable** | ✅ (Quadrants) | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Parallel envs** | ✅ (1000+) | ✅ | ✅ (MJX) | ✅ | ✅ | ✅ |
| **Sensors (built-in)** | ✅ (comprehensive) | ✅ | ✅ (basic) | ✅ | ✅ | ✅ |
| **Robot formats** | URDF, MJCF, USD, OBJ | USD, URDF | MJCF, XML | URDF, SDF | ❌ | SUNCG, HM3D |
| **License** | Apache 2.0 | NVIDIA EULA | Apache 2.0 | MIT | Apache 2.0 | MIT |
| **Open source** | ✅ | ❌ (binary) | ✅ | ✅ | ✅ | ✅ |

### 6.2 Detailed Comparison

#### Genesis vs Isaac Sim

| Aspect | Genesis | Isaac Sim |
|--------|---------|----------|
| **Philosophy** | Unified multi-physics | Omniverse platform |
| **Physics scope** | All in one | Requires extensions |
| **Performance** | 10-80x faster (claimed) | Fast |
| **Rendering** | Nyx (custom) | Omniverse RTX |
| **Cost** | Free (Apache 2.0) | Paid (enterprise) |
| **Customization** | Full source | Limited |

**Winner for:** Research, academia, budget-conscious teams

#### Genesis vs MuJoCo + MJX

| Aspect | Genesis | MuJoCo |
|--------|---------|--------|
| **Physics scope** | Multi-physics | Rigid + simple constraints |
| **Deformables** | ✅ (FEM, IPC) | Limited |
| **Fluids** | ✅ | ❌ |
| **GPU acceleration** | Native | MJX (good) |
| **Speed** | 10-80x faster | Good |
| **Sim-to-real** | 89% correlation | Lower |

**Winner for:** Complex manipulation, deformable objects

#### Genesis vs PyBullet

| Aspect | Genesis | PyBullet |
|--------|---------|----------|
| **Performance** | Much faster | CPU-based |
| **GPU** | ✅ | ❌ |
| **Rendering** | Nyx | Basic |
| **Scope** | Multi-physics | Rigid only |

**Winner for:** Production workloads

#### Genesis vs Habitat

| Aspect | Genesis | Habitat |
|--------|---------|----------|
| **Focus** | Physics simulation | Visual simulation |
| **Physics fidelity** | High | Low |
| **Realistic rendering** | ✅ | ✅ |
| ** embodied AI** | ✅ | ✅ |

**Winner for:** When physics accuracy matters more than photorealism

### 6.3 Pros/Cons Summary

#### Genesis World Pros

✅ Unified multi-physics (one API for everything)  
✅ 10-80x faster than competitors  
✅ Photo-realistic rendering built-in  
✅ Full differentiability  
✅ Open source (Apache 2.0)  
✅ Comprehensive sensors out of the box  
✅ Heterogeneous parallel environments  

#### Genesis World Cons

❌ Newer project (less mature than MuJoCo)  
❌ Smaller community than established tools  
❌ Some features require additional setup (IPC, Nyx)  
❌ Hardware requirements for best performance  

---

## 7. When to Use Genesis

### 7.1 Ideal Use Cases

**Use Genesis when you need:**

1. **Multi-physics simulation** — Robot manipulating cloth, fluid, or granular material
2. **Fast iteration** — Large-scale RL training, data generation
3. **Unified pipeline** — Single tool for physics + rendering + sensors
4. **Differentiable simulation** — Gradient-based optimization
5. **Sim-to-real transfer** — Evaluation that correlates with real world

### 7.2 When NOT to Use Genesis

**Consider alternatives when:**

1. **Mature, proven physics only** — Use MuJoCo if you need battle-tested rigid body dynamics
2. **Isaac ecosystem** — Already invested in NVIDIA Omniverse
3. **Visual-only simulation** — Use Habitat or Blender for pure visual tasks
4. **Very simple needs** — PyBullet may be simpler for basic rigid body

### 7.3 Decision Flowchart

```
START: Do you need multi-physics coupling?
│
├─ NO → Is speed critical for large-scale training?
│   │
│   ├─ YES → Genesis ✅
│   │
│   └─ NO → Do you need GPU acceleration?
│       │
│       ├─ YES → MuJoCo MJX or Brax
│       │
│       └─ NO → PyBullet or MuJoCo
│
└─ YES → Do you need photo-realistic rendering?
    │
    ├─ YES → Genesis ✅ (with Nyx)
    │
    └─ NO → Genesis ✅
```

---

## 8. Quick Reference Decision Table

| Your Need | Recommended Tool | Alternative |
|----------|---------------|------------|
| **Unified multi-physics** | **Genesis** | MuJoCo + separate tools |
| **Fast rigid body only** | **Genesis** or Brax | Isaac Gym |
| **Deformable objects** | **Genesis** (IPC) | MuJoCo (limited) |
| **Fluids/cloth** | **Genesis** | FLIP solvers |
| **Photo-realistic + physics** | **Genesis** | Isaac Sim |
| **Sim-to-real research** | **Genesis** | MuJoCo |
| **Academic/education** | **Genesis** or MuJoCo | PyBullet |
| **Large-scale data generation** | **Genesis** | Isaac Gym |
| **Differentiable simulation** | **Genesis** | Brax |

---

## 9. Study Plan

### 9.1 Prerequisites

Before diving into Genesis, ensure you have:

| Category | Knowledge | Resources |
|----------|-----------|----------|
| **Programming** | Python proficiency | Python docs |
| **Linear Algebra** | Matrix operations, transforms | 3Blue1Brown Essence of Linear Algebra |
| **Physics** | Rigid body dynamics, basic continuum mechanics | Goldstein Classical Mechanics (ch. 1-4) |
| **ML/RL** | Basic reinforcement learning | Sutton & Barto |
| **Robotics** | URDF, forward/inverse kinematics | Modern Robotics (Lynch & Park) |

### 9.2 Learning Phases

#### Phase 1: Foundation (Week 1-2)

**Goal:** Install Genesis and run basic examples

| Milestone | Tasks | Duration |
|-----------|------|----------|
| **1.1** Install Genesis | `pip install genesis-world` + PyTorch | 2 hours |
| **1.2** Run first example | `examples/rigid/single_franka.py` | 1 hour |
| **1.3** Explore API | Read `genesis-world.readthedocs.io` | 4 hours |
| **1.4** Run 5+ basic examples | Try rigid, collision, rendering examples | 6 hours |

**Files to Read (in order):**

1. README.md — High-level overview
2. docs/getting_started.md — Installation
3. docs/physics/overview.md — Physics architecture
4. examples/tutorials/basic — First tutorials

**Hands-on Exercises:**

- [ ] Run `examples/rigid/franka_cube.py`
- [ ] Run `examples/rendering/follow_entity.py`
- [ ] Modify cube size and observe physics changes

#### Phase 2: Core Physics (Week 3-4)

**Goal:** Understand each physics solver

| Milestone | Tasks | Duration |
|-----------|------|----------|
| **2.1** Rigid body dynamics | Read rigid solver docs, run rigid examples | 4 hours |
| **2.2** FEM for deformables | Run `examples/fem_hard_and_soft_constraint.py` | 3 hours |
| **2.3** MPM for granular | Run `examples/tutorials/mpm.py` | 3 hours |
| **2.4** PBD for cloth | Run `examples/tutorials/pbd_cloth.py` | 3 hours |
| **2.5** SPH for fluids | Run `examples/pbd_liquid.py` | 3 hours |

**Files to Read:**

1. `genesis/physics/solvers/` — Solver implementations
2. `genesis/materials/` — Material definitions
3. `examples/tutorials/` — Comprehensive tutorials

**Hands-on Exercises:**

- [ ] Create a scene with two rigid bodies
- [ ] Create a cloth that falls on a rigid object
- [ ] Create a container filled with granular material

#### Phase 3: Multi-Physics Coupling (Week 5)

**Goal:** Master physics coupling

| Milestone | Tasks | Duration |
|-----------|------|----------|
| **3.1** Coupler architecture | Read coupler docs | 2 hours |
| **3.2** Rigid + cloth coupling | Run `examples/coupling/cloth_on_rigid.py` | 2 hours |
| **3.3** Rigid + MPM coupling | Run `examples/coupling/sand_wheel.py` | 2 hours |
| **3.4** IPC solver | Run `examples/IPC_Solver/ipc_robot_cloth_teleop.py` | 3 hours |
| **3.5** SAP coupler | Run `examples/sap_coupling/franka_grasp_rigid_cube.py` | 2 hours |

**Hands-on Experiments:**

- [ ] Switch between couplers in one scene
- [ ] Create robot + deformable object interaction
- [ ] Experiment with heterogeneous environments

#### Phase 4: Sensors & Perception (Week 6)

**Goal:** Use the sensor system

| Milestone | Tasks | Duration |
|-----------|------|----------|
| **4.1** Camera sensors | Run `examples/sensors/depth_camera_custom_vverts.py` | 2 hours |
| **4.2** Tactile sensors | Run `examples/sensors/tactile_sandbox.py` | 2 hours |
| **4.3** IMU sensors | Run `examples/sensors/imu_franka.py` | 1 hour |
| **4.4** LiDAR | Run `examples/sensors/lidar_teleop.py` | 2 hours |
| **4.5** Contact force | Run `examples/sensors/contact_force_go2.py` | 1 hour |

**Hands-on Exercises:**

- [ ] Attach camera to robot end-effector
- [ ] Use tactile feedback for grasping
- [ ] Build multi-sensor perception pipeline

#### Phase 5: Rendering with Nyx (Week 7)

**Goal:** Master photo-realistic rendering

| Milestone | Tasks | Duration |
|-----------|------|----------|
| **5.1** Install Nyx | `pip install gs-nyx-plugin` | 1 hour |
| **5.2** Basic Nyx | Run `genesis-nyx/examples/01_hello_nyx.py` | 2 hours |
| **5.3** PBR materials | Run `genesis-nyx/examples/03_materials.py` | 2 hours |
| **5.4** Lighting | Run `genesis-nyx/examples/04_light_types.py` | 2 hours |
| **5.5** 3D Gaussian splat | Run `genesis-nyx/examples/05_gaussian_splat.py` | 3 hours |

**Files to Read:**

1. `genesis-nyx/docs/` — Nyx documentation
2. `genesis-nyx/examples/` — All Nyx examples

#### Phase 6: Controls & Controllers (Week 8)

**Goal:** Control robots effectively

| Milestone | Tasks | Duration |
|-----------|------|----------|
| **6.1** Joint control | Run `examples/gui/imgui_joint_control.py` | 2 hours |
| **6.2** Inverse kinematics | Run `examples/tutorials/batched_IK.py` | 3 hours |
| **6.3** Diff-IK | Run `examples/rigid/diffik_controller.py` | 3 hours |
| **6.4** Domain randomization | Run `examples/rigid/domain_randomization.py` | 2 hours |

**Hands-on Projects:**

- [ ] Implement joint position controller
- [ ] Implement end-effector position control via IK

#### Phase 7: Parallel & Heterogeneous Environments (Week 9)

**Goal:** Scale to many parallel environments

| Milestone | Tasks | Duration |
|-----------|------|----------|
| **7.1** Batched environments | Run `examples/rigid/heterogeneous_simulation.py` | 3 hours |
| **7.2** Multi-GPU | Run `examples/ddp_multi_gpu.py` | 4 hours |
| **7.3** Domain randomization | Scale up environment variations | 3 hours |

#### Phase 8: Differentiable Simulation (Week 10)

**Goal:** Use gradients for optimization

| Milestone | Tasks | Duration |
|-----------|------|----------|
| **8.1** Autodiff basics | Read differentiation docs | 2 hours |
| **8.2** Forward-mode AD | Run `examples/differentiable_push.py` | 3 hours |
| **8.3** Custom gradients | Implement custom gradient | 4 hours |

#### Phase 9: Mini-Projects (Week 11-12)

**Goal:** Build complete projects

Choose one of:

| Project | Description | Difficulty |
|---------|------------|-----------|
| **P1: Robotic Grasping** | Train robot to grasp object using RL | Medium |
| **P2: Cloth Folding** | Robot folds cloth autonomously | Hard |
| **P3: Fluid Pouring** | Robot pours fluid into container | Hard |
| **P4: Quadruped Locomotion** | Train Go1 robot to walk | Medium |
| **P5: Drone Hover** | Train drone to hover | Easy |

**Project Structure:**

1. Define environment (Genesis scene + robot)
2. Design reward function
3. Implement RL algorithm
4. Train in simulation
5. Evaluate and iterate

### 9.3 Week-by-Week Timeline Summary

| Week | Focus | Key Deliverables |
|------|-------|---------------|
| 1 | Setup & Basics | Working Genesis, first simulation |
| 2 | API Exploration | Comfortable with core API |
| 3-4 | Physics Solvers | Each solver understood |
| 5 | Coupling | Multi-physics scenes |
| 6 | Sensors | Perception pipeline |
| 7 | Nyx Rendering | Photo-realistic images |
| 8 | Control | Robot controllers |
| 9 | Scaling | Parallel environments |
| 10 | Differentiable Sim | Gradient computation |
| 11-12 | Mini-Project | Complete project |

### 9.4 Resources for Deeper Dives

#### Papers & Technical Reports

| Paper | Link | Relevance |
|-------|------|----------|
| Genesis (original) | github.com/Genesis-Embodied-AI/genesis-world | Core paper |
| IPC Solver | github.com/spiriMirror/libuipc | Deformable contact |
| SimplerEnv | arxiv.org/abs/2405.05941 | Evaluation metrics |

#### Related Repositories

| Repo | Description |
|------|------------|
| genesis-nyx | Nyx renderer |
| quadrants | Compiler |
| libuipc | IPC backend |
| genesis-doc | Documentation |

#### Community

| Resource | Link |
|----------|------|
| Discord | discord.gg/nukCuhB47p |
| GitHub Discussions | github.com/Genesis-Embodied-AI/genesis-world/discussions |
| Issues | github.com/Genesis-Embodied-AI/genesis-world/issues |

---

## 10. References & Resources

### 10.1 Official Links

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/Genesis-Embodied-AI/genesis-world |
| PyPI | https://pypi.org/project/genesis-world/ |
| Documentation | https://genesis-world.readthedocs.io/ |
| Blog Post | https://genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward |
| Discord | https://discord.gg/nukCuhB47p |

### 10.2 Related Projects

| Project | URL |
|---------|-----|
| Nyx Renderer | https://github.com/Genesis-Embodied-AI/genesis-nyx |
| Quadrants | https://github.com/Genesis-Embodied-AI/quadrants |
| libuipc | https://github.com/spiriMirror/libuipc |

### 10.3 Alternative Simulators

| Simulator | URL |
|-----------|-----|
| MuJoCo | https://github.com/google-deepmind/mujoco |
| Isaac Sim | https://developer.nvidia.com/isaac-sim |
| PyBullet | https://github.com/bulletphysics/bullet3 |
| Brax | https://github.com/google/brax |
| Habitat | https://github.com/facebookresearch/habitat-sim |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | June 2026 | Genesis Survey Agent | Initial comprehensive survey |

---

*This document is maintained as part of the Genesis World study resources. Last updated: June 2026*
# Genesis World - Thorough Survey & Study Plan

## 1. What is Genesis World? (Intuition)

Genesis World is a **simulation platform for physical AI and robotics** — think of it as a virtual world where robots can learn to interact with realistic physics. It combines:

- **A multi-physics engine** — simulate rigid bodies, soft tissues, fluids, cloth, sand
- **A photo-realistic renderer** (Nyx) — ray-traced visuals for training vision-based agents
- **A cross-platform compiler** (Quadrants) — speeds up simulation 10-80x faster than existing tools
- **A Pythonic API** — easy to read, extend, and embed in research code

It's designed to scale from a laptop to datacenter GPUs, making it viable for both quick prototyping and large-scale data generation.

---

## 2. The Problem It Solves

Existing simulators have trade-offs:

| Simulator | Strength | Weakness |
|-----------|----------|---------|
| **MuJoCo** | Accurate, widely used | Slow, single-threaded |
| **Isaac Gym** | Fast (GPU), parallel | NVIDIA-only, limited material support |
| **PyBullet** | Easy, free | Slow, less accurate |
| **Drake** | Sophisticated dynamics | Complex API, not GPU-accelerated |

**Genesis tackles this by offering:**
- GPU acceleration (10-80x faster) with multi-backend support (CUDA, AMD, Metal, Vulkan)
- Unified physics (rigid + soft + fluid + cloth) in one scene
- Differentiable simulation for end-to-end RL
- Python-first, fully transparent codebase

---

## 3. Architecture Deep Dive

### Four-Layer Stack

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

### Physics Solvers Explained

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

---

## 4. Actual Code Examples

### 4.1 Installation

```bash
# PyPI (stable)
pip install genesis-world

# Latest from git
pip install git+https://github.com/Genesis-Embodied-AI/genesis-world.git
```

### 4.2 Basic Scene — Franka Cube Manipulation

Real code from `examples/rigid/franka_cube.py`:

```python
import numpy as np
import genesis as gs

# Initialize
gs.init(backend=gs.gpu, precision="32")

# Create scene
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    rigid_options=gs.options.RigidOptions(box_box_detection=True),
    show_viewer=True,
)

# Add entities
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
scene.build()

# Control
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)
franka.set_dofs_kp([100.0, 100.0], fingers_dof)
franka.set_dofs_kv([10.0, 10.0], fingers_dof)

# Move to grasp
end_effector = franka.get_link("hand")
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.135]),
    quat=np.array([0, 1, 0, 0]),
)
)
franka.control_dofs_position(qpos[:-2], motors_dof)

# Simulation loop
for i in range(1000):
    scene.step()
```

### 4.3 PD Control Example

```python
import numpy as np
import genesis as gs

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True)

franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build()

# Get joint indices
joints_name = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "finger_joint2")
motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]

# Set gains
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]), motors_dof_idx)
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]), motors_dof_idx)

# Control loop
for i in range(1000):
    # Position control
    franka.control_dofs_position(
        np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
        motors_dof_idx
    )
    scene.step()
```

### 4.4 Key API Quick Reference

| Pattern | Code |
|---------|------|
| Initialize | `gs.init(backend=gs.gpu)` |
| Create Scene | `scene = gs.Scene(...)` |
| Add Entity | `scene.add_entity(gs.morphs.MJCF(...))` |
| Build | `scene.build()` |
| Step | `scene.step()` |
| Position Control | `robot.control_dofs_position(qpos, dofs_idx)` |
| Velocity Control | `robot.control_dofs_velocity(qvel, dofs_idx)` |
| Force Control | `robot.control_dofs_force(force, dofs_idx)` |
| Inverse Kinematics | `robot.inverse_kinematics(link, pos, quat)` |

---

## 5. Comparison to Alternatives

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

## 6. When to Use Genesis World

**✅ Great for:**
- Training manipulation policies (grasp, push, fold)
- Sim-to-real transfer research
- Large-scale data generation (1000s of envs)
- Fluid/granular manipulation tasks
- Differentiable RL / sim-to-real

**⚠️ Consider alternatives if:**
- You need MuJoCo-specific features (native ROS integration)
- You already have Isaac Gym workflows (lock-in to NVIDIA)
- You need physics verification (Genesis prioritizes speed)

---

## 7. Study Plan

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
- **Discord**: https://discord.gg/nukCuhB47p

---

## 8. Summary

Genesis World is a **next-gen robotics simulator** that unifies physics, rendering, and differentiation in one Pythonic framework. It's fastest in class (10-80x vs existing tools), supports diverse materials, and is fully differentiable — making it ideal for:

- **Learning-based manipulation** (grasp, push, cloth folding)
- **Sim-to-real transfer** (differentiable physics)
- **Large-scale data generation** (parallel GPU envs)

If you're building embodied AI agents, Genesis World is worth the learning curve. Start with the basics, then branch into your specific use case (control, sensing, RL).

---

*Survey compiled: 2026-06-09*
*Updated with real code: 2026-06-10*
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

### Rendering Options

| Renderer | Type | Best For |
|----------|------|----------|
| **Nyx** | Ray-tracing (photo-realistic) | Training vision-based policies |
| **Luisa** | DSL ray-tracer | Custom rendering pipelines |
| **Pyrender** | Rasterization | Fast visualization |

---

## 4. Key Features for Robotics Research

### Sensors (Out of the Box)
- **Tactile** — pressure distribution (differentiable!)
- **IMU** — acceleration/gyro
- **LiDAR** — 3D depth scanning
- **Depth Camera** — RGB-D
- **Contact Force** — wrench at contact points
- **Surface Distance** — proximity sensing
- **Temperature Grid** — thermal sensing

### Differentiable Simulation
- Full autodiff via Quadrants compiler
- Backprop through physics → enables end-to-end RL
- Useful for sim-to-real transfer learning

### Parallel & Heterogeneous Environments
- Run thousands of environments in parallel on GPU
- Mix different robot types in one batch
- Domain randomization built-in

---

## 5. Code Structure & Examples

### Installation
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

### Quick Start
```python
import genesis as gs

# Initialize
gs.init(backend=gs.cuda)  # or gs.vulkan, gs.metal, gs.cpu

# Create scene
scene = gs.Scene()

# Load robot (URDF)
franka = scene.load_asset("franka_panda.urdf")

# Step simulation
for _ in range(1000):
    scene.step()
```

### Example Categories (in repo)

**Physics:**
- `examples/rigid/` — Franka cube manipulation, collision towers
- `examples/fem/` — Soft body constraints
- `examples/tutorials/mpm.py` — Granular materials
- `examples/coupling/` — Multi-physics (cloth+rigid, sand wheel)

**Rendering:**
- `examples/rendering/` — Follow camera, moving camera
- `genesis-nyx/examples/` — Nyx-specific: PBR materials, Gaussian splatting

**Sensors:**
- `examples/sensors/` — LiDAR, tactile, IMU, depth camera, contact force

**Control:**
- `examples/tutorials/control_your_robot.py` — Basic joint control
- `examples/rigid/diffik_controller.py` — Differentiable IK

---

## 6. Comparison to Alternatives

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

## 7. When to Use Genesis World

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

## 9. Summary

Genesis World is a **next-gen robotics simulator** that unifies physics, rendering, and differentiation in one Pythonic framework. It's fastest in class (10-80x vs existing tools), supports diverse materials, and is fully differentiable — making it ideal for:

- **Learning-based manipulation** (grasp, push, cloth folding)
- **Sim-to-real transfer** (differentiable physics)
- **Large-scale data generation** (parallel GPU envs)

If you're building embodied AI agents, Genesis World is worth the learning curve. Start with the basics, then branch into your specific use case (control, sensing, RL).

---

*Survey compiled: 2026-06-09*
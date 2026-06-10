# Genesis World: From Zero to Simulation

**Target:** Someone who knows Python but has never done physics simulation

---

## Day 1: Just Run One Thing

### The Absolute Minimum

```python
# 1 line to start
import genesis as gs
gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())  # floor
scene.add_entity(gs.morphs.Box(pos=(0, 0, 1)))  # box in the air
scene.build()

for _ in range(100):
    scene.step()  # physics runs here — box falls!
```

**What just happened:**
- `gs.init()` starts the simulation engine
- `Scene` is the world — holds everything
- `morphs.Plane()` creates a floor
- `morphs.Box()` creates a box
- `scene.build()` compiles the physics
- `scene.step()` advances physics by one timestep

The box falls because gravity is on by default.

---

## Day 2: What Is Physics Simulation?

### Intuition

A **physics simulator** is like a tiny video game engine that:

1. **Stores positions** of every object (x, y, z)
2. **Applies forces** — gravity, springs, contacts
3. **Updates positions** based on velocities
4. **Handles collisions** — objects don't pass through each other

```
Real world          Simulator
─────────────────────────────
Object position    (x, y, z) array
Gravity           -9.8 m/s² on z-axis
Spring force      F = -k * displacement
Collision        push objects apart
```

### Why It Matters for Robotics

Before putting a robot in the real world (expensive, slow, can break), we:
1. **Train** it in simulation (fast, cheap, infinite resets)
2. **Transfer** to real robot (sim-to-real)

---

## Day 3: Understanding the Genesis API

### Core Concepts

| Concept | What It Is | Genesis Code |
|---------|-----------|--------------|
| **Scene** | The world | `scene = gs.Scene()` |
| **Entity** | An object in the world | `scene.add_entity(...)` |
| **Morph** | Shape/geometry | `gs.morphs.Box(...)`, `gs.morphs.MJCF(...)` |
| **Material** | Physics properties | `gs.materials.Rigid(...)`, `gs.materials.PBD.Cloth()` |
| **Step** | Advance one timestep | `scene.step()` |

### Morphs (Shapes)

```python
# Simple shapes
gs.morphs.Plane()           # infinite floor
gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(x, y, z))
gs.morphs.Sphere(radius=0.05)
gs.morphs.Cylinder(height=1, radius=0.1)

# From files
gs.morphs.MJCF(file="robot.xml")     # MuJoCo format
gs.morphs.URDF(file="robot.urdf")   # ROS format
gs.morphs.OBJ(file="mesh.obj")     # 3D mesh
```

### Materials (Physics)

```python
# Rigid body — solid objects that collide
gs.materials.Rigid()

# PBD Cloth — soft fabric
gs.materials.PBD.Cloth()

# SPH Liquid — water/fluid
gs.materials.SPH.Liquid(mu=0.01)
```

---

## Day 4: Your First Robot

### What's a Robot in Simulation?

A robot = **links** (rigid bodies) + **joints** (connections)

```
    Link 0 (base)
        |
     Joint 0
        |
     Link 1
        |
     Joint 1
        |
     Link 2 (end effector)
```

### Load a Robot

```python
import genesis as gs
gs.init()

scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())  # floor

# Load Franka robot from MuJoCo XML
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

scene.build()
```

### Control Joints

Joints have **positions** (angles). We control them:

```python
import numpy as np

# Get joint indices
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
joint_indices = [franka.get_joint(name).dofs_idx_local[0] for name in joint_names]

# Set target positions (in radians for rotation joints)
target_positions = np.array([0.5, 0.3, 0.0, -0.5, 0.2, 0.1, 0.0])

# Send position command
franka.control_dofs_position(target_positions, joint_indices)

# Run simulation
for _ in range(500):
    scene.step()
```

---

## Day 5: Inverse Kinematics (IK)

### The Problem

We know **WHERE** we want the hand to go (x, y, z). We need to figure out **WHAT JOINT ANGLES** get us there.

This is called Inverse Kinematics.

### Genesis IK

```python
# Get the hand link
hand = franka.get_link("hand")

# Compute joint angles to reach target position
target_pos = np.array([0.5, 0.0, 0.3])  # x, y, z
target_quat = np.array([0, 1, 0, 0])  # rotation (quaternion)

joint_angles = franka.inverse_kinematics(
    link=hand,
    pos=target_pos,
    quat=target_quat
)

# Now control to those angles
franka.control_dofs_position(joint_angles[:-2], joint_indices)

for _ in range(500):
    scene.step()
```

---

## Day 6: Cloth Simulation

### What Is PBD?

**Position-Based Dynamics** — instead of forces, we directly fix positions.

```python
# Create scene with cloth
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.004, substeps=10),
    show_viewer=True
)

# Add cloth mesh from file
cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(file="meshes/cloth.obj", scale=2.0, pos=(0, 0, 0.5)),
    surface=gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0))
)

scene.build()

# Pin corners so it doesn't fall
cloth.fix_particles(cloth.find_closest_particle((-1, -1, 1.0)))
cloth.fix_particles(cloth.find_closest_particle((1, -1, 1.0)))

# Watch it hang!
for _ in range(1000):
    scene.step()
```

---

## Day 7: Sensors

### LiDAR — Laser Scanner

```python
# Add robot
robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf"))

# Add LiDAR sensor
lidar = scene.add_sensor(
    gs.sensors.Lidar(
        pattern=gs.sensors.SphericalPattern(),  # rays in sphere
        entity_idx=robot.idx,
        pos_offset=(0.3, 0.0, 0.1),  # mounted on top
        draw_debug=True  # show rays in viewer
    )
)

scene.build()

# Read sensor
for _ in range(100):
    distances = lidar.read()  # array of distances
    print(f"Min distance: {distances.min()}")
    scene.step()
```

---

## Key Insights to Remember

### 1. Scene Holds Everything

```
scene
├── entities (robots, boxes, cloth)
├── sensors (camera, lidar)
└── physics options
```

### 2. Build Before Step

```python
scene.build()      # create physics world ONCE
for _ in range(1000):
    scene.step()  # advance physics many times
```

### 3. Control Modes

| Mode | What It Does | Use Case |
|-----|--------------|---------|
| Position | Move to angle | Standard control |
| Velocity | Set joint speed | Smooth motion |
| Force | Apply torque | Physical interaction |

### 4. Materials Determine Physics

- `Rigid()` → solid objects, collide
- `PBD.Cloth()` → fabric, stretches
- `SPH.Liquid()` → water, flows

---

## Run Order for Beginners

```
1. pip install genesis-world
2. python examples/rigid/franka_cube.py     # watch first
3. python examples/tutorials/control_your_robot.py  # understand control
4. python examples/tutorials/pbd_cloth.py   # different physics
5. Create your own!
```

---

## What Each Example Teaches

| Example | Teaches |
|---------|--------|
| `franka_cube.py` | IK, grasping, basic control |
| `control_your_robot.py` | PD control, position/velocity/force |
| `pbd_cloth.py` | Cloth physics |
| `sph_rigid.py` | Fluid + rigid coupling |
| `lidar_teleop.py` | Sensors |

---

*Learn by running — start with Day 1 and work up.*
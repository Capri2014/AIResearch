# Genesis World: Complete Beginner's Guide

## Why This Guide Exists

When I first saw physics simulation code, I thought:
- "What are all these parameters?"
- "Why does the box fall?"
- "What even IS inverse kinematics?"
- "How do I get started?"

This guide fixes that. We'll start from absolute zero and build understanding step by step.

---

# Part 1: What IS Physics Simulation?

## The Core Idea (No Code)

A physics simulator is a program that:

1. **Remembers where things are** (position: x, y, z)
2. **Applies forces** (gravity pulls down, springs push back)
3. **Updates positions** (objects move based on velocity)
4. **Handles collisions** (things don't pass through each other)

```
Think of it like a video game:
- Objects have positions
- Each frame: apply forces → update velocity → update position → check collisions
- Repeat 60 times per second = smooth animation
```

## Why Use Simulation for Robotics?

| Real Robot | Simulation |
|-----------|------------|
| Expensive ($100K+) | Free |
| Breaks | Can't break |
| Slow (real-time) | Fast (can speed up 100x) |
| One try | Infinite tries |

**Workflow:**
1. Train policy in simulation
2. Deploy on real robot
3. Fix what breaks in sim
4. Repeat

This is called **sim-to-real** transfer.

---

# Part 2: Your First Simulation

## The Simplest Possible Code

```python
import genesis as gs

gs.init()                           # 1. Start the engine
scene = gs.Scene(show_viewer=True)     # 2. Create the world

# 3. Add a floor
scene.add_entity(gs.morphs.Plane())

# 4. Add a box in the air (x=0, y=0, z=1 meter up)
scene.add_entity(gs.morphs.Box(pos=(0, 0, 1)))

# 5. Build physics world
scene.build()

# 6. Run physics
for _ in range(100):
    scene.step()
```

**What you'll see:** A box appears 1 meter in the air, then falls and hits the floor.

## Understanding Every Line

| Line | What It Does | Why It Matters |
|------|-------------|---------------|
| `gs.init()` | Starts GPU/CPU physics engine | Required before anything |
| `gs.Scene()` | Creates simulation world | Holds all objects |
| `show_viewer=True` | Opens 3D window | See what's happening |
| `gs.morphs.Plane()` | Creates floor | Objects need something to hit |
| `gs.morphs.Box()` | Creates box | Basic object shape |
| `pos=(x, y, z)` | Position in meters | Origin is (0, 0, 0) |
| `scene.build()` | Compiles physics | Must call before stepping |
| `scene.step()` | Advances 1 timestep | Default: 0.01 seconds |

## Common Mistakes

### ❌ Forgetting to build
```python
# WRONG
scene.add_entity(gs.morphs.Box())
scene.step()  # Won't work!

# RIGHT
scene.add_entity(gs.morphs.Box())
scene.build()  # Build first!
scene.step()  # Then step
```

### ❌ Building twice
```python
# WRONG
scene.build()
scene.build()  # Can't build twice!

# RIGHT
scene.build()  # Call once
for _ in range(100):
    scene.step()  # Step many times
```

---

# Part 3: Robots in Genesis

## What IS a Robot?

In simulation, a robot = **links** (rigid parts) + **joints** (connections that move)

```
       Link 0 (base)
          │
       Joint 0 (shoulder) ─── allows rotation
          │
       Link 1 (upper arm)
          │
       Joint 1 (elbow) ─── allows rotation
          │
       Link 2 (forearm)
          │
       Joint 2 (wrist) ─── allows rotation
          │
       Link 3 (hand)
```

Each joint has an **angle** (position). We control joints to move the robot.

## Loading a Robot

Genesis supports two robot formats:

| Format | Description | Use Case |
|--------|------------|---------|
| **MJCF** | MuJoCo XML | Most common in research |
| **URDF** | ROS format | Robot operating systems |

```python
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

# Load Franka Panda robot
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

scene.build()
```

## Finding Joints

Each joint has a name. We need to find the joint indices to control them:

```python
# Get all joint names
joint_names = [
    "joint1",    # shoulder pan
    "joint2",    # shoulder lift
    "joint3",    # elbow
    "joint4",    # wrist 1
    "joint5",    # wrist 2
    "joint6",    # wrist 3
    "joint7",    # wrist 4 (for gripper)
    "finger_joint1",  # gripper finger
    "finger_joint2",  # gripper finger
]

# Get their indices (internal IDs)
joint_indices = []
for name in joint_names:
    joint_indices.append(franka.get_joint(name).dofs_idx_local[0])

# Now joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
```

## Control Modes

There are three ways to control a joint:

| Mode | What It Does | Code |
|------|------------|------|
| **Position** | Move to target angle | `control_dofs_position()` |
| **Velocity** | Set rotation speed | `control_dofs_velocity()` |
| **Force** | Apply torque | `control_dofs_force()` |

### Position Control (Most Common)

```python
import numpy as np

# Target angles (in radians for rotation joints)
target = np.array([0.5, 0.3, 0.0, -0.5, 0.2, 0.1, 0.0, 0.04, 0.04])

# Send command
franka.control_dofs_position(target, joint_indices)

# Run
for _ in range(500):
    scene.step()
```

### Setting Gains

**PD Control** = Proportional-Derivative control:
- **P (Proportional)** = how hard to reach target
- **D (Derivative)** = how hard to stop overshooting

```python
# Set proportional gain (stiffness)
# Higher = stiffer, reaches target faster
franka.set_dofs_kp(
    kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local=joint_indices
)

# Set derivative gain (damping)
# Higher = more damping, less oscillation
franka.set_dofs_kv(
    kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local=joint_indices
)
```

### Reading State

```python
# Read current joint positions
current_positions = franka.get_dofs_position(joint_indices)

# Read applied forces
forces = franka.get_dofs_force(joint_indices)
control_forces = franka.get_dofs_control_force(joint_indices)

print(f"Current: {current_positions}")
print(f"Forces: {forces}")
```

---

# Part 4: Inverse Kinematics (IK)

## The Problem

We know WHERE we want the hand (x, y, z position). We need to find WHAT JOINT ANGLES get us there.

```
Given: hand target position (0.5, 0.0, 0.3)
Find: joint angles [θ1, θ2, θ3, θ4, θ5, θ6, θ7]
```

This is called **Inverse Kinematics (IK)**.

## Genesis IK

```python
# Get the hand link
hand = franka.get_link("hand")

# Target position (x, y, z in meters)
target_pos = np.array([0.5, 0.0, 0.3])

# Target rotation (quaternion: x, y, z, w)
target_quat = np.array([0, 1, 0, 0])

# Compute joint angles!
joint_angles = franka.inverse_kinematics(
    link=hand,
    pos=target_pos,
    quat=target_quat
)

# joint_angles = [0.52, 0.31, -0.19, ...] (the angles to reach target)

# Now control to those angles
franka.control_dofs_position(joint_angles[:-2], joint_indices[:-2])

for _ in range(500):
    scene.step()
```

## IK + Grasping Example

Here's a complete grasp-and-lift sequence:

```python
import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)

# Add environment
scene.add_entity(gs.morphs.Plane())

# Add robot and object
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))

scene.build()

# Get indices
motors = np.arange(7)
fingers = np.arange(7, 9)

# Set gains
franka.set_dofs_kp([100.0, 100.0], fingers)
franka.set_dofs_kv([10.0, 10.0], fingers)

# Phase 1: Move to grasp position
hand = franka.get_link("hand")
grasp_pos = np.array([0.65, 0.0, 0.135])
qpos = franka.inverse_kinematics(link=hand, pos=grasp_pos, quat=np.array([0, 1, 0, 0]))
franka.control_dofs_position(qpos[:-2], motors)

for _ in range(100):  # wait to settle
    scene.step()

# Phase 2: Close fingers
franka.control_dofs_position(np.array([0.0, 0.0]), fingers)
for _ in range(100):
    scene.step()

# Phase 3: Lift
lift_pos = np.array([0.65, 0.0, 0.3])
qpos = franka.inverse_kinematics(link=hand, pos=lift_pos, quat=np.array([0, 1, 0, 0]))
franka.control_dofs_position(qpos[:-2], motors)
for _ in range(200):
    scene.step()
```

---

# Part 5: Cloth & Materials

## Materials Determine Physics

Different materials behave differently:

| Material | Behavior | Use Case |
|----------|----------|---------|
| `Rigid()` | Solid, collides | Boxes, robots |
| `PBD.Cloth()` | Flexible, stretches | Fabric, cloth |
| `SPH.Liquid()` | Flows, splashes | Water, fluids |
| `FEM()` | Deformable | Soft robotics |

## Cloth Simulation

```python
# Create scene with cloth settings
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.004,      # smaller timestep for cloth
        substeps=10,      # more accuracy
    ),
    show_viewer=True
)

scene.add_entity(gs.morphs.Plane())

# Add cloth mesh from file
cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),  # cloth physics
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=2.0,
        pos=(0, 0, 0.5),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0)
    )
)

scene.build()

# Pin corners so it hangs
cloth.fix_particles(cloth.find_closest_particle((-1, -1, 1.0)))
cloth.fix_particles(cloth.find_closest_particle((1, -1, 1.0)))

# Watch it hang!
for _ in range(1000):
    scene.step()
```

### Key Cloth Concepts

| Function | What It Does |
|----------|------------|
| `fix_particles()` | Pin a point (won't move) |
| `find_closest_particle(pos)` | Find particle near position |
| `release_particles()` | Unpin |

## Fluid + Rigid Coupling

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
    sph_options=gs.options.SPHOptions(
        lower_bound=(0.0, -1.0, 0.0),
        upper_bound=(1.0, 1.0, 2.4),
    ),
    show_viewer=True
)

scene.add_entity(gs.morphs.Plane())

# Add water (SPH liquid)
water = scene.add_entity(
    material=gs.materials.SPH.Liquid(mu=0.01),
    morph=gs.morphs.Box(pos=(0.5, 0.0, 0.6), size=(0.9, 1.6, 1.2)),
    surface=gs.surfaces.Default(color=(0.5, 0.7, 0.9, 1.0))
)

# Add rigid body that will interact with fluid
cube = scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True),  # enable coupling
    morph=gs.morphs.Box(pos=(0.5, 0.0, 2.4), size=(0.2, 0.2, 0.2))
)

scene.build()

for _ in range(500):
    scene.step()
```

---

# Part 6: Sensors

## Why Sensors?

Sensors let the robot **perceive** the world:

| Sensor | What It Measures |
|--------|--------------|
| `LiDAR` | Distance to objects (laser) |
| `DepthCamera` | RGB + depth image |
| `Tactile` | Contact pressure |
| `IMU` | Acceleration, rotation |

## LiDAR

```python
# Add robot
robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf"))

# Add LiDAR
lidar = scene.add_sensor(
    gs.sensors.Lidar(
        pattern=gs.sensors.SphericalPattern(),  # rays in sphere
        entity_idx=robot.idx,
        pos_offset=(0.3, 0.0, 0.1),  # mount position
        return_world_frame=True,
        draw_debug=True,  # show rays
    )
)

scene.build()

# Read distances
for _ in range(100):
    distances = lidar.read()  # array of distances
    print(f"Min: {distances.min():.3f}m, Max: {distances.max():.3f}m")
    scene.step()
```

### LiDAR Patterns

```python
# Spherical — rays in a sphere (most common)
gs.sensors.SphericalPattern()

# Grid — rays in a grid
gs.sensors.GridPattern()

# Depth — depth camera image
gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern())
```

## Reading Camera

```python
camera = scene.add_sensor(
    gs.sensors.DepthCamera(
        pattern=gs.sensors.DepthCameraPattern(),
        entity_idx=robot.idx,
        pos_offset=(0.0, 0.0, 0.5),
    )
)

scene.build()

for _ in range(100):
    rgb, depth = camera.read_image()
    # rgb = (H, W, 3) uint8
    # depth = (H, W) float32
    scene.step()
```

---

# Part 7: Debugging Guide

## It's Not Working

### Box falls through floor
- Did you call `scene.build()`?
- Is the floor `fixed=True`? (default is yes for Plane)

### Robot jitters wildly
- Gains too high → lower them
- Gains too low → raise them
- Start with: kp=1000, kv=100

### Robot doesn't move
- Are you using correct joint indices?
- Is the robot `fixed=False`? (should be for base)

### IK fails
- Target unreachable (too far, angle limits)
- Try closer target

### Nothing displays
- `show_viewer=True` in Scene?
- Is GPU working? Try `gs.init(backend=gs.cpu)`

## Print Debug

```python
# List all entities
print(scene.entities)

# List all joints
for joint in franka.joints:
    print(joint.name, joint.dofs_idx_local)

# Print sensor data
print(lidar.read())
```

---

# Part 8: Learning Path

## Week 1: Basics

| Day | Goal | Exercise |
|-----|------|----------|
| 1 | Run code | Run `franka_cube.py` |
| 2 | Understand scene | Add box, sphere, cylinder |
| 3 | Control joints | Move each joint one by one |
| 4 | IK | Reach different positions |
| 5 | Grasping | Pick and place cube |
| 6 | Cloth | Run `pbd_cloth.py` |
| 7 | Sensors | Read LiDAR |

## Common Code Snippets

### Minimal Setup
```python
import genesis as gs
gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(pos=(0, 0, 1)))
scene.build()
for _ in range(100): scene.step()
```

### Load Robot
```python
robot = scene.add_entity(gs.morphs.MJCF(file="path/to/robot.xml"))
scene.build()
joint_indices = [robot.get_joint(n).dofs_idx_local[0] for n in joint_names]
```

### Control Loop
```python
for _ in range(1000):
    robot.control_dofs_position(target, indices)
    scene.step()
```

---

*Start here. Run the code. Then explore.*
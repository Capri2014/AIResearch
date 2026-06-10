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
---

# Appendix A: Study Tutorial - Phase 1 (Week 1)

## Day 1: Installation & First Run

### Install Genesis

```bash
pip install genesis-world
```

If you want the latest from git:

```bash
pip install git+https://github.com/Genesis-Embodied-AI/genesis-world.git
```

### Run Your First Example

```bash
cd genesis-world
python examples/rigid/franka_cube.py --vis
```

You should see a window with a Franka robot and a cube. The robot picks up the cube.

### What Just Happened?

1. `gs.init(backend=gs.gpu)` — Started GPU physics
2. Created scene with camera, viewer
3. Added plane (floor), robot (from MJCF), cube
4. Built physics world
5. Used IK to compute joint angles
6. Controlled robot to reach, grasp, lift cube

### Exercise 1.1: Just Run Code
- Try running other examples in `examples/` folder
- Change camera position in ViewerOptions
- See what changes

---

## Day 2: Core API - Scene, Entities, Stepping

### The Scene Object

```python
scene = gs.Scene(
    # Physics options
    sim_options=gs.options.SimOptions(
        dt=0.01,           # timestep (seconds)
        gravity=(0, 0, -9.8),  # gravity direction
    ),
    
    # Viewer options  
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),  # where camera is
        camera_lookat=(0, 0, 0.5),    # what camera looks at
        camera_fov=30,                    # field of view
    ),
    
    show_viewer=True,  # open 3D window
)
```

### Entity Types

| Type | Code | Description |
|------|------|-------------|
| Floor | `gs.morphs.Plane()` | Infinite ground |
| Box | `gs.morphs.Box(size=(w,h,d), pos=(x,y,z))` | Rectangular box |
| Sphere | `gs.morphs.Sphere(radius, pos)` | Ball |
| Cylinder | `gs.morphs.Cylinder(height, radius, pos)` | Cylinder |
| Capsule | `gs.morphs.Capsule(radius, height, pos)` | Capsule |

### From Files

```python
# MuJoCo format (most common)
scene.add_entity(gs.morphs.MJCF(file="path/to/robot.xml"))

# ROS URDF
scene.add_entity(gs.morphs.URDF(file="path/to/robot.urdf"))

# 3D mesh
scene.add_entity(gs.morphs.Mesh(file="path/to/model.obj"))
```

### Stepping

```python
scene.build()  # MUST call before stepping

# Step once
scene.step()

# Step many times
for _ in range(1000):
    scene.step()
    
# Or use built-in loop
scene.step_n(1000)  # same thing
```

### Exercise 2.1: Create Your Own Scene
1. Create a scene with floor and 3 boxes at different heights
2. Change gravity to be sideways (0, -9.8, 0)
3. Add a ramp (rotated plane)

---

## Day 3: Loading & Controlling a Robot

### Load Robot

```python
# Load from MJCF
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)
```

### Find Joints

```python
# All joints in Franka
joint_names = [
    "joint1", "joint2", "joint3", "joint4", 
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2"
]

# Get indices
joint_indices = []
for name in joint_names:
    joint_indices.append(franka.get_joint(name).dofs_idx_local[0])
```

### Control Modes

```python
import numpy as np

# Position control (most common)
target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04])
franka.control_dofs_position(target, joint_indices)

# Velocity control
velocity = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
franka.control_dofs_velocity(velocity, joint_indices)

# Force control
force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
franka.control_dofs_force(force, joint_indices)
```

### PD Gains

```python
# Proportional gain (stiffness)
kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
franka.set_dofs_kp(kp, joint_indices)

# Derivative gain (damping)
kv = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
franka.set_dofs_kv(kv, joint_indices)
```

### Exercise 3.1: Joint Control
1. Load robot
2. Move joint 1 to position 0.5 radians
3. Then move joint 2 to -0.5
4. Then return to zero

---

## Day 4: Inverse Kinematics

### The Problem

Forward: Given joint angles → where is the hand?
Inverse: Given hand position → what joint angles?

### Genesis IK

```python
# Get hand link
hand = franka.get_link("hand")

# Target: 30cm forward, 15cm up
target_pos = np.array([0.3, 0.0, 0.15])
target_quat = np.array([0, 1, 0, 0])  # rotation (quaternion)

# Solve IK
joint_angles = franka.inverse_kinematics(
    link=hand,
    pos=target_pos,
    quat=target_quat
)

# Now move to those angles
franka.control_dofs_position(joint_angles[:-2], motor_indices)
```

### IK + Control Loop

```python
# Move to a sequence of positions
positions = [
    np.array([0.3, 0.0, 0.1]),
    np.array([0.3, 0.0, 0.2]),
    np.array([0.4, 0.1, 0.2]),
    np.array([0.5, 0.0, 0.15]),
]

for target_pos in positions:
    # Solve IK
    qpos = franka.inverse_kinematics(link=hand, pos=target_pos)
    
    # Move there
    for _ in range(100):
        franka.control_dofs_position(qpos[:-2], motor_indices)
        scene.step()
```

### Exercise 4.1: IK Practice
1. Use IK to reach 5 different positions
2. Move smoothly between them
3. Add a cube and try to touch it with IK

---

## Day 5: Grasping

### Complete Grasp Example

```python
import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

# Add robot and cube
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

# Get hand
hand = franka.get_link("hand")

# ===== PHASE 1: Approach =====
qpos = franka.inverse_kinematics(link=hand, pos=(0.65, 0.0, 0.135))
for _ in range(100):
    franka.control_dofs_position(qpos[:-2], motors)
    scene.step()

# ===== PHASE 2: Lower =====
qpos = franka.inverse_kinematics(link=hand, pos=(0.65, 0.0, 0.08))
for _ in range(100):
    franka.control_dofs_position(qpos[:-2], motors)
    scene.step()

# ===== PHASE 3: Grasp =====
franka.control_dofs_position(np.array([0.0, 0.0]), fingers)
for _ in range(50):
    scene.step()

# ===== PHASE 4: Lift =====
qpos = franka.inverse_kinematics(link=hand, pos=(0.65, 0.0, 0.3))
for _ in range(200):
    franka.control_dofs_position(qpos[:-2], motors)
    scene.step()
```

### Exercise 5.1: Pick and Place
1. Pick up cube
2. Move to another location
3. Release
4. Return to home

---

## Day 6: Cloth Physics

### PBD Cloth

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.004, substeps=10),
    show_viewer=True
)
scene.add_entity(gs.morphs.Plane())

# Add cloth
cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(file="meshes/cloth.obj", scale=2.0),
    surface=gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0))
)

scene.build()

# Pin corners
cloth.fix_particles(cloth.find_closest_particle((-1, -1, 1.0)))
cloth.fix_particles(cloth.find_closest_particle((1, -1, 1.0)))

for _ in range(1000):
    scene.step()
```

### Exercise 6.1: Cloth Experiments
1. Pin only one corner — watch it swing
2. Pin all four corners — it becomes a tent
3. Add a box under the cloth — cloth drapes over it

---

## Day 7: Sensors

### LiDAR

```python
robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf"))

lidar = scene.add_sensor(
    gs.sensors.Lidar(
        pattern=gs.sensors.SphericalPattern(),
        entity_idx=robot.idx,
        pos_offset=(0.3, 0.0, 0.1),
        draw_debug=True,
    )
)
scene.build()

for _ in range(100):
    distances = lidar.read()
    print(f"Objects at: {distances.min():.3f}m to {distances.max():.3f}m")
    scene.step()
```

### Depth Camera

```python
camera = scene.add_sensor(
    gs.sensors.DepthCamera(
        pattern=gs.sensors.DepthCameraPattern(),
        entity_idx=robot.idx,
        pos_offset=(0.0, 0.0, 0.5),
    )
)

for _ in range(100):
    rgb, depth = camera.read_image()
    # rgb = (H, W, 3) RGB image
    # depth = (H, W) depth map
    scene.step()
```

### Exercise 7.1: Sensor Reading
1. Read LiDAR and print distances
2. Use depth camera to save an image
3. Move robot and observe sensor changes

---

# Appendix B: Study Tutorial - Phase 2 (Week 2)

## Rigid Body Dynamics

### Collision Detection

```python
# Enable collision detection
scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        box_box_detection=True,  # box-box collisions
        box_sphere_detection=True,  # box-sphere
    )
)
```

### Constraints

```python
# Fixed joint (doesn't move)
joint = robot.get_joint("joint1")
joint.set_type(gs.joints.Fixed)

# Revolute joint (rotates)
joint.set_type(gs.joints.Revolute)

# Prismatic joint (slides)
joint.set_type(gs.joints.Prismatic)
```

## Multi-Physics Coupling

### Cloth + Rigid

```python
# Cloth entity
cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(file="cloth.obj")
)

# Rigid body that interacts with cloth
obj = scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True),  # enable coupling
    morph=gs.morphs.Box(pos=(0, 0, 0.5))
)
```

### SPH + Rigid

```python
water = scene.add_entity(
    material=gs.materials.SPH.Liquid(mu=0.01),
    morph=gs.morphs.Box(pos=(0.5, 0.0, 0.6), size=(0.9, 1.6, 1.2))
)

cube = scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True, coup_friction=0.0),
    morph=gs.morphs.Box(pos=(0.5, 0.0, 2.4))
)
```

# Appendix C: Study Tutorial - Phase 3 (Week 3)

## Differentiable IK

```python
# Compute gradients through IK
grad = franka.compute_ik_gradient(
    link=hand,
    target_pos,
)

# Use for learning
loss = (end_effector_pos - target_pos).sum()
loss.backward()  # backprop through simulation
```

## Domain Randomization

```python
# Randomize physics parameters
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        gravity=np.random.uniform(-10, -9.8),  # vary gravity
    ),
)

# Randomize object positions
for obj in objects:
    obj.set_pos(np.random.uniform(-0.5, 0.5, 3))
```

## RL Integration (Simple)

```python
import torch

# Simple policy network
policy = torch.nn.Sequential(
    torch.nn.Linear(obs_dim, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, action_dim),
)

# Training loop
for episode in range(1000):
    obs = scene.reset()
    total_reward = 0
    
    for step in range(200):
        # Get action
        action = policy(obs).detach()
        
        # Apply action
        robot.control_dofs_position(action.numpy(), joint_indices)
        scene.step()
        
        # Get reward
        obs = get_observation()
        reward = compute_reward()
        total_reward += reward
        
        # Store in replay buffer
        replay_buffer.push(obs, action, reward)
    
    # Update policy
    update_policy(replay_buffer)
```

# Appendix D: Study Tutorial - Phase 4 (Week 4+)

## Custom Environments

```python
class MyEnv:
    def __init__(self):
        self.scene = gs.Scene(show_viewer=True)
        self.setup()
    
    def setup(self):
        # Add floor, robot, objects
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="robot.xml"))
        self.target = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.05, pos=(0.5, 0, 0.1)))
    
    def reset(self):
        # Randomize positions
        self.scene.build()
        return self.get_observation()
    
    def step(self, action):
        # Apply action
        self.robot.control_dofs_position(action)
        self.scene.step()
        
        # Get obs, reward, done
        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.is_done()
        
        return obs, reward, done
    
    def get_observation(self):
        # Return sensor data, joint positions, etc.
        return np.concatenate([
            self.robot.get_dofs_position(),
            self.target.get_pos(),
        ])
    
    def compute_reward(self):
        # Reward for reaching target
        dist = np.linalg.norm(self.robot.get_end_pos() - self.target.get_pos())
        return -dist
    
    def is_done(self):
        return np.linalg.norm(
            self.robot.get_end_pos() - self.target.get_pos()
        ) < 0.01
```

## Nyx Rendering (Photo-realistic)

```python
scene = gs.Scene(
    renderer=gs.renderers.Nyx(),  # Photo-realistic
    viewer_options=...,
)
```

## Differentiable Simulation

```python
# Forward pass
scene.step()

# Backward pass (differentiable!)
scene.backward(loss)

# Use gradients for RL
loss = compute_loss()
loss.backward()  # backprop through physics
```

---

# Quick Reference: Common Patterns

## Minimal Script
```python
import genesis as gs
gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(pos=(0, 0, 1)))
scene.build()
for _ in range(100): scene.step()
```

## Load Robot
```python
robot = scene.add_entity(gs.morphs.MJCF(file="robot.xml"))
scene.build()
joints = [robot.get_joint(n).dofs_idx_local[0] for n in names]
```

## Control
```python
robot.control_dofs_position(target, joints)
scene.step()
```

## IK
```python
qpos = robot.inverse_kinematics(link=hand, pos=target)
robot.control_dofs_position(qpos, joints)
```

## Sensors
```python
sensor = scene.add_sensor(gs.sensors.Lidar(...))
distances = sensor.read()
```

---

*Start with Appendix A (Week 1). Move at your pace.*

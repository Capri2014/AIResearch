# Genesis World: Executable Study Tutorial

This is a hands-on tutorial. Each section contains code you can copy, save as a .py file, and run.

---

# Week 1: Core Skills

## Day 1: Your First Simulation

Save as `01_basic.py`:

```python
#!/usr/bin/env python3
"""Day 1: Your first physics simulation"""

import genesis as gs

# 1. Initialize the physics engine
gs.init()

# 2. Create a simulation world
scene = gs.Scene(show_viewer=True)

# 3. Add a floor
scene.add_entity(gs.morphs.Plane())

# 4. Add a box that will fall
scene.add_entity(gs.morphs.Box(
    size=(0.1, 0.1, 0.1),  # width, depth, height
    pos=(0.0, 0.0, 1.0)   # x, y, z position (1m up)
))

# 5. Build the physics world
scene.build()

# 6. Run the simulation
print("Box falling... Watch it drop!")
for i in range(300):
    scene.step()
    if i % 50 == 0:
        print(f"Step {i}")

print("Done! The box hit the floor.")
```

Run:
```bash
python 01_basic.py
```

**What you see:** A box appears in the air, falls, and hits the floor.

**What you learn:**
- `gs.init()` starts the engine
- `Scene` holds everything
- `morphs.Box` creates shapes
- `scene.step()` advances physics

---

## Day 2: Adding Multiple Objects

Save as `02_objects.py`:

```python
#!/usr/bin/env python3
"""Day 2: Multiple objects and materials"""

import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)

# Floor
scene.add_entity(gs.morphs.Plane())

# Stack of boxes (they will tumble)
for i in range(3):
    scene.add_entity(gs.morphs.Box(
        size=(0.2, 0.2, 0.2),
        pos=(0.0, 0.0, 0.1 + i * 0.21),  # stacked
        fixed=False
    ))

# A sphere that will roll
scene.add_entity(gs.morphs.Sphere(
    radius=0.1,
    pos=(0.5, 0.0, 0.1)
))

# A cylinder
scene.add_entity(gs.morphs.Cylinder(
    height=0.3,
    radius=0.1,
    pos=(-0.5, 0.0, 0.15)
))

scene.build()

for i in range(500):
    scene.step()
```

**What you learn:**
- Multiple entities
- Different shapes: Box, Sphere, Cylinder
- `fixed=False` means it can move

---

## Day 3: Load a Robot

First, find where Genesis stores robot files:

```python
import genesis as gs
print(gs.__file__)  # shows where genesis is installed
```

Then look in `genesis/assets/` for robot XML files.

Save as `03_robot.py`:

```python
#!/usr/bin/env python3
"""Day 3: Load and control a robot"""

import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)

# Floor
scene.add_entity(gs.morphs.Plane())

# Load Franka robot (check the path in your genesis installation)
# Common locations:
# - xml/franka_emika_panda/panda.xml
# - assets/xml/franka_emika_panda/panda.xml

try:
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
    )
except:
    # Try alternative path
    franka = scene.add_entity(
        gs.morphs.MJCF(file="genesis/assets/xml/franka_emika_panda/panda.xml")
    )

scene.build()

# Get joint names
joint_names = [
    "joint1", "joint2", "joint3", "joint4", 
    "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2"
]

# Get joint indices
joint_indices = []
for name in joint_names:
    try:
        joint_indices.append(franka.get_joint(name).dofs_idx_local[0])
    except:
        print(f"Joint {name} not found")

print(f"Joint indices: {joint_indices}")

# Move to home position
home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04])

# Control loop
for i in range(500):
    franka.control_dofs_position(home, joint_indices)
    scene.step()
```

---

## Day 4: Joint Control Modes

Save as `04_control.py`:

```python
#!/usr/bin/env python3
"""Day 4: Different control modes"""

import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

scene.build()

# Get joints
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
joint_idx = [franka.get_joint(n).dofs_idx_local[0] for n in joint_names]

# Set PD gains (stiffness and damping)
kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000])
kv = np.array([450, 450, 350, 350, 200, 200, 200])
franka.set_dofs_kp(kp, joint_idx)
franka.set_dofs_kv(kv, joint_idx)

# Different targets over time
targets = [
    ([0.5, 0.3, 0.0, -0.5, 0.2, 0.1, 0.0], "Pose 1"),
    ([-0.5, 0.5, 0.5, -1.0, 0.3, 0.5, -0.3], "Pose 2"),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Home"),
    ([0.8, 0.8, 1.0, -1.5, 0.5, 0.8, 0.5], "Pose 3"),
]

for target, name in targets:
    print(f"Moving to {name}...")
    for _ in range(100):
        franka.control_dofs_position(np.array(target), joint_idx)
        scene.step()

print("Done!")
```

---

## Day 5: Inverse Kinematics

Save as `05_ik.py`:

```python
#!/usr/bin/env python3
"""Day 5: Inverse Kinematics - reach any position"""

import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

scene.build()

# Get the hand link
hand = franka.get_link("hand")

# Target positions to reach
targets = [
    (np.array([0.3, 0.0, 0.2]), "Close"),
    (np.array([0.4, 0.2, 0.3]), "Right-Up"),
    (np.array([0.4, -0.2, 0.3]), "Left-Up"),
    (np.array([0.5, 0.0, 0.15]), "Forward-Low"),
]

for target_pos, name in targets:
    print(f"Reaching {name} at {target_pos}...")
    
    # Solve IK
    qpos = franka.inverse_kinematics(
        link=hand,
        pos=target_pos,
        quat=np.array([0, 1, 0, 0])  # identity rotation
    )
    
    # Move there
    motors = np.arange(7)
    for _ in range(150):
        franka.control_dofs_position(qpos[:-2], motors)
        scene.step()

print("IK demo complete!")
```

---

## Day 6: Complete Grasp and Lift

Save as `06_grasp.py`:

```python
#!/usr/bin/env python3
"""Day 6: Complete grasp and lift sequence"""

import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)

# Environment
scene.add_entity(gs.morphs.Plane())

# Robot
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

# Cube to grasp
cube = scene.add_entity(gs.morphs.Box(
    size=(0.04, 0.04, 0.04),
    pos=(0.65, 0.0, 0.02)
))

scene.build()

# Joints
motors = np.arange(7)
fingers = np.arange(7, 9)

# Gripper gains
franka.set_dofs_kp([100.0, 100.0], fingers)
franka.set_dofs_kv([10.0, 10.0], fingers)

hand = franka.get_link("hand")

print("=== Phase 1: Approach ===")
target = np.array([0.65, 0.0, 0.15])
qpos = franka.inverse_kinematics(link=hand, pos=target, quat=np.array([0,1,0,0]))
for _ in range(100):
    franka.control_dofs_position(qpos[:-2], motors)
    scene.step()

print("=== Phase 2: Lower ===")
target = np.array([0.65, 0.0, 0.08])
qpos = franka.inverse_kinematics(link=hand, pos=target, quat=np.array([0,1,0,0]))
for _ in range(100):
    franka.control_dofs_position(qpos[:-2], motors)
    scene.step()

print("=== Phase 3: Grasp ===")
franka.control_dofs_position(np.array([0.0, 0.0]), fingers)
for _ in range(50):
    scene.step()

print("=== Phase 4: Lift ===")
target = np.array([0.65, 0.0, 0.25])
qpos = franka.inverse_kinematics(link=hand, pos=target, quat=np.array([0,1,0,0]))
for _ in range(200):
    franka.control_dofs_position(qpos[:-2], motors)
    scene.step()

print("Grasp complete!")
```

---

## Day 7: Cloth Simulation

Save as `07_cloth.py`:

```python
#!/usr/bin/env python3
"""Day 7: Cloth simulation with PBD"""

import genesis as gs

gs.init()

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.004,     # smaller timestep for cloth
        substeps=10,    # more accuracy
    ),
    show_viewer=True
)

scene.add_entity(gs.morphs.Plane())

# Cloth material
cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",  # check path in your installation
        scale=2.0,
        pos=(0, 0, 0.5),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0)
    )
)

scene.build()

# Pin two corners
cloth.fix_particles(cloth.find_closest_particle((-1, -1, 1.0)))
cloth.fix_particles(cloth.find_closest_particle((1, -1, 1.0)))

print("Simulating cloth...")
for i in range(1000):
    scene.step()
    if i % 200 == 0:
        print(f"Step {i}")

print("Cloth done!")
```

---

# Week 2: Sensors & Advanced

## Day 8: LiDAR Sensor

Save as `08_lidar.py`:

```python
#!/usr/bin/env python3
"""Day 8: LiDAR sensor"""

import numpy as np
import genesis as gs

gs.init()

scene = gs.Scene(
    sim_options=gs.options.SimOptions(gravity=(0, 0, -1)),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(-3, 0, 2),
        camera_lookat=(0, 0, 0.5)
    ),
    show_viewer=True
)

scene.add_entity(gs.morphs.Plane())

# Add some obstacles
for i in range(8):
    angle = i * np.pi / 4
    x = 2 * np.cos(angle)
    y = 2 * np.sin(angle)
    scene.add_entity(gs.morphs.Cylinder(
        height=1, radius=0.1,
        pos=(x, y, 0.5), fixed=True
    ))

# Robot (or simple box)
robot = scene.add_entity(gs.morphs.Box(
    size=(0.1, 0.1, 0.1),
    pos=(0, 0, 0.2), fixed=True
))

# LiDAR sensor
lidar = scene.add_sensor(
    gs.sensors.Lidar(
        pattern=gs.sensors.SphericalPattern(),
        entity_idx=robot.idx,
        pos_offset=(0, 0, 0.1),
        draw_debug=True
    )
)

scene.build()

print("LiDAR reading distances...")
for i in range(200):
    distances = lidar.read()
    if i % 20 == 0:
        valid = distances[distances > 0]
        if len(valid) > 0:
            print(f"Step {i}: min={valid.min():.3f}m, max={valid.max():.3f}m, count={len(valid)}")
    scene.step()
```

---

## Day 9: Camera Sensor

Save as `09_camera.py`:

```python
#!/usr/bin/env python3
"""Day 9: Depth camera"""

import genesis as gs

gs.init()

scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

# Add objects to see
scene.add_entity(gs.morphs.Box(pos=(1, 0, 0.5), size=(0.3, 0.3, 0.3)))
scene.add_entity(gs.morphs.Sphere(radius=0.2, pos=(-1, 0.5, 0.2)))

# Robot with camera
robot = scene.add_entity(gs.morphs.Box(pos=(0, 0, 0.2)))

# Depth camera
camera = scene.add_sensor(
    gs.sensors.DepthCamera(
        pattern=gs.sensors.DepthCameraPattern(),
        entity_idx=robot.idx,
        pos_offset=(0, 0, 0.5),
    )
)

scene.build()

print("Reading camera...")
for i in range(100):
    rgb, depth = camera.read_image()
    if i % 20 == 0:
        print(f"RGB shape: {rgb.shape if rgb is not None else 'None'}")
        print(f"Depth shape: {depth.shape if depth is not None else 'None'}")
        if depth is not None:
            print(f"Depth range: {depth.min():.3f} to {depth.max():.3f}")
    scene.step()
```

---

## Day 10: Fluid + Rigid Coupling

Save as `10_fluid.py`:

```python
#!/usr/bin/env python3
"""Day 10: SPH fluid interacting with rigid body"""

import genesis as gs

gs.init()

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
    sph_options=gs.options.SPHOptions(
        lower_bound=(0, -1, 0),
        upper_bound=(1, 1, 2.5),
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2, -2, 2),
        camera_lookat=(0.5, 0, 0.5)
    ),
    show_viewer=True
)

scene.add_entity(gs.morphs.Plane())

# SPH Liquid
water = scene.add_entity(
    material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
    morph=gs.morphs.Box(
        pos=(0.5, 0, 0.6),
        size=(0.8, 1.5, 1.0)
    ),
    surface=gs.surfaces.Default(color=(0.3, 0.6, 0.9, 0.8))
)

# Rigid body that falls into water
cube = scene.add_entity(
    material=gs.materials.Rigid(needs_coup=True, coup_friction=0.0),
    morph=gs.morphs.Box(
        pos=(0.5, 0, 2.2),
        size=(0.2, 0.2, 0.2),
        euler=(30, 20, 0)
    )
)

scene.build()

print("Fluid simulation with coupling...")
for i in range(500):
    scene.step()
    if i % 100 == 0:
        print(f"Step {i}")

print("Fluid demo done!")
```

---

# Week 3: Control & RL

## Day 11: PD Control Deep Dive

Save as `11_pd.py`:

```python
#!/usr/bin/env python3
"""Day 11: Understanding PD control"""

import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

scene.build()

# Get joints
joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
joint_idx = [franka.get_joint(n).dofs_idx_local[0] for n in joints]

# Different gain settings to see the effect
gain_sets = [
    (np.array([100, 100, 100, 100, 100, 100, 100]),   # weak
    (np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000]),  # medium
    (np.array([5000, 5000, 5000, 5000, 5000, 5000, 5000]),  # strong
]

target = np.array([0.5, 0.3, 0.0, -0.5, 0.2, 0.1, 0.0])

for i, (kp,) in enumerate(gain_sets):
    kv = kp / 10  # damping = 10% of proportional
    franka.set_dofs_kp(kp, joint_idx)
    franka.set_dofs_kv(kv, joint_idx)
    
    print(f"Gain set {i+1}: kp={kp[0]}, kv={kv[0]}")
    for _ in range(100):
        franka.control_dofs_position(target, joint_idx)
        scene.step()
```

---

## Day 12: Velocity Control

Save as `12_velocity.py`:

```python
#!/usr/bin/env python3
"""Day 12: Velocity control"""

import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

scene.build()

joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
joint_idx = [franka.get_joint(n).dofs_idx_local[0] for n in joints]

# Velocity control - move joints at constant speed
velocities = [
    ([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Joint 1 forward"),
    ([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], "Joint 2 forward"),
    ([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0], "Joint 3 forward"),
    ([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Joint 1 backward"),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Stop"),
]

for vel, name in velocities:
    print(f"Velocity: {name}")
    for _ in range(50):
        franka.control_dofs_velocity(np.array(vel), joint_idx)
        scene.step()
```

---

## Day 13: Force Control

Save as `13_force.py`:

```python
#!/usr/bin/env python3
"""Day 13: Force/torque control"""

import numpy as np
import genesis as gs

gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
)

scene.build()

joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
joint_idx = [franka.get_joint(n).dofs_idx_local[0] for n in joints]

# Force control - apply torque directly
forces = [
    ([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Torque joint 1"),
    ([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Torque joint 2"),
    ([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0], "Torque joint 3"),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "No torque"),
]

for force, name in forces:
    print(f"Force: {name}")
    for _ in range(50):
        franka.control_dofs_force(np.array(force), joint_idx)
        scene.step()
```

---

## Day 14: Simple RL Environment

Save as `14_rl_env.py`:

```python
#!/usr/bin/env python3
"""Day 14: Simple RL environment structure"""

import numpy as np
import genesis as gs

class SimpleReachEnv:
    """Simple reaching environment for RL"""
    
    def __init__(self):
        gs.init()
        self.scene = gs.Scene(show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane())
        
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.target = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.05, pos=(0.4, 0, 0.1))
        
        self.scene.build()
        
        # Joint indices
        self.joints = np.arange(7)
        
        # Get hand
        self.hand = self.robot.get_link("hand")
    
    def reset(self):
        """Reset environment"""
        # Could randomize here
        return self._get_obs()
    
    def step(self, action):
        """Apply action, return obs, reward, done"""
        # Action is target joint positions (7 joints)
        self.robot.control_dofs_position(action, self.joints)
        self.scene.step()
        
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()
        
        return obs, reward, done
    
    def _get_obs(self):
        """Get observation"""
        joint_pos = self.robot.get_dofs_position(self.joints)
        hand_pos = self.hand.get_pos()
        target_pos = self.target.get_pos()
        return np.concatenate([joint_pos, hand_pos, target_pos])
    
    def _get_reward(self):
        """Reward = negative distance to target"""
        hand_pos = self.hand.get_pos()
        target_pos = self.target.get_pos()
        dist = np.linalg.norm(hand_pos - target_pos)
        return -dist
    
    def _is_done(self):
        """Done when close enough"""
        hand_pos = self.hand.get_pos()
        target_pos = self.target.get_pos()
        return np.linalg.norm(hand_pos - target_pos) < 0.02


# Test the environment
print("Creating environment...")
env = SimpleReachEnv()

print("Running episodes...")
for episode in range(3):
    obs = env.reset()
    total_reward = 0
    
    for step in range(50):
        # Random action (replace with policy in real RL)
        action = np.random.uniform(-0.5, 0.5, 7)
        
        obs, reward, done = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode+1}: reward={total_reward:.3f}")

print("RL env demo done!")
```

---

# Quick Reference

## Common Tasks

| Task | Code |
|------|-----|
| Initialize | `gs.init()` |
| Create world | `scene = gs.Scene(show_viewer=True)` |
| Add floor | `scene.add_entity(gs.morphs.Plane())` |
| Add box | `scene.add_entity(gs.morphs.Box(size=(w,h,d), pos=(x,y,z))` |
| Load robot | `scene.add_entity(gs.morphs.MJCF(file="path.xml"))` |
| Build | `scene.build()` |
| Step | `scene.step()` |
| Control | `robot.control_dofs_position(target, joints)` |
| IK | `robot.inverse_kinematics(link, pos, quat)` |

## File Paths (check your installation)

```
genesis/
├── xml/
│   └── franka_emika_panda/panda.xml
├── meshes/
│   └── cloth.obj
├── urdf/
│   └── go2/urdf/go2.urdf
└── examples/
    └── ...
```

---

*Run one script per day. Start with 01_basic.py.*

# GitHub Open Source Robotic Arm Projects Survey

**Date:** 2026-02-21  
**Surveyed by:** Agent (pipeline)  
**Source:** WeChat article - "GitHub硬核开源机械臂项目" (Hardcore Open Source Robotic Arm Projects on GitHub)

## TL;DR

A comprehensive survey of 7 mature open-source robotic arm projects ranging from desktop to industrial-grade. AR4 and Dummy-Robot are recommended as the two "top-tier" projects for different learning paths.

## Project Overview

| Project | Type | Complexity | Target Use Case |
|---------|------|------------|-----------------|
| AR4 | 6-axis desktop industrial | ⭐⭐⭐⭐ | Desktop collaborative platform |
| Dummy-Robot | 6-axis mini | ⭐⭐⭐⭐ | Full-stack learning |
| Niryo One | 6-axis 3D printed | ⭐⭐⭐ | Education/lab |
| BCN3D Moveo | 6-axis 3D printed | ⭐⭐⭐ | Reproducible builds |
| PAROL6 | 6-axis desktop | ⭐⭐⭐⭐ | Desktop industrial + GUI |
| SO-ARM100 | Standard 6-axis | ⭐⭐⭐ | AI data collection |
| MeArm | 4-DOF pocket | ⭐⭐ | STEAM education |

## Detailed Project Surveys

### 1. AR4 (Annin Robotics)

**Overview:** Desktop-grade "industrial style" 6-axis open-source arm with the most complete documentation system.

**Repository:** https://anninrobotics.com/downloads/

**What's Included:**
- Control application (Python source code)
- Teensy/Arduino firmware
- 3D printable STL files
- Assembly manual

**Best For:** Those who want a "desktop industrial collaborative arm platform" with complete planning + simulation + control chain.

**Key Features:**
- Industrial design aesthetic
- Complete mechanical + electrical + software stack
- One-stop download

### 2. Dummy-Robot (稚晖君)

**Overview:** Ultra-mini but full-stack hardcore project. Highest learning and secondary development value.

**Repository:** https://github.com/peng-zhihui/Dummy-Robot

**What's Included:**
- Core firmware
- Algorithm structure (detailed)
- Driver modules
- Engineering organization methods

**Best For:** Systematically learning "how to productize a complete robotic arm project" (hardware + firmware + upper computer + toolchain).

**Key Features:**
- Detailed firmware/algorithm explanation
- Covers firmware, algorithms, drivers
- Good for learning engineering best practices
- Full-stack reference implementation

### 3. Niryo One

**Overview:** 3D printed 6-axis open collaborative arm, commonly used in education/labs.

**Repositories:**
- STL: https://github.com/NiryoRobotics/niryo_one
- ROS: https://github.com/NiryoRobotics/niryo_one_ros

**Best For:** Teaching and rapid prototyping.

**Key Features:**
- Complete ROS software stack (GPLv3)
- 3D printable STL structures
- Educational ecosystem
- Good documentation

### 4. BCN3D Moveo

**Overview:** Classic open-source 3D printed robotic arm with complete CAD/STL/BOM/firmware.

**Repository:** https://github.com/BCN3D/BCN3D-Moveo

**What's Included:**
- CAD files
- STL files
- Bill of Materials (BOM)
- Firmware
- User/assembly manual

**Best For:** High reproducibility builds.

**Key Features:**
- Very complete documentation
- Clear BOM for sourcing parts
- Well-tested design

### 5. PAROL6

**Overview:** High-performance 3D printed desktop 6-axis with GUI / Python API / ROS2 simulation.

**Repository:** https://github.com/PCrnjak/PAROL6-Desktop-robot-arm

**What's Included:**
- Control software
- GUI
- STL files
- BOM + assembly instructions
- Python API
- ROS2/MoveIt simulation

**Best For:** "Desktop industrial route" with complete planning + simulation + control.

**Key Features:**
- Modern GUI
- Python API for easy control
- ROS2 simulation support
- MoveIt integration

### 6. SO-ARM100

**Overview:** Standard open-source robotic arm (compatible with LeRobot).

**Repository:** https://github.com/TheRobotStudio/SO-ARM100

**What's Included:**
- STEP files
- STL files
- Simulation files
- Assembly guide
- BOM for self-sourcing

**Best For:** AI data collection and training.

**Key Features:**
- Designed for "open, reproducible, AI data collection/training"
- Standard arm design
- LeRobot compatible

### 7. MeArm

**Overview:** Pocket-sized 4-DOF open-source robotic arm, low-cost, education-friendly.

**Repository:** https://github.com/MeArm/MeArm

**Best For:** STEAM education and quick hands-on projects.

**Key Features:**
- Very low cost
- Simple to build
- Cutting files + assembly instructions
- Great for beginners

## Recommendations

### Choose AR4 If:
- You want a "desktop industrial collaborative arm platform"
- You need complete planning + simulation + control chain
- You prefer industrial-grade components

### Choose Dummy-Robot If:
- You want to learn "how to productize a complete robotic arm project"
- You're interested in hardware + firmware + upper computer + toolchain
- You want to understand engineering organization

### Learning Path Summary

```
Beginner:
  MeArm (4-DOF, simple) → Niryo One (6-axis, ROS)

Intermediate:
  BCN3D Moveo (complete build) → PAROL6 (with GUI/API)

Advanced:
  AR4 (industrial platform) → Dummy-Robot (full-stack mastery)
```

## Action Items

- [ ] Choose project based on learning goal
- [ ] Start with AR4 or Dummy-Robot for maximum value
- [ ] Consider: integrate with our waypoint prediction work
- [ ] Explore: LeRobot + SO-ARM100 for AI policy training

## Related Reading

- ROS MoveIt tutorial
- Arduino/Teensy firmware development
- Python control APIs
- Robot kinematics basics

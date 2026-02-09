# Overview

This repository is organized around a common loop:

**Simulation → Data → Model → Evaluation → (iterate)**

We aim to support two domains:
- **Autonomous driving** (CARLA-first)
- **Robotics** (Isaac and/or MuJoCo tracks)

The most important outcome is not “a sim that runs,” but an evaluation harness that can answer:
- Did the model complete the task?
- Did it collide / violate constraints?
- How smooth/comfortable was the behavior?
- How does performance change under controlled stress tests (fog, latency, occlusion, friction, etc.)?

See:
- `docs/sim_training_eval.md`
- `models/policy_interface.py`
- `sim/common/metrics.py`
# Octo: Open Generalist Robot Policy — Digest

**Date:** 2026-03-12  
**Survey type:** Public anchor digest (Survey PR #3)  
**Sources:** https://octo-models.github.io/ | GitHub: https://github.com/octo-models/octo

---

## TL;DR (3 bullets)

- **Octo** is an open-source generalist robot policy (27M–93M params) trained on **Open X-Embodiment** (~800k episodes from 25+ datasets), achieving **80% zero-shot success** on WidowX and **72% with just 100 finetuning demos**.
- Uses **Transformer + Diffusion** objective — models action uncertainty, outputs action sequences (8–16 steps), supports language instructions + goal images + observation history.
- **Best public baseline** for robotics foundation model thesis; full code + pretrained checkpoints available on GitHub.

---

## Problem

Robotics learning typically trains separate policies per robot/task, making it hard to leverage data across robots. The question: can we train a single generalist policy on heterogeneous robot data and efficiently adapt to new embodiments/tasks?

This directly maps to Tesla/Ashok's "one foundational network" thesis for robotics.

---

## Dataset / Inputs / Outputs

### Dataset: Open X-Embodiment

| Metric | Value |
|--------|-------|
| Episodes | ~800k |
| Source datasets | 25+ |
| Robot embodiments | 20+ (arms, bi-manual, mobile manipulators) |
| Institutions | 20+ academic labs |
| Format | RLDS (episode sequences) |

**Diversity covers:** Multiple sensors (wrist cameras, force-torque), different labels (language instructions, goal images), various action spaces.

### Model inputs
- **Images:** Multiple workspace RGB camera views (configurable)
- **Language:** Task instruction string (e.g., "pick up the cube and place on the left")
- **Goal image:** Optional target observation for goal-conditioned tasks
- **Observation history:** Temporal context (past few frames)
- **Proprio:** Robot joint positions / end-effector pose when available

### Model outputs (action contract)
- **7D end-effector** in gripper frame: `[x, y, z, roll, pitch, yaw, gripper]`
- Supports **joint positions** or other action spaces via finetuning
- Action output as **diffusion denoising** — predicts full action sequence (8–16 steps), not just next step

---

## Training objective (BC / diffusion / etc.)

Octo uses **Transformer + Diffusion** objective:

1. **Observation encoding:** CNN/Transformer encoder processes images + language + proprio into tokens
2. **Diffusion process:** Models full action distribution as diffusion process (DDPM-style denoising)
3. **Multi-step prediction:** Outputs action sequences (typically 8–16 steps) rather than single-step actions
4. **Language conditioning:** Task string tokenized and attended via cross-attention

**Key advantage over RT-X's behavior cloning:** Diffusion models action uncertainty explicitly, produces smoother trajectories.

---

## Evaluation setup

### Zero-shot evaluation (WidowX UR5)
| Model | Success Rate |
|-------|-------------|
| RT-1-X | 0.60 |
| RT-2-X (55B) | 0.85 |
| **Octo** | **0.80** |

Octo outperforms RT-1-X significantly, competitive with RT-2-X while being much smaller (93M vs 55B).

### Finetuning evaluation (100 demos)
| Baseline | Avg Success |
|----------|-------------|
| From Scratch | 0.20 |
| VC-1 | 0.15 |
| **Octo** | **0.72** |

**Key finding:** Finetuning Octo outperforms next best by **52%** — validates efficient domain transfer.

### Flexibility evaluations
- Tested with **different camera configs** (workspace, wrist, multiple views)
- Successfully adapted to **new action spaces** (joint positions vs end-effector)
- Works with **additional sensors** (force-torque) via finetuning

---

## What maps cleanly to Tesla / Ashok talk claims vs what doesn't

### Maps cleanly ✓

| Tesla Claim | Octo Finding |
|-------------|--------------|
| "One foundation model for multiple robots" | Octo = single policy across 20+ embodiments |
| "Pretrain + finetune" paradigm | 100 demos = 72% success — validates efficient transfer |
| "Language as API" | Task string conditioning works across robots |
| "End-to-end from pixels" | Image → action, no handcrafted perception pipeline |
| "Diffusion for action" | Octo uses diffusion — models uncertainty, smooth trajectories |
| "Multi-modal inputs" | Handles images + language + proprio — aligns with Tesla's 8-camera input stack |

### Doesn't map ✗

| Gap | Detail |
|-----|--------|
| **Driving scale** | Octo = manipulation (short-horizon, single-arm); Tesla = ~2B tokens/day fleet data |
| **End-to-end to wheels** | Octo outputs end-effector (7D), not throttle/steering |
| **Humanoid / full-body** | No balance, locomotion, or whole-body control |
| **Fleet learning loop** | Static pretrained model; no continuous deployment/online learning |
| **Real-time 36 Hz** | Diffusion decoding heavier than single-step BC; not characterized for real-time |
| **3D scene understanding** | Octo is a policy, not a world model/simulator |

---

## Action items for AIResearch (interfaces / contracts to copy)

### Immediate (copy these interfaces)
- [ ] **Adopt diffusion policy** for waypoint/trajectory prediction — models action uncertainty explicitly
- [ ] **Use Octo's input contract** as template: images + language goal + observation history → action sequence
- [ ] **Standardize action contract** to 7D end-effector; document absolute vs delta vs velocity

### Near-term (architecture decisions)
- [ ] **Leverage pretrained vision encoder** from Octo — efficiency gain vs training from scratch
- [ ] **Study finetuning recipe:** learning rate, epochs, data augmentation for 100-demo transfer
- [ ] **Action space design:** Decide on end-effector vs joint-level vs throttle/steering

### Evaluation protocol
- [ ] **Define internal "WidowX-style" benchmark** — small real-world eval set for quick iteration
- [ ] **Low-data transfer benchmark** — 100-demo finetuning to measure sample efficiency
- [ ] **Closed-loop eval** — for driving: need simulator (CARLA/scenario runner)

### Long-term (Tesla-scale gaps)
- [ ] **Fleet learning loop** — Octo is static; Tesla has continuous data collection
- [ ] **World model integration** — Octo is a policy; consider pairing with GAIA-1 / DriveDreamer
- [ ] **Multi-camera 3D reasoning** — Octo uses single workspace view; Tesla needs 8-camera consistent world model

---

## Citations + Links

- **Project site:** https://octo-models.github.io/
- **GitHub (code + checkpoints):** https://github.com/octo-models/octo
- **Paper (arXiv):** https://arxiv.org/abs/2405.19685
- **Open X-Embodiment dataset:** https://robotics-transformer-x.github.io/
- **Diffusion Policy (Chi et al.):** https://diffusionpolicy.cs.berkeley.edu/
- **RT-X comparison:** https://github.com/google-deepmind/open_x_embodiment

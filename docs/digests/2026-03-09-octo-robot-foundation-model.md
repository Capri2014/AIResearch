# Octo: Open Generalist Robot Policy — Digest

Source: https://octo-models.github.io/ | GitHub: https://github.com/octo-models/octo

## TL;DR (5 bullets)
- **Octo** is an open-source generalist robot policy built on the **Open X-Embodiment** dataset (~800k episodes from 25 datasets).
- Two model sizes: **Octo-Small (27M)** and **Octo-Base (93M parameters)** — transformer + diffusion policy architecture.
- Achieves **80% zero-shot success** on WidowX UR5 and **72% with only 100 finetuning demos** (52% better than next best).
- Supports **language instructions, goal images, observation history** — flexible multi-modal interface.
- Cleanest public baseline for "foundation model for robotics" — full code + pretrained checkpoints available.

## Problem
Robotics learning typically trains separate policies per robot/task, making it hard to leverage data across robots. The question: can we train a single generalist policy that works across multiple robot embodiments and tasks, then efficiently adapt to new ones?

## Dataset / Inputs / Outputs

### Dataset (Open X-Embodiment)
- **Scale:** ~800k robot episodes from 25 datasets, pooled from 20+ academic labs.
- **Diversity:** Multiple robot embodiments (arms, bi-manual, mobile manipulators), different sensors (wrist cameras, force-torque), different labels (language instructions, goal images).
- **Format:** RLDS episode format (sequence of observations + actions).

### Model inputs
- **Images:** Multiple workspace RGB camera views (configurable).
- **Language:** Task instruction string (e.g., "pick up the cube and place it on the left").
- **Goal image:** Optional target observation for goal-conditioned tasks.
- **Observation history:** Temporal context (past few frames).
- **Proprio:** Robot joint positions / end-effector pose when available.

### Model outputs (action contract)
- Flexible action head — defaults to **7D end-effector** (x, y, z, roll, pitch, yaw, gripper open/close).
- Supports **joint positions** or other action spaces via finetuning.
- Action is output as a **diffusion denoising process** — predicts full action sequence, not just next step.

## Training objective (BC / diffusion / etc.)

Octo uses a **Transformer + Diffusion** objective:

1. **Observation encoding:** CNN/Transformer encoder processes images + language + proprio into tokens.
2. **Diffusion process:** Instead of predicting a single action, models the full action distribution as a diffusion process (DDPM-style denoising).
3. **Multi-step prediction:** Outputs action sequences (typically 8-16 steps) rather than single-step actions.
4. **Language conditioning:** Task string is tokenized and attended to via cross-attention.

This is more expressive than RT-X's behavior cloning — diffusion allows modeling **action uncertainty** and produces smoother trajectories.

## Evaluation setup

### Zero-shot evaluation (WidowX UR5)
| Model | Success Rate |
|-------|-------------|
| RT-1-X | 0.60 |
| RT-2-X (55B) | 0.85 |
| **Octo** | **0.80** |

Octo outperforms RT-1-X significantly and is competitive with RT-2-X while being much smaller (93M vs 55B).

### Finetuning evaluation (100 demos)
| Baseline | Avg Success |
|----------|-------------|
| From Scratch | 0.20 |
| VC-1 | 0.15 |
| **Octo** | **0.72** |

Key finding: Finetuning Octo outperforms the next best by **52%** — demonstrates efficient domain transfer.

### Flexibility evaluations
- Tested with **different camera configs** (workspace, wrist, multiple views).
- Successfully adapted to **new action spaces** (joint positions vs end-effector).
- Works with **additional sensors** (force-torque) via finetuning.

## What maps cleanly to Tesla / Ashok talk claims vs what doesn't

### Maps cleanly
- **"One foundation model for multiple robots"**: Octo is literally this — single policy trained on heterogeneous data, works across robot embodiments.
- **Efficient finetuning**: Tesla likely needs to adapt to new tasks/vehicles; Octo shows 100 demos = strong performance — validates the "pretrain + finetune" paradigm.
- **Language as interface**: Task string conditioning matches "text-based reasoning" / language as API.
- **Diffusion for action**: Tesla's policy likely outputs continuous actions; diffusion is a principled way to model action uncertainty.
- **Multi-modal inputs**: Octo handles images + language + proprio — aligns with Tesla's "8 cameras + navigation + kinematics" input stack.

### Doesn't map (or not demonstrated)
- **Autonomous driving scale**: Octo is manipulation-focused (short-horizon, single-arm). Tesla's ~2B tokens/day fleet data and long-horizon driving are orders of magnitude beyond.
- **End-to-end to wheels**: Octo outputs end-effector commands, not throttle/steering. The action space gap is significant.
- **Fleet learning loop**: No deployment/continuous learning component — static pretrained model.
- **Real-time closed-loop at 36 Hz**: Octo's diffusion decoding is heavier than single-step BC; inference speed not explicitly characterized for real-time robotics.
- **3D scene understanding / world model**: Octo is a policy, not a scene representation or simulator.

## Action items for AIResearch (interfaces / contracts to copy)

- [ ] **Adopt diffusion policy** for waypoint/trajectory prediction — models action uncertainty explicitly, produces smoother outputs than MSE regression.
- [ ] **Use Octo's input contract** as template: images + language goal + observation history → action sequence. This is a clean, extensible interface.
- [ ] **Leverage pretrained vision encoder** from Octo for downstream tasks — potential efficiency gain vs training from scratch.
- [ ] **Study finetuning recipe**: Octo shows efficient adaptation with 100 demos; document the recipe (learning rate, epochs, data augmentation).
- [ ] **Action space design**: Decide on end-effector vs joint-level vs throttle/steering — Octo supports flexible heads, so start with whichever is easiest and iterate.
- [ ] **Evaluation benchmark**: Define a "WidowX-style" internal benchmark — small real-world eval set for quick iteration before fleet testing.

## Citations / links

- **Project site:** https://octo-models.github.io/
- **GitHub (code + checkpoints):** https://github.com/octo-models/octo
- **Paper (arXiv):** https://arxiv.org/abs/2405.19685 (Octo: An Open Generalist Robot Policy)
- **Open X-Embodiment dataset:** https://robotics-transformer-x.github.io/
- **Diffusion Policy (Chi et al.):** https://diffusionpolicy.cs.berkeley.edu/ — foundational work that Octo builds on
- **Survey reference:** docs/surveys/2026-02-21-robotics-foundation-models.md

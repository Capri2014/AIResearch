# Octo: Open Generalist Robot Policy — Digest

Source: https://octo-models.github.io/ (paper: https://arxiv.org/abs/2405.12213 ; code: https://github.com/octo-models/octo)

## TL;DR (5 bullets)
- **Octo** is an **open-source generalist robot policy** built on Open X-Embodiment data, with **released model weights** (27M / 93M params) and full inference code.
- Uses a **Transformer + Diffusion** architecture — predicts actions as a diffusion process over continuous action tokens, unlike RT-X's discrete action token classification.
- Trained on **800k robot episodes** from 25 datasets, supporting multiple embodiments ( WidowX, UR5, Franka, bi-manual, mobile manipulators).
- Achieves **0.80 zero-shot success** on WidowX (competitive with RT-2-X's 0.85) and **0.72 with just 100 finetuning demos** — outperforms next best by **52%**.
- **Best reproducibility** of any robotics foundation model: pip-installable, HF Hub weights, colab notebooks, and clean finetuning API.

## Problem
Robotics learning historically trains separate policies per robot/task. The field needs:
1. A **generalist pretrained policy** that works across robots out-of-the-box
2. **Efficient finetuning** so domain experts can adapt to new tasks with modest data
3. **Open, reproducible** baselines the community can build on

Octo addresses all three.

## Dataset / Inputs / Outputs

### Dataset (Open X-Embodiment subset)
- **800k episodes** from 25 datasets in the Open X-Embodiment ecosystem
- **25 diverse datasets** with varying:
  - Robot embodiments (6-DoF arms, bi-manual, mobile manipulators)
  - Sensors (RGB, depth, wrist cameras, force-torque)
  - Supervision (language instructions, goal images)
- Released in **RLDS format** (standardized observation/action schema)

### Model inputs
- **Images**: multiple camera views (1–4 cameras supported)
- **Language**: task instruction string (optional)
- **Goal image**: optional goal-conditioned approach
- **Observation history**: temporal context (multiple past steps)
- Supports custom observation keys — flexible for new sensor configs

### Model outputs (action contract)
- **Continuous action vector** via diffusion decoding
- Default: 7D end-effector actions (x, y, z, roll, pitch, yaw, gripper)
- Supports **custom action spaces** — can adapt to joint positions, delta commands, etc.
- Action predictions are **sequence-valued** (multi-step action predictions)

## Training objective (BC / diffusion / etc.)
Octo uses a **diffusion policy** objective:

1. **Tokenization**: Observations and actions are tokenized into discrete tokens
2. **Transformer backbone**: Processes observation tokens + language/goal tokens
3. **Diffusion decoding**: Actions are generated via a **denoising diffusion process** — the model predicts noise to subtract, iteratively refining action predictions
4. **Multi-step action prediction**: Outputs a sequence of future actions (not just next-step)

This differs fundamentally from RT-1-X (discrete action token classification) and RT-2-X (VLM co-tuning with action-as-token). Diffusion enables **better handling of multi-modal action distributions** and **continuous control refinement**.

## Evaluation setup

### Zero-shot evaluation (WidowX UR5)
| Model | Success Rate |
|-------|-------------|
| RT-1-X | 0.60 |
| RT-2-X (55B VLM) | 0.85 |
| **Octo (93M)** | **0.80** |

Octo matches RT-2-X performance at 1/60th the parameter count.

### Finetuning evaluation (100 demos)
| Method | Avg Success |
|--------|-------------|
| From scratch | 0.20 |
| VC-1 (pretrained visual RL) | 0.15 |
| **Octo finetuned** | **0.72** |

Key finding: **Finetuning Octo outperforms next best by 52%** with minimal data.

### Evaluation details
- Tested on ** WidowX UR5** arm for zero-shot
- Finetuning experiments use **100 demonstrations** from target task
- Tasks include: picking, placing, pushing, manipulation in clutter
- Evaluation is real-robot (not simulation-only)

### Reproducibility strengths
- Model weights on **HuggingFace Hub** (`octo-base`, `octo-small`)
- **pip-installable**: `pip install octo-model`
- **Colab notebooks** for inference and finetuning
- Clear API: `OctoModel.load_pretrained()`, `model.predict_action()`

## What maps cleanly to Tesla / Ashok talk claims vs what doesn’t

### Maps cleanly
- **"One foundational network across robots"**: Octo is literally this — one model, multiple embodiments, unified action interface
- **"Fleet data + pretraining"**: 800k episodes from many labs is a proxy for "fleet-scale" diversity; shows transfer works
- **"Language as API"**: Task string conditioning is explicit and works
- **"Efficient fine-tuning"**: 100 demos = strong performance mirrors "data efficiency" goals
- **"End-to-end, camera-first"**: Octo uses RGB observations (no depth required by default)
- **Diffusion objective**: Tesla may use diffusion-style output (world model → action), aligning with this trend

### Doesn’t map (or not demonstrated)
- **Humanoid / full-body control**: Octo is primarily **manipulation** (7D end-effector); no locomotion, balance, or whole-body control demonstrated
- **Long-horizon autonomy**: Evaluations are **short-horizon** (single task, seconds); no hours-long autonomy or planning
- **Real-time fleet deployment**: This is a research model; no deployed continuous learning loop
- **3D scene understanding / Gaussian Splatting**: Octo is 2D image-to-action; no explicit 3D geometric reasoning like Tesla's generative GS
- **Multi-camera video generation**: Octo outputs actions, not video; Tesla's world simulator generates video for evaluation
- **Safety / rollback / verification**: No explicit safety gating or rollback mechanisms

## Action items for AIResearch (interfaces / contracts to copy)

- [ ] **Adopt diffusion policy** for waypoint/action prediction — aligns with Octo architecture and Tesla's reported output stochasticity
- [ ] **Use Octo as pretraining backbone** for driving policy: load `octo-base`, replace action head with driving-specific outputs (steering, throttle, brake)
- [ ] **Mirror RLDS episode schema** for on-disk data format — enables loading Open X-Embodiment datasets directly
- [ ] **Implement goal-conditioned driving**: Octo supports goal images; for driving, use "goal frame" = future desired position
- [ ] **Explore HuggingFace Hub integration**: Octo weights are HF-native; our models could follow same pattern for easy loading
- [ ] **Finetuning recipe**: Start with Octo + 100–500 driving demonstrations to test transfer; measure vs. training from scratch
- [ ] **Action contract**: Define unified 7D (or similar) action space that can map from driving (steer, throttle) to manipulation (EEF) — keep contract flexible

## Citations / links

- Project site: https://octo-models.github.io/
- Paper (arXiv): https://arxiv.org/abs/2405.12213
- GitHub: https://github.com/octo-models/octo
- HuggingFace Hub (weights): https://huggingface.co/octo-models
- Open X-Embodiment dataset: https://github.com/google-deepmind/open_x_embodiment
- Diffusion Policy original (Chi et al.): https://diffusionpolicy.cs.cmu.edu/
- Octo colab (inference): https://colab.research.google.com/github/octo-models/octo/blob/main/octo/colab/inference_demo.ipynb
- Octo colab (finetuning): https://colab.research.google.com/github/octo-models/octo/blob/main/octo/colab/finetuning_demo.ipynb

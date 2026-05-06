# Octo: Robotics Foundation Model Anchor — Digest

**Source:** [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | [Project](https://octo-models.github.io/) | [Code](https://github.com/octo-models/octo) | [Models](https://huggingface.co/rail-berkeley)

## TL;DR (3 bullets)
- **Octo** is the most **reproducible open-source** robotics foundation model — transformer + diffusion policy pretrained on **800k trajectories** from Open X-Embodiment.
- Provides **pretrained base (93M) + small (27M) models** on HuggingFace with full training code; single-GPU finetuning works in hours.
- **Maps to Tesla/Ashok claims**: strong on unified data contracts, cross-embodiment transfer, diffusion actions; gaps on humanoid/full-body, factory-scale, long-horizon autonomy.

---

## Problem

Robotics lacks a **ImageNet moment** — a reusable pretrained model that transfers across robots/tasks. The Tesla/Ashok vision requires:
1. **Unified data contracts** at scale
2. **Cross-embodiment transfer** — one model, many robots
3. **Fast adaptation** to new action spaces

Octo is the strongest **open-source** attempt to deliver this.

---

## Dataset / Inputs / Outputs

### Pretraining data
- **Source:** Open X-Embodiment — 800k trajectories (filtered from 1M+ total)
- **Scale:** ~1.2TB preprocessed
- **Format:** RLDS episode format (sequence of episodes/steps with obs/action/metadata)
- **Robots:** 22 embodiments, 60 datasets from 34 labs

### Model inputs
| Input | Type | Notes |
|-------|------|-------|
| RGB images | Multiple cameras | Workspace view + optional wrist; tokenized via CNN |
| Language | Text string | Task instruction ("pick up the spoon") |
| Goal images | Image | Alternative to language for goal-conditioned policy |
| History | 2 timesteps | Current + previous observation |

### Model outputs (action contract)
| Field | Value |
|------|-------|
| Action chunk | 4 timesteps |
| Default space | 7D end-effector (xyz + rpy + gripper) |
| Modality | Continuous-valued diffusion (not discrete tokens) |

---

## Training objective (BC / diffusion / etc.)

**Architecture:** Transformer encoder-decoder with diffusion policy head

| Component | Details |
|-----------|---------|
| Objective | Denoising diffusion (DDPM-style) over action sequences |
| Backbone | Transformer processing multimodal tokens |
| Visual encoder | ResNet / CNN |
| Language encoder | Transformer encoder |
| Action tokenization | Continuous diffusion over action chunks |
| Finetuning modes | `head_only`, `head_mlp_only`, `full` |

**Key differentiator vs RT-X:** Octo uses **continuous diffusion** rather than discrete action tokens — more natural for continuous robot control.

---

## Evaluation setup

### Multi-platform transfer (9 robots)
- **Tested on:** WidowX, Franka, xArm, UR5, and 5 others
- **Finetuning:** Small target-domain datasets (100-500 trajectories)
- **Result:** Strong positive transfer vs. training from scratch in low-data regimes

### Reproducibility
| Metric | Value |
|--------|-------|
| Pretraining time | ~8-14h on TPUv4-128 |
| Finetuning | Single NVIDIA 4090 in hours |
| Inference throughput | ~13 it/sec (base) / ~17 it/sec (small) |
| Data loader | Standalone PyTorch provided |

### Gaps
- Evaluations are **simulation / lab robot** — not factory floor
- Real-world success rates not fully quantified
- No long-horizon autonomy benchmarks

---

## What maps cleanly to Tesla / Ashok talk claims vs what doesn't

### Maps cleanly ✓

| Claim | Octo alignment |
|-------|---------------|
| "Data is the moat" | Open X-Embodiment RLDS format — standardized contract |
| Cross-embodiment transfer | 9 robots, positive transfer demonstrated |
| Diffusion for actions | Continuous diffusion policy, not discrete tokens |
| Modular architecture | Swappable tokenizers + readout heads |
| Language/instruction interface | Text task string + goal images |
| Fast adaptation | 3 finetuning modes, works in hours |
| Open source | Full code + pretrained models + data loaders |

### Doesn't map (or not demonstrated) ✗

| Gap | Details |
|-----|---------|
| Humanoid / full-body | Only manipulation (arm + gripper); no locomotion |
| Long-horizon autonomy | Short-horizon, single-task evaluations |
| Factory-scale deployment | Lab setups; no grease/clutter/safety robustness |
| Fleet learning | Static pretraining; no continuous fleet updates |
| End-to-end sensing-control | Modular tokenization; not fully monolithic |

---

## Action items for AIResearch (interfaces / contracts to copy)

### Immediate actions

1. **Adopt RLDS episode format** — canonical on-disk schema for manipulation data
   - Fields: `observations`, `actions`, `rewards`, `discounts`, `metadata`
   - Sequence of episodes → steps within episodes

2. **Standardize action contract to 7D** — end-effector in gripper frame
   - `xyz` (position), `rpy` (rotation), `gripper` (open/close)
   - Document: absolute vs delta vs velocity per dataset
   - Handle missing dimensions: zero-fill vs mask

3. **Build modular tokenization layer**
   - Separate visual, language, action tokenizers
   - Configurable observation/action spaces via finetuning

4. **Implement action chunking (size=4)** — predict multiple future actions
   - Enables receding horizon control
   - Smoother execution than single-step prediction

### Medium-term actions

5. **Benchmark transfer from Octo** — "finetune from Octo" vs "train from scratch"
   - Quantify data efficiency gains in low-data regimes
   - Compare finetuning modes (`head_only`, `head_mlp_only`, `full`)

6. **Provide standalone data loader** — mirror Octo's PyTorch dataloader
   - Support RLDS format directly
   - Enable efficient batched loading

7. **Design multi-platform policy architecture** — support new robots from day 1
   - Arbitrary observation configurations via tokenizers
   - Configurable readout heads for different action spaces

### Questions to resolve

- Should we adopt Octo as the **base checkpoint** for internal manipulation models?
- What internal datasets map cleanly to the RLDS format?
- Should we use continuous diffusion or discrete tokens for our action head?

---

## Citations + links

| Resource | URL |
|----------|-----|
| Paper | https://arxiv.org/abs/2405.12213 |
| Project website | https://octo-models.github.io/ |
| GitHub repo | https://github.com/octo-models/octo |
| HuggingFace models | https://huggingface.co/rail-berkeley |
| Inference Colab | https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz |
| Open X-Embodiment | https://robotics-transformer-x.github.io/ |
| RLDS format | https://github.com/google-research/rlds |

---

## Related digests

- [Open X-Embodiment / RT-X](/docs/digests/2026-02-14-open-x-embodiment-rtx.md) — upstream dataset + Google's internal model
- [Octo Generalist Robot Policy](/docs/digests/2026-05-04-octo-generalist-robot-policy.md) — earlier digest with more technical details
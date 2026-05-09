# Octo: Robotics Foundation Model Anchor — Digest

**Source:** [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | [Project](https://octo-models.github.io/) | [Code](https://github.com/octo-models/octo) | [Models](https://huggingface.co/rail-berkeley)

**Last updated:** 2026-05-09

---

## TL;DR (3 bullets)

- **Octo** (v0.2, May 2026) — transformer + diffusion policy, pretrained on 800k trajectories from Open X-Embodiment; most reproducible open-source robotics foundation model.
- **Pretrained models**: 93M base + 27M small on HuggingFace; full training code; single-GPU finetuning in hours. v0.2 improves cross-attention + language rephrasing.
- **Maps to Tesla/Ashok claims**: strong on unified data contracts, cross-embodiment transfer, diffusion actions; gaps vs Optimus Gen 3 (1,000+ deployed Jan 2026, 22-DOF hands), factory-scale, long-horizon autonomy.

---

## Problem

Robotics lacks a **ImageNet moment** — a reusable pretrained model that transfers across robots/tasks. The Tesla/Ashok vision requires:

1. **Unified data contracts** at scale
2. **Cross-embodiment transfer** — one model, many robots
3. **Fast adaptation** to new action spaces

Octo is the strongest **open-source** attempt to deliver this. Chosen over RT-X for better reproducibility (fully open code + models), diffusion-based objective (aligns with Tesla's simulator interests), and modular architecture.

**Why Octo over RT-X:**
- Open-source checkpoints (RT-X limited/restricted)
- Diffusion-based (continuous actions, not discrete tokens)
- Modular tokenization (easier to adapt)
- Active community + HuggingFace-hosted models

---

## Dataset / Inputs / Outputs

### Pretraining data

| Aspect | Details |
|--------|---------|
| **Source** | Open X-Embodiment (filtered 800k from ~1M+) |
| **Scale** | ~1.2TB preprocessed |
| **Datasets** | 60 robot datasets from 34 research labs |
| **Robots** | 22 embodiments (arms, bi-manual, quadrupeds) |
| **Format** | RLDS episode format (episode → steps with obs/action/metadata) |

### Model inputs

| Input | Type | Notes |
|-------|------|-------|
| RGB images | Multiple cameras | Workspace view + optional wrist; tokenized via CNN/ResNet |
| Language | Text string | Task instruction ("pick up the spoon", "open drawer") |
| Goal images | Image | Alternative conditioning to language |
| History | 2 timesteps | Current + previous observation |

### Model outputs (action contract)

| Field | Value |
|------|-------|
| **Action chunk** | 4 timesteps (predict next 4 actions) |
| **Default space** | 7D end-effector (xyz + rpy + gripper) |
| **Modality** | Continuous-valued diffusion (not discrete tokens) |

**Action space modularity:** Finetuning supports new action spaces via configurable readout heads — enables adapting to different robot morphologies.

---

## Training objective

### Architecture: Transformer-based diffusion policy

| Component | Details |
|-----------|---------|
| **Objective** | Denoising diffusion (DDPM-style) over action sequences |
| **Backbone** | Transformer encoder-decoder processing multimodal tokens |
| **Visual encoder** | ResNet / CNN backbone |
| **Language encoder** | Transformer encoder (text → tokens) |
| **Action tokenization** | Continuous diffusion over action chunks |
| **Finetuning modes** | `head_only`, `head_mlp_only`, `full` |
| **v0.2 update (May 2026)** | Cross-attention: repeat language tokens at every timestep; dropout off in diffusion head; fixed attention mask bug; image augmentations use fresh seeds |

### Comparison to alternatives

| Method | Action representation | Open? | Reproducibility |
|--------|----------------------|-------|-----------------|
| **Octo** | Continuous diffusion | ✅ Full | High (code + models) |
| RT-1-X | Discrete tokens | Partial | Medium (limited checkpoints) |
| RT-2-X | Discrete tokens (VLM-style) | ❌ | Low |
| Diffusion Policy | Continuous diffusion | ✅ | Medium |

**Key differentiator:** Octo uses **continuous diffusion** rather than discrete action tokens — more natural for continuous robot control, smoother action trajectories.

---

## Evaluation setup

### Multi-platform transfer (9 robots)

- **Tested on:** WidowX, Franka, xArm, UR5, and 5 additional platforms
- **Finetuning:** Small target-domain datasets (100-500 trajectories)
- **Result:** Strong positive transfer vs. training from scratch in low-data regimes

### Reproducibility benchmarks

| Component | Time / Resource | Notes |
|-----------|-----------------|-------|
| Pretraining (small, 27M) | ~8h on TPUv4-128 | Single pod |
| Pretraining (base, 93M) | ~14h on TPUv4-128 | Single pod |
| Finetuning | Hours on single NVIDIA 4090 | Per robot |
| Inference | ~13 it/sec (base) / ~17 it/sec (small) | On NVIDIA 4090 |
| Data loader | Standalone PyTorch provided | Full batch loading |

### v0.2 Release (May 2025)

Recent updates from Octo release notes (May 24, 2025):

- **Cross-attention improvement**: Repeats language tokens at every timestep in the context window for stronger visual-language alignment
- **Language augmentation**: GPT-3.5 rephrases added to language instructions for data diversity
- **Bug fixes**: Dropout disabled in diffusion head (incompatible with layer norm); attention mask off-by-one error fixed; image augmentations now use fresh random seeds

### What's NOT evaluated

- Long-horizon autonomy (hours+ of continuous operation)
- Factory-scale deployment (grease, clutter, safety)
- Fleet learning (continuous updates from deployed robots)
- Full-body / humanoid control (manipulation only, arm + gripper) — Tesla Optimus has 22-DOF hands + bipedal locomotion
- Real-world success rates under clutter/occlusion

---

## What maps cleanly to Tesla / Ashok talk claims vs what doesn't

### Maps cleanly ✓

| Tesla/Ashok claim | Octo alignment | Gap notes |
|------------------|---------------|-----------|
| "End-to-end neural network" | Transformer encoder-decoder, pixels→actions | Octo uses modular tokenizers; Tesla may be more monolithic |
| "Fleet data + interesting data mining" | Open X-Embodiment (60 datasets, 34 labs) | Tesla has 500 years/day; Octo has 800k trajectories |
| "Cross-embodiment transfer" | 9-robot transfer results | Octo shows positive transfer; Tesla claims one model for driving+Optimus |
| "Diffusion / generative for actions" | Diffusion objective | Direct alignment — continuous diffusion |
| "Language as API" | Text task string conditioning | Matches — language or goal images |
| "3D geometric reasoning" | ❌ Not in Octo | Tesla has generative Gaussian Splatting |
| "World simulator" | ❌ Not in Octo | Tesla trains video-generation network |
| "Long-horizon autonomy" | ❌ Short-horizon only | Single-task, not multi-hour |
| "Factory / humanoid" | ❌ Manipulation only | Arm + gripper, no balance/locomotion |
| "Fast adaptation" | 3 finetuning modes, hours on 4090 | Works in low-data regimes |

### What Octo proves (publicly)

1. **RLDS schema works** — standardized data contract enables cross-dataset, cross-robot training at scale
2. **Continuous diffusion > discrete tokens** — smoother action generation for manipulation
3. **Positive transfer is real** — mixture-trained > single-dataset even with small finetuning data
4. **Reproducibility is achievable** — full training code + pretrained models + PyTorch loader provided

### What Octo does NOT prove

- Scaling to humanoid / full-body (not evaluated)
- Long-horizon planning (not benchmarked)
- Factory-grade robustness (lab settings only)
- End-to-end differentiable stack (modular tokenizers)

---

## Action items for AIResearch (interfaces / contracts to copy)

### Priority 1: Contracts (do first)

- [ ] **Adopt RLDS episode format** — canonical on-disk schema for manipulation data
  - Fields: `observations`, `actions`, `rewards`, `discounts`, `metadata`
  - Sequence of episodes → steps within episodes

- [ ] **Make "task string" a first-class field** — explicit guidelines for allowed verbs/objects
  - Treat as required instruction channel for language-conditioned policies
  - Document: vocabulary, syntax, required fields

- [ ] **Lock portable action contract to 7D** — end-effector in gripper frame
  - `xyz` (position), `rpy` (rotation), `gripper` (open/close)
  - Document: absolute vs delta vs velocity per dataset
  - Handle missing dimensions: zero-fill vs mask

### Priority 2: Architecture (do second)

- [ ] **Build modular tokenization layer** — separate visual, language, action tokenizers
  - Enables swapping sensors without retraining
  - Supports arbitrary observation configurations

- [ ] **Implement action chunking (size=4)** — predict multiple future actions
  - Enables receding horizon execution
  - Smoother than single-step prediction

- [ ] **Use diffusion over discrete tokens** for action generation
  - Continuous diffusion produces smoother trajectories

### Priority 3: Benchmarking (do third)

- [ ] **Run "finetune from Octo" vs "train from scratch"** — quantify transfer gains
  - Use target-domain dataset (100-500 trajectories)
  - Compare three finetuning modes: head_only, head_mlp_only, full

- [ ] **Evaluate on target robot** — report success rates on manipulation tasks
  - Use standard manipulation benchmarks (LIBERO, RLBench, or own tasks)

- [ ] **Build "golden loader"** — visualize episodes, batch data, run forward pass, overlay predictions vs GT

### Priority 4: Extension (later)

- [ ] **Add 3D geometry** — incorporate Gaussian Splatting or similar for view consistency
- [ ] **World model** — add action-conditioned video prediction for simulation
- [ ] **Long-horizon** — extend from single-task to multi-step / hierarchical

---

## Citations + links

| Resource | URL |
|----------|-----|
| Paper (arXiv) | https://arxiv.org/abs/2405.12213 |
| Project website | https://octo-models.github.io/ |
| GitHub repo | https://github.com/octo-models/octo |
| HuggingFace models | https://huggingface.co/rail-berkeley |
| Inference Colab | https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz |
| Finetuning Colab | https://colab.research.google.com/github/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb |
| Open X-Embodiment | https://robotics-transformer-x.github.io/ |
| RLDS format | https://github.com/google-research/rlds |

---

## Related digests

- [Open X-Embodiment / RT-X](/docs/digests/2026-02-14-open-x-embodiment-rtx.md) — upstream dataset + Google's internal model
- [Octo Generalist Robot Policy](/docs/digests/2026-05-04-octo-generalist-robot-policy.md) — earlier technical details

---

## PR metadata

- **Digest for:** Survey PR #3
- **Target:** docs/digests/
- **Status:** Ready for commit
# Octo: Public Anchor for Robotics Foundation Models — Digest

**Source:** [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | [Project](https://octo-models.github.io/) | [Code](https://github.com/octo-models/octo) | [Models](https://huggingface.co/rail-berkeley)

**Last updated:** 2026-05-05

## TL;DR (3 bullets)

- **Octo** is the closest public approximation to "foundation model for robotics" — a **transformer-based diffusion policy** pretrained on **800k trajectories** from Open X-Embodiment.
- Provides **pretrained base (93M) and small (27M) models** on HuggingFace with full training/finetuning code; reproducible in ~8-14h on TPUv4-128 or hours on NVIDIA 4090.
- **Action items for AIResearch:** adopt the RLDS schema + modular tokenization + diffusion objective + action chunking as baseline contracts before building proprietary models.

---

## Problem: The Anchor Gap

Tesla/Ashok claims a "foundational network" for robotics trained on fleet data. The public equivalent is **Octo** — the most reproducible open-source generalist robot policy. This digest establishes Octo as the **reference baseline** for comparing proprietary claims against.

**Why Octo over RT-X:**
- Open-source (RT-X checkpoints are limited/restricted)
- Diffusion-based (aligns with Tesla's generative video/simulator interest)
- Modular architecture (easier to adapt and extend)
- Active community and HuggingFace-hosted models

---

## Dataset / Inputs / Outputs

### Pretraining data
| Aspect | Details |
|--------|---------|
| **Source** | Open X-Embodiment (filtered 800k from ~1M+) |
| **Scale** | ~1.2TB preprocessed, 60 datasets, 22 robot embodiments |
| **Format** | RLDS episode format (sequence of episodes/steps with obs/action/metadata) |

### Model inputs
| Input | Description |
|-------|-------------|
| **Visual** | Multiple RGB cameras (workspace view, optional wrist); arbitrary camera configs via tokenization |
| **Language** | Text instruction / task string (e.g., "pick up the spoon") |
| **Goal images** | Alternative conditioning to language |
| **History** | 2 timesteps (current + previous); configurable |

### Model outputs (action contract)
| Aspect | Details |
|--------|---------|
| **Action chunking** | Pretrained with **action chunk size of 4** — predicts next 4 actions; enables receding horizon execution |
| **Action space** | Variable by finetuning — defaults to 7D end-effector (position + rotation + gripper); modular readout heads support new action spaces |
| **Modality** | **Continuous-valued diffusion** — not discrete tokens |

---

## Training objective

**Transformer-based diffusion policy:**
- **Diffusion objective:** Denoising diffusion over action sequences (DDPM-style)
- **Backbone:** Transformer encoder-decoder processing multimodal tokens
- **Tokenization:** Images via CNN/ResNet; language via transformer encoder; actions tokenized for diffusion
- **Finetuning modes:** Three options — `head_only`, `head_mlp_only`, `full` (full transformer)

**Comparison to alternatives:**
| Method | Action representation | Open? | Modularity |
|--------|----------------------|-------|-------------|
| **Octo** (this) | Continuous diffusion | ✅ | High (tokenizers) |
| RT-1-X | Discrete tokens | Partial | Low |
| RT-2-X | Discrete tokens (VLM) | ❌ | Low |
| Diffusion Policy | Continuous diffusion | ✅ | Medium |

---

## Evaluation setup

### Multi-platform transfer (9 robots)
- Tested on **9 different robot platforms**: WidowX, Franka, xArm, UR5, and others.
- **Finetuning with small target-domain datasets** (a few hundred trajectories) shows strong positive transfer vs. training from scratch.
- Outperforms training from scratch in **low-data regimes** (consistent with RT-X findings).

### Reproducibility benchmarks
| Component | Time / Resource | Notes |
|-----------|-----------------|-------|
| Pretraining (small) | ~8h on TPUv4-128 | 27M params |
| Pretraining (base) | ~14h on TPUv4-128 | 93M params |
| Finetuning | Hours on single NVIDIA 4090 | Per robot |
| Inference | ~13-17 it/sec on 4090 | Depends on model size |

### What's NOT evaluated
- Long-horizon autonomy (hours+)
- Factory-scale deployment
- Fleet learning
- Full-body / humanoid control

---

## Mapping to Tesla / Ashok claims

### ✅ Maps cleanly (anchored by Octo)

| Tesla claim | Octo equivalent | Gap notes |
|------------|-----------------|-----------|
| "End-to-end neural network" | Transformer encoder-decoder processing pixels→actions | Octo uses modular tokenization; Tesla may be more monolithic |
| "Fleet data + interesting data mining" | Open X-Embodiment pooling from 34 labs | Tesla has 500 years/day; Octo has 800k trajectories |
| "Cross-embodiment transfer" | 9-robot transfer results | Octo shows positive transfer; Tesla claims one model for driving+Optimus |
| "Diffusion / generative for actions" | Diffusion objective | Direct alignment |
| "Language as API" | Task string conditioning | Matches |
| "3D geometric reasoning" | ❌ Not in Octo | Tesla has generative Gaussian Splatting |
| "World simulator" | ❌ Not in Octo | Tesla trains video-generation network |
| "Long-horizon autonomy" | ❌ Short-horizon only | Single-task, not multi-hour |
| "Factory / humanoid" | ❌ Manipulation only | Arm + gripper, no balance/locomotion |

### What Octo proves (publicly)

1. **RLDS schema works** — standardized data contract enables cross-dataset, cross-robot training
2. **Diffusion over discrete tokens** — continuous action generation is viable and outperforms discrete for manipulation
3. **Positive transfer is real** — mixture-trained > single-dataset even with small finetuning data
4. **Reproducibility is achievable** — full training code + pretrained models + PyTorch loader

### What Octo does NOT prove

1. Scaling to humanoid / full-body (not evaluated)
2. Long-horizon planning (not evaluated)
3. Factory-grade robustness (lab settings only)
4. End-to-end differentiable stack (modular tokenizers)

---

## Action items for AIResearch

### Priority 1: Contracts (do first)

- [ ] **Adopt RLDS episode format** as canonical on-disk schema for manipulation data
- [ ] **Make "task string" a first-class field** — explicit guidelines for allowed verbs/objects; treat as required instruction channel
- [ ] **Lock portable action contract**: 7D end-effector in gripper frame (+ absolute/delta/velocity choice); document missing-dimension handling

### Priority 2: Architecture (do second)

- [ ] **Build modular tokenization layer** — separate visual, language, action tokenizers; enables swapping sensors without retraining
- [ ] **Implement action chunking (size=4)** — predict multiple future actions; enables receding horizon
- [ ] **Use diffusion over discrete tokens** for action generation

### Priority 3: Benchmarking (do third)

- [ ] **Run "finetune from Octo" vs "train from scratch"** — quantify transfer gains in low-data regimes
- [ ] **Evaluate on target robot** — report success rates on manipulation tasks
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
| Open X-Embodiment | https://robotics-transformer-x.github.io/ |
| RLDS format | https://github.com/google-research/rlds |
| Tesla/Ashok talk | https://www.youtube.com/watch?v=LFh9GAzHg1c |

---

## PR metadata

- **Digest for:** Survey PR #3
- **Target:** docs/digests/
- **Related:** 2026-05-04-octo-generalist-robot-policy.md (superseded by this anchor version)
- **Status:** Ready for commit
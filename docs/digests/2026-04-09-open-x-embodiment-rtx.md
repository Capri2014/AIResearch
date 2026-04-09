# Open X-Embodiment / RT-X — Public Anchor Digest (Updated)

**Source:** [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) | [Project](https://robotics-transformer-x.github.io/) | [Code/Data](https://github.com/google-deepmind/open_x_embodiment)

**Date:** April 9, 2026 | **Anchor Type:** Robotics Foundation Model Baseline | **Selection Rationale:** Best open code + reproducibility among public multi-robot datasets; RT-X is the direct predecessor to Octo.

---

## TL;DR (3 bullets)

- **Open X-Embodiment** pools **60 datasets**, **22 robot embodiments**, **1M+ trajectories** from **34 labs** into a unified RLDS format — the foundational "data moat" contract for robotics foundation models.
- **RT-X** (RT-1-X / RT-2-X) demonstrates **positive transfer** across robots: mixture-trained policies outperform per-dataset baselines, especially in low-data regimes (~50% improvement).
- **Maps to Tesla/Ashok claims**: standardized data contracts ✅, cross-embodiment transfer ✅, unified action space ✅; gaps: humanoid/full-body, factory-scale, long-horizon autonomy.

---

## Problem

Robotics learning historically trains **separate policies per robot/task**. The Tesla/Ashok talk frames "data as the moat" and "one foundational network" across robots. Open X-Embodiment asks: can we standardize heterogeneous robot data into a unified contract, then train a single "generalist" policy that transfers across embodiments?

**Key contribution:** Not the model architecture per se, but the **standardized data contract** (RLDS format) that enables pooling diverse datasets into a shared pretraining corpus.

---

## Dataset / Inputs / Outputs

### Pretraining data

| Item | Detail |
|------|--------|
| **Scale** | 1M+ real robot trajectories |
| **Composition** | 60 datasets, 22 robot embodiments |
| **Source labs** | 34 academic/industry labs |
| **Format** | RLDS episode format (sequence of episodes with obs/action/metadata) |
| **Access** | TFDS loading, standalone PyTorch dataloader, Google Colabs |

### Model inputs (RT-1-X / RT-2-X)

| Input | Description |
|-------|-------------|
| **Vision** | Single workspace RGB camera (default) |
| **Language** | Task string / instruction (e.g., "pick up the cup") |
| **Proprio** | Varies by dataset; missing dims zero-filled |
| **History** | Not explicitly used in released checkpoint |

### Model outputs (action contract)

| Item | Detail |
|------|--------|
| **Action space** | 7D end-effector in gripper frame: x, y, z, roll, pitch, yaw, gripper open/close |
| **Representation** | Absolute / delta / velocity (dataset-dependent) |
| **RT-1-X** | Discrete action tokens (transformer policy) |
| **RT-2-X** | Discrete action tokens emitted by VLM (co-finetuned RT-2) |

---

## Training Objective (BC / diffusion / etc.)

### RT-1-X: Transformer BC

- **Architecture:** RT-1style transformer (ViT + tokenization)
- **Objective:** Supervised behavior cloning — predict next action token given (image, task string, history)
- **Action discretization:** Continuous actions discretized into bins; trained with cross-entropy classification

### RT-2-X: VLM co-finetuning

- **Architecture:** RT-2-style VLM (large language model + vision encoder)
- **Objective:** Same as RT-1-X but action output integrated into language model token space
- **Claim:** "Emergent behaviors" from large-scale VLM pretraining

### Key distinction from Octo

| Dimension | RT-X | Octo |
|-----------|------|------|
| **Action modality** | Discrete tokens | Continuous diffusion |
| **Code** | Dataset + loaders open; limited model release | Full training code open |
| **Multi-camera** | Single workspace (default) | Modular tokenization |
| **Goal images** | No | Yes |

---

## Evaluation Setup

### RT-1-X: In-distribution transfer

- **Platforms:** Tested across academic labs (UC Berkeley, Stanford, USC, NYU, Univ. Freiburg)
- **Metric:** Success rate vs. "Original Method" (dataset creator's own method trained only on that dataset)
- **Headline:** RT-1-X **outperforms per-dataset baselines**; ~50% better in small-data regimes

### RT-2-X: Emergent skills

- **Focus:** Language-conditioned tasks with spatial prepositions ("on" vs "near", relative placement)
- **Headline:** RT-2-X **outperforms RT-2 by ~3×** on emergent skill evaluations

### Reproducibility caveats

- **Dataset/loaders:** Highly reproducible (TFDS, RLDS, Colabs)
- **Model checkpoints:** RT-1-X checkpoint released; RT-2-X more limited
- **Real-robot eval:** Hard to reproduce outside contributing labs

---

## What Maps Cleanly to Tesla/Ashok Talk Claims vs What Doesn't

### Maps Cleanly ✅

| Claim from Talk | Open X-Embodiment / RT-X Support |
|-----------------|-----------------------------------|
| "Data is the moat" | ✅ Unified RLDS format = portable contract; 1M+ trajectories |
| Foundation model for robotics | ✅ Single policy transfers across 22 robot embodiments |
| "Rapid adaptation" | ✅ Positive transfer in low-data regimes |
| "Language as API" | ✅ Task string as instruction interface |
| Cross-embodiment transfer | ✅ RT-1-X runs on multiple robots without retraining |
| Unified action representation | ✅ 7D end-effector contract in gripper frame |
| End-to-end neural network | ✅ Single transformer: image + language → actions |

### Doesn't Map (or not demonstrated) ❌

| Claim | Gap |
|-------|-----|
| Humanoid / full-body | End-effector manipulation only; no locomotion, balance, whole-body |
| Factory-scale deployment | Lab setups; no robustness to grease, clutter, safety constraints |
| Long-horizon autonomy | Short-horizon, single-task evaluations; no multi-hour autonomy |
| 8-camera video input | Single workspace camera (default); limited multi-view |
| Fleet learning | Static dataset; no continuous fleet updates |
| Diffusion for actions | Discrete token prediction; Octo uses diffusion instead |

---

## Action Items for AIResearch (interfaces / contracts to copy)

1. **Adopt RLDS episode format** as canonical on-disk schema — matches Open X-Embodiment standard and enables TFDS loading

2. **Define unified action contract early** — 7D end-effector (x, y, z, roll, pitch, yaw, gripper); document absolute/delta/velocity handling

3. **Make "task string" first-class** — explicit guidelines for allowed verbs/objects; treat as required instruction channel

4. **Build multi-camera tokenization** — RT-X uses single camera; Tesla uses 8 cameras; design for arbitrary camera configs from day 1

5. **Consider diffusion over discrete tokens** — Octo shows diffusion is viable; continuous actions map more naturally to robot control

6. **Benchmark "mixture-trained" vs "single-dataset"** — quantify transfer gains in low-data regimes per RT-X protocol

7. **Provide Colab-like golden loader** — visualize episodes, batch data, run forward pass, overlay predicted vs GT actions

8. **Design for multi-embodiment from day 1** — architecture should support arbitrary action spaces via configurable readout heads

---

## Cross-Comparison: Open X-Embodiment/RT-X vs Octo

| Dimension | RT-X | Octo |
|-----------|------|------|
| **Code** | Dataset + loaders open; model limited | Full training code open |
| **Action modality** | Discrete tokens | Continuous diffusion |
| **Pretraining data** | 1M+ trajectories (full OXE) | 800k trajectories (filtered OXE) |
| **Multi-camera** | ❌ Single workspace | ✅ Modular tokenization |
| **Goal images** | ❌ No | ✅ Yes |
| **Action chunking** | ❌ Single-step | ✅ Chunk size 4 |
| **Finetuning flexibility** | Limited | High (3 modes) |
| **Inference speed** | Not benchmarked | 13-17 it/sec (4090) |

**Recommendation:** Use **Open X-Embodiment as the data contract** (RLDS format, standardized fields) and **Octo as the model baseline** (open code, diffusion, modular architecture). RT-X demonstrates transfer is real; Octo makes it reproducible.

---

## Citations + Links

- **Paper (arXiv):** https://arxiv.org/abs/2310.08864
- **Project website:** https://robotics-transformer-x.github.io/
- **GitHub repo (code + data):** https://github.com/google-deepmind/open_x_embodiment
- **Dataset spreadsheet:** https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit
- **RLDS format reference:** https://github.com/google-research/rlds
- **Octo (model follow-up):** https://octo-models.github.io/ | https://github.com/octo-models/octo
- **HuggingFace (Octo models):** https://huggingface.co/rail-berkeley

### BibTeX Citation
```bibtex
@article{open_x_embodiment_2023,
  title={Open X-Embodiment: Robotic Learning Datasets and RT-X Models},
  author = {Octo Model Team and Pieter Abbeel and et al.},
  journal = {arXiv preprint arXiv:2310.08864},
  year = {2023},
}
```

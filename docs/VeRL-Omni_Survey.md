# VeRL-Omni — Research Survey

**Date:** 2026-05-26  
**Status:** Complete  
**Focus:** RL training framework for diffusion and omni-modality models

---

## Overview

VeRL-Omni is a general RL post-training framework for multimodal generative models, built on top of verl and vLLM-Omni. It handles diffusion models (image/video/audio) and omni-modality models (text + image + video + audio) with efficient rollout, flexible rewards, and modular training backends.

**Key features:**
- Efficient multimodal rollout via vLLM-Omni (~25% higher throughput than diffusers-based flow_grpo)
- Flexible reward pipelines (rule-based, model-based, VLM judges)
- Modular training backends (FSDP, USP, TP)
- Supports NVIDIA GPUs and Ascend NPUs

**Supported models:**
- Qwen-Image (DiT, text→image)
- Wan2.2 (DiT, text→video)
- BAGEL (unified understand + gen)
- Qwen3-Omni-Thinker (omni-modality)

---

## Table of Contents

1. [Background](#1-background)
2. [Core Concepts](#2-core-concepts)
3. [Key Methods](#3-key-methods)
4. [Comparison & Tradeoffs](#4-comparison--tradeoffs)
5. [Applications](#5-applications)
6. [Open Problems](#6-open-problems)
7. [References](#7-references)

---

<a name="1-background"></a>
## 1. Background

### What is this field/problem about?

Multimodal generative RL differs from text-only LLM RL in several critical ways:
- **Model structure**: Diffusion transformers, mixed AR-DiT, omni-modality
- **I/O patterns**: Continuous latent spaces instead of discrete tokens
- **Compute characteristics**: Higher memory peaks, heterogeneous pipelines
- **Runtime bottlenecks**: Denoising trajectories, multi-stage model invocation

Existing LLM RL frameworks (like verl) were not designed for these workloads.

### Why does it matter for our work?

- **End-to-end training** for diffusion-based image/video generation
- **Omni-modality support** for text + image + audio + video models
- **High throughput** via vLLM-Omni integration
- **Modular backends** that integrate with existing parallelism

### Historical context — how did we get here?

| Year | Development |
|------|-------------|
| 2024 | verl (original RL framework for LLMs) |
| 2025 | vLLM-Omni (multimodal inference) |
| 2026 | VeRL-Omni (multimodal RL training) |

### Key papers timeline (with arXiv links)

- [FlowGRPO](https://arxiv.org/abs/2505.05470) — RL for flow matching models
- [MixGRPO](https://arxiv.org/abs/2507.21802) — Mixed reward RL
- [GSPO](https://arxiv.org/abs/2507.18071) — Omni-modality RL
- [GRPO-Guard](https://arxiv.org/abs/2510.22319) — Guardrail for generation

---

<a name="2-core-concepts"></a>
## 2. Core Concepts

### 2.1 Diffusion Model RL Training

**Intuition:** Unlike autoregressive LLM generation, diffusion models generate by iteratively denoising from random noise. RL training must handle these denoising trajectories differently.

**The Problem It Solves:** Standard LLM RL assumes discrete token generation. Diffusion generates in continuous latent space with multiple steps.

**How It Works:**
1. **Rollout**: Generate denoising trajectory (multi-step sampling)
2. **Reward**: Score generated images/videos via reward model
3. **Optimize**: Update policy using FlowGRPO CLIP-style loss
4. **Sync**: Synchronize policy weights to rollout workers

**The Math:**
```
# FlowGRPO loss (CLIP-style)
L = -E[log(sigmoid(A - A'))]  where A = advantage
```

**Code Example:**
```python
# From verl_omni/trainer/main_diffusion.py
from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    DirectPreferenceRayTrainer,
    PolicyGradientRayTrainer,
)

@hydra.main(config_path="./config", config_name="diffusion_trainer", version_base=None)
def main(config):
    """Main entry point for diffusion model training."""
    run_diffusion(config)

def run_diffusion(config, task_runner_class=None):
    """Initialize Ray and run distributed diffusion training."""
    if not ray.is_initialized():
        ray.init(**ray_init_kwargs)
    
    # Run trainer
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))
```

**Cross-Comparison:**

| Framework | Model Type | Throughput | Modality |
|-----------|-----------|----------|----------|
| diffusers only | Low | High | Single (image) |
| **VeRL-Omni** | **High** | **High** | **Multi** |
| Naive impl | Low | Low | Single |

**When to Use:** Use for any diffusion model RL training.

---

### 2.2 vLLM-Omni Integration

**Intuition:** Efficient rollout is critical for training throughput. vLLM-Omni provides high-throughput async serving for multimodal generation.

**The Problem It Solved:** Diffusers-based rollout is slow. Need batched, asynchronous generation.

**How It Works:**
1. vLLM-Omni serves rollout requests asynchronously
2. Step-wise continuous batching maximizes GPU utilization
3. Embedding caching avoids redundant computations
4. Reward computation overlaps with ongoing rollout

**Code Example:**
```python
# Integration pattern (from documentation)
# Launch with vLLM-Omni for high-throughput rollout
# Config in examples/flowgrpo_trainer/config.yaml:
worker:
  rollout:
    backend: vllm_omni  # Use vLLM-Omni backend
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.8
```

**Performance:**
```
| Mode              | Throughput (images/GPU/s) | Time/Step (s) |
|------------------|------------------------|---------------|
| FlowGRPO colocation | 0.305              | 420           |
| FlowGRPO async   | 0.280               | 360 (-14%)    |
| Full-model ft    | 0.510               | 250           |
```

**Cross-Comparison:**

| Backend | Batching | Async | Quantization |
|--------|----------|-------|-------------|
| diffusers | ✗ | ✗ | ✗ |
| **vLLM-Omni** | **✓** | **✓** | **✓** |

**When to Use:** Always use vLLM-Omni for production training.

---

### 2.3 Reward Pipeline

**Intuition:** Multimodal RL requires diverse reward computation - not just accuracy, but OCR, VLM-as-judge, aesthetic scores.

**The Problem It Solved:** Text-only rewards don't work for images/videos.

**How It Works:**
- **Rule-based**: Length, format, exact match
- **Model-based**: VLM judges, aesthetic models
- **Multimodal**: OCR, image quality, video coherence
- Async reward computation overlaps with rollout

**Code Example:**
```python
# From verl_omni/pipelines reward
worker:
  reward:
    reward_type: function
    compute_score: ocr  # OCR-based scoring
```

**Cross-Comparison:**

| Reward Type | Use Case | Complexity |
|------------|----------|------------|
| Rule-based | Exact match | Low |
| VLM-as-judge | Preferences | Medium |
| OCR | Text rendering | Medium |
| Aesthetic | Image quality | Low |

**When to Use:** Choose based on task - OCR for text rendering, VLM for preferences.

---

<a name="3-key-methods"></a>
## 3. Key Methods

### FlowGRPO

**Paper:** [FlowGRPO: Flow Matching based RL for Diffusion Models](https://arxiv.org/abs/2505.05470) | **Year:** 2025 | **Venue:** ArXiv

**Summary:** Online policy method for flow-matching models using multi-step SDE sampling and model-based rewards.

**Key Contributions:**
- Multi-step SDE sampling for diffusion policy exploration
- Model-based rewards for generation quality assessment
- CLIP-style loss for policy optimization
- ~25% higher throughput than diffusers-based implementation

**Architecture:**
```
Rollout → Reward Model → Advantage Computation
                ↓
Policy → FlowGRPO Loss → Weight Update
```

**Results:**

| Mode | Throughput | Step Time |
|------|------------|----------|
| Co-located | 0.305 img/GPU/s | 420s |
| Async reward | 0.280 img/GPU/s | 360s |
| Full-model | 0.510 img/GPU/s | 250s |

**Supports:** Qwen-Image, BAGEL

---

### MixGRPO

**Paper:** [MixGRPO](https://arxiv.org/abs/2507.21802) | **Year:** 2025 | **Venue:** ArXiv

**Summary:** RL with mixed reward signals for unified understanding + generation models.

**Key Contributions:**
- Handles both understanding and generation tasks
- Mixed reward aggregation
- Stable training across modalities

**Supports:** HunyuanImage-3.0

---

### GSPO

**Paper:** [GSPO: Generalist Sub-Policy Optimization](https://arxiv.org/abs/2507.18071) | **Year:** 2025 | **Venue:** ArXiv

**Summary:** RL for omni-modality models handling text, image, video, and audio jointly.

**Key Contributions:**
- Unified framework for all modalities
- Handles heterogeneous I/O patterns
- Cross-modal learning

**Supports:** Qwen3-Omni-Thinker

---

<a name="4-comparison--tradeoffs"></a>
## 4. Comparison & Tradeoffs

### Summary Comparison Table

| Aspect | VeRL-Omni | diffusers + naive | verl (LLM) |
|--------|-----------|----------------|-------------|
| Model support | Diffusion + Omni | Diffusion only | LLM only |
| Throughput | High | Low | N/A |
| Modality | Multi (img,vid,aud,txt) | Single | Text only |
| Hardware | GPU + NPU | GPU only | GPU only |

### Tradeoffs Analysis

**LoRA vs Full-model:**
- What you gain: LoRA uses less memory
- What you lose: Some quality with full-model
- When it matters: GPU-constrained environments

**Async vs Sync Reward:**
- What you gain: ~14% faster with async
- What you lose: Added complexity
- When it matters: Production training

### Quick Reference Decision Guide

| If you need... | Use | Why |
|---------------|-----|-----|
| Fast training | VeRL-Omni + vLLM | 25% faster |
| Text generation | verl (not VeRL-Omni) | Different stack |
| Diffusion | VeRL-Omni | Specialized |
| NPU support | VeRL-Omni | Supports Ascend |

---

<a name="5-applications"></a>
## 5. Applications

### Robotics / Autonomous Driving

- **Image generation** — Diffusion-based world models
- **Video prediction** — Future scene rendering
- **Multimodal planning** — Text + image instruction following
- **Reward modeling** — VLM-based evaluation

### Other Domains

- **Text-to-image** — Qwen-Image, SD3.5
- **Text-to-video** — Wan2.2
- **Unified understanding + generation** — BAGEL, HunyuanImage

### What's NOT applicable

- Pure text LLM RL (use verl instead)
- Real-time inference (use vLLM-Omni alone)
- Single GPU extremely small scale (overhead not worth it)

---

<a name="6-open-problems"></a>
## 6. Open Problems

1. **Algorithm expansion:** More algorithms like DiffusionNFT under development
   - Current: FlowGRPO, MixGRPO, GSPO, DPO
   - Coming: DiffusionNFT

2. **Full async RL:** Currently async-reward, working toward full async
   - Actor, rollout, reward all async
   - Higher GPU utilization

3. **More model support:**
   - SD3.5 DPO (WIP)
   - DanceGRPO for Wan2.2 (WIP)

---

<a name="7-references"></a>
## 7. References

- [VeRL-Omni GitHub](https://github.com/verl-project/verl-omni) — Official code
- [Documentation](https://verl-omni.readthedocs.io/en/latest/index.html) — Full docs
- [Blog announcement](https://vllm-project.github.io/2026/05/14/verl-omni.html) — vLLM blog
- [FlowGRPO paper](https://arxiv.org/abs/2505.05470) — Core algorithm
- [MixGRPO paper](https://arxiv.org/abs/2507.21802) — Mixed rewards
- [GSPO paper](https://arxiv.org/abs/2507.18071) — Omni-modality

---

## Quick Start

```bash
# Installation
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
pip install -e .

# Quickstart (FlowGRPO)
cd examples/flowgrpo_trainer
python run.py config.yaml
```

**Requirements:**
- NVIDIA GPU (H800 recommended) or Ascend NPU
- 8+ GPUs for training
- vLLM-Omni for rollout

---

## Notes

*VeRL-Omni is the go-to framework for any diffusion or omni-modality RL training. Integrates tightly with vLLM-Omni for high throughput. For text-only LLM RL, use the original verl instead.*

*For our pipeline: Consider VeRL-Omni for diffusion-based world model training when RL refinement is needed.*

---

*Created using the standard paper survey format from AGENTS.md*
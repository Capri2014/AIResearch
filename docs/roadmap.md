# Roadmap / TODOs

This document mirrors (and can expand) the **TODO (roadmap)** section in the repo README.

## microgpt backbone (CoT for autonomous driving)

Goal: incorporate a **microgpt** component that provides the backbone for *reasoning/tracing* (CoT-style, but ideally structured) in the autonomous-driving stack.

Key decisions:
- **Integration shape:** library module (import) vs service (API/daemon).
- **Trace format:** prefer structured traces (JSON/tool traces) over free-form "CoT text".

Planned follow-up:
- **LoRA/PEFT update tricks:** train head + LoRA adapters (instead of full backbone finetune).
  - Provide a clean config switch: `--freeze-backbone`, `--lora-rank`, etc.
  - Define checkpoint format: backbone base ref + LoRA weights + head weights.

## Model/provider eval

Try **MLG / MinMax / Doubao**
- Identify official endpoints/SDKs, auth method, and pricing.
- Run a small repeatable benchmark suite:
  - coding task (repo-aware)
  - long-context summarization/survey
  - tool use / function calling / structured output
  - refusal/safety behavior quality
- Record results + choose recommendations (default + when to use each).

## Survey: Karpathy microgpt/nanoGPT gist (atomic GPT)

Status: **paused** (per Qi, 2026-02-14).

Source: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Goal: extract reusable abstractions/patterns for our autonomy stack:
- "atomic decomposition" (params / forward / cache / loss / update / sampling)
- stepwise inference + caching semantics
- minimal eval harness mindset

Deliverables:
- a concrete **trace schema** draft for autonomy planning
- a minimal **eval harness** outline (scenarios + metrics + regression gates)

## PPO improvements for large models

Goal: improve PPO stability and explore model-based RL approaches that scale well with large foundation models.

Planned follow-up:
- **Stable PPO**: Implement advances from successful RLHF pipelines (clipping tricks, value function centering, advantage normalization, GAE tuning)
- **Model-based RL**: Explore world model integration (e.g., GAIA-2 style latent dynamics) for more sample-efficient policy learning
- **Large model compatibility**: Ensure delta-waypoint head works with scaled encoders/backbones; consider LoRA for the RL head
- **Training stability**: Add KL divergence constraints, learning rate scheduling, checkpoint selection by policy entropy

## CoT data generation and finetuning for autonomous driving

Goal: explore Chain-of-Thought reasoning data generation and finetuning for improved driving decisions.

Planned follow-up:
- **CoT data synthesis**: Generate reasoning traces from expert drivers or rule-based planners
- **Structured trace format**: Define schema for driving reasoning (perception → prediction → planning → action)
- **Finetuning**: Fine-tune vision-language or action models on CoT-augmented data
- **Evaluation**: Measure decision quality improvements from CoT reasoning

## RL algorithm upgrade: GRPO / Agent RL

Goal: move beyond basic PPO to more capable RL algorithms suitable for autonomous driving.

Planned follow-up:
- **GRPO (Group Relative Policy Optimization)**: Implement GRPO for autonomous driving tasks
- **Agent RL**: Explore agent-centric RL formulations for long-horizon driving decisions
- **Benchmark**: Compare PPO vs GRPO on waypoint prediction and control tasks
- **Scaling**: Ensure algorithm works with large foundation models and high-dimensional inputs

## DeepSeek RL pipeline: Pre-train → Reasoning SFT → RL → SFT/RLHF

Goal: survey DeepSeek's RL pipeline and upgrade our current pipeline accordingly.

Planned follow-up:
- **Survey DeepSeek RL**: Study DeepSeek's training recipe (pre-train → reasoning SFT → RL → SFT/RLHF)
- **Gap analysis**: Compare current pipeline (SFT → RL refinement) with DeepSeek's approach
- **Reasoning SFT**: Add explicit CoT/Reasoning SFT stage before RL
- **GRPO integration**: Replace or supplement PPO with GRPO in the RL stage
- **Iterative improvement**: Plan SFT/RLHF cycles for continuous refinement
- **Evaluation**: Define metrics for reasoning quality and driving safety


## Survey: Kimi RL + LengthControl (Moonshot AI)

Goal: survey Kimi's unique RL approach and LengthControl mechanism.

Planned follow-up:
- **Source**: Kimi/Moonshot AI papers or technical reports on their RL training methodology
- **LengthControl**: Understand their approach to managing output length in RL training
- **RL design**: Extract unique insights in their RL algorithm design that differ from PPO/GRPO
- **Applicability**: Assess which insights apply to autonomous driving (long-horizon planning, safety constraints)
- **Comparison**: Compare with DeepSeek's RL pipeline approach

## Survey: DeepSeek Engram (Memory-Augmented Language Models)

Goal: survey DeepSeek's Engram paper on efficient knowledge retrieval via N-gram embeddings and hash lookup.

Context: DeepSeek Engram builds on two research lines:
- **Memory research**: Transformer FFN as KV Memory, Product Key Memory, RETRO, External Memory
- **N-gram research**: N-Grammer, Scaling Embedding Layer

Core ideas:
- **N-gram Embedding + Hash lookup**: O(1) complexity knowledge retrieval
- **Sparsity**: Key to breaking "impossible triangle" (Performance / Compute / Model Size)
- **Gating + Memory Hierarchy**: Efficient knowledge storage and retrieval
- **Relationship to MoE**: Engram complements MoE architectures

Planned follow-up:
- **Survey Engram paper**: Study the core idea of N-gram Embedding + Hash for efficient retrieval
- **Technical points**: Understand Sparsity, Gating mechanisms, Memory Hierarchy
- **Applicability**: Assess relevance to autonomous driving (scene understanding, rule retrieval, safety knowledge)
- **Implementation**: Explore adding external memory to vision backbone for efficient knowledge access
- **Comparison**: Contrast with traditional FFN as implicit memory

Source: https://www.bilibili.com/video/BV1x3zWB6EU6/

---

## Survey: GigaBrain (VLA + World Model RL)

**Status:** **NEW** (added 2026-02-16)

**Source:** https://arxiv.org/abs/2602.12099  
**Project:** https://gigabrain05m.github.io/

**TL;DR:** VLA model trained via world model-based RL with RAMP (Reinforcement leArning via world Model-conditioned Policy)

**Key Results:**
- Pre-trained on 10,000+ hours of robotic manipulation data
- RAMP achieves ~30% improvement over RECAP baseline on Laundry Folding, Box Packing, Espresso Preparation
- Reliable long-horizon execution for complex manipulation tasks

**Core Innovation:**
```
GigaBrain-0.5 (pre-trained on 10K hours robot data)
        │
        ▼
┌─────────────────┐
│  World Model    │  Video-based world model for dynamics forecasting
│  (pre-trained)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     RAMP        │  World Model-Conditioned Policy
│                 │  (RL with world model conditioning)
└────────┬────────┘
         │
         ▼
GigaBrain-0.5M* (VLA with world model RL)
```

**Relevance to Our Pipeline:**

| Our Component | GigaBrain Application |
|--------------|----------------------|
| **World Model** | Use pre-trained world model for trajectory forecasting |
| **RAMP** | Conditioning policy on world model predictions |
| **Long-horizon** | Apply to complex driving scenarios |
| **VLA** | Vision-Language-Action for driving decisions |

**Planned follow-up:**
- **Survey RAMP**: Understand world model-conditioned policy learning
- **World Model Integration**: Explore using world models (GAIA-2 style) for driving
- **RAMP for Driving**: Adapt RAMP to autonomous driving scenarios
- **Long-horizon Planning**: Use world model predictions for trajectory planning
- **Comparison**: Contrast with our current PPO approach

---

## Survey Candidate Pipeline

The following papers are queued for future survey:

1. **GigaBrain** - VLA + World Model RL (NEW, 2026-02-16)
2. **GenAD** - Generalized predictive model for driving (CVPR 2024)
3. **VAD** - Vectorized autonomous driving planning
4. **UniAD** - Unified autonomous driving framework
5. **BEVFormer** - Bird's-eye view transformer

**Selection Criteria:**
- Relevance to driving/robotics
- Novel methodology
- Open-source implementation
- Scalability to our pipeline

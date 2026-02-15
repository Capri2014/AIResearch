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


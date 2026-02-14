# Roadmap / TODOs

This document mirrors (and can expand) the **TODO (roadmap)** section in the repo README.

## microgpt backbone (CoT for autonomous driving)

Goal: incorporate a **microgpt** component that provides the backbone for *reasoning/tracing* (CoT-style, but ideally structured) in the autonomous-driving stack.

Key decisions:
- **Integration shape:** library module (import) vs service (API/daemon).
- **Trace format:** prefer structured traces (JSON/tool traces) over free-form “CoT text”.

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

Source: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Goal: extract reusable abstractions/patterns for our autonomy stack:
- “atomic decomposition” (params / forward / cache / loss / update / sampling)
- stepwise inference + caching semantics
- minimal eval harness mindset

Deliverables:
- a concrete **trace schema** draft for autonomy planning
- a minimal **eval harness** outline (scenarios + metrics + regression gates)


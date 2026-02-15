# Roadmap / ToDos

## Research / Evaluation

### Survey: Karpathy microgpt/nanoGPT gist (atomic GPT in one file)

Goal: extract reusable abstractions/patterns to borrow for our **autonomous-driving CoT backbone** (trace schema + eval harness + stepwise inference semantics), using the gist as conceptual scaffolding.

Checklist:
- Summarize components: dataset/tokenizer, KV cache, MHA/MLP blocks, loss, Adam, sampling.
- Identify what to keep as *conceptual scaffolding* vs what to replace (PyTorch kernels, tokenizer, eval harness).
- Draft a minimal test harness we can reuse for future model experiments.

### Try MLG / MinMax / Doubao

Goal: evaluate these models/providers for quality, latency, cost, and reliability for our typical workflows.

Checklist:
- Identify official endpoints + SDKs, auth method, pricing.
- Run a small, repeatable benchmark suite:
  - Coding task (repo-aware).
  - Long-context summarization/survey.
  - Tool use (function calling / structured output).
  - Safety/guardrail behavior (refusal quality).
- Record results (notes + example prompts + outputs) and pick a default/where each fits.

---

## Repo

### Cooperate the `microgpt` project into this repo

Goal: bring the external `microgpt` project into this repository in a clean, maintainable way.

Context: intended to serve as the **backbone of CoT (chain-of-thought) for autonomous driving**.

Add later: **LoRA/PEFT update tricks** (train head + LoRA adapters instead of full backbone finetune; define a clean config switch + checkpoint format).

Steps:
1. **Clarify source + desired integration mode**
   - Where is `microgpt` currently hosted (URL / local path)?
   - Prefer: git submodule vs subtree vs vendor-copy (default recommendation: **subtree** if we want one repo, **submodule** if we want to keep histories separate).
2. **Decide target location**
   - e.g. `projects/microgpt/` or `packages/microgpt/`.
3. **Unify tooling**
   - Align Node/Python versions, lockfiles, lint/format, CI.
4. **Wire it into the repo**
   - Add build/test commands to root scripts / CI.
   - Add docs pointer from root README.
5. **Smoke test**
   - Run microgpt quickstart inside the monorepo layout.

Acceptance criteria:
- `microgpt` code is present under the repo with a documented update workflow.
- Root-level docs + CI can run `microgpt` tests (or explicitly skip with rationale).

### PPO improvements for large models

Goal: improve PPO stability and explore model-based RL approaches that scale well with large foundation models.

Checklist:
- Implement stable PPO advances (clipping tricks, value function centering, advantage normalization, GAE tuning)
- Explore model-based RL integration (e.g., GAIA-2 style latent dynamics) for sample-efficient policy learning
- Ensure delta-waypoint head works with scaled encoders/backbones; consider LoRA for RL head
- Add KL divergence constraints, learning rate scheduling, checkpoint selection by policy entropy

### CoT data generation and finetuning for autonomous driving

Goal: explore Chain-of-Thought reasoning data generation and finetuning for improved driving decisions.

Checklist:
- CoT data synthesis: generate reasoning traces from expert drivers or rule-based planners
- Structured trace format: define schema for driving reasoning (perception → prediction → planning → action)
- Finetuning: fine-tune vision-language or action models on CoT-augmented data
- Evaluation: measure decision quality improvements from CoT reasoning

### RL algorithm upgrade: GRPO / Agent RL

Goal: move beyond basic PPO to more capable RL algorithms suitable for autonomous driving.

Checklist:
- GRPO (Group Relative Policy Optimization): implement for driving tasks
- Agent RL: explore agent-centric formulations for long-horizon decisions
- Benchmark: compare PPO vs GRPO on waypoint prediction and control
- Scaling: ensure algorithm works with large foundation models

### DeepSeek RL pipeline upgrade

Goal: survey DeepSeek's training recipe and upgrade our pipeline accordingly.

Checklist:
- Survey DeepSeek RL: pre-train → reasoning SFT → RL → SFT/RLHF
- Gap analysis: compare current pipeline (SFT → RL) with DeepSeek's approach
- Reasoning SFT: add explicit CoT/Reasoning SFT stage before RL
- GRPO integration: replace/supplement PPO with GRPO in RL stage
- Iterative improvement: plan SFT/RLHF cycles for continuous refinement

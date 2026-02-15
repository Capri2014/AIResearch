# AIResearch - Physical AI for Autonomous Driving + Robotics (Sim/Data + Large Models)

This repo is a **starter skeleton** for two tightly-coupled tracks:

1) **Large models for autonomy**: policies that map (vision, state, optional language) → actions for driving + robots.
2) **Simulation & data**: using simulation to generate training data and run repeatable evaluation suites.

> Status: initial scaffold commit. The goal is to make the project structure "real" first, then iterate.

---

## Why simulation?
Simulation is our multiplier:
- **Training**: generate large-scale rollouts, rare edge cases, and perfect labels.
- **Evaluation**: run deterministic scenario suites for regression testing and safety gating.
- **Sim→Real**: use domain randomization + system ID to bridge gaps.

See: `docs/sim_training_eval.md`.

Also see: `docs/roadmap.md` (expanded TODOs).

---

## Repo map
- `docs/` - summaries + references
- `sim/` - simulation env adapters, scenarios, evaluation
  - `sim/driving_carla/` - CARLA-first driving sim scaffold
  - `sim/robotics/` - robotics sim scaffold (Isaac and MuJoCo tracks)
- `models/` - unified policy interface + stubs
- `demos/` - "run it" entrypoints (thin wrappers)
- `data/` - dataset schemas + tooling stubs
- `results/` - metrics outputs (tracked or gitignored later)

---

## Quickstart (placeholder)
This scaffold does **not** install CARLA/Isaac/MuJoCo yet. Start by reading:

- `docs/overview.md`
- `sim/driving_carla/README.md`
- `sim/robotics/README.md`

When you're ready, we'll add:
- a minimal CARLA docker-compose workflow
- a minimal scenario runner + eval that outputs JSON metrics
- a small dataset format + recorder

---

## Design principles
- **One policy interface** across domains (`models/policy_interface.py`)
- **Scenario-driven evaluation** (YAML/JSON scenario specs)
- **Episode recorder** writes unified dataset format (`sim/common/recorder.py`)
- Prefer **reproducible metrics** over flashy demos.

---

## TODO (roadmap)

### microgpt backbone (CoT for autonomous driving)
- Cooperate a **microgpt** component into this repo as the backbone for CoT-style reasoning/tracing in driving.
  - Decide integration shape: library module vs service.
  - Add later: **LoRA/PEFT update tricks** (train head + LoRA adapters vs full-backbone finetune; clean config switch + checkpoint format).

### Model/provider eval
- Try **MLG / MinMax / Doubao**
  - Evaluate quality/latency/cost + tool-use/structured-output reliability.
  - Create a tiny repeatable benchmark suite + write up results.

### Survey: Karpathy microgpt/nanoGPT gist (atomic GPT)
- **Paused** for now (per Qi).

### Agent Swarm (reusable framework)
- Build a reusable **agent swarm framework** for this repo (task router + planner + worker agents + aggregator):
  - supports "research → implement → eval" parallel lanes
  - emits structured artifacts (notes, citations, code PRs, eval reports)
  - standard prompt templates + checklists for consistency

### Robotics VLA base (Xiaomi)
- Review and integrate Xiaomi's robotics **VLA base** ideas into our roadmap:
  - source: https://mp.weixin.qq.com/s/FEMJBekqPHpMpE60b8kV4A
  - extract: model interface (obs/action), data recipe, training stages (pretrain → BC/SFT → RL), and evaluation protocol
  - decide: what we can reuse for our driving-first plan + later robotics track

### RL fine-tuning (behind SFT)
- Add `training/rl/train_ppo.py` MVP: PPO on waypoint policy
  - Use CARLA/ScenarioRunner as env for online RL rollouts
  - Minimal: single-worker, CPU env, JSON metrics output
  - First target: improve waypoint accuracy beyond BC baseline

### Drive-JEPA inspired pretrain (post-RL)
- Implement **JEPA (Joint Embedding Predictive Architecture)** for driving SSL:
  - Masked encoder training: predict future embeddings from past
  - Proposal/scoring heads: learn which spatial regions matter most
  - Multi-cam fusion: aggregate from multiple views with learned attention
- Reference: XPeng Drive-JEPA (arXiv:2601.22032) for training recipe

### Drive-JEPA waypoint head (post-RL)
- Add **multi-step waypoint head** inspired by Drive-JEPA:
  - Predict N waypoints (e.g., 20 @ 10Hz = 2s future)
  - Per-waypoint confidence scores (proposal scoring)
  - Ego-frame coordinate output (x, y in meters)
- Integrates with BC → RL pipeline after baseline waypoint head works

### Driving world model (Waymo)
- Add and review the **Waymo world model** paper:
  - distill: what's being modeled (BEV/occupancy/video), conditioning signals, rollout horizon
  - training objective(s) + dataset recipe
  - how it plugs into planning (MPC / sampling / value estimation)
  - what we can reuse for encoder pretraining + ScenarioRunner evaluation

### XPeng Drive-JEPA (vs Waymo WorldModel)
- Review and integrate **XPeng Drive-JEPA** into our roadmap:
  - source: https://mp.weixin.qq.com/s/JqJMY_f0gh7Zn3b2KXV3AQ
  - extract: training stages (pretrain encoder/world model → distill/BC/RL), inputs/outputs (BEV/video), eval protocol
  - compare vs Alpamayo-R1: what's "JEPA/world-model" vs what's "VLA policy"; where each fits in our pipeline

### WorldModel rollout (Genie 3 vs Lingbot)
- Study whether we should use **Lingbot** instead of **Genie 3** (Waymo) for world-model rollout:
  - clarify: which Genie 3 capability we need (video prediction? BEV rollout? action-conditioned simulation?)
  - evaluate: Lingbot's inputs/outputs, training recipe, licensing/availability
  - decide: integration plan with Waymo pretrain + CARLA ScenarioRunner evaluation

### Production-ready pipeline optimization (later)
- Revisit our **episodes vs TFRecord streaming** choice and optimize towards a production-ready training pipeline:
  - reduce storage/IO overhead (e.g., WebDataset tar shards, LMDB, or parquet-backed metadata)
  - deterministic sharding + caching for distributed training
  - data versioning + conversion reproducibility
  - keep optional TFRecord streaming as a backend if it becomes necessary at scale

### Flywheel efficiency: PySpark + Delta Lake (exploration)
- Explore using **PySpark + Delta Lake** to improve data flywheel efficiency:
  - episode/frame indexing at scale (faster slicing/filtering than per-JSON scans)
  - incremental conversion + versioned datasets

### Survey: GAIA-2 (Wayve)
- Review **GAIA-2** (Wayve's video-generative world model):
  - source: https://wayve.ai/thinking/gaia-2/
  - extract: video tokenizer + latent diffusion architecture, conditioning (ego-action, weather, geometry), geographic diversity, safety-critical scenario generation
  - compare vs DriveArena WorldDreamer vs SceneDiffuser: what's different in world model approach?
  - decide: potential for synthetic data augmentation in our pipeline
  - simple dataset statistics/QA jobs (camera coverage, missing frames, label sanity)

### ClawBot LLM cost optimization
- Optimize ClawBot's LLM usage to reduce cost while preserving quality:
  - model tiering (cheap default for routine CLs; upgrade for deep research/debugging)
  - agent-router for trivial vs complex tasks
  - context minimization (small rolling state file; avoid large diffs unless needed)
  - batch tool calls; avoid browser flows unless necessary

### World Arena benchmark
- Study **World Arena** (world-arena.ai) and assess relevance to our world-model + autonomy evaluation:
  - project: http://world-arena.ai
  - paper: http://arxiv.org/abs/2602.08971
  - leaderboard: https://huggingface.co/spaces/WorldArena/WorldArena
  - code/data: https://github.com/tsinghua-fib-lab/WorldArena
  - extract: tasks, metrics, environment setup, and how to integrate as an eval lane

### HoloBrain VLA (Horizon Robotics)
- Study **HoloBrain VLA** and assess what's reusable for our robotics track:
  - tech report: https://arxiv.org/abs/2602.12062
  - model code: https://github.com/HorizonRobotics/RoboOrchardLab
  - real-robot infra: https://github.com/HorizonRobotics/RoboOrchard
  - project: https://horizonrobotics.github.io/robot_lab/holobrain
  - extract: data recipe, model interface (obs/action), training stages, evaluation on real hardware

### PPO improvements for large models
- Improve PPO stability and explore model-based RL approaches that scale well with large foundation models:
  - Implement stable PPO advances (clipping tricks, value centering, GAE tuning)
  - Explore model-based RL integration (GAIA-2 style latent dynamics)
  - Ensure delta-waypoint head works with scaled encoders; consider LoRA for RL head
  - Add KL divergence constraints, LR scheduling, entropy-based checkpointing

### CoT data generation and finetuning for autonomous driving
- Explore Chain-of-Thought reasoning data generation and finetuning for improved driving decisions:
  - CoT data synthesis: generate reasoning traces from expert drivers or rule-based planners
  - Structured trace format: define schema for driving reasoning (perception → prediction → planning → action)
  - Finetuning: fine-tune vision-language or action models on CoT-augmented data
  - Evaluation: measure decision quality improvements from CoT reasoning

### RL algorithm upgrade: GRPO / Agent RL
- Move beyond basic PPO to more capable RL algorithms suitable for autonomous driving:
  - GRPO (Group Relative Policy Optimization): implement for driving tasks
  - Agent RL: explore agent-centric formulations for long-horizon decisions
  - Benchmark: compare PPO vs GRPO on waypoint prediction and control
  - Scaling: ensure algorithm works with large foundation models

### DeepSeek RL pipeline upgrade
- Survey DeepSeek's training recipe and upgrade our pipeline accordingly:
  - Survey DeepSeek RL: pre-train → reasoning SFT → RL → SFT/RLHF
  - Gap analysis: compare current pipeline (SFT → RL) with DeepSeek's approach
  - Reasoning SFT: add explicit CoT/Reasoning SFT stage before RL
  - GRPO integration: replace/supplement PPO with GRPO in RL stage
  - Iterative improvement: plan SFT/RLHF cycles for continuous refinement

---

## Maintenance
This repository is maintained with help from **ClawBot (OpenClaw)**.

- I will open PRs from `feature/*` branches.
- You review and merge into `main`.
- No secrets are committed to the repo; credentials live only on the VPS.

See `docs/branch-pr-workflow.md` for branch naming conventions and PR guidelines.

## License
Add a license in `LICENSE` (MIT/Apache-2.0 are common).
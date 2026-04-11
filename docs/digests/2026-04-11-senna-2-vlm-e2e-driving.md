# Senna-2: Aligning VLM and End-to-End Driving Policy — Digest

**Date:** 2026-04-11  
**Status:** Survey Complete  
**Source:** arXiv:2603.11219 (Mar 2026), [Project Page](https://ambitious-idiot.github.io/senna2-project), [Code](https://github.com/hustvl/Senna), [Model](https://huggingface.co/rb93dett/Senna)

---

## TL;DR (5 bullets)

- **Dual-System Consistency Focus**: First E2E stack explicitly aligning VLM high-level decisions with E2E low-level trajectories — addresses the "say-do gap" in VLM-augmented driving.
- **Decision Adapter**: Learnable module translating VLM semantic embeddings → E2E latent conditioning — eliminates brittle discrete command passing.
- **Closed-Loop HRL**: Bottom-up hierarchical RL in 3DGS sims for safety alignment beyond open-loop imitation — key differentiator from OpenDriveVLA/ORION.
- **SOTA Metrics**: +19.3% F1 (consistency), -5.7% FDE (open-loop), -30.6% AF-CR (closed-loop safety) vs prior VLM+E2E.
- **Post-UniAD + Camera-First**: Pure camera input, no LiDAR — directly maps to Tesla "vision-only" philosophy with semantic grounding.

---

## Problem

1. **VLM-E2E Misalignment**: VLM decides "turn left" but E2E trajectory may drift — existing systems treat them as loose composing modules.
2. **Discrete Command Bridging**: Many VLM+E2E stacks use discrete meta-actions (e.g., "lane change") to communicate — loses nuance and context.
3. **Open-Loop Bias**: Most VLM driving papers report only L2/ADE metrics — no closed-loop safety validation (collision rate, intervention frequency).
4. **Long-Tail Reasoning**: VLM provides commonsense, but E2E policy can override or ignore it — no enforcement mechanism.
5. **Training Instability**: Joint VLM+E2E training often diverges — frozen VLM + frozen E2E leaves no gradient path for consistency.

---

## Method

### System Decomposition

```
[Multi-view Cameras (6x)] → [Vision Encoder]
                                ↓
                    ┌───────────────────────────────┐
                    │     Senna-VLM (LVLM)         │
                    │  ┌─────────────────────────┐  │
                    │  │  LLaVA-based 7B/13B   │  │
                    │  │  → Semantic Decisions  │  │
                    │  └─────────────────────────┘  │
                    │            ↓                │
                    │   [Decision Adapter]        │
                    │   (embedding projector)   │
                    └───────────────────────────────┘
                                ↓
                    ┌───────────────────────────────┐
                    │     Senna-E2E (Planner)    │
                    │  ┌─────────────────────────┐  │
                    │  │  Transformer Planner  │  │
                    │  │  (trajectory decode)   │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                ↓
                    [Waypoints (x, y, θ) × T]
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Senna-VLM** | LLaVA-based LVLM (vicuna-7B/13B backbone), generates semantic decisions + scene understanding |
| **Decision Adapter** | Learnable MLP projecting VLM embeddings → E2E latent conditioning vectors |
| **Senna-E2E** | Transformer-based trajectory planner, receives conditioning from adapter |
| **3DGS Env** | 3D Gaussian Splatting simulator for closed-loop RL alignment |

### Training Objectives (Three-Stage)

| Stage | Objective | Key Mechanism |
|-------|-----------|---------------|
| **Stage 1** | Driving pre-training | Decision adapter learns VLM→E2E mapping, frozen VLM, fine-tune adapter+E2E |
| **Stage 2** | Open-loop consistency alignment | Minimize FDE while maximizing decision-F1 (semantic alignment) |
| **Stage 3** | Closed-loop HRL | Bottom-up hierarchical RL in 3DGS envs — rewards safety + efficiency + consistency |

**Losses**:
- Stage 1: L2 waypoint loss + auxiliary VQA loss
- Stage 2: L2 + Decision F1 (VLM decision vs E2E implicit decision)
- Stage 3: RL reward = -λ₁·collision - λ₂·off-road - λ₃·inconsistency + λ₄·efficiency

### Inputs / Outputs

| Input | Details |
|-------|---------|
| Multi-view cameras | 6 cameras (nuScenes setup) |
| Ego state | Speed, heading, timestamp |
| (Optional) Text command | "turn left at intersection" |

| Output | Details |
|--------|---------|
| Trajectory | 2-second horizon, 4-8 waypoints |
| Semantic decision | High-level decision string (e.g., "yield to pedestrian") |
| Planning embedding | Latent vector (internal) |

---

## Data / Training

- **Pretraining**: DriveX (large-scale, 1M+ frames, pseudo-labels from rule-based planner)
- **Fine-tuning**: nuScenes (20K training, 6-camera)
- **Closed-loop**: nuScenes-based 3DGS reconstruction + CARLA towns
- **QA Data**: LLaVA-generated planning-oriented Q&A (see original Senna)

---

## Evaluation

### Metrics

| Metric | Description | Senna-2 vs Prior |
|--------|-------------|------------------|
| **FDE** | Final Displacement Error (m) | -5.7% vs VLM+E2E baselines |
| **ADE** | Average Displacement Error (m) | Comparable |
| **Decision F1** | VLM decision vs ground-truth semantic alignment | +19.3% (primary contribution) |
| **AF-CR** | Average Failure Collision Rate (%) | -30.6% (closed-loop) |

### Benchmarks

- **nuScenes** (open-loop planning)
- **Bench2Drive** (CARLA closed-loop) — note: not explicitly mentioned in Senna-2 paper but compatible
- **3DGS Sim** (custom nuScenes reconstructions for RL)

---

## Tesla / Ashok Alignment

| Tesla Claim | Senna-2 Mapping | Gap |
|------------|----------------|-----|
| **Camera-first** | ✅ Pure 6-camera input, no LiDAR | ✅ Direct alignment |
| **Long-tail handling** | VLM provides commonsense reasoning for rare scenarios | Limited — still relies on imitation +RL, no explicit "shadow mode" |
| **End-to-end** | ✅ Camera→trajectory E2E gradient (Stage 2), augmented with semantic signal | Partially modular: VLM frozen; adapter learns cross-modal mapping |
| **Regression testing** | Closed-loop AF-CR measures collision/intervention | ✅ Concrete metric |
| **No HD maps** | ✅ Map-free from camera | ✅ Direct alignment |
| **Neural planner** | VLM as "reasoning copilot" + E2E as "executor" | Tesla uses single monolithic network; Senna-2 uses dual-system with adapter |

**What Senna-2 doesn't address**:
- Real-time VLM inference latency (current LVLM ~3-5s/frame on A100)
- Fleet learns no explicit "shadow mode" — safety depends on RL reward shaping
- No "take-over" protocol for LLM hallucinations
- Notesla-style "影像记忆" (video memory) for long-horizon context

---

## What to Borrow for AIResearch

### High Priority

1. **Decision Adapter Architecture**
   - Simple MLP + layer norm between VLM embedding space and E2E latent
   - Enables gradient-based alignment without retraining frozen VLM
   - Can swap VLM backbone (LLaVA → GPT-4V, etc.)

2. **Closed-Loop HRL Protocol**
   - 3DGS environments for realistic rendering + low inference cost
   - Bottom-up hierarchical RL: high-level decision policy + low-level trajectory policy
   - Composite reward: collision + off-road + efficiency + **consistency penalty**

3. **Decision F1 Metric**
   - Compute semantic alignment between VLM's decoded decision and E2E's implicit trajectory intent
   - Novel evaluation angle not in prior E2E papers

### Medium Priority

- **Planning-Oriented Q&A Generation** (from original Senna): Use LLaVA to auto-generate "why did you brake?" → "pedestrian crossing" pairs
- **Three-Stage Training**: Even for production, staging helps stability

### Lower Priority

- Full-parameter fine-tuning (vs LoRA) — may want LoRA for faster iteration

---

## Citations

```
@article{jiang2024senna,
  title={Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving},
  author={Bo Jiang and Shaoyu Chen and Bencheng Liao and Xingyu Zhang and Wei Yin and Qian Zhang and Chang Huang and Wenyu Liu and Xinggang Wang},
  year={2024},
  eprint={2410.22313},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2410.22313},
}

@article{song2026senna2,
  title={Senna-2: Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning},
  author={Yuehao Song and Shaoyu Chen and Hao Gao and Yifan Zhu and Weixiang Yue and Jialv Zou and Bo Jiang and Zihao Lu and Yu Wang and Qian Zhang and Xinggang Wang},
  year={2026},
  eprint={2603.11219},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.11219},
}
```

- **Paper**: https://arxiv.org/abs/2603.11219  
- **Code**: https://github.com/hustvl/Senna  
- **Project**: https://ambitious-idiot.github.io/senna2-project  
- **Model**: https://huggingface.co/rb93dett/Senna

---

## PR Link

[Create PR: senna-2-vlm-e2e-driving digest](https://github.com/YourOrg/repo/pull/new/feature/senna-2-digest)

---

## Summary (3 bullets)

- **Senna-2 (Mar 2026)** is the latest VLM+E2E driving stack from HUST+Horizon Robotics — it introduces a **Decision Adapter** that bridges VLM semantic decisions to E2E trajectory planning with explicit **dual-system consistency alignment** (Decision F1 metric).
- The key innovation is **Stage 3 closed-loop HRL in 3DGS environments** — unlike OpenDriveVLA (open-loop only) and ORION (generative planner), Senna-2 enforces safety beyond imitation by reinforcing consistency between "what VLM says" and "what E2E does."
- For AIResearch, prioritize borrowing the **Decision Adapter** (simple embed projector) + **Decision F1 metric** + **closed-loop HRL protocol** when building VLM-augmented driving stacks — these address the core "say-do gap" in camera-first E2E systems.
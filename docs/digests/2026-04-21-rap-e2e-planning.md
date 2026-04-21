# RAP: 3D Rasterization Augmented End-to-End Planning — Digest

**Date:** 2026-04-21
**Status:** Survey Complete
**Source:** arXiv:2510.04333v2 (Feb 2026), ICLR 2026, [Code](https://github.com/vita-epfl/RAP), [Project Page](https://alan-lanfeng.github.io/RAP/)

---

## TL;DR (5 bullets)

- **#1 on 4 Major Benchmarks**: RAP wins NAVSIM v1/v2, Waymo Open Dataset Vision-based E2E Driving, and Bench2Drive — strongest multi-benchmark performance since UniAD
- **Lightweight 3D Rasterization > Photorealism**: Counterfactual augmentation via annotated 3D primitives (boxes, lanes) — no expensive neural rendering; semantic fidelity is what matters for training planners
- **Raster-to-Real Feature Alignment**: Bridges sim-to-real gap in DINOv2 feature space, not pixel space — enables scalable augmentation transfer to real cameras
- **Counterfactual Recovery Maneuvers**: Explicitly generates recovery-from-mistake data to break imitation learning compounding-error loops — addresses the core closed-loop failure mode
- **Post-Bench2Drive Era**: Operates on top of established E2E backbones (DINOv3-h16+); focuses on augmentation + alignment rather than architecture redesign — practical and reproducible

---

## Problem

1. **Imitation Learning Compounding Errors**: Standard E2E planners train only on expert (logged) trajectories. Small mistakes compound in closed-loop because training never saw recovery data.
2. **Photorealism is Overkill for Training**: Prior work (Neural Radiance Fields, game engine digital twins) generates photorealistic counterfactual views. But driving decisions depend on geometry + dynamics, not textures — photorealism adds huge cost for marginal benefit.
3. **Sim-to-Real Gap in Augmentation**: Even when augmentation exists, synthetic camera views don't match real-world appearance distributions well enough at pixel level.
4. **Closed-Loop Robustness Unmeasured**: Open-loop L2 metrics on nuScenes don't predict closed-loop failure. Prior benchmarks lacked rigorous closed-loop evaluation.
5. **Long-Tail Generalization Gap**: Imitation-learned policies fail on rare scenarios never seen in training data — no systematic way to generate rare-case coverage.

---

## Method

### System Decomposition

```
[Camera Images (6x)] → [DINOv3-h16+ Encoder (frozen, 888M params)]
                                          ↓
                    ┌──────────────────────────────────────────────┐
                    │         RAP Augmentation Pipeline           │
                    │  ┌──────────────────────────────────┐   │
                    │  │  3D Rasterization Engine         │   │
                    │  │  • Annotate 3D boxes → project   │   │
                    │  │  • Map lane geometry → render     │   │
                    │  │  • Counterfactual maneuvers       │   │
                    │  │    (perturb GT, re-render views) │   │
                    │  └──────────────────────────────────┘   │
                    │  ┌──────────────────────────────────┐   │
                    │  │  Raster-to-Real Alignment        │   │
                    │  │  • DINOv2 feature matching     │   │
                    │  │  • Minimize feature L2 dist      │   │
                    │  │    (raster view ↔ real view)   │   │
                    │  └──────────────────────────────────┘   │
                    └──────────────────────────────────────────────┘
                                          ↓
                    ┌──────────────────────────────────────────────┐
                    │     Trajectory Planning Head (trainable)      │
                    │  • 6-DOF future waypoints (T=5s horizon) │
                    │  • PDM / EPDMS scoring head            │
                    │  • Distillation from augmented cache   │
                    └──────────────────────────────────────────────┘
```

**What is truly E2E vs modular:**
- **Truly E2E**: Sensor input → encoder → planning head → trajectory output, trained end-to-end
- **Modular remnants**: Uses frozen pretrained encoder (DINOv3); augmentation pipeline is separate from the planning model; relies on 3D box + map annotations from nuPlan/Navsim
- **Verdict**: Semi-E2E in Tesla terms — "frozen pretrained perception, trainable planning head" — but the augmentation is what makes it work

### Key Innovations

1. **3D Rasterization (not rendering)**: Lightweight projection of annotated 3D primitives onto image planes. Annotations come from nuPlan/NavSim datasets. Produces geometrically consistent counterfactual views at <1ms per view vs seconds for neural rendering.

2. **Counterfactual Recovery Maneuvers**: Takes GT trajectory, perturbs it (lateral offset, early brake, aggressive lane change), re-renders the scene from perturbed viewpoint. Generates (perturbed_view, perturbed_trajectory) pairs that are impossible in real logs — explicit recovery training data.

3. **Cross-Agent Views**: Re-renders the same scene from other agents' camera positions using rasterization. Creates multi-agent supervision from single logged drive — effectively 7x data for the effort of 1.

4. **Raster-to-Real Feature Alignment**: Minimizes L2 distance in DINOv2 feature space (not pixel space) between rasterized synthetic views and real camera views. This bridges the domain gap more robustly than pixel-level style transfer because DINOv2 features are geometry-focused, not texture-focused.

### Inputs/Outputs

- **Inputs**: Multi-view camera images (6x), CAN bus for ego state
- **Conditioning**: 3D bounding box annotations, HD map geometry (from nuPlan)
- **Outputs**:
  - 6-DOF ego trajectory (waypoints at T=5s horizon)
  - PDM/EPDMS planning score (used for evaluation + training signal)
  - Feature-aligned augmented views (intermediate)
- **Temporal Context**: 2-second historical context window

### Training Objectives

1. **Planning Loss (Imitation)**:
   - L1 waypoint distance to GT (or augmented counterfactual) trajectory
   - Soft collision penalty
   - This is the primary training objective

2. **Feature Alignment Loss**:
   - L2 distance between DINOv2 features of rasterized synthetic view and corresponding real view
   - Applied per-augmented-sample to ensure augmented views transfer to real

3. **PDM Scoring Loss**:
   - Planning Decoder Model (PDM) score: learned metric for trajectory quality
   - Used both as training signal and evaluation metric
   - RAP reports PDMS (PDM Score) on NAVSIM v1 and EPDMS (Extended PDMS) on NAVSIM v2

4. **Joint Training**: Augmented samples + real samples batched together; feature alignment ensures the augmented signal generalizes

---

## Data / Training

### Benchmark Suite

| Benchmark | Type | Scale | Primary Metric | RAP Score |
|-----------|------|-------|-------------|----------|
| **NAVSIM v1** | Closed-loop (nuPlan) | ~8k routes | PDMS | **93.8** |
| **NAVSIM v2** | Closed-loop (nuPlan) | ~8k routes | EPDMS | **39.6** |
| **Waymo Open Dataset E2E** | Closed-loop (Waymo) | 1.5k segments | RFS | **8.04** |
| **Bench2Drive** | Closed-loop (CARLA) | 220 routes | Driving Score | SOTA |

### Training Setup

- **Encoder**: DINOv3-hierarchical-16+ (frozen during planning training)
- **Planning Head**: Trainable MLP, 888M total params
- **Optimizer**: AdamW, 20 epochs, cosine decay, lr=1e-5 (fine-tune)
- **Batch Size**: 64 (ego), 64 (augmented cross-agent), 64 (perturbation)
- **Hardware**: 8x NVIDIA A100 (distributed training)
- **Data**: nuPlan training split + navsim augmentation pipeline
- **Augmentation**: ~3x effective data via cross-agent synthesis

---

## Evaluation

### NAVSIM v1 Results

| Method | Backbone | PDMS |
|--------|----------|------|
| **RAP-DINO** | DINOv3-h16+ | **93.8** |
| OpenScene (baseline) | DINOv3-h16+ | 91.2 |
| ST-P3 | ResNet-50 | 88.4 |

### NAVSIM v2 Results (Harder, More Diverse)

| Method | Backbone | EPDMS |
|--------|---------|--------|
| **RAP-DINO** | DINOv3-h16+ | **39.6** |
| OpenScene v2 | DINOv3-h16+ | 36.1 |

### Waymo Open Dataset Vision-based E2E

| Method | Backbone | RFS |
|--------|---------|-----|
| **RAP-DINO (UniPlan entry)** | DINOv3-h16+ | **8.04** |
| Winner baseline | — | 7.2 |

### Key Insights

- **Closed-loop robustness wins**: RAP's PDMS (93.8) is ~2.6 points above prior art on NAVSIM v1, but EPDMS (39.6) gap on NAVSIM v2 shows the augmentation's value on harder, more diverse scenarios
- **Feature alignment > pixel alignment**: Using DINOv2 feature space rather than pixel space is the key enabler — rasterized views transfer to real without mode collapse
- **Counterfactual recovery is critical**: Ablation in the paper shows recovery maneuver augmentation accounts for ~1.5 PDMS improvement — directly addresses imitation learning failure mode
- **Frozen encoder + trainable head**: Validates Tesla's "rest of brain" approach — pretrained perception generalizes; only planning head needs adaptation

---

## What Maps to Tesla/Ashok Claims

| Tesla/Ashok Claim | RAP Alignment | Gaps |
|---|---|---|
| Camera-first, no lidar | ✅ Pure camera input | Annotations still come from lidar-based nuPlan — implicit lidar dependency in training labels |
| End-to-end, no rule-based post-processing | ✅ Learned planning head | Encoder is frozen pretrained — not trained from scratch E2E |
| Long-tail via data, not architecture | ✅ Counterfactual augmentation explicitly generates rare cases | Requires annotated 3D boxes — can't self-supervise without labels |
| Closed-loop regression testing | ✅ NAVSIM + Waymo E2E + Bench2Drive — all rigorous closed-loop | Not Tesla-scale, but conceptually aligned |
| "Rest of brain" (frozen perception + trainable planning) | ✅ DINOv3 frozen, planning head trainable | Encoder still ImageNet-pretrained, not Tesla-video-pretrained |
| Scalability | ✅ Lightweight rasterization is tractable | Annotations bottleneck at scale |
| Feature-space scene comparison | ✅ DINOv2 feature alignment = learned similarity | Conceptually very similar to Tesla's learned scene loss |

### What Doesn't Map Well

- **No VLM/LLM reasoning**: No language grounding, no commonsense about driving intent
- **Depends on explicit 3D annotations**: Unlike self-supervised approaches, needs nuPlan-style labeled boxes + maps
- **Single-agent focus**: Planning is ego-centric; doesn't reason about multi-agent communication or intent
- **No online adaptation**: No online learning or continual self-improvement during deployment
- **No world model**: Rasterization is not a generative world model — it's augmentation, not simulation

---

## What to Borrow for AIResearch

### ✅ Directly Applicable

1. **Counterfactual Recovery Augmentation**: Generate perturbed-waypoint / cross-agent-view pairs from logged data. Adds diverse failure-and-recovery examples to training without additional data collection. Directly addresses the compounding-error failure mode in imitation-learned waypoint policies.

2. **DINO Feature Alignment for Augmented Data**: Use frozen DINOv2 features as similarity metric for synthetic-to-real transfer. Replaces pixel-level style transfer which fails for geometric reasoning. AIResearch could align simulated training views to real-world camera views this way.

3. **Closed-Loop Eval Harness (NAVSIM + Waymo)**: Adopt NAVSIM v1/v2 as the standard eval harness for waypoint heads. Both provide rigorous closed-loop metrics (PDM/EPDMS) that predict real-world performance better than open-loop L2.

4. **PDM Scoring Head as Training Signal**: Train a learned trajectory quality scorer (PDM) to provide training gradients beyond simple L2 to GT. Enables learning from diverse trajectory hypotheses, not just imitation.

### ⚠️ Need Adaptation

1. **Annotate → Self-Supervise Transition**: RAP needs 3D box labels. For AIResearch, consider using a pretrained detector (DDAD, BEVDepth) to auto-label training data, then fine-tune the planning head with RAP augmentation.

2. **Faster 3D Rasterization in the Loop**: RAP's rasterization requires map + box annotations. For Waymo-free training, explore depth-supervised pseudo-labels that enable lightweight rasterization without HD maps.

3. **Multi-Agent Reasoning**: RAP is ego-centric. For multi-agent scenarios, extend cross-agent view synthesis to planning-level reasoning (not just visual augmentation).

4. **World Model Integration**: RAP's rasterization doesn't generate full future scenes. Combine with a generative world model (GAIA-1/2, DriveMJ) for future-conditioned planning with RAP-style augmentation.

### Action Items for This Repo

- [ ] Add NAVSIM v1 eval harness for waypoint head closed-loop evaluation
- [ ] Implement counterfactual recovery augmentation (perturb waypoints → re-render) using existing logs
- [ ] Train PDM scoring head on top of waypoint features for learned training signal
- [ ] Evaluate with DINOv2 feature alignment as augmentation quality metric
- [ ] Compare closed-loop robustness with vs without recovery augmentation

---

## Citations

- **RAP** — "RAP: 3D Rasterization Augmented End-to-End Planning" [arXiv:2510.04333](https://arxiv.org/abs/2510.04333), ICLR 2026
- **Project Page** — [alan-lanfeng.github.io/RAP](https://alan-lanfeng.github.io/RAP)
- **Code** — [github.com/vita-epfl/RAP](https://github.com/vita-epfl/RAP)
- **Weights** — [HuggingFace](https://huggingface.co/Lanl11/RAP_ckpts/tree/main)
- **NAVSIM** — "NAVSIM: Neural Autonomous Vision and Planning Benchmark" [HuggingFace Spaces](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navtest)
- **Waymo E2E Challenge 2025** — [waymo.com/open/challenges/2025/e2e-driving](https://waymo.com/open/challenges/2025/e2e-driving)
- **Bench2Drive** — "Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving" [thinklab-sjtu.github.io/Bench2Drive](https://thinklab-sjtu.github.io/Bench2Drive/)
- **DINOv3** — "DINOv3: Large Scale Self-Supervised Pre-Training for Vision Tasks" [self-supervised]
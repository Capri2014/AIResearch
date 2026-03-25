# Open X-Embodiment & RT-X: Robotics Foundation Model Baseline — Public Anchor (Survey PR #3)

**Date**: March 25, 2026  
**Source**: https://robotics-transformer-x.github.io/ (paper: https://arxiv.org/abs/2310.08864 ; code: https://github.com/google-deepmind/open_x_embodiment)

---

## TL;DR (3 bullets)

- **Open X-Embodiment** is the foundational dataset release — **1M+ trajectories**, **22 robot embodiments**, **60 datasets** from **34 labs** — unified in RLDS format.
- **RT-X** (RT-1-X / RT-2-X) models trained on this mixture demonstrate **positive cross-embodiment transfer** — single policy improves multiple robots.
- **Action contract**: 7D end-effector (x,y,z,roll,pitch,yaw,gripper) in gripper frame — clean mapping to driving control outputs.

---

## TL;DR Extended

Open X-Embodiment was chosen as the public anchor for Survey PR #3 because it represents the **foundational dataset** that enabled subsequent models like Octo. While Octo offers better code reproducibility, Open X-Embodiment/RT-X is the canonical baseline that established:

1. **Data standardization as the moat** — RLDS episode schema统一了异构机器人数据
2. **Cross-embodiment transfer is real** — 训练一个模型服务于多个机器人本体
3. **Unified action representation** — 7D末端执行器控制协议

RT-X is the direct baseline that Tesla/Ashok talk claims map to most cleanly.

---

## Dataset / Inputs / Outputs

### Dataset (Open X-Embodiment)
| Dimension | Value |
|-----------|-------|
| **Trajectories** | 1M+ real robot episodes |
| **Robot embodiments** | 22 ( WidowX, UR5, Franka, bi-manual, quadrupeds ) |
| **Source datasets** | 60 datasets from 34 labs |
| **Skills** | 527 unique skills, 160k+ tasks |
| **Format** | RLDS episode format (sequence of steps) |

### Inputs (RT-1-X / RT-2-X)
- **RGB image**: Single workspace camera (no depth, no wrist camera by default)
- **Language**: Task string instruction ("move red pepper to tray")
- **Observation history**: Temporal context (3Hz = 333ms intervals)

### Outputs (Action Contract)
- **7D vector**: x, y, z, roll, pitch, yaw, gripper open/close
- **Frame of reference**: Robot gripper frame
- **Representation modes**: absolute / delta / velocity (configurable per dataset)
- **Missing dimensions**: Zero-filled during training

---

## Training Objective

### RT-1-X: Behavior Cloning
- **Architecture**: Transformer-based (EFFICIENT backbone)
- **Objective**: Supervised cross-entropy on discretized action tokens
- **Parameters**: ~80M
- **Action discretization**: Actions tokenized into discrete bins, trained via classification

### RT-2-X: VLM Co-Finetuning
- **Architecture**: Large Vision-Language Model (2B+ params)
- **Objective**: Co-finetune so robot actions are emitted as **language tokens**
- **Key insight**: Leverage pre-trained VLM knowledge for "emergent" semantic understanding
- **Parameters**: ~2B (RT-2-X 55B variant also exists)

### Training Details
| Model | Data | Objective | Params |
|-------|------|-----------|--------|
| RT-1-X | Open X-Embodiment mixture | BC (action token classification) | 80M |
| RT-2-X | Open X-Embodiment mixture | VLM co-finetuning | 2B |

---

## Evaluation Setup

### RT-1-X: In-Distribution Skills (Real Robot)
| Metric | Value |
|--------|-------|
| **Outperforms** | RT-1 or Original Methods trained on single datasets |
| **Small-data gain** | ~50% better in low-data regimes |
| **Evaluation sites** | UC Berkeley RAIL, Stanford IRIS, NYU CILVR, USC CLVR, Univ. Freiburg AiS |

### RT-2-X: Emergent Skills (Language Sensitivity)
| Metric | Value |
|--------|-------|
| **Emergent capabilities** | Spatial understanding (on vs near), preposition modulation |
| **RT-2-X vs RT-2** | ~3x improvement on emergent skill evaluations |
| **Zero-shot language** | Understands novel task strings without finetuning |

### Reproducibility
- ✅ Dataset loading via TFDS/RLDS
- ✅ Colab notebooks for visualization
- ⚠️ RT-1-X checkpoint available (JAX/TF)
- ⚠️ Real-robot evaluation limited to contributing labs

---

## What Maps to Tesla/Ashok Claims vs What Doesn't

### Maps Cleanly ✅

| Tesla/Ashok Claim | RT-X Alignment |
|------------------|----------------|
| "One foundational network across robots" | ✅ RT-X IS single policy serving 22 embodiments |
| "Fleet data + pretraining" | ✅ 1M+ trajectories pooled across 34 labs |
| "Data standardization is the moat" | ✅ RLDS schema + unified action contracts |
| "Language as API" | ✅ Task string as instruction interface |
| "Cross-embodiment transfer" | ✅ Positive transfer documented |
| "Efficient fine-tuning" | ✅ 50% gain in small-data regimes |

### Doesn't Map ❌

| Gap | Details |
|-----|---------|
| **Humanoid / full-body** | Manipulation only (7D end-effector), no locomotion/balance |
| **Long-horizon autonomy** | Short-horizon, reactive tasks (seconds per episode) |
| **Real-time fleet deployment** | Research baseline, no deployed continuous learning |
| **Factory-grade robustness** | Lab distribution, not factory floor |
| **Video generation** | Action prediction, not world model |
| **Real-time control (36 Hz)** | 3Hz evaluation, not high-frequency closed-loop |

---

## Comparison: RT-X vs Octo

| Dimension | RT-X (RT-1-X / RT-2-X) | Octo |
|-----------|------------------------|------|
| **Code reproducibility** | ⚠️ Partial (dataset, some checkpoints) | ✅ pip install, HF Hub, Colab |
| **Dataset** | ✅ 1M+ trajectories | ~800k (subset) |
| **Action space** | Fixed 7D | Customizable via config |
| **Objective** | BC (RT-1-X), VLM co-tune (RT-2-X) | Diffusion |
| **Parameters** | 80M (RT-1-X), 2B (RT-2-X) | 35M |
| **Zero-shot WidowX** | 0.60 (RT-1-X), 0.85 (RT-2-X) | 0.80 |
| **Multi-camera** | ❌ Single camera | ✅ 1-4 cameras |

**Rationale for selection**: RT-X chosen as "public anchor" because it is the **foundational dataset release** that enables all subsequent work (Octo builds on Open X-Embodiment data). RT-X also provides the cleanest mapping to Tesla/Ashok claims about data standardization and cross-embodiment transfer.

---

## Action Items for AIResearch (Interfaces/Contracts to Copy)

### High Priority
- [ ] **Adopt RLDS episode schema** for on-disk robot data format — enables direct loading of Open X-Embodiment and future standardization
- [ ] **Lock action contract**: 7D end-effector in gripper frame — document absolute/delta/velocity mode and missing-dimension handling
- [ ] **Make task string first-class**: explicit instruction field, guidelines for allowed verbs/objects

### Medium Priority
- [ ] **Benchmark baseline**: Implement RT-1-X style BC policy as internal baseline before diffusion/other objectives
- [ ] **Test cross-domain transfer**: RT-X style mixture training on driving data (multiple sensors, vehicles)
- [ ] **Define "Original Method" baseline**: per-domain baseline for comparison (mirrors paper's methodology)

### Low Priority / Exploratory
- [ ] **Explore VLM co-tuning**: RT-2-X approach of leveraging pre-trained VLMs for semantic understanding
- [ ] **Multi-camera input**: Extend beyond single workspace camera (wrist, in-hand, surround views)
- [ ] **Goal-conditioned learning**: Add goal image/frame conditioning (not in RT-X, but Octo supports)

---

## Contract Summary (for implementation)

```
Observation:
  - image: (B, H, W, 3)           # Single RGB workspace camera
  - task: str                     # Language instruction string
  - timestamp: float              # 3Hz (333ms) intervals

Action (7D End-Effector):
  - position: x, y, z             # Cartesian position
  - orientation: roll, pitch, yaw  # Euler angles
  - gripper: open/close           # Binary or continuous
  
Representation:
  - absolute: direct target pose
  - delta: change from current
  - velocity: rate of change
  
Missing dimensions: zero-filled during training
```

---

## Citations / Links

- Paper (arXiv): https://arxiv.org/abs/2310.08864
- Project site: https://robotics-transformer-x.github.io/
- GitHub (data/code): https://github.com/google-deepmind/open_x_embodiment
- RLDS format: https://github.com/google-research/rlds#dataset-format
- Dataset spreadsheet: https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit
- Citation: https://robotics-transformer-x.github.io/citation.txt
- RT-1-X JAX checkpoint: `gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax`

---

## PR Summary

- **PR**: [Survey PR #3] Open X-Embodiment RT-X — Public Anchor
- **Choice**: Open X-Embodiment/RT-X as foundational baseline (Octo builds on this data)
- **Key insight**: 1M+ trajectories + 22 embodiments establish that cross-robot transfer is real; RLDS schema is the contract to copy
- **Action items**: Adopt RLDS schema, lock 7D action contract, implement RT-1-X style BC baseline, test domain transfer

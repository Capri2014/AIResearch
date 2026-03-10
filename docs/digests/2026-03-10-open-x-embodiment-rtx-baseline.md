# Open X-Embodiment + RT-X: Public Robotics Foundation Model Baseline — Digest

**Date:** 2026-03-10  
**Survey type:** Public anchor digest (Survey PR #3)  
**Sources:** https://robotics-transformer-x.github.io/ | https://github.com/google-deepmind/open_x_embodiment

---

## TL;DR (3 bullets)

- **Open X-Embodiment** = 1M+ trajectories from 22 robot embodiments across 60 datasets from 34 labs — the largest open unified robot dataset.
- **RT-X models** (RT-1-X, RT-2-X) = generalist policies trained on this mixture; RT-1-X achieves **80% zero-shot** on new robots, **50%+ improvement** over single-robot baselines in low-data regime.
- **Best public baseline** for "one foundation model across robots" thesis; cleanly maps to Tesla's pretrain+finetune narrative but **doesn't scale to driving/humanoid** or fleet learning.

---

## Problem

Tesla/Ashok argues for **one foundational network** across robotics (driving + humanoid). The closest public analog: can we train a single policy on heterogeneous robot data and transfer to new robots/tasks?

Open X-Embodiment is the dataset; RT-X is the baseline model family answering this question.

---

## Dataset / Inputs / Outputs

### Dataset: Open X-Embodiment

| Metric | Value |
|--------|-------|
| Trajectories | **1M+** |
| Datasets | 60 |
| Robot embodiments | 22 |
| Institutions | 34 |
| Format | RLDS (episode sequences) |

**Diversity covers:**
- Arms (Sawyer, WidowX, UR5, Franka, KUKA)
- Bi-manual (BAXTER, two-arm setups)
- Mobile manipulators (Trossen WidowX + mobile base)
- Sensors: RGB workspace, wrist cameras, depth, force-torque
- Labels: language instructions, goal images, task strings

### Model inputs (RT-1-X checkpoint)

- **Vision:** single workspace RGB camera (configurable)
- **Language:** task string (e.g., "pick up the red block")
- **History:** optional observation history for temporal context

**Not used in released checkpoint:** depth, wrist cameras, multi-view by default.

### Model outputs (action contract)

- **7D end-effector** in gripper frame: `[x, y, z, roll, pitch, yaw, gripper]`
- Each dimension can be absolute / delta / velocity (dataset-dependent)
- Missing dimensions **zero-filled** during training

---

## Training objective

### RT-1-X: Behavior Cloning
- **Transformer policy** (RT-1 family)
- Predict next action given (image, task string, history)
- Cross-entropy on discretized action tokens
- ~27M parameters (small enough to run in real-time)

### RT-2-X: VLM Co-finetuning
- **RT-2 family** (large VLM, 55B params)
- Actions emitted as **tokens** via language modeling head
- Co-finetuned on robot data + web-scale VLM data
- Shows "emergent" language-sensitive behaviors (spatial reasoning)

### Key difference from Octo
- **RT-X = BC** (predict next action)
- **Octo = Diffusion** (models action distribution)
- Diffusion is more expressive but heavier; BC is simpler and faster

---

## Evaluation setup

### Zero-shot transfer (RT-1-X on novel robots)

| Model | Success Rate |
|-------|-------------|
| RT-1 (single robot) | 0.40 |
| RT-1-X (mixture) | **0.80** |
| VC-1 (pretrained visual) | 0.35 |

**Key result:** 2× improvement over single-robot training just by mixing data.

### Low-data finetuning (100 demos)

| Method | Avg Success |
|--------|-------------|
| From scratch | 0.20 |
| RT-1 (single dataset) | 0.35 |
| RT-1-X → finetune | **0.50** |

**Key result:** +50% vs single-robot baseline in low-data regime.

### RT-2-X emergent skills
- Tested on language-sensitive spatial tasks ("put on" vs "put near")
- RT-2-X outperforms RT-2 by **~3×** on these

---

## What maps cleanly to Tesla / Ashok claims vs what doesn't

### Maps cleanly ✓

| Tesla Claim | RT-X Finding |
|-------------|--------------|
| "One foundational network for multiple robots" | RT-1-X = single policy across 22 embodiments |
| "Fleet data + pretrain + finetune" | Mixture training → efficient transfer with 100 demos |
| "Language as API" | Task string conditioning works across robots |
| "End-to-end from pixels" | Image → action, no handcrafted perception pipeline |
| "Cross-domain transfer" | Positive transfer from training on heterogeneous data |

### Doesn't map ✗

| Gap | Detail |
|-----|--------|
| **Driving scale** | RT-X = manipulation (short-horizon, single-arm); Tesla = 2B tokens/day fleet |
| **End-to-end to throttle/steering** | RT-X outputs 7D end-effector, not vehicle control |
| **Humanoid / full-body** | No balance, locomotion, or whole-body control |
| **Real-time 36 Hz** | RT-2-X is 55B (too large for real-time); RT-1-X is ~27M but slower than dedicated policies |
| **Fleet learning loop** | Static pretrained model; no continuous deployment/online learning |
| **World simulator** | RT-X is a policy, not a generative world model |
| **3D scene understanding** | No explicit geometric reasoning / neural rendering |

---

## Action items for AIResearch

### Immediate (copy these interfaces)

- [ ] **Adopt RLDS episode schema** as internal data format — sequence of (observation, action, reward) steps
- [ ] **Standardize action contract** to 7D end-effector in gripper frame; document absolute vs delta vs velocity
- [ ] **Make task string first-class** — every episode needs a language label; establish style guide (verb + object + preposition)

### Near-term (architecture decisions)

- [ ] **Choose BC vs Diffusion** — RT-X uses BC (simpler, faster); Octo uses Diffusion (more expressive). For driving: diffusion may better model uncertainty at intersections.
- [ ] **Start with RT-1-X architecture** (27M transformer) for baseline; scale up if compute allows
- [ ] **Vision encoder:** Use pretrained (RT-1 uses EfficientNet; Octo uses ViT) — don't train from scratch

### Evaluation protocol

- [ ] **Define "novel robot" benchmark** — train on mixture, eval on held-out robot/task
- [ ] **Low-data transfer benchmark** — 100-demo finetuning to measure sample efficiency
- [ ] **Closed-loop eval** — RT-X is real-robot; for driving: need simulator (CARLA/scenario runner)

### Long-term (Tesla-scale gaps)

- [ ] **Fleet learning loop** — RT-X is static; Tesla has continuous data collection
- [ ] **World model** — RT-X doesn't have this; consider pairing with GAIA-1 / DriveDreamer
- [ ] **Multi-camera 3D reasoning** — RT-X uses single workspace view; Tesla needs 8-camera consistent world model

---

## Citations + Links

- **Paper (arXiv):** https://arxiv.org/abs/2310.08864
- **Project site:** https://robotics-transformer-x.github.io/
- **GitHub (code + data):** https://github.com/google-deepmind/open_x_embodiment
- **RT-1-X checkpoint:** Available via GitHub (TF SavedModel)
- **RLDS format:** https://github.com/google-research/rlds
- **Octo comparison:** https://github.com/octo-models/octo (diffusion-based alternative)

---

## PR Summary

- **PR:** (create PR with this digest)
- **3-bullet summary:**
  1. Open X-Embodiment (1M trajectories, 22 robots) is the largest open unified robot dataset — RT-X models trained on it show **2× zero-shot transfer** and **50%+ low-data finetuning** gains.
  2. Maps cleanly to Tesla's "one foundation model" thesis (pretrain on fleet → finetune for new tasks), but gaps remain: manipulation≠driving, no world model, no fleet learning loop.
  3. **Action:** copy RLDS data schema + 7D action contract + task-string labeling; start with RT-1-X architecture (27M transformer BC) as baseline before scaling to diffusion.

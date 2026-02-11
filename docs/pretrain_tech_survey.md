# Pretraining tech survey (robotics + autonomous driving)

This is a **living** survey of *pretraining* (and near-pretraining) techniques that show up repeatedly in modern robotics and autonomous driving stacks.

> Note: written from offline knowledge (no live web search in this environment). Treat as a curated starting point; we can later add exact bib entries + links.

---

## 1) What “pretraining” means in Physical AI

In practice you’ll see 4 distinct layers that people loosely call “pretrain”:

1) **Vision / video encoder pretraining**
   - self-supervised or supervised on internet + driving/robotics video
   - output: strong visual representations

2) **Multimodal (VLM) pretraining**
   - image/video + text alignment (captioning, contrastive)
   - output: representations that support language grounding

3) **Behavior pretraining (imitation / BC at scale)**
   - train a policy on large logged action datasets
   - output: action-conditional policy prior

4) **World-model / dynamics pretraining**
   - learn latent dynamics and/or sensor models
   - output: predictive model used for planning or as an auxiliary loss

A lot of “foundation model” results are some blend of these.

---

## 2) Robotics: the dominant pretrain patterns

### A) Large-scale imitation from robot episodes (multi-task)
**Why it works:** robot data is expensive; pooling many tasks gives generality.

Common ingredients:
- action tokenization (discrete bins) or continuous regression
- transformer policy over a horizon
- goal conditioning (language, images of goal state, task id)

Representative directions to anchor on:
- **RT-1 / RT-2 style**: transformer policies trained on many robot tasks; RT-2 injects VLM-style knowledge into action generation.
- **Open X-Embodiment / RT-X style**: multi-robot dataset unification + policy transfer.
- **Octo**: open generalist policy trained on many robot datasets (often used as a “pretrained policy” you fine-tune).

### B) Diffusion policies (action sequence generation)
**Why it works:** diffusion is good at multi-modal action distributions and long-horizon sequence modeling.

Core idea:
- model a trajectory distribution with denoising steps
- condition on observation (and sometimes language)

Practical implications:
- you evaluate with rollouts; training can be stable
- easy to add classifier-free guidance style conditioning

### C) Latent/action representation learning
You’ll often see:
- discretized action vocabularies
- VQ-style latent codes for trajectories
- hierarchical policies (skills/options)

These are “pretraining” because they create reusable primitives.

---

## 3) Autonomous driving: the dominant pretrain patterns

### A) BEV-centric perception pretraining
Driving stacks increasingly standardize on **BEV** (bird’s-eye view) intermediate representations.

Pretraining targets:
- 3D detection, map elements (lanes), occupancy
- multi-camera fusion
- temporal aggregation

Why it matters for policy:
- BEV is closer to planning/control than raw pixels
- it provides a stable interface for downstream modules or end-to-end heads

### B) End-to-end “perception → planning” training (often imitation)
Trends:
- supervised trajectory prediction (waypoints / curvature / accel)
- imitation on logged expert trajectories
- sometimes cost heads (collision risk, offroad, red-light)

In many papers this is *the* “pretrain” before closed-loop fine-tuning.

### C) World models for driving
Typical goals:
- predict future frames/BEV occupancy
- learn latent dynamics for planning (model predictive control in latent space)

Benefits:
- can exploit unlabeled video at scale
- can generate rollouts for planning or as training augmentation

### D) Language-conditioned driving (early but growing)
Language is used for:
- route-level goals (“take next left”)
- interactive driving (“yield”, “merge behind that car”)

This tends to start from VLM pretraining + driving-specific action heads.

---

## 4) Datasets to treat as “pretraining fuel”

Robotics (episodes):
- multi-task robot manipulation datasets (often aggregated across labs/robots)

Driving (logs):
- **nuScenes**, **Waymo Open Motion/Perception**, **Argoverse**, **nuPlan**

Simulators (for pretraining + eval):
- **CARLA** (+ ScenarioRunner)
- robotics: **Isaac Sim**, **MuJoCo**

---

## 5) Concrete plan for this repo

If we want a pragmatic path:

1) **Codify interfaces** (obs/action + env + metrics)
2) Add a tiny **SFT/BC** runnable (CPU-only) to prove wiring
3) Add schemas for:
   - episodes
   - scenarios
   - metrics
4) Pick the first “real” pretraining target:
   - Driving: BEV encoder + small policy head (imitation)
   - Robotics: Octo-style fine-tune on a small public dataset

---

## 6) What I recommend you decide next

To avoid wandering:

1) Driving track first? (CARLA closed-loop eval) **or** robotics track first? (MuJoCo/Isaac tasks)
2) Preferred policy family for the first real experiment:
   - transformer BC
   - diffusion policy
   - world-model + planner

Once you pick, we can turn this survey into a tight reading list + a 2-week build plan.

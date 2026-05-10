# SNG-VLA: Sequential Navigation Guidance for End-to-End Autonomous Driving

**Paper:** [arXiv:2604.12208](https://arxiv.org/abs/2604.12208)  
**Title:** Unveiling the Surprising Efficacy of Navigation Understanding in End-to-End Autonomous Driving  
**Authors:** Zhihua Hua, Junli Wang, Pengfei Li, Qihao Jin, Bo Zhang, Kehua Sheng, Yilun Chen, Zhongxue Gan, Wenchao Ding  
**Affiliation:** Fudan University, Didi Chuxing (Tsinghua), Chinese Academy of Sciences  
**Venue:** ICRA 2026  
**GitHub:** [SNG-VLA](https://fudan-magic-lab.github.io/SNG-VLA-web)

---

## TL;DR

Current end-to-end autonomous driving systems **ignore global navigation** and still perform fine — that's a surprising red flag. This paper discovers that removing navigation entirely barely hurts performance, revealing a fundamental gap in how these systems understand "where they're going."

The fix: **Sequential Navigation Guidance (SNG)** — combining a sparse navigation path (40m trajectory ahead) with turn-by-turn (TBT) real-time instructions. The resulting **SNG-VLA** model achieves SOTA on both Bench2Drive (Driving Score 67.17) and NAVSIM (PDM-Score 88.24), without any auxiliary perception losses. Results are modest but real, and the diagnostic work is genuinely valuable.

---

## 1. Intuition

### What is this paper about?

This paper asks a simple but shocking question: **Do end-to-end autonomous driving systems actually understand navigation?**

In human driving, you'd expect knowing "turn left at the next intersection" to matter a lot. But when the authors removed or corrupted navigation inputs in existing E2E driving systems, performance barely dropped. In some cases, it *improved*. That's insane.

The root cause: Current systems use **driving commands** (like "Turn Left", "Go Forward") as the sole navigation signal. These are too coarse — they don't capture:
- Where exactly the vehicle should go over the next 40 meters
- What the upcoming turn actually looks like (distance, timing)
- Lane-level guidance for complex maneuvers

### Why it matters

If systems don't use navigation, they can't:
- Follow complex routes accurately
- Handle roundabout exits correctly
- Do advance lane-changing for far-ahead turns
- Generalize to new environments

The fix requires better **representation** of navigation information, not more data or bigger models.

---

## 2. The Problem It Solves

### What's broken in current E2E driving?

| Issue | Description | Impact |
|-------|------------|--------|
| **Coarse navigation** | Driving commands are one-hot (4-5 classes), miss spatial detail | Can't distinguish "turn now" vs "prepare to turn" |
| **Annotation ambiguity** | Same command for different situations (roundabout exit vs turn) | Wrong learning signal |
| **BVR blind spots** | Beyond-visual-range lane changes | No advance warning for far turns |
| **Privileged information** | Perfect waypoints available in simulation | Doesn't transfer to real world |
| **Overfitting to perception** | Systems ignore nav because perception is easier | Nav signal doesn't matter enough |

### The key experiment (Table IV in paper)

The authors ablated Transfuser on NAVSIM by corrupting driving commands:

| Input | PDM-Score |
|-------|----------|
| Original (GT command) | 84.0 |
| No command | 84.4 |
| Random command | 84.7 |
| Fixed "Left" | 84.3 |
| Fixed "Right" | 84.4 |

**Removing navigation entirely = almost no change.** This is the paper's killer result.

### What wasn't working before

Previous approaches:
- UniAD: Integrated perception + planning, but navigation = latent embedding
- VAD: Vectorized scene representation, no explicit navigation path
- BEVFormer: Bird's-eye view, navigation as additional tokens
- TCP/TransFuser: Concatenate command with ego state

None explicitly model the **spatial trajectory** of where the car should go.

---

## 3. How It Works

### The SNG Framework

SNG has two components:

```
SNG = {Navigation Path, Turn-by-Turn Information}
```

#### 3.1 Navigation Path P

- **What:** 40m trajectory ahead (sparse points)
- **Sampling:** Road centerlines, sampled every 10m (4 points)
- **Coordinate:** Transformed from world → vehicle frame
- **Noise:** Added substantial noise to simulate real-world localization errors (prevleges info problem)

```
P = {(x₁,y₁), (x₂,y₂), ..., (x_N,y_N)} where N = 4
```

Why sparse? Too dense = overfitting to perfect sim data. Too sparse = no spatial guidance. 4 points at 10m intervals = optimal (see ablation Table V).

#### 3.2 Turn-by-Turn (TBT) Information

- **What:** Real-time high-level guidance
- **Categories (8):** turn left, turn right, U-turn, straight, keep left, keep right, enter roundabout, none
- **Supplementary (9):** highway entry, tunnel, right-turn lane, left-turn lane, etc.
- **Format:** Text-like tokens fed to VLM

Example TBT:
```
Current: "Turn right" (50m, 5s)
Future: "Turn left" (200m, 15s)
Supplementary: "Right-turn lane"
```

#### 3.3 SNG-QA Dataset

For aligning global + local planning:
- **100K QA pairs** from NAVSIM annotations
- **Three reasoning stages:**
  1. Global planning (summarize navigation)
  2. Local planning (explain trajectory given nav + scene)
  3. Trajectory generation (waypoints)

Built using **Qwen2.5 VL 72B** with 3-stage validation:
- Accuracy verification
- Consistency validation  
- Language refinement

### The SNG-VLA Model

```
SNG-VLA = LLaVA architecture
  - Backbone: Qwen2.5-0.5B
  - Vision: SigLIP-So400M (patch=14, image=384)
  - Input: Front view (single camera) + nav path + TBT + ego state
  - Output: Text reasoning + trajectory waypoints
```

#### Architecture

1. **Vision Encoder:** SigLIP processes front/rear cameras → features
2. **Navigation Encoder:** MLP encodes path points P → features
3. **TBT Encoder:** LLM tokenizer processes TBT text → features
4. **Ego State Encoder:** Attention-based dropout (SDE) → features
5. **Fusion:** Concatenate all features + waypoint query → Transformer
6. **Output:** Autoregressively generate text, then trajectory

Key insight: **No auxiliary perception loss.** Just predict trajectory from SNG.

---

## 4. The Math

### 4.1 Navigation Path Encoding

```
P = {(x₁,y₁), ..., (x_N,y_N)}  ∈ ℝ^{N×2}

F_P = MLP(P)  ∈ ℝ^{N×H}
```
where MLP projects 2D points to hidden dimension H.

### 4.2 TBT Encoding

```
T = {action, distance, time, future_action, supplementary}

F_T = Tokenizer(T)  ∈ ℝ^{N_T×H}
```
Using Qwen2.5 tokenizer.

### 4.3 Feature Fusion

```
F = Concat(F_T, F_P, F_M, F_E, Q_W)
```
Where:
- F_T: TBT features
- F_P: Path features  
- F_M: Vision features
- F_E: Ego state features
- Q_W: Waypoint query

### 4.4 Trajectory Prediction

```
τ̂ = MLP(Transformer(F))
L = ||τ̂ - τ||₁
```
Simple L1 loss between predicted and ground truth trajectory.

### 4.5 Ablation: Optimal Path Density (Table V)

| ID | Path Points | TBT | NC | DAC | PDMS |
|----|-----------|-----|-----|-----|------|
| 3 | 2×20m | No | 97.8 | 95.1 | 86.4 |
| 4 | 2×20m | Yes | 97.5 | 96.1 | 87.6 |
| 5 | 4×10m | No | 97.5 | 96.6 | 87.7 |
| **6** | **4×10m** | **Yes** | **98.9** | **96.5** | **88.2** |
| 7 | 8×5m | No | 97.5 | 96.2 | 87.2 |
| 8 | 8×5m | Yes | 97.5 | 96.6 | 87.6 |

**Takeaway:** 4 points at 10m intervals + TBT = optimal.

---

## 5. Code Examples

### 5.1 Navigation Path Feature Encoding

```python
import torch
import torch.nn as nn

class NavigationPathEncoder(nn.Module):
    """Encode sparse navigation path points."""
    
    def __init__(self, hidden_dim=1280, num_points=4):
        super().__init__()
        self.num_points = num_points
        self.fc = nn.Linear(2, hidden_dim)  # (x, y) → hidden
    
    def forward(self, path_points):
        """
        Args:
            path_points: (B, num_points, 2) in vehicle frame
        Returns:
            features: (B, num_points, hidden_dim)
        """
        # path_points already in vehicle coordinates
        features = self.fc(path_points)  # (B, N, H)
        return features
```

### 5.2 TBT Information Processing

```python
class TBTEncoder(nn.Module):
    """Encode turn-by-turn instructions."""
    
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, tbt_tokens):
        """
        Args:
            tbt_tokens: (B, seq_len) token IDs
        Returns:
            features: (B, seq_len, hidden_dim)
        """
        return self.embedding(tbt_tokens)
```

### 5.3 SNG-VLA Forward Pass

```python
class SNGVLA(nn.Module):
    """SNG-VLA model."""
    
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = SigLIPEncoder(config)
        self.path_encoder = NavigationPathEncoder(config)
        self.tbt_encoder = TBTEncoder(config)
        self.ego_encoder = EgoStateEncoder(config)  # with dropout
        self.transformer = nn.Transformer(...)
        self.traj_head = nn.Linear(config.hidden, config.num_waypoints * 2)
    
    def forward(self, images, path_points, tbt_tokens, ego_state):
        # Encode each modality
        F_M = self.vision_encoder(images)
        F_P = self.path_encoder(path_points)
        F_T = self.tbt_encoder(tbt_tokens)
        F_E = self.ego_encoder(ego_state)
        
        # Concatenate all features
        F = torch.cat([F_T, F_P, F_M, F_E], dim=1)
        
        # Transform and predict
        hidden = self.transformer(F)
        traj = self.traj_head(hidden)
        
        return traj  # (B, num_waypoints, 2)
```

### 5.4 Training Loop

```python
def train_step(model, batch, optimizer):
    images = batch['images']      # (B, C, H, W)
    path_points = batch['path']  # (B, 4, 2)
    tbt_tokens = batch['tbt']     # (B, seq_len)
    ego_state = batch['ego']        # (B, 4)
    target_traj = batch['target']   # (B, T, 2)
    
    # Forward
    pred_traj = model(images, path_points, tbt_tokens, ego_state)
    
    # L1 loss
    loss = F.l1_loss(pred_traj, target_traj)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## 6. Cross-Comparison

### 6.1 Navigation Representation Methods

| Method | Representation | Spatial | Temporal | SOTA? |
|--------|--------------|---------|----------|--------|
| Driving Commands | One-hot (4-5) | ❌ | ❌ | ❌ |
| Waypoints | Dense (50+) | ✅ | ✅ | Partial |
| **SNG (this)** | **Sparse + TBT** | ✅ | ✅ | ✅ |
| latent-Nav (UniAD) | Latent embed | Partial | ❌ | Partial |

### 6.2 E2E Driving Methods Comparison

| Method | Nav Input | NAVSIM PDMS | Bench2Drive DS | Latency |
|--------|----------|------------|--------------|----------|
| UniAD-Base | Command | 45.81 | 16.36 | 663ms |
| VAD | Command | 42.35 | 15.00 | 278ms |
| Transfuser | Command | 84.0 | - | 100ms |
| DriveTransformer | Command | 63.46 | 35.01 | 212ms |
| **SNG-VLA** | **SNG** | **88.24** | **67.17** | **160ms** |

### 6.3 Vision-Language-Action Models

| Model | Backbone | Nav Strategy | Planning | Interpretable |
|-------|--------|-------------|----------|-------------|
| DriveGPT4 | GPT-4V | Command | Trajectory | ✅ (text) |
| LMDrive | LLaVA-1.5 | Command | Trajectory | ✅ |
| DriveAdapter | ViT-L | Latent | Trajectory | Partial |
| **SNG-VLA** | **Qwen2.5** | **SNG** | **Trajectory** | ✅ |

### 6.4 Ablation: Navigation Component Contribution

| Config | PDM-Score | Δ |
|--------|----------|---|
| No navigation | 85.9 | - |
| Driving command only | 86.1 | +0.2 |
| TBT only | 86.4 | +0.5 |
| Path only (4×10m) | 87.7 | +1.8 |
| **Full SNG** | **88.2** | **+2.3** |

---

## 7. When to Use SNG-VLA

### Decision Guide

**Use SNG when:**
- [ ] You're working on end-to-end autonomous driving
- [ ] You have access to navigation API (Google Maps, etc.)
- [ ] Your current model ignores navigation inputs
- [ ] You need better route-following in complex scenarios
- [ ] Open-loop metrics don't match closed-loop performance

**Use alternative when:**
- [ ] You're doing pure perception (object detection, segmentation)
- [ ] Simulation-only (waypoints already perfect)
- [ ] Latency critical (<50ms required)
- [ ] You have a working modular pipeline

### Integration Options

**Option A: Add SNG to existing E2E model (Simplest)**
- Replace driving command with SNG input
- Keep all other architecture same
- Expected improvement: +2-4 PDM-Score

**Option B: Full SNG-VLA**
- Use LLaVA architecture
- Train from scratch with SNG-QA
- Best performance, most effort

**Option C: Hybrid**
- Existing model + SNG as auxiliary input
- Multi-task learning
- Middle ground

---

## 8. Applications to Our Work

### 8.1 RL Planning Pipeline

Our GRPO pipeline for waypoint prediction could benefit from SNG in several ways:

**Current state:**
- Input: 4D state (x, y, heading, speed)
- No explicit navigation
- Reactive (not predictive)

**With SNG integration:**

```
Option A: Navigation as Auxiliary Loss (Simplest)
- Keep GRPO intact
- Add navigation-following reward
- r_total = r_grpo + λ * r_nav

Option B: Imagination with Nav (Dreamer-style)
- Train world model with SNG
- Latent planning using CEM
- More ambitious, could reason about routes

Option C: Hybrid (Recommended)
- Input: GRPO state + navigation path
- Separate encoder → concat
- Single policy network
```

### 8.2 Relevant Datasets

| Dataset | Nav Available | Suitable |
|--------|--------------|----------|
| nuScenes | Command (implicit) | Partial |
| NAVSIM | ✅ | Best fit |
| Bench2Drive | ✅ (waypoints) | Good |
| CARLA | ✅ | Good |

### 8.3 Code: Navigation-Aware GRPO

```python
class NavigationGRPO:
    """GRPO with navigation inputs."""
    
    def __init__(self, config):
        self.policy = PolicyNetwork(
            state_dim=config.state_dim,
            nav_dim=config.nav_dim,  # NEW: navigation features
            action_dim=config.action_dim
        )
        self.nav_encoder = NavigationPathEncoder(config)
    
    def compute_reward(self, states, actions, nav_path):
        # Original GRPO reward
        r_grpo = self.base_reward(states, actions)
        
        # Navigation alignment reward
        r_nav = self.nav_reward(actions, nav_path)
        
        # Combined
        r_total = r_grpo + 0.1 * r_nav
        
        return r_total
    
    def nav_reward(self, actions, nav_path):
        """How well does action follow navigation?"""
        # Simple: distance to nearest path point
        dist = torch.cdist(actions[:, :2], nav_path[:, :, :2])
        return -dist.mean()  # Closer = higher reward
```

---

## 9. Honest Pros and Cons

### Pros

✅ **Strong diagnostic work** — Navigation ablation is eye-opening  
✅ **Clean framework** — SNG is well-motivated  
✅ **No auxiliary losses** — Simple single-task training  
✅ **Real-world validation** — Tested on actual vehicle  
✅ **Good code release** — Project page available  
✅ **Modular design** — Path + TBT can be used separately  
✅ **Optimal density found** — 4×10m is well-ablated  

### Cons

❌ **Incremental contribution** — Fixes representation, not paradigm  
❌ **Modest absolute gains** — +0.1 PDMS vs DiffusionDrive  
❌ **Single camera** — Could use more views  
❌ **No real-time TBT** — Assumes perfect API  
❌ **Limited BVR testing** — Only qualitative examples  
❌ **Two-Room failure** — Can't generalize to novel layouts  
❌ **Latency not competitive** — 160ms vs TCP's 83ms  

### When to Use

| Scenario | Recommendation |
|----------|-------------|
| NAVSIM benchmark | ✅ Strong fit |
| CARLA closed-loop | ✅ Good |
| Real-world deployment | ⚠️ Needs TBT API |
| Latency-critical | ❌ Use TCP instead |
| Pure perception | ❌ Not relevant |

---

## References

- [SNG-VLA Paper](https://arxiv.org/abs/2604.12208) — Main paper
- [SNG-VLA Project Page](https://fudan-magic-lab.github.io/SNG-VLA-web) — Code & demo
- [NAVSIM](https://arxiv.org/abs/2406.20041) — Benchmark
- [Bench2Drive](https://arxiv.org/abs/2405.12241) — Benchmark
- [LLaVA](https://arxiv.org/abs/2304.08485) — Architecture
- [Qwen2.5](https://arxiv.org/abs/2412.15115) — Language model
- [SigLIP](https://arxiv.org/abs/2303.15343) — Vision encoder
- [UniAD](https://arxiv.org/abs/2212.10156) — Prior E2E
- [Transfuser](https://arxiv.org/abs/2205.15997) — Prior E2E

---

## Appendix: Key Results Tables

### NAVSIM (Table I)

| Method | Nav Input | NC↑ | DAC↑ | TTC↑ | Comf↑ | EP↑ | PDMS↑ |
|--------|----------|-----|-----|-----|-------|-----|-------|
| UniAD | Command | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Transfuser | Command | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| Transfuser† | SNG | 97.8 | 94.5 | 93.5 | 100 | 80.0 | 85.5 |
| Hydra-MDP | Command | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive | Command | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| **SNG-VLA** | **SNG** | **98.9** | **96.5** | **92.9** | **100** | **83.8** | **88.24** |

### Bench2Drive (Table II)

| Method | Avg L2↓ | DS↑ | SR↑ | Eff↑ | Comfort↑ | Latency |
|--------|---------|-----|-----|------|---------|--------|
| UniAD-Base | 0.73 | 45.81 | 16.36 | 129.21 | 43.58 | 663ms |
| VAD | 0.91 | 42.35 | 15.00 | 157.94 | 46.01 | 278ms |
| DriveTransformer | 0.62 | 63.46 | 35.01 | 100.64 | 20.78 | 212ms |
| **SNG-VLA** | **0.82** | **67.17** | **35.90** | **158.58** | **22.30** | **160ms** |

---

*Survey completed: 2026-05-09*
---

## Appendix A: Extended Method Details

### A.1 Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        SNG-VLA Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│  Input Modalities                                              │
│  ├─ Front View Image (384×384)                                  │
│  ├─ Rear View Image (optional)                                   │
│  ├─ Navigation Path (4 points × 2)                          │
│  ├─ TBT Information (8 categories + 9 supplementary)            │
│  └─ Ego State (vx, vy, ax, ay)                                │
│                          ↓                                   │
│  ┌────────────────────┐    ┌────────────────┐                   │
│  │  SigLIP Encoder  │    │   MLP Layer   │                   │
│  │   (Vision)    │    │  (Path → H)  │                   │
│  └───────↑───────┘    └───────↑─────┘                   │
│        F_M (feature)      F_P (feature)                     │
│                          ↓                                 │
│        ┌──────────────────────────────────────┐            │
│        │       TBT Encoder (Qwen2.5)        │            │
│        └───────────────↑───────────────────────┘            │
│                    F_T (feature)                           │
│                          ↓                                 │
│        ┌──────────────────────────────────────┐            │
│        │   Ego State Encoder (SDE + dropout)   │            │
│        └───────────────↑───────────────────────┘            │
│                    F_E (feature)                           │
│                          ↓                                 │
│        ┌─────────────────────────────────────────────┐      │
│        │     Feature Fusion (Concat) + Waypoint Query   │      │
│        └──────────────────↑────────────────────────┘      │
│                          ↓                               │
│        ┌───────────────────────────────────────────┐        │
│        │   Unified Transformer Backbone (Qwen2.5)    │        │
│        └──────────────────↑───────────────────────┘        │
│                          ↓                               │
│        ┌────────────────────────────┐                      │
│        │   Autoregressive Decode    │                      │
│        ├────────────────────────────┤                      │
│        │  1. Text Reasoning      │ ← Language output         │
│        │  2. Trajectory       │ ← Waypoint output       │
│        └────────────────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### A.2 Detailed SNG-QA Construction

The SNG-QA dataset has 100K samples with three reasoning stages:

**Stage 1: Global Planning Summary**
```
Input: SNG (navigation path + TBT)
Prompt: "Summarize the navigation intent for the upcoming trajectory."
Output: "The vehicle should proceed forward for 50m, then turn 
         right at the intersection, staying in the right lane."
```
- Uses Qwen2.5 VL 72B for generation
- 3-way validation for quality

**Stage 2: Local Planning Explanation**
```
Input: SNG + Scene Understanding + Global Summary
Prompt: "Given the navigation intent and current scene, explain 
         the local planning rationale."
Output: "The pedestrian crossing ahead requires slowing down.
         The right-turn lane is clear. The vehicle should 
         maintain speed until 30m before the turn, then decelerate
         to 2m/s for the turn."
```
- Incorporates object detection labels
- Explains causal relationships

**Stage 3: Trajectory Generation**
```
Input: All above + Waypoint Query
Output: [(x₁,y₁), (x₂,y₂), ..., (x_T,y_T)]
```
- T = 2 seconds @ 2Hz = 4 waypoints
- Ground truth from expert trajectories

### A.3 Noise Injection for Realism

To prevent overfitting to perfect simulation data, the authors added noise to navigation paths:

```python
def add_path_noise(path, noise_std=2.0):
    """
    Add Gaussian noise to simulate localization errors.
    
    Args:
        path: (N, 2) navigation path in vehicle frame
        noise_std: Standard deviation of noise (meters)
    
    Returns:
        noisy_path: (N, 2) path with noise
    """
    noise = torch.randn_like(path) * noise_std
    noisy_path = path + noise
    return noisy_path
```

Why 2.0m? Typical GPS uncertainty in urban environments.

### A.4 State Dropout Encoder (SDE)

Inspired by prior work showing ego state overfitting, the SDE applies dropout:

```python
class StateDropoutEncoder(nn.Module):
    """Ego state encoder with dropout for regularization."""
    
    def __init__(self, state_dim=4, hidden_dim=1280, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(state_dim, hidden_dim)
    
    def forward(self, ego_state):
        """
        Args:
            ego_state: (B, 4) — [vx, vy, ax, ay]
        Returns:
            features: (B, hidden_dim)
        """
        # Apply dropout to each state channel independently
        out = self.dropout(ego_state)
        features = self.fc(out)
        return features
```

---

## Appendix B: Extended Experimental Analysis

### B.1 Full NAVSIM Results (Table I - Complete)

| Method | Nav Input | NC↑ | DAC↑ | TTC↑ | Comf↑ | EP↑ | PDMS↑ |
|--------|----------|-----|-----|-----|-------|-----|-------|
| UniAD | Command | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| PARA-Drive | Command | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LTF | Command | 97.4 | 92.8 | 92.4 | 100 | 79.0 | 83.8 |
| Transfuser | Command | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| Transfuser† | SNG | 97.8 | 94.5 | 93.5 | 100 | 80.0 | 85.5 |
| DRAMA | Command | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| Hydra-MDP | Command | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive | Command | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| **SNG-VLA** | **SNG** | **98.9** | **96.5** | **92.9** | **100** | **83.8** | **88.24** |
| SNG-VLA-QA | SNG | 98.4 | 96.7 | 93.1 | 100 | 83.4 | 88.21 |

### B.2 Full Bench2Drive Open-Loop (Table II)

| Method | Avg L2↓ | DS↑ | SR↑ | Eff↑ | Comfort↑ | Latency |
|--------|---------|-----|-----|------|---------|--------|--------|
| AD-MLP | 3.64 | 18.05 | 0.00 | 48.45 | 22.63 | 3ms |
| UniAD-Tiny | 0.80 | 40.73 | 13.18 | 123.92 | 47.04 | 420ms |
| UniAD-Base | 0.73 | 45.81 | 16.36 | 129.21 | 43.58 | 663ms |
| VAD | 0.91 | 42.35 | 15.00 | 157.94 | 46.01 | 278ms |
| DriveTransformer-Large | 0.62 | 63.46 | 35.01 | 100.64 | 20.78 | 212ms |
| **SNG-VLA** | **0.82** | **67.17** | **35.90** | **158.58** | **22.30** | **160ms** |
| TCP* | 1.70 | 40.70 | 15.00 | 54.26 | 47.80 | 83ms |
| TCP-ctrl* | - | 30.47 | 7.27 | 55.97 | 51.51 | 83ms |
| TCP-traj* | 1.70 | 59.90 | 30.00 | 76.54 | 18.08 | 83ms |

### B.3 Multi-Ability Results (Table III - Complete)

Ability scores on 5 challenging scenarios:

| Method | M↑ | O↑ | EB↑ | GW↑ | TS↑ | Mean↑ |
|--------|----|----|-----|-----|-----|-------|
| AD-MLP | 0.0 | 0.0 | 0.0 | 0.0 | 4.4 | 0.9 |
| UniAD-Tiny | 8.9 | 9.3 | 20.0 | 20.0 | 15.4 | 14.7 |
| UniAD-Base | 14.1 | 17.8 | 21.7 | 10.0 | 14.2 | 15.6 |
| VAD | 8.1 | 24.4 | 18.6 | 20.0 | 19.2 | 18.1 |
| DriveTransformer | 17.6 | 35.0 | 48.4 | 40.0 | 52.1 | 38.6 |
| **SNG-VLA** | **33.8** | **11.1** | **46.6** | **50.0** | **50.0** | **38.1** |
| TCP* | 16.12 | 20.0 | 20.0 | 10.0 | 7.0 | 14.6 |
| TCP-ctrl* | 10.3 | 4.4 | 10.0 | 10.0 | 6.5 | 8.2 |
| TCP-traj* | 8.9 | 24.3 | 51.7 | 40.0 | 46.3 | 34.2 |
| ThinkTwice* | 27.4 | 18.4 | 35.8 | 50.0 | 54.2 | 37.2 |
| DriveAdapter* | 28.8 | 26.4 | 48.8 | 50.0 | 56.4 | 42.1 |

Where: M=Merge, O=Overtake, EB=Emergency Brake, GW=Give Way, TS=Traffic Sign

### B.4 Ablation: Command vs SNG Components

| ID | Path | TBT | Command | NC | DAC | PDMS |
|----|------|-----|---------|-----|-----|------|
| 0 | - | - | - | 97.2 | 95.1 | 85.9 |
| 1 | - | - | ✓ | 97.5 | 95.3 | 86.1 |
| 2 | - | ✓ | - | 97.6 | 95.2 | 86.4 |
| 3 | 2×20m | - | - | 97.8 | 95.1 | 86.4 |
| 4 | 2×20m | ✓ | - | 97.5 | 96.1 | 87.6 |
| 5 | 4×10m | - | - | 97.5 | 96.6 | 87.7 |
| **6** | **4×10m** | **✓** | **-** | **98.9** | **96.5** | **88.2** |
| 7 | 8×5m | - | - | 97.5 | 96.2 | 87.2 |
| 8 | 8×5m | ✓ | - | 97.5 | 96.6 | 87.6 |

---

## Appendix C: Extended Integration Code

### C.1 Complete Navigation-Aware GRPO Training Loop

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NavigationGRPO:
    """GRPO with navigation path integration."""
    
    def __init__(self, config):
        self.config = config
        # State encoder (4D: x, y, heading, speed)
        self.state_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        # Navigation path encoder (NEW)
        self.nav_encoder = nn.Sequential(
            nn.Linear(config.nav_points * 2, 256),  # 4 points × 2 coords
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        # Combined policy
        self.policy = nn.Sequential(
            nn.Linear(256 + 256, 256),  # state + nav
            nn.ReLU(),
            nn.Linear(256, config.action_dim)  # waypoints
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def encode_state(self, state, nav_path):
        """Encode both state and navigation."""
        state_feat = self.state_encoder(state)
        nav_feat = self.nav_encoder(nav_path)
        combined = torch.cat([state_feat, nav_feat], dim=-1)
        return self.policy(combined)
    
    def compute_grpo_loss(self, group_rewards, advantages):
        """Group-relative policy gradient."""
        # Standard GRPO loss
        policy_loss = -advantages.mean()
        return policy_loss
    
    def compute_nav_reward(self, actions, nav_path):
        """
        Navigation alignment reward: encourage actions to follow nav path.
        
        Args:
            actions: (B, T, 2) planned trajectory
            nav_path: (B, N, 2) navigation path points
        
        Returns:
            reward: (B,) alignment score
        """
        # Find minimum distance from each action point to nav path
        distances = torch.cdist(actions, nav_path)  # (B, T, N)
        min_distances = distances.min(dim=-1)[0]   # (B, T)
        
        # Reward = negative distance (closer = better)
        reward = -min_distances.mean(dim=-1)       # (B,)
        return reward
    
    def train_step(self, batch, group_rewards):
        """
        Single training step.
        
        Args:
            batch: dict with 'state', 'nav_path', 'actions'
            group_rewards: (B,) rewards for each sample
        """
        state = batch['state']
        nav_path = batch['nav_path']
        actions = batch['actions']
        
        # Forward pass
        pred_actions = self.encode_state(state, nav_path)
        
        # GRPO loss
        advantages = self.compute_advantages(group_rewards)
        grpo_loss = self.compute_grpo_loss(group_rewards, advantages)
        
        # Navigation reward (auxiliary)
        nav_reward = self.compute_nav_reward(actions, nav_path)
        
        # Combined loss
        total_loss = grpo_loss - 0.1 * nav_reward.mean()
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {'total': total_loss.item(), 'grpo': grpo_loss.item(), 'nav': nav_reward.mean().item()}
    
    def compute_advantages(self, rewards):
        """Compute group-relative advantages."""
        # Group rewards by task/environment
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        advantages = (rewards - mean) / std
        return advantages
```

### C.2 Navigation Path Preprocessing

```python
import numpy as np

def extract_navigation_path(route, max_distance=40, point_interval=10):
    """
    Extract sparse navigation path from route.
    
    Args:
        route: Full route from navigation API (list of [x, y] points)
        max_distance: Maximum distance ahead (meters)
        point_interval: Spacing between points (meters)
    
    Returns:
        path: (N, 2) array of [x, y] in vehicle frame
    """
    if len(route) < 2:
        return np.zeros((4, 2))
    
    # Calculate cumulative distance along route
    distances = np.zeros(len(route))
    for i in range(1, len(route)):
        distances[i] = distances[i-1] + np.linalg.norm(route[i] - route[i-1])
    
    # Sample points at intervals
    target_distances = np.arange(0, max_distance, point_interval)
    path = []
    
    for d in target_distances:
        # Find closest route point
        idx = np.argmin(np.abs(distances - d))
        path.append(route[idx])
    
    return np.array(path)

def transform_to_vehicle_frame(path, vehicle_pose):
    """
    Transform path from world to vehicle frame.
    
    Args:
        path: (N, 2) path in world frame
        vehicle_pose: [x, y, theta] of vehicle
    
    Returns:
        path_vehicle: (N, 2) path in vehicle frame
    """
    if len(path) == 0:
        return np.zeros((4, 2))
    
    vx, vy, theta = vehicle_pose
    
    # Translation
    path_centered = path - np.array([vx, vy])
    
    # Rotation
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    path_vehicle = path_centered @ rotation.T
    
    return path_vehicle
```

### C.3 TBT Information Extraction

```python
from dataclasses import dataclass
from enum import Enum

class DrivingAction(Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    U_TURN = "u_turn"
    STRAIGHT = "straight"
    KEEP_LEFT = "keep_left"
    KEEP_RIGHT = "keep_right"
    ENTER_ROUNDABOUT = "enter_roundabout"
    NONE = "none"

class SupplementaryAction(Enum):
    HIGHWAY_ENTRY = "highway_entry"
    TUNNEL = "tunnel"
    RIGHT_TURN_LANE = "right_turn_lane"
    LEFT_TURN_LANE = "left_turn_lane"
    BUS_LANE = "bus_lane"
    BIKE_LANE = "bike_lane"
    SCHOOL_ZONE = "school_zone"
    CONSTRUCTION_ZONE = "construction_zone"
    NONE = "none"

@dataclass
class TBTInformation:
    """Turn-by-turn navigation instruction."""
    current_action: DrivingAction
    current_distance: float  # meters
    current_time: float        # seconds
    future_action: DrivingAction
    future_distance: float
    supplementary: SupplementaryAction

def extract_tbt_from_api(nav_api_response):
    """Extract TBT from navigation API response."""
    # Simplified - real implementation would parse API
    maneuvers = nav_api_response.get('maneuvers', [])
    
    if len(maneuvers) == 0:
        return TBTInformation(
            current_action=DrivingAction.STRAIGHT,
            current_distance=50.0,
            current_time=5.0,
            future_action=DrivingAction.NONE,
            future_distance=0.0,
            supplementary=SupplementaryAction.NONE
        )
    
    current = maneuvers[0]
    future = maneuvers[1] if len(maneuvers) > 1 else None
    
    return TBTInformation(
        current_action=DrivingAction(current['action']),
        current_distance=current['distance'],
        current_time=current['time'],
        future_action=DrivingAction(future['action']) if future else DrivingAction.NONE,
        future_distance=future['distance'] if future else 0.0,
        supplementary=SupplementaryAction(current.get('supplementary', 'none'))
    )
```

---

## Appendix D: Limitations and Failure Cases

### D.1 Known Limitations

1. **Two-Room Environment Failure**
   - SNG-VLA fails in novel layouts not seen in training
   - Can't generalize to unseen roundabout configurations
   - Fix: More diverse training data

2. **Long Horizon Planning**
   - 2-second prediction window
   - Doesn't handle multi-minute routes
   - Fix: Hierarchical planning

3. **Multi-Task Generalization**
   - Trained on single-task NAVSIM
   - Doesn't transfer to other benchmarks
   - Fix: Multi-domain pre-training

4. **Real-Time TBT Dependency**
   - Assumes perfect navigation API
   - Degrades with imperfect API
   - Fix: Robust TBT extraction

5. **Latency**
   - 160ms vs TCP's 83ms
   - Not suitable for latency-critical apps
   - Fix: Model compression

### D.2 Qualitative Failure Examples

From the paper's qualitative analysis:

**Failure 1: Complex Roundabout**
- Expert: Exit 2 (right turn)
- SNG-VLA predicted: Continue (wrong)
- Cause: Ambiguous TBT in multi-exit roundabout

**Failure 2: Occluded Intersection**
- Expert: Turn left after pedestrian
- SNG-VLA predicted: Go straight
- Cause: Occluded view, no TBT guidance

**Failure 3: Construction Zone**
- Expert: Merge to alternate lane
- SNG-VLA predicted: Stop
- Cause: No supplementary action for construction

---

## Appendix E: Related Survey Cross-References

For deeper dives into related topics, see:

- **[SURVEY_LeWorldModel_LeWM.md](SURVEY_LeWorldModel_LeWM.md)** — World models for planning, JEPA, SIGReg
- **[DEEPSEEK_RL_TECHNIQUES_GUIDE.md](DEEPSEEK_RL_TECHNIQUES_GUIDE.md)** — GRPO, process rewards, RL for LLMs
- **SURVEY_WorldModels.md** — Generative vs predictive world models, MPC, Dreamer
- **Driving Models Survey** — UniAD, VAD, BEVFormer, Transfuser comparison

---

## Appendix F: Quick Reference Card

| Aspect | Details |
|--------|--------|
| **Paper** | arXiv:2604.12208 |
| **Venue** | ICRA 2026 |
| **Core Method** | SNG (Navigation Path + TBT) |
| **Model** | SNG-VLA (Qwen2.5 + SigLIP) |
| **Best Benchmark** | NAVSIM (88.24 PDMS) |
| **Best Benchmark** | Bench2Drive (67.17 DS) |
| **Key Insight** | Nav ablation barely hurts |
| **Limitation** | Latency (160ms) |
| **Code** | fudan-magic-lab.github.io |

---

*Expanded survey completed: 2026-05-09*

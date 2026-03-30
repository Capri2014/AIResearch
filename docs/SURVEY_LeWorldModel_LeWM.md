# LeWorldModel (LeWM) — Research Survey

**Date:** 2026-03-29  
**Status:** Complete  
**Focus:** End-to-end JEPA world model that finally solves representation collapse

---

## Overview

Joint Embedding Predictive Architectures (JEPAs) are a promising approach for learning world models in compact latent spaces. However, existing JEPA methods are notoriously fragile — they require complex multi-term losses, exponential moving averages (EMA), pre-trained encoders, or auxiliary supervision to prevent **representation collapse**. 

LeWorldModel (LeWM), by Lucas Maes et al. (Mila/NYU/Brown/Samsung), is the first JEPA that trains stably **end-to-end from raw pixels** using only:
1. A next-embedding prediction loss (simple MSE)
2. SIGReg — a Gaussian regularizer that provably prevents collapse

**Key results:**
- 15M parameters, trainable on a single GPU in hours
- Planning: **<1 second** (48× faster than DINO-WM's ~47 seconds)
- Competitive with foundation-model-based world models on robotics tasks
- Latent space **naturally encodes physical structure** — spatial relationships, object locations, angles
- Can detect physically implausible events (teleportation, color changes)

---

## Table of Contents

1. [Background](#1-background)
2. [Core Concepts](#2-core-concepts)
3. [Key Method: LeWorldModel](#3-key-method-leworldmodel)
4. [Comparison & Tradeoffs](#4-comparison--tradeoffs)
5. [Applications to Our Work](#5-applications-to-our-work)
6. [Open Problems](#6-open-problems)
7. [References](#7-references)

---

<a name="1-background"></a>
## 1. Background

### 1.1 World Models — What Are They?

**Intuition:** A world model learns "what happens next" in an environment. Given the current state and an action, it predicts the future state. This lets an agent **plan in imagination** — simulate many possible futures, pick the best action — without actually executing each one.

World models are central to the dream of autonomous agents that can:
- Learn from offline data (without interacting with the real environment)
- Reason about counterfactual actions ("what if I do X instead of Y?")
- Improve their own behavior without trial-and-error in the real world

**The classic approach:** Learn a dynamics model `p(s' | s, a)` that predicts the next state given current state and action. Then use this model for planning or policy learning.

**The problem:** High-dimensional inputs (images, pixels) are hard to model. Pixel-space prediction is computationally expensive and requires modeling irrelevant details (shadows, textures) rather than semantic content.

---

### 1.2 JEPA — Joint Embedding Predictive Architecture

**Paper:** LeCun (2022) — [arXiv:2206.07769](https://arxiv.org/abs/2206.07769)

**Intuition:** Instead of predicting pixels, JEPA predicts in a **compact latent space**. An encoder compresses observations into low-dimensional embeddings, and a predictor models the dynamics in that space.

```
Raw pixels → Encoder → Compact latent z → Predictor → Next latent ẑ
                                   ↑                              ↑
                              learned via               learned via
                              prediction loss           prediction loss
```

**Why this is better than pixel prediction:**
- **Efficiency:** Latent space is smaller (192 dimensions vs. 128×128×3 = 49K pixels)
- **Semantic:** Embeddings capture what matters (object positions, spatial relationships) vs. what doesn't (pixel noise, textures)
- **Planning:** Rolls out are fast — no pixel generation needed, just forward through the predictor

---

### 1.3 The Core Problem: Representation Collapse

**The Problem It Solves:** JEPA is highly prone to **collapse** — a failure mode where the model maps all inputs to nearly identical embeddings to trivially satisfy the prediction objective.

```
Normal training:
  Different frames → Different embeddings → Predict next → Learn dynamics ✓

Collapse mode:
  All frames → Same embedding [0, 0, 0, ...] → Prediction is trivial → Learn nothing ✗
```

**Why collapse happens:** The prediction objective alone doesn't force the encoder to produce diverse, informative representations. There's no penalty for collapsing — the predictor just learns to output the constant and the loss is still minimized.

**Existing "fixes" are all problematic:**

| Heuristic | How it works | Problem |
|-----------|-------------|---------|
| **Exponential Moving Average (EMA)** | Maintain a slowly-moving copy of the encoder as target | Doesn't minimize a well-defined objective; theoretical understanding is limited |
| **Stop-gradient (SG)** | Don't backprop through the target encoder | Same issue — not grounded in optimization theory |
| **Pre-trained encoder** | Freeze a foundation model (DINO, CLIP) | Limits expressivity; can't adapt to task-specific dynamics |
| **Multi-term losses (VICReg)** | Add diversity loss + covariance loss + prediction loss | 6 hyperparameters to tune; training is unstable |
| **Reward signals** | Use RL rewards as auxiliary supervision | Requires reward annotations; not task-agnostic |

**The ideal solution:** A single, principled regularizer that provably prevents collapse without any heuristics.

---

### 1.4 Key Papers Timeline

| Year | Paper | Contribution | Relevance |
|------|-------|-------------|-----------|
| 2022 | LeCun — "A theory of learning from pixels" | JEPA conceptual introduction | Foundational |
| 2023 | I-JEPA (Assran et al.) | Self-supervised image learning via JEPA | Uses EMA/SG |
| 2023 | V-JEPA (Facebook AI) | Video prediction via JEPA | Uses EMA/SG |
| 2024 | PLDM (Cai et al.) | End-to-end action-conditioned JEPA | Uses VICReg, unstable |
| 2024 | DINO-WM (Wu et al.) | Foundation-model JEPA world model | Pre-trained, fast enough |
| 2025 | LeJEPA (Balestriero et al.) | SIGReg theory | Foundation for LeWM |
| **2026** | **LeWorldModel (Maes et al.)** | **End-to-end JEPA without heuristics** | **This paper** |

---

<a name="2-core-concepts"></a>
## 2. Core Concepts

### 2.1 SIGReg — Sketched Isotropic Gaussian Regularizer

**Paper:** [LeJEPA (arXiv:2511.08544)](https://arxiv.org/abs/2511.08544) — Balestriero et al., Nov 2025

**Intuition:** The key insight is mathematical: **for a JEPA to not collapse, its embeddings must be non-degenerate**. The simplest non-degenerate distribution in high dimensions is an **isotropic Gaussian** (mean zero, equal variance in all directions).

SIGReg enforces this by projecting embeddings onto random directions and checking if each projection follows a Gaussian distribution.

**Why this works:**
- If all 1D projections are Gaussian → by Cramér-Wold theorem → the full joint distribution is Gaussian
- An isotropic Gaussian in latent space has:
  - Full rank (no collapse)
  - Equal variance in all directions
  - Maximum entropy for a given variance
- The prediction loss then forces these embeddings to be **useful** (not just non-collapsed)

**The Math:**

```
Let Z ∈ R^(N × B × d) be latent embeddings:
  N = history length (e.g., 8 frames)
  B = batch size
  d = embedding dimension (e.g., 192)

SIGReg(Z) = (1/M) Σ_m=1^M T(h^(m))
  where h^(m) = Z · u^(m)  (projection onto random direction)
  and u^(m) ∈ S^(d-1)  (random unit-norm direction)
  and T(·) = Epps-Pulley statistical test for normality
```

The Epps-Pulley test measures how Gaussian a distribution is using empirical CDF quantiles.

**The complete loss:**
```
L_LeWM = L_pred + λ * SIGReg(Z)
  where L_pred = ||ẑ_{t+1} - z_{t+1}||² (simple MSE on latent embeddings)
```

**Only 1 hyperparameter to tune:** `λ` (SIGReg weight, typically 0.1)

---

**Code Implementation (from LeJEPA paper):**

```python
import torch
import numpy as np

def epps_pulley_test(h: torch.Tensor) -> torch.Tensor:
    """
    Epps-Pulley test for univariate normality.
    
    Measures how Gaussian a 1D distribution is.
    Returns 0 if perfectly Gaussian, higher values if non-Gaussian.
    """
    h = h.flatten().cpu().numpy()
    n = len(h)
    if n < 8:
        return torch.tensor(0.0, device=h.device)
    
    # Sort values
    h_sorted = np.sort(h)
    
    # Empirical quantiles
    ts = (np.arange(1, n + 1) - 0.5) / n
    phi_inv_t = norm.ppf(ts)  # Inverse CDF of standard normal
    
    # Compute statistics
    h_mean = h.mean()
    h_std = h.std() + 1e-8
    h_norm = (h_sorted - h_mean) / h_std
    
    # Simplified EP statistic: measures deviation from normality
    z_diff = np.diff(h_norm)
    T = np.sum((phi_inv_t[1:] - phi_inv_t[:-1]) * z_diff ** 2)
    
    return torch.tensor(T / (n + 1), device=h.device)


def sigreg_loss(embeddings: torch.Tensor, num_projections: int = 1024, reg_weight: float = 0.1) -> torch.Tensor:
    """
    Compute SIGReg loss for JEPA anti-collapse regularization.
    
    Args:
        embeddings: [B, d] — batch of latent embeddings
        num_projections: number of random directions to test
        reg_weight: weight λ for SIGReg
    
    Returns:
        SIGReg loss (scalar)
    """
    B, d = embeddings.shape
    
    # Random projection directions (fixed for efficiency)
    if not hasattr(sigreg_loss, 'projections'):
        sigreg_loss.projections = torch.randn(num_projections, d)
        sigreg_loss.projections = sigreg_loss.projections / sigreg_loss.projections.norm(dim=-1, keepdim=True)
    
    projections = sigreg_loss.projections.to(embeddings.device)
    
    # Project onto random directions
    h = embeddings @ projections.T  # [B, M]
    
    # Apply EP test to each projection, then average
    ep_stats = torch.stack([
        epps_pulley_test(h[:, i])
        for i in range(num_projections)
    ])  # [M]
    
    return reg_weight * ep_stats.mean()


def leworldmodel_loss(predicted_next: torch.Tensor, target_next: torch.Tensor,
                      embeddings: torch.Tensor, lambda_reg: float = 0.1) -> dict:
    """
    Full LeWM training objective.
    
    Args:
        predicted_next: [B, T-1, d] predicted next embeddings
        target_next: [B, T-1, d] ground truth next embeddings
        embeddings: [B, T, d] all embeddings (for SIGReg)
        lambda_reg: weight for SIGReg
    
    Returns:
        dict with total_loss, pred_loss, sigreg_loss
    """
    # 1. Prediction loss (MSE on latent space)
    pred_loss = torch.nn.functional.mse_loss(predicted_next, target_next)
    
    # 2. SIGReg (enforce Gaussian embeddings)
    all_embeddings = embeddings.reshape(-1, embeddings.shape[-1])  # [B*T, d]
    sigreg_loss = sigreg_loss(all_embeddings, reg_weight=lambda_reg)
    
    # Total loss
    total_loss = pred_loss + sigreg_loss
    
    return {
        'total_loss': total_loss,
        'pred_loss': pred_loss,
        'sigreg_loss': sigreg_loss,
    }
```

**Cross-Comparison: Anti-Collapse Methods**

| Method | Principle | Hyperparameters | Stability | Theoretical Basis |
|--------|-----------|-----------------|-----------|------------------|
| EMA encoder | Slowly track encoder | decay rate | Medium | Weak (no loss) |
| Stop-gradient | Asymmetric updates | none | Medium | Weak (no loss) |
| Pre-trained frozen | Fixed features | none | High | Transfer learning |
| VICReg + friends | Multi-objective | 6+ | Low | Empiricism |
| Diversity loss | Explicit variance | 2-3 | Medium | Weak |
| **SIGReg** | **Gaussian regularizer** | **1** | **High** | **Strong (Cramér-Wold)** |

---

### 2.2 End-to-End JEPA Training

**Intuition:** Instead of using pre-trained features (DINO-WM) or frozen targets (EMA), train the encoder and predictor **jointly** from scratch using only:
1. Raw pixel observations
2. Associated actions
3. The two-term loss (prediction + SIGReg)

**Why this matters:** Pre-trained encoders limit expressivity. A frozen DINO encoder is optimized for ImageNet classification, not for predicting robot arm movements in a warehouse. End-to-end training allows the encoder to learn task-specific representations.

**The training loop:**

```python
def train_step(encoder, predictor, optimizer, observations, actions, lambda_reg=0.1):
    """
    Single training step for LeWM.
    
    Key insight: Only 2 loss terms. No EMA, no stop-gradient, no tricks.
    """
    # Encode all frames
    embeddings = encoder(observations)  # [B, T, d]
    
    # Predict next frame embedding (teacher forcing)
    predicted_next = predictor(
        embeddings[:, :-1],    # Current embeddings
        actions[:, :-1]         # Actions taken
    )  # [B, T-1, d]
    
    # Target next frame embedding
    target_next = embeddings[:, 1:].detach()  # [B, T-1, d]
    
    losses = leworldmodel_loss(predicted_next, target_next, embeddings, lambda_reg)
    
    losses['total_loss'].backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return losses
```

---

### 2.3 Latent Planning with Cross-Entropy Method (CEM)

**Intuition:** At test time, given a start image and a goal image, find the action sequence that brings us closest to the goal — all in **latent space**.

**The planning loop:**
```
1. Encode start image → z_start (192-dim)
2. Encode goal image → z_goal (192-dim)
3. Initialize action sequence A = [a_0, a_1, ..., a_H]
4. Repeat until convergence:
   a. Sample candidate action sequences from Gaussian around A
   b. Roll out each candidate through the predictor:
      z_1 = pred(z_start, a_0)
      z_2 = pred(z_1, a_1)
      ...
      z_H = pred(z_{H-1}, a_{H-1})
   c. Score by: dist(z_H, z_goal) — how close is final state to goal?
   d. Keep top-k candidates, update A to their mean
5. Execute first action of best sequence
6. Repeat (replan at each step)
```

**Why it's fast:** Each step is just a forward pass through the predictor (~10M params). No pixel generation. With only ~200 tokens per frame (vs DINO-WM's ~40K), planning is **48× faster**.

---

### 2.4 Physical Understanding in Latent Space

**Intuition:** A good world model should understand physics — not just predict pixels, but encode meaningful physical structure in its latent space.

**LeWM shows three emergent properties:**

1. **Spatial Probing:** Linear probes can predict physical quantities (block position, angle) from latent embeddings with high accuracy.

2. **t-SNE Structure:** Latent embeddings preserve spatial neighborhood relationships. Close objects in the scene → close in latent space.

3. **Surprise Detection:** The model assigns higher prediction error to physically implausible events (teleportation, color changes) vs. normal motion.

---

<a name="3-key-method-leworldmodel"></a>
## 3. Key Method: LeWorldModel (LeWM)

**Paper:** [LeWorldModel (arXiv:2603.19312)](https://arxiv.org/abs/2603.19312) — Maes, Le Lidec, Scieur, LeCun, Balestriero  
**Affiliation:** Mila, NYU, Samsung SAIL, Brown University  
**Published:** March 2026

### 3.1 Architecture

**Encoder (ViT-Tiny):**
```
Input: 128×128 RGB image → 1 token per frame (192-dim)
Architecture: ViT-Tiny (5M params)
  - Patch size: 14
  - Layers: 12
  - Attention heads: 3
  - Hidden dim: 192
  - Output: [CLS] token → 1-layer MLP → 192-dim embedding
```

**Predictor (Action-conditioned transformer):**
```
Input: N frame embeddings + N actions → next frame embedding
Architecture: Transformer (10M params)
  - Layers: 6
  - Attention heads: 16
  - Dropout: 10%
  - Action conditioning: Adaptive Layer Normalization (AdaLN)
  - Output: 192-dim latent prediction

Key: AdaLN parameters initialized to zero → stable training
```

**Total: ~15M parameters, single GPU, ~4 hours to train**

### 3.2 Quantitative Results

**Planning Performance (Success Rate)**

| Environment | Description | LeWM | DINO-WM | PLDM |
|-------------|-------------|------|---------|------|
| Two-Room | 2D navigation | 52% | 97% | 58% |
| Reacher | 2-joint arm | **95%** | 84% | 51% |
| Push-T | Block manipulation | **96%** | 88% | 40% |
| OGBench-Cube | 3D pick-and-place | 39% | **59%** | 15% |

**Key observations:**
- LeWM **outperforms DINO-WM** on Reacher and Push-T (pure pixels!)
- LeWM **beats DINO-WM on Push-T** even though DINO-WM uses extra proprioceptive input
- DINO-WM has edge on **visually complex 3D** tasks (ImageNet pretraining helps)
- LeWM **struggles on Two-Room** (low intrinsic dimensionality issue)

**Planning Speed Comparison**

| Method | Planning Time | Tokens/Frame | Relative Speed |
|--------|--------------|--------------|---------------|
| DINO-WM | ~47s | ~40K | 1× |
| PLDM | ~0.8s | ~200 | ~60× |
| **LeWM** | **~1s** | **~200** | **~48×** |

---

### 3.3 Ablation Studies (from paper)

1. **SIGReg weight (λ):** Robust across 0.01 to 1.0 (unlike VICReg which is sensitive)
2. **Number of projections (M):** Negligible impact beyond 256
3. **Architecture:** Works with ResNets, ViTs, ConvNets
4. **Without SIGReg:** Training collapses within 10K steps
5. **Without prediction loss:** Embeddings become uniform Gaussian (no dynamics)

---

<a name="4-comparison--tradeoffs"></a>
## 4. Comparison & Tradeoffs

### 4.1 JEPA Methods Comparison

| Method | End-to-End | Heuristics | Hyperparameters | Planning Speed | Performance | Compute |
|--------|-----------|------------|----------------|----------------|-------------|---------|
| I-JEPA / V-JEPA | No (uses EMA) | Yes (SG, EMA) | 2-3 | N/A (representation) | SOTA on images | Medium |
| PLDM | Yes | Yes (VICReg, 6 terms) | 6 | Fast (<1s) | Low | Medium |
| DINO-WM | No (frozen encoder) | None | 1 | Slow (47s) | High | Very High |
| **LeWM** | **Yes** | **None** | **1** | **Fast (<1s)** | **High** | **Low** |

### 4.2 World Model Approaches Comparison

| Approach | Representation | Training Data | Planning | Main Limitation |
|----------|---------------|---------------|----------|-----------------|
| **Generative (Dreamer)** | Pixel latent | Reward-required | Imagination | Needs rewards |
| **TD-MPC** | Compact latent | Reward-required | CEM | Needs rewards |
| **Oracle dynamics** | State-based | Perfect | Fast | Not scalable |
| **DINO-WM** | Frozen vision | Offline | Slow | Not task-specific |
| **PLDM** | Learned | Offline | Fast | Unstable |
| **LeWM** | Learned | Offline | Fast | Low-dim tasks |

### 4.3 Quick Reference Decision Guide

| If you need... | Use | Why |
|---------------|-----|-----|
| Best pixel accuracy on complex scenes | DINO-WM | Large-scale pretraining |
| Fast planning + good performance | LeWM | 48× faster, single GPU |
| Task-agnostic, no pretraining | LeWM | End-to-end, no frozen features |
| Simple, stable training | LeWM | 1 hyperparameter, no heuristics |
| Rewards available for RL | DreamerV3 | Imagination-based policy learning |
| Rapid iteration, many baselines | PLDM | Fast to train but unstable |

---

<a name="5-applications-to-our-work"></a>
## 5. Applications to Our Work

### 5.1 Relevance to Waypoint Prediction

**Direct applicability:** LeWM's architecture is very similar to what we've built:

```
Our system:                    LeWM:
Encoder → [CLS] → 192-dim  →  ✓ Same (ViT + projection)
Predictor → next waypoints   →  ✓ Similar (action-conditioned transformer)
GRPO for policy learning      →  Planning via CEM

Difference:
- Ours: RL (GRPO) for policy learning
- LeWM: Planning (CEM) for goal-conditioned control
```

**Key insight:** LeWM's encoder/predictor could be used as the world model in our RL pipeline:
- Encode current observation → latent state
- Predict next latent state given action
- Use predicted trajectory for planning or as auxiliary loss in GRPO

### 5.2 How LeWM Could Improve Our System

**Option 1: World Model as Auxiliary Loss**

```python
# In our waypoint prediction training:
class WaypointWithWorldModel(nn.Module):
    def __init__(self, waypoint_model, lewm_encoder, lewm_predictor):
        self.waypoint_model = waypoint_model
        self.lewm_encoder = lewm_encoder
        self.lewm_predictor = lewm_predictor
        self.alpha = 0.1  # Weight for world model loss
    
    def forward(self, observations, actions):
        # Waypoint prediction (our main task)
        waypoints = self.waypoint_model(observations, actions)
        
        # World model prediction (auxiliary task)
        obs_emb = self.lewm_encoder(observations)
        next_emb_pred = self.lewm_predictor(obs_emb, actions)
        
        # World model loss: prediction should be consistent
        world_model_loss = (next_emb_pred - obs_emb[:, 1:]).norm()
        
        return waypoints + self.alpha * world_model_loss
```

**Why this helps:** The world model provides a **consistency regularizer** — the same encoder features that predict waypoints should also predict future states.

**Option 2: SIGReg for Our JEPA**
- If we ever want to build a JEPA-style world model, SIGReg is the principled anti-collapse regularizer
- Much more stable than VICReg or other approaches

### 5.3 LeJEPA — SIGReg's Theoretical Foundation

The SIGReg regularizer comes from [LeJEPA (arXiv:2511.08544)](https://arxiv.org/abs/2511.08544) — Balestriero et al., Nov 2025.

**Key theoretical contributions:**
1. **Optimal distribution identified:** Isotropic Gaussian minimizes downstream prediction risk
2. **SIGReg implementation:** Sketch-based statistical test (Epps-Pulley) for normality
3. **Linear complexity:** O(N) time and memory via random projections
4. **No heuristics:** No EMA, stop-gradient, teacher-student, or hyperparameter schedulers
5. **Proven stability:** Tested across 10+ datasets, 60+ architectures, varying scales and domains

**Why isotropic Gaussian is optimal:**
- For a fixed variance, the isotropic Gaussian maximizes entropy
- Maximum entropy → least assumptions about structure → most generalizable
- Any deviation from isotropy would introduce bias
- The prediction loss then forces these embeddings to be **useful for prediction**

### 5.4 Key Lessons from LeWM for Our System

1. **SIGReg is a powerful anti-collapse regularizer** — if we ever need to train a JEPA, use SIGReg
2. **Two-term losses are enough** — don't over-engineer with 6+ loss terms
3. **AdaLN for action conditioning** — better than concatenation for transformers
4. **Single hyperparameter is achievable** — LeWM reduces 6 hyperparams to 1
5. **Physical structure emerges naturally** — good representations capture meaningful structure

---

<a name="6-open-problems"></a>
## 6. Open Problems

1. **Why does LeWM struggle on Two-Room?** — SIGReg's Gaussian prior may be too strong for simple, low-dimensional tasks. Adaptive regularization could help.

2. **Scaling laws** — LeWM tested at 15M params. Does SIGReg remain stable at 1B+ params?

3. **Long-horizon planning** — LeWM plans for 8-16 steps. Error accumulation at 100+ step horizons?

4. **Real-world deployment** — All experiments on simulation. Real-world performance unknown.

---

<a name="7-references"></a>
## 7. References

- [LeWorldModel (arXiv:2603.19312)](https://arxiv.org/abs/2603.19312) — Maes et al., Mila/NYU/Brown (Mar 2026) **← Primary source**
- [LeJEPA / SIGReg (arXiv:2511.08544)](https://arxiv.org/abs/2511.08544) — Balestriero et al. (Nov 2025) — SIGReg theory and implementation
- [PLDM (NeurIPS 2024)](https://arxiv.org/abs/2410.06991) — Cai et al. — End-to-end JEPA, baseline comparison
- [DINO-WM (ICLR 2025)](https://arxiv.org/abs/2410.06991) — Wu et al. — Foundation model JEPA, baseline comparison
- [I-JEPA (CVPR 2023)](https://arxiv.org/abs/2301.08264) — Assran et al. — Image JEPA with EMA/SG
- [JEPA (LeCun 2022)](https://arxiv.org/abs/2206.07769) — LeCun — Original JEPA conceptual paper
- [World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122) — Foundational world model paper
- [DreamerV3 (ICLR 2023)](https://arxiv.org/abs/2301.04104) — Imagination-based RL with world models

---

## Notes

*Key takeaways:*
- SIGReg + 2-term loss = stable end-to-end JEPA training
- No EMA, no stop-gradient, no pre-trained encoder needed
- 48× faster planning than DINO-WM with competitive performance
- Physical structure emerges naturally in latent space
- 15M params, single GPU, hours to train

*Questions to follow up:*
- Can SIGReg be applied to our RL training as an auxiliary loss?
- Should we consider LeWM-style planning vs. our GRPO policy approach?
- How does LeWM perform on real-world (CARLA) data?

---

*Survey completed by OpenClaw agent — 2026-03-29*
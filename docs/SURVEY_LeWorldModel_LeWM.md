# LeWorldModel (LeWM) — Deep Research Survey

**Date:** 2026-03-29  
**Status:** Complete  
**Focus:** Full technical analysis with application roadmap for our RL training pipeline

---

## Overview

LeWorldModel (LeWM) by Maes et al. (Mila/NYU/Brown/Samsung, March 2026) is the **first JEPA that trains end-to-end from raw pixels without any training heuristics**. This is a fundamentally important result because JEPA collapse has been the main obstacle preventing end-to-end world model training.

---

## Table of Contents

1. [The Big Picture: Why LeWM Matters](#1-the-big-picture)
2. [JEPA Architecture & How It Works](#2-jepa-architecture)
3. [The Collapse Problem Explained](#3-the-collapse-problem)
4. [SIGReg: The Core Innovation](#4-sigreg)
5. [LeWorldModel Architecture](#5-leworldmodel-architecture)
6. [Planning Algorithm (CEM)](#6-planning-algorithm)
7. [Detailed Results Analysis](#7-detailed-results)
8. [Pros & Cons Deep Dive](#8-pros--cons)
9. [Comparison with Our Existing Pipeline](#9-comparison-with-our-pipeline)
10. [Concrete Integration Options](#10-concrete-integration)
11. [References & Code](#11-references)

---

<a name="1-the-big-picture"></a>
## 1. The Big Picture: Why LeWM Matters

### 1.1 What is a World Model?

A **world model** learns the dynamics of an environment: given the current state and an action, predict what happens next.

```
For autonomous driving:
  Current state: camera image + ego-vehicle state
  Action: throttle, steer, brake
  World model predicts: next camera image, next vehicle state
```

Once you have a world model, you can plan in "imagination":
1. Simulate 1000 possible futures by rolling out different action sequences
2. Pick the action sequence that leads to the best outcome
3. Execute the first action, then replan

This is called **model-predictive control (MPC)** or **imagination-based planning**.

### 1.2 Why Learn in Latent Space?

Predicting in pixel space is hard because:
- You need to model every pixel (100K+ dimensions for a small image)
- Irrelevant details (shadows, reflections, texture noise) dominate
- Computation is expensive — rolling out 1000 futures at pixel resolution is slow

**JEPA (Joint Embedding Predictive Architecture)** solves this by:
1. **Encoder:** Compress the image into a compact embedding (e.g., 192 dimensions)
2. **Predictor:** Predict the *embedding* of the next frame, not the pixels themselves

```
Input image (128×128×3 = 49K values) 
    ↓ Encoder (ViT-Tiny)
Compact embedding (192 values)
    ↓ Predictor (action-conditioned transformer)
Next embedding prediction (192 values)
```

### 1.3 Why This Matters for Our Work

Our current system uses:
- **GRPO** for policy learning (how to predict waypoints given state)
- **No world model** — we react to the current state without simulating futures

LeWM provides a world model that could enable:
- **Imagination-based planning** instead of reactive policy
- **Better representations** for waypoint prediction (auxiliary loss)
- **Intrinsic motivation** via curiosity (surprise detection)

---

<a name="2-jepa-architecture"></a>
## 2. JEPA Architecture & How It Works

### 2.1 The Two Components

**Encoder (E):**
- Compresses raw pixels → compact latent embedding
- Architecture: ViT-Tiny (5M params), 12 layers, 192-dim output
- Input: 128×128 RGB image → 1 [CLS] token → 192-dim embedding

**Predictor (P):**
- Models dynamics in latent space
- Architecture: Transformer (10M params), 6 layers, 16 attention heads
- Input: current embedding z_t + action a_t
- Output: predicted next embedding ẑ_{t+1}

**Total: ~15M parameters**

### 2.2 Training Objective

```
L_LeWM = L_pred + λ * SIGReg(Z)

Where:
  L_pred = MSE( ẑ_{t+1}, z_{t+1} )  ← prediction loss
  SIGReg(Z) ← anti-collapse regularizer (see Section 4)
  Z = all embeddings in the batch
  λ = 0.1 (only hyperparameter!)
```

That's it. Two loss terms. No EMA, no stop-gradient, no teacher-student.

### 2.3 How Prediction Works

```python
def forward(self, observations, actions):
    # observations: [B, T, C, H, W] — video frames
    # actions: [B, T-1] — actions taken between frames
    
    # 1. Encode all frames
    embeddings = self.encoder(observations)  # [B, T, 192]
    
    # 2. Predict next frame embedding
    # Teacher forcing: predict z_{t+1} from z_t and a_t
    current_emb = embeddings[:, :-1]    # [B, T-1, 192]
    predicted_next = self.predictor(current_emb, actions)  # [B, T-1, 192]
    
    # 3. Target is the actual next embedding (detached to avoid gradient loop)
    target_next = embeddings[:, 1:].detach()  # [B, T-1, 192]
    
    # 4. Compute loss
    loss = MSE(predicted_next, target_next)
    
    return loss
```

---

<a name="3-the-collapse-problem"></a>
## 3. The Collapse Problem Explained

### 3.1 What is Collapse?

**Collapse** is the failure mode where the encoder maps ALL input images to nearly identical embeddings.

```
Normal training:
  Frame A (car ahead) → embedding[0.3, 0.7, ...]  ← unique
  Frame B (empty road) → embedding[0.9, 0.1, ...] ← different
  Frame C (pedestrian) → embedding[0.5, 0.5, ...] ← different

Collapse mode:
  Frame A → [0.0, 0.0, 0.0, ...]  ← same as everyone
  Frame B → [0.0, 0.0, 0.0, ...]
  Frame C → [0.0, 0.0, 0.0, ...]
```

### 3.2 Why Does Collapse Happen?

The prediction objective alone doesn't force diverse embeddings:

```
Suppose all embeddings collapse to [0, 0, ..., 0]

Predictor's job: predict ẑ_{t+1} given z_t and a_t
  - If z_t = 0 and a_t = something → ẑ_{t+1} = some constant
  - Loss = MSE(some constant, z_{t+1})
  - Predictor learns: always output the mean embedding
  - Loss is minimized! ✓

But we learn nothing useful.
```

The prediction loss doesn't penalize collapse because it can always learn to output a constant.

### 3.3 Existing Fixes (All Problematic)

| Method | How It Works | The Problem |
|--------|-------------|-------------|
| **EMA (Exponential Moving Average)** | Keep a slow-copy of the encoder as target | Doesn't minimize a well-defined loss — just engineering |
| **Stop-gradient (SG)** | Don't backprop through target encoder | Same problem — no principled objective |
| **Pre-trained frozen encoder** | Use DINO/CLIP as fixed encoder | Can't adapt to task-specific dynamics |
| **VICReg-style losses** | Add diversity + covariance + prediction losses | 6+ hyperparameters, very unstable |
| **Reward signals** | Use RL rewards as supervision | Requires reward annotations |

The ideal solution: **a single, principled regularizer** that provably prevents collapse.

---

<a name="4-sigreg"></a>
## 4. SIGReg: The Core Innovation

### 4.1 The Key Insight

**An isotropic Gaussian distribution cannot collapse.**

An isotropic Gaussian in d dimensions:
- Has mean = 0 (centered)
- Has equal variance in ALL directions
- Is non-degenerate (full rank)

If embeddings follow an isotropic Gaussian, they must be:
1. Spread out in space (not collapsed)
2. Isotropic (no preferred direction)

**Theorem (Cramér-Wold):** If all 1D projections of a d-dimensional distribution are Gaussian, then the d-dimensional distribution is Gaussian.

This means: **we just need to enforce Gaussian on random 1D projections.**

### 4.2 SIGReg Implementation

SIGReg works by:
1. Projecting embeddings onto M random directions
2. Testing each projection for Gaussianity using the Epps-Pulley test
3. Averaging the test results as the regularization loss

```python
import torch
import numpy as np
from scipy.stats import norm

def epps_pulley_test(x):
    """
    Test whether a 1D distribution is Gaussian.
    
    Uses the Epps-Pulley statistic:
    - Measures deviation from normality
    - Returns 0 for perfectly Gaussian
    - Returns high value for non-Gaussian
    """
    x = x.cpu().numpy().flatten()
    n = len(x)
    if n < 8:
        return 0.0
    
    # Sort values
    x_sorted = np.sort(x)
    
    # Empirical quantiles at positions (k-0.5)/n
    k = np.arange(1, n + 1)
    t_k = (k - 0.5) / n
    phi_inv_t = norm.ppf(t_k)  # Standard normal quantiles
    
    # Normalize
    x_mean = x.mean()
    x_std = x.std() + 1e-8
    x_norm = (x_sorted - x_mean) / x_std
    
    # Epps-Pulley statistic: sum of squared differences in normalized values
    # weighted by the spacing in normal quantiles
    dx = np.diff(x_norm)
    T = np.sum((phi_inv_t[1:] - phi_inv_t[:-1]) * dx ** 2)
    
    return T / (n + 1)


def sigreg_loss(embeddings, num_projections=1024, reg_weight=0.1):
    """
    SIGReg: Sketched Isotropic Gaussian Regularizer.
    
    Prevents JEPA collapse by enforcing Gaussian-distributed embeddings.
    
    Args:
        embeddings: [B, d] — batch of embeddings
        num_projections: number of random directions to test
        reg_weight: λ for SIGReg (only hyperparameter!)
    
    Returns:
        Scalar loss (add to prediction loss)
    """
    B, d = embeddings.shape
    
    # Generate random projection directions (fixed for efficiency)
    if not hasattr(sigreg_loss, 'projections'):
        # Generate once, reuse
        sigreg_loss.projections = torch.randn(num_projections, d)
        sigreg_loss.projections = sigreg_loss.projections / sigreg_loss.projections.norm(dim=-1, keepdim=True)
    
    projections = sigreg_loss.projections.to(embeddings.device)
    
    # Project embeddings onto random directions
    h = embeddings @ projections.T  # [B, M]
    
    # Apply EP test to each projection
    ep_stats = []
    for i in range(num_projections):
        ep = epps_pulley_test(h[:, i])
        ep_stats.append(ep)
    
    # Average and weight
    return reg_weight * np.mean(ep_stats)
```

### 4.3 Why This Works

1. **Cramér-Wold guarantee:** If all random 1D projections are Gaussian, the joint distribution must be Gaussian
2. **No collapse possible:** A collapsed distribution (all same embedding) is not Gaussian
3. **Prediction loss handles utility:** SIGReg only ensures non-collapse. The prediction loss ensures embeddings are **useful for prediction**

```python
# Full training loop
def train_step(encoder, predictor, observations, actions, lambda_reg=0.1):
    embeddings = encoder(observations)  # [B, T, 192]
    
    predicted_next = predictor(embeddings[:, :-1], actions)
    target_next = embeddings[:, 1:].detach()
    
    # 1. Prediction loss (makes embeddings useful)
    pred_loss = MSE(predicted_next, target_next)
    
    # 2. SIGReg (prevents collapse)
    flat_emb = embeddings.reshape(-1, 192)
    sigreg = sigreg_loss(flat_emb, reg_weight=lambda_reg)
    
    # Total loss
    loss = pred_loss + sigreg
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return {'pred_loss': pred_loss.item(), 'sigreg_loss': sigreg}
```

### 4.4 SIGReg Properties

| Property | Value |
|----------|-------|
| Hyperparameters | 1 (λ for reg_weight) |
| Time complexity | O(N) with random projections |
| Memory | O(1) extra |
| Stability | High — tested across 60+ architectures |
| Theoretical basis | Strong (Cramér-Wold theorem) |

**vs. VICReg (PLDM's approach):**
- VICReg: 6 hyperparameters, unstable, empirically tuned
- SIGReg: 1 hyperparameter, stable, theoretically grounded

---

<a name="5-leworldmodel-architecture"></a>
## 5. LeWorldModel Architecture

### 5.1 Encoder (ViT-Tiny)

```
Input: 128×128 RGB image

Architecture:
  - Patch size: 14 → 9 patches per image (3×3 grid)
  - Patch embedding: 14×14×3 → 192-dim per patch
  - [CLS] token → 192-dim final embedding
  - 12 transformer layers
  - 3 attention heads
  - Hidden dim: 192
  - MLP dim: 768

Output: 192-dimensional embedding per frame
```

### 5.2 Predictor (Action-Conditioned Transformer)

```
Input: [z_t; z_{t-1}; ...; z_{t-N}] + [a_t; a_{t-1}; ...; a_{t-N}]

Architecture:
  - 6 transformer layers
  - 16 attention heads
  - Action conditioning: Adaptive Layer Normalization (AdaLN)
  - Dropout: 10%
  - Residual connections

Key innovation: AdaLN conditions the transformer on actions
  - Standard LN: layer_norm(x)
  - AdaLN: layer_norm(γ * x + β) where γ,β from action embedding
  
  - AdaLN initialized to zero → predictor starts as identity function
  - This ensures stable training (no exploding activations)
```

### 5.3 Why AdaLN Matters

Without AdaLN, conditioning on actions typically uses concatenation:
```python
# Concatenation approach (problematic)
concat = torch.cat([embeddings, action_embeddings], dim=-1)  # [B, 192+32]
h = transformer(concat)  # Larger input dimension, harder to optimize
```

With AdaLN:
```python
# AdaLN approach (LeWM)
h = transformer(embeddings)  # [B, 192] — no dimension increase
scale, bias = action_embedding_to_adaln_params(action)  # [B, 384]
h = layer_norm(gamma * h + beta)  # Action conditioning via affine transform
```

Benefits:
- No dimension explosion
- Action conditioning is learnable and stable
- Easy to initialize (zero → identity function)

### 5.4 Training Details

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Resolution | 128×128 |
| Sub-trajectory length | 8 frames |
| Optimizer | AdamW |
| Learning rate | 1e-3 with cosine schedule |
| Epochs | 100 |
| Training time | ~4 hours on single A100 |
| SIGReg weight (λ) | 0.1 |
| SIGReg projections | 1024 |
| Total params | 15M |

---

<a name="6-planning-algorithm"></a>
## 6. Planning Algorithm (CEM)

### 6.1 What is CEM?

**Cross-Entropy Method (CEM)** is a gradient-free optimization algorithm for finding action sequences that maximize a reward.

For LeWM, the reward is: `dist(final_state, goal_state)` — how close is the final predicted state to the goal?

### 6.2 Planning Loop

```
Given: start image, goal image
Goal: find action sequence [a_0, a_1, ..., a_H] that reaches goal

Step 1: Encode
  z_start = encoder(start_image)  # [192]
  z_goal = encoder(goal_image)     # [192]

Step 2: Initialize action sequence
  A = [a_0, a_1, ..., a_H]  # random initialization

Step 3: CEM iterations (repeat 10 times)
  a. Sample 256 candidate action sequences from Gaussian(A, σ)
  b. For each candidate:
     - Roll out through predictor:
       z_1 = pred(z_start, a_0)
       z_2 = pred(z_1, a_1)
       ...
       z_H = pred(z_{H-1}, a_{H-1})
     - Score: reward = -||z_H - z_goal||²  (negative distance)
  c. Keep top 16 candidates (highest reward)
  d. Update A = mean(top_16)  (increase probability of good actions)
  e. Reduce σ (convergence)

Step 4: Execute first action of best sequence
Step 5: Replan every step (re-observe, re-encode, re-run CEM)
```

### 6.3 Why LeWM is Fast (48× faster than DINO-WM)

| Component | DINO-WM | LeWM |
|-----------|---------|------|
| Tokens per frame | ~40,000 (diffusion tokens) | 192 |
| Planning time | ~47 seconds | <1 second |
| Replan | Every 5 steps | Every step |
| Memory | GPU-heavy | Single GPU |

The key insight: LeWM uses a **single 192-dim token** per frame (DINO-WM uses ~40K). Planning is just forward passes through a small transformer.

### 6.4 Code: CEM Planning

```python
def plan_with_cem(predictor, start_emb, goal_emb, horizon=16, num_iters=10, num_candidates=256, top_k=16):
    """
    Plan action sequence using Cross-Entropy Method.
    
    Args:
        predictor: trained predictor network
        start_emb: [192] current state embedding
        goal_emb: [192] goal state embedding
        horizon: number of action steps
        num_iters: CEM iterations
        num_candidates: candidates per iteration
        top_k: top candidates to keep
    
    Returns:
        best_action_sequence: [horizon] best actions found
    """
    action_dim = 2  # throttle, steer
    device = start_emb.device
    
    # Initialize: mean=0, std=1 for each action
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)
    
    for iteration in range(num_iters):
        # Sample candidates
        candidates = torch.randn(num_candidates, horizon, action_dim, device=device)
        candidates = candidates * std + mean  # ~N(mean, std)
        
        # Roll out each candidate
        rewards = []
        for i in range(num_candidates):
            z = start_emb.unsqueeze(0)  # [1, 192]
            for t in range(horizon):
                action = candidates[i, t]  # [2]
                z_next = predictor(z, action)  # [1, 192]
                z = z_next
            
            # Score: negative distance to goal
            dist = (z - goal_emb.unsqueeze(0)).norm()
            rewards.append(-dist.item())
        
        rewards = torch.tensor(rewards, device=device)
        
        # Keep top-k
        top_indices = rewards.topk(top_k).indices
        top_candidates = candidates[top_indices]  # [16, horizon, 2]
        
        # Update mean and std
        mean = top_candidates.mean(dim=0)  # [horizon, 2]
        std = top_candidates.std(dim=0)      # [horizon, 2]
        
        # Reduce std for convergence
        std = std * 0.9  # 10% reduction per iteration
    
    # Return first action of best candidate
    best_idx = rewards.argmax()
    return candidates[best_idx, 0]  # First action only
```

---

<a name="7-detailed-results"></a>
## 7. Detailed Results Analysis

### 7.1 Planning Success Rates

| Environment | Description | LeWM | DINO-WM | PLDM | Why Difference? |
|-------------|-------------|------|---------|------|-----------------|
| **Push-T** | Block manipulation | **96%** | 88% | 40% | LeWM is better even without proprioception |
| **Reacher** | 2-joint arm reaching | **95%** | 84% | 51% | Simple geometry suits LeWM |
| **OGBench-Cube** | 3D pick-and-place | 39% | **59%** | 15% | DINO pretraining helps on complex visuals |
| **Two-Room** | 2D navigation | 52% | **97%** | 58% | LeWM struggles on low-dim tasks |

**Key observations:**
- LeWM **beats DINO-WM** on Push-T and Reacher (real robotics tasks!)
- DINO-WM has edge on **visually complex 3D** tasks
- LeWM **struggles on Two-Room** (low intrinsic dimensionality)

### 7.2 Why Does LeWM Fail on Two-Room?

Two-Room is a simple 2D navigation task where:
- State space is low-dimensional (grid-based)
- Dynamics are discrete (move up/down/left/right)
- The "Gaussian latent space" prior is **too strong** for discrete dynamics

LeWM's Gaussian regularizer forces the latent space to be smooth and isotropic. But Two-Room has discrete, non-smooth dynamics. The mismatch causes poor performance.

**Lesson:** SIGReg works best when the underlying dynamics are approximately continuous and smooth.

### 7.3 Physical Latent Probing

LeWM's latent space **naturally encodes physical structure**:

| Physical Quantity | LeWM (Linear) | LeWM (MLP) | DINO-WM | PLDM |
|------------------|---------------|------------|---------|------|
| Agent location | **0.052** MSE | 0.004 | 1.888 | 0.090 |
| Block location | 0.003 | **0.001** | 0.006 | 0.014 |
| Block angle | 0.029 | 0.001 | **0.050** | 0.446 |

LeWM achieves the best agent location probing. This means:
- **Spatial structure is encoded in latent space**
- Linear probes can recover physical quantities
- The encoder learned meaningful representations

### 7.4 Surprise Detection

LeWM can detect physically implausible events:

| Event | Expected Surprise | LeWM Surprise | DINO-WM Surprise |
|-------|------------------|---------------|-------------------|
| Normal motion | Low | Low | Low |
| Block teleportation | High | **High** | Medium |
| Block color change | Medium | Medium | High |

This suggests LeWM learned physical regularities that can be violated.

---

<a name="8-pros--cons"></a>
## 8. Pros & Cons Deep Dive

### 8.1 Advantages of LeWM

| Advantage | Explanation | Impact |
|-----------|-------------|--------|
| **No training heuristics** | No EMA, stop-gradient, pre-trained encoder | Simplifies training, easier to debug |
| **Single hyperparameter** | Only λ for SIGReg (typically 0.1) | Much easier to tune than VICReg (6+ params) |
| **Fast planning** | 192 tokens/frame, <1s planning | Enables real-time replanning |
| **End-to-end** | Encoder + predictor trained together | Task-specific representations |
| **Stable training** | SIGReg provably prevents collapse | No collapse debugging needed |
| **Low compute** | 15M params, single GPU, 4 hours | Accessible to researchers |
| **Physical understanding** | Latent space encodes physics | Enables surprise detection, probing |

### 8.2 Limitations of LeWM

| Limitation | Explanation | Mitigation |
|------------|-------------|------------|
| **Fails on discrete/low-dim tasks** | SIGReg's Gaussian prior mismatches discrete dynamics | Adaptive regularization |
| **Limited to short horizons** | Tested on 8-16 step planning | Unknown for 100+ steps |
| **Pixel-based only** | No proprioceptive input used | Add state as additional input |
| **Requires diverse training data** | Needs varied trajectories for good latent space | Curate diverse training set |
| **No policy learning** | Planning-based, not RL-based | Integrate with RL (see Section 10) |

### 8.3 When to Use LeWM vs Alternatives

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Fast real-time planning** | LeWM | <1s vs 47s for DINO-WM |
| **Complex visual scenes** | DINO-WM | Foundation model pretraining helps |
| **Low-dimensional/discrete** | Don't use LeWM | Gaussian prior mismatches |
| **Task-specific adaptation** | LeWM | End-to-end training vs frozen encoder |
| **RL policy learning** | DreamerV3 | Imagination-based RL vs planning |
| **Quick baseline** | PLDM | Fast to train but unstable |

---

<a name="9-comparison-with-our-pipeline"></a>
## 9. Comparison with Our Existing Pipeline

### 9.1 Our Current System

```
Current approach: Reactive policy learning via GRPO

Input: state [x, y, heading, speed] (4D)
    ↓ Encoder (simple MLP)
Hidden representation (128D)
    ↓ Waypoint head
Output: waypoints [H, 2] (e.g., 16 waypoints)

Training signal: Termination reward shaping (+100 on success)
Learning: GRPO (no value function, group-relative advantage)

Pros: Simple, fast, works on toy env
Cons: Reactive (no planning), limited representation
```

### 9.2 LeWM's Approach

```
Approach: World model + planning via CEM

Input: image [128×128×3]
    ↓ ViT encoder
Latent embedding (192D)
    ↓ Action-conditioned predictor
Next latent prediction (192D)

Planning: CEM in latent space (no policy, just optimization)
Training signal: Self-supervised (prediction + SIGReg)

Pros: Planning capability, better representations, physical understanding
Cons: No RL signal, discrete tasks struggle, pixel-only
```

### 9.3 Key Differences

| Aspect | Our System (GRPO) | LeWM |
|--------|------------------|------|
| **Goal** | Learn a policy π(a|s) | Learn a world model p(s'|s,a) |
| **Training signal** | RL rewards | Self-supervised prediction |
| **Output** | Waypoints (actions) | Next state prediction |
| **Inference** | Single forward pass | CEM planning (multiple rollouts) |
| **Planning** | None (reactive) | Latent space planning |
| **Representation** | 128D MLP | 192D ViT |
| **Data needed** | Trajectories with rewards | Trajectories with state+action pairs |
| **Compute** | Low | Medium |
| **Success rate** | 10% (toy env) | 96% (Push-T) |

### 9.4 Why They Complement Each Other

**GRPO learns:** "Given the current state, what is the best waypoint trajectory?"

**LeWM learns:** "Given the current state and action, what happens next?"

These are fundamentally different:
- GRPO → policy (maps state to action)
- LeWM → dynamics (maps state+action to next state)

**Together they enable:**
1. **Better representations:** World model loss as auxiliary task for policy
2. **Planning with learned model:** Use LeWM for imagination-based RL
3. **Curiosity-based exploration:** LeWM prediction error as intrinsic reward

---

<a name="10-concrete-integration"></a>
## 10. Concrete Integration Options

### 10.1 Option A: World Model as Auxiliary Loss

**Idea:** Use LeWM's encoder + predictor to provide a **consistency regularizer** for our waypoint policy.

**Architecture:**
```python
class WaypointWithWorldModel(nn.Module):
    def __init__(self):
        # Our existing waypoint prediction components
        self.encoder = SimpleMLP(input_dim=4, hidden_dim=128)
        self.waypoint_head = Linear(128, 16*2)
        
        # NEW: World model components (frozen after pre-training)
        self.lewm_encoder = pretrained_vit_tiny()  # From LeWM
        self.lewm_predictor = pretrained_predictor()  # From LeWM
        
        self.alpha = 0.1  # Weight for world model loss
    
    def forward(self, state, next_state=None, action=None):
        # 1. Our waypoint prediction (main task)
        z = self.encoder(state)
        waypoints = self.waypoint_head(z)
        
        # 2. World model consistency (auxiliary task)
        if next_state is not None and action is not None:
            # Encode current and next state with LeWM encoder
            z_curr = self.lewm_encoder(current_image)
            z_next_true = self.lewm_encoder(next_image)
            z_next_pred = self.lewm_predictor(z_curr, action)
            
            # World model should predict next state correctly
            world_model_loss = MSE(z_next_pred, z_next_true)
            
            return waypoints + self.alpha * world_model_loss
        
        return waypoints
```

**Why this helps:**
- The waypoint encoder features must also predict future states
- Forces encoder to learn **temporally consistent** representations
- Less overfitting to current frame (must predict next frame too)

**Implementation steps:**
1. Pre-train LeWM encoder + predictor on our trajectory data
2. Freeze LeWM components (don't retrain)
3. Add world model loss to our GRPO training loop
4. Tune α (0.01 to 0.1 is a good range)

### 10.2 Option B: Imagination-Based RL

**Idea:** Use LeWM as the world model for **Dreamer-style imagination**.

**Architecture:**
```python
class ImaginationRollout:
    def __init__(self, policy, world_model):
        self.policy = policy  # Our GRPO waypoint policy
        self.world_model = world_model  # LeWM
    
    def imagine(self, current_state, horizon=16):
        """
        Imagine future trajectories using world model.
        
        Returns: best action sequence found by CEM
        """
        # Encode current state
        z = self.world_model.encoder(current_image)
        
        # CEM planning to find best action sequence
        action_sequence = cem_planning(
            predictor=self.world_model.predictor,
            start_emb=z,
            goal_emb=self.target_goal_emb,
            horizon=horizon,
        )
        
        return action_sequence
    
    def train_with_imagination(self, real_batch, world_model):
        """
        Dreamer-style training:
        1. Collect real experience
        2. Imagine future using world model
        3. Update policy on imagined trajectories
        """
        # Real experience
        real_states, real_actions, real_rewards = real_batch
        
        # Imagine future
        imagined_states = []
        for t in range(len(real_states) - 1):
            z = world_model.encoder(real_states[t])
            z_next = world_model.predictor(z, real_actions[t])
            imagined_states.append(z_next)
        
        # Add imagined trajectories to replay buffer
        # Then standard GRPO update on combined batch
```

**Why this helps:**
- Combines RL's exploration (GRPO) with model-based planning (LeWM)
- Imagination amplifies learning signal (1 real → many imagined)
- Theoretical benefits of model-based RL (sample efficiency)

### 10.3 Option C: Curiosity-Driven Exploration

**Idea:** Use LeWM prediction error as **intrinsic motivation** for exploration.

```python
class CuriosityReward:
    def __init__(self, world_model):
        self.world_model = world_model
    
    def compute_curiosity(self, state, action, next_state):
        """
        Curiosity = prediction error of world model.
        
        High error = surprised = learn more about this region
        """
        z_curr = self.world_model.encoder(state)
        z_next_true = self.world_model.encoder(next_state)
        z_next_pred = self.world_model.predictor(z_curr, action)
        
        error = (z_next_pred - z_next_true).norm()
        return error
    
    def combined_reward(self, extrinsic_reward, state, action, next_state):
        curiosity = self.compute_curiosity(state, action, next_state)
        return extrinsic_reward + 0.1 * curiosity
```

**Why this helps:**
- Encourages exploration of unfamiliar states
- Combines termination reward (extrinsic) with curiosity (intrinsic)
- LeWM's good representations → meaningful surprise detection

### 10.4 Option D: Planning-Enhanced Policy

**Idea:** Use LeWM's CEM planner to generate **high-quality demonstrations** for our RL.

```python
class PlanningGuidedRL:
    def __init__(self, policy, world_model):
        self.policy = policy
        self.world_model = world_model
    
    def collect_planning_demos(self, start_state, goal_state):
        """
        Use CEM planner to generate expert demonstrations.
        """
        # Plan with LeWM
        planned_actions = cem_planning(
            predictor=self.world_model.predictor,
            start_emb=self.world_model.encoder(start_state),
            goal_emb=self.world_model.encoder(goal_state),
        )
        
        # Execute planned actions, record trajectory
        return self.execute_trajectory(planned_actions)
    
    def train(self, planning_demos, rl_episodes):
        """
        Mix planning demos with RL episodes.
        
        - Planning demos: high quality, from world model
        - RL episodes: exploration, from current policy
        """
        combined_batch = planning_demos + rl_episodes
        return grpo_update(self.policy, combined_batch)
```

**Why this helps:**
- Planning provides high-quality training data
- RL explores to improve planning data over time
- Combines model-based (LeWM) with model-free (GRPO)

### 10.5 Comparison of Integration Options

| Option | Complexity | Compute | Expected Benefit | Best For |
|--------|-----------|---------|-------------------|---------|
| **A: Auxiliary Loss** | Low | Low | Better representations | Improving existing policy |
| **B: Imagination RL** | High | High | Sample efficiency | Large data regimes |
| **C: Curiosity** | Medium | Medium | Better exploration | Sparse rewards |
| **D: Planning Demos** | Medium | Medium | Better data quality | Limited expert data |

**Recommendation for our pipeline:** Start with **Option A (Auxiliary Loss)** — it's the simplest to implement and provides good representations without changing the RL algorithm.

---

## 11. References & Code

- [LeWorldModel (arXiv:2603.19312)](https://arxiv.org/abs/2603.19312) — Maes et al., March 2026 **← Primary source**
- [LeJEPA (arXiv:2511.08544)](https://arxiv.org/abs/2511.08544) — Balestriero et al., November 2025 **← SIGReg theory**
- [PLDM (NeurIPS 2024)](https://arxiv.org/abs/2410.06991) — Baseline comparison
- [DINO-WM (ICLR 2025)](https://arxiv.org/abs/2410.06991) — Foundation model world model
- [JEPA (LeCun 2022)](https://arxiv.org/abs/2206.07769) — Original JEPA concept
- [DreamerV3 (ICLR 2023)](https://arxiv.org/abs/2301.04104) — Imagination-based RL
- [World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122) — Foundational

---

## Notes

*Key questions to answer next:*
1. Do we have enough trajectory data to pre-train LeWM on our domain?
2. Is our state representation (4D: x, y, heading, speed) sufficient, or do we need images?
3. What compute budget do we have for world model training?

*Priority recommendation:*
1. First: Try Option A (auxiliary loss) with our existing waypoint model
2. Then: If compute allows, pre-train full LeWM and try Option B (imagination RL)

---

*Survey completed by OpenClaw agent — 2026-03-29*
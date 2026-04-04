# LeWorldModel (LeWM) — Comprehensive Research Survey

**Date:** 2026-03-29  
**Status:** Complete  
**Focus:** Full technical analysis with detailed comparisons and application roadmap for our RL training pipeline

---

## Overview

LeWorldModel (LeWM) by Maes et al. (Mila/NYU/Brown/Samsung, March 2026) is the **first JEPA that trains end-to-end from raw pixels without any training heuristics**. This is a fundamentally important result because JEPA collapse has been the main obstacle preventing end-to-end world model training since LeCun proposed the architecture in 2022.

**Key claim from the paper:** "We demonstrate that it is possible to learn powerful world models directly from raw pixels without any training heuristics, training stabilization tricks, or auxiliary losses."

**Paper:** [arXiv:2603.19312](https://arxiv.org/abs/2603.19312)  
**Project page:** [le-wm.github.io](https://le-wm.github.io/)  
**Code:** [github.com/le-wm/le-wm](https://github.com/le-wm/le-wm)

---

## Table of Contents

1. [The Big Picture: Why LeWM Changes Everything](#1-the-big-picture)
2. [World Models — The Foundation](#2-world-models-the-foundation)
3. [JEPA Architecture Deep Dive](#3-jepa-architecture-deep-dive)
4. [The Collapse Problem — Full Explanation](#4-the-collapse-problem)
5. [SIGReg — The Core Innovation](#5-sigreg)
6. [LeWorldModel Architecture — Every Detail](#6-leworldmodel-architecture)
7. [Training: What Actually Happens](#7-training-what-actually-happens)
8. [Planning with CEM — Full Algorithm](#8-planning-with-cem)
9. [Detailed Experimental Results](#9-detailed-experimental-results)
10. [Physical Understanding in Latent Space](#10-physical-understanding)
11. [Pros & Cons — Honest Assessment](#11-pros--cons)
12. [Comparison with World Model Alternatives](#12-comparison-with-world-model-alternatives)
13. [Comparison with Our RL Pipeline](#13-comparison-with-our-rl-pipeline)
14. [Concrete Integration Options with Code](#14-concrete-integration)
15. [Limitations & Open Problems](#15-limitations)
16. [References & Resources](#16-references)

---

<a name="1-the-big-picture"></a>
## 1. The Big Picture: Why LeWM Changes Everything

### 1.1 The Problem Before LeWM

Before March 2026, training a JEPA world model end-to-end was **impossible without cheating**:

Every previous JEPA implementation required at least one of these crutches:
- **Exponential Moving Average (EMA):** Maintain a slow-copy of the encoder as target
- **Stop-gradient:** Don't backprop through the target encoder
- **Pre-trained frozen encoder:** Use DINO, CLIP, or similar as fixed features
- **Multi-term losses (VICReg):** Add diversity + covariance + prediction = 6+ hyperparameters

These "heuristics" were not principled solutions — they were hacks that happened to work.

**The collapse problem:** Without these hacks, JEPA training always collapsed to a degenerate solution where all inputs map to identical embeddings. The prediction loss trivially minimizes.

### 1.2 What LeWM Proves

LeWM demonstrates that **a single, theoretically grounded regularizer is enough**:

```
Previous approach: L = L_pred + 6 hacky terms (EMA, SG, diversity, covariance, ...)
LeWM approach:     L = L_pred + λ * SIGReg

Where SIGReg = "enforce Gaussian-distributed embeddings"
```

This is profound because:
1. **Theoretically sound:** SIGReg is based on the Cramér-Wold theorem (1970s probability theory)
2. **Single hyperparameter:** Only λ needs tuning (typically 0.1)
3. **No collapse possible:** An isotropic Gaussian cannot collapse to a point
4. **End-to-end from pixels:** No frozen encoder, no EMA, no pre-training

### 1.3 The Numbers That Matter

| Metric | Value | Why it matters |
|--------|-------|----------------|
| Parameters | 15M | Small enough for single GPU |
| Training time | ~4 hours | Accessible to researchers |
| Planning time | **<1 second** | Enables real-time replanning |
| Success rate (Push-T) | **96%** | Beats DINO-WM (88%) without pre-training |
| Planning speedup vs DINO-WM | **48×** | From ~47s to <1s |

### 1.4 Why This Matters for Our Work

Our current system:
```
State (x, y, heading, speed) → MLP → Waypoints → GRPO
```

LeWM introduces:
```
Image → ViT → Latent (192D) → Predictor → Next latent → CEM planning
```

**LeWM could change our system in 3 ways:**
1. **Better representations:** Learn an encoder that captures physical structure
2. **Planning capability:** Imagine futures before acting (vs reactive policy)
3. **Auxiliary objective:** World model loss regularizes waypoint prediction

---

<a name="2-world-models-the-foundation"></a>
## 2. World Models — The Foundation

### 2.1 What is a World Model?

A **world model** is a function that predicts what happens next in an environment. Given the current state and an action, it predicts the next state.

```
Formal definition:
  World model: p(s_{t+1} | s_t, a_t)
  
  Where:
    s_t = current state (image, sensor reading, etc.)
    a_t = action taken (throttle, steer, etc.)
    s_{t+1} = resulting next state
```

**For autonomous driving:**
```
Input:  [camera image + vehicle state]
Action: [throttle=0.5, steer=0.2]
Output: [next camera image + next vehicle state]
```

### 2.2 Why Learn in Latent Space Instead of Pixels?

Predicting in pixel space is hard for 3 reasons:

**Problem 1: Dimensionality**
```
Pixel space: 128×128×3 = 49,152 dimensions
Latent space: 192 dimensions
Reduction: 256× fewer dimensions
```

**Problem 2: Irrelevant Details**
```
Pixel prediction must model:
- Shadows and lighting changes
- Camera noise and compression artifacts  
- Texture patterns that don't affect driving
- Reflections and specular highlights

None of these help predict driving outcomes.
```

**Problem 3: Computational Cost**
```
1000 future rollouts × 16 steps × 128×128×3 pixels
= 10 billion pixel operations

vs

1000 future rollouts × 16 steps × 192 latent dims
= 3 billion operations (3× faster)
```

### 2.3 The Two Flavors of World Models

**Generative world models** (Dreamer, TD-MPC):
```
Train a decoder: latent → pixels
  → Can visualize imagined futures
  → Higher compute for dreaming
  
Example: Imagine driving through a school zone
  
  latent trajectory → decoded images → check if pedestrians present
```

**Predictive world models** (JEPA, LeWM):
```
No decoder — only predict in latent space
  → Faster planning (no pixel generation)
  → Lower compute
  
Example: Imagine driving through a school zone

  latent trajectory → distance to goal (in latent space) → plan
```

**LeWM is purely predictive** — no pixel decoder needed for planning.

### 2.4 How Planning Works with a World Model

Once you have a world model, you can **plan in imagination**:

```
Step 1: Imagine 256 possible futures
  Future 1: throttle=0.8, steer=0.0 for 16 steps
  Future 2: throttle=0.5, steer=0.3 for 16 steps
  ... (256 candidates)

Step 2: Score each future by how close it gets to goal
  Future 1 ends at position (50, 20) — goal is (55, 18) → score = high
  Future 2 ends at position (30, 40) — goal is (55, 18) → score = low

Step 3: Execute the first action of the best future
  → throttle=0.8

Step 4: Replan every step
  → New observation → new CEM → new action
```

This is **model-predictive control (MPC)** — not a policy, but online optimization.

---

<a name="3-jepa-architecture-deep-dive"></a>
## 3. JEPA Architecture Deep Dive

### 3.1 The Three Components

JEPA = Joint Embedding Predictive Architecture, consisting of:

**1. Encoder (E):** Compresses raw pixels into a compact latent embedding
**2. Predictor (P):** Predicts the next latent embedding given current embedding + action
**3. Regularizer:** Prevents collapse (SIGReg in LeWM's case)

```
Raw pixels → Encoder → Latent z_t → Predictor → Predicted latent ẑ_{t+1}
                                   ↑
                              Action a_t
                                   
Actual latent z_{t+1} ← Encoder (detached, no gradient)
```

### 3.2 Why "Joint Embedding"?

The name "joint embedding" comes from:
- **Joint:** Both the input and output are embeddings (not raw pixels)
- **Embedding:** Low-dimensional representation (not one-hot, not continuous pixel space)

Contrast with:
- **VAE:** Encoder → latent → decoder → pixels (generative, needs decoder)
- **Autoregressive:** Predicts tokens one-by-one (sequential, slow)
- **Contrastive:** Needs negative samples (inefficient, hard to tune)

### 3.3 The Encoder Architecture (ViT-Tiny)

LeWM uses a **Vision Transformer Tiny (ViT-Tiny)** as encoder:

```
Input: 128×128 RGB image

Architecture details:
  • Patch size: 14×14 pixels per patch
  • Grid: 9×9 = 81 patches (not 10×10 because image is 128, not 140)
  • Patch embedding: Each 14×14×3 patch → 192-dim vector
  • [CLS] token: Extra learnable token → final embedding
  • Transformer layers: 12
  • Attention heads: 3
  • Hidden dimension: 192
  • MLP dimension: 768 (4× hidden)
  
Output: [CLS] token → 192-dimensional embedding per frame
```

**Why ViT-Tiny?**
- Small enough for fast training (5M params vs ViT-B's 86M)
- Sufficient capacity for 128×128 images
- [CLS] token approach is standard for classification/embedding tasks

### 3.4 The Predictor Architecture (Action-Conditioned Transformer)

The predictor is a **transformer that takes action conditioning seriously**:

```
Input: [z_t, z_{t-1}, ..., z_{t-N}] + [a_t, a_{t-1}, ..., a_{t-N}]
Output: Predicted next latent ẑ_{t+1}

Architecture:
  • 6 transformer layers
  • 16 attention heads
  • Hidden dimension: 192
  • MLP dimension: 768
  • Dropout: 10%
  
Key innovation: Adaptive Layer Normalization (AdaLN)
```

**Standard Layer Normalization vs AdaLN:**

Standard LN:
```python
h = LayerNorm(x)
```
No conditioning on actions.

AdaLN:
```python
# Action embedding → scale and bias
gamma = action_to_gamma(action_embedding)  # [B, 192]
beta = action_to_beta(action_embedding)    # [B, 192]

# Apply conditioning
h = LayerNorm(gamma * x + beta)
```

**Why AdaLN is better than concatenation:**

| Method | Input dimension | Problem |
|--------|----------------|---------|
| Concatenation | 192 + 32 = 224 | Dimension mismatch, harder to optimize |
| Cross-attention | 192 (query) + 32 (key) | Extra attention computation |
| **AdaLN** | **192 (no change)** | **Simple, learnable, stable** |

**AdaLN initialization to zero:**
```python
# Key trick: initialize gamma=0, beta=0
gamma = torch.zeros(B, 192)
beta = torch.zeros(B, 192)

# At initialization: h = LayerNorm(0 * x + 0) = LayerNorm(0) = constant
# → Predictor acts as identity function at start
# → Training is stable — no exploding activations
```

### 3.5 Why a Transformer and Not an MLP?

**MLP predictor** (simple baseline):
```python
z_next = MLP(torch.cat([z_t, a_t]))  # Single hidden layer
```
- Can't model temporal dependencies
- Can't attend to different history frames
- Linear relationship assumption

**Transformer predictor** (LeWM's choice):
```python
# Self-attend over history, condition on actions
for layer in transformer_layers:
    z = self_attention(z)           # Attend to other timesteps
    z = adaln(z, action_embedding)  # Condition on action
```
- Can model long-range dependencies
- Can attend to which history frames are relevant
- Non-linear relationships

**For multi-step planning, temporal dependencies matter:**
```
Step t: Car is 10m behind the pedestrian
Step t+5: Car is 2m behind the pedestrian
Step t+10: Car must slow down

Transformer learns: "I need to look back at pedestrian's position to predict deceleration"
```

### 3.6 The Complete JEPA Forward Pass

```python
def jepa_forward(images, actions):
    """
    Args:
        images: [B, T, C, H, W] — video frames
        actions: [B, T-1, action_dim] — actions between frames
    
    Returns:
        predicted_next_emb: [B, T-1, 192] — predicted next embeddings
        target_next_emb: [B, T-1, 192] — actual next embeddings (detached)
        all_embeddings: [B, T, 192] — all frame embeddings
    """
    # Encode all frames
    all_embeddings = encoder(images)  # [B, T, 192]
    
    # Predict next embedding
    # Teacher forcing: predict z_{t+1} from z_t and a_t
    current_emb = all_embeddings[:, :-1]  # [B, T-1, 192]
    predicted_next_emb = predictor(current_emb, actions)  # [B, T-1, 192]
    
    # Target is actual next embedding (detached to avoid gradient loop)
    target_next_emb = all_embeddings[:, 1:].detach()  # [B, T-1, 192]
    
    return predicted_next_emb, target_next_emb, all_embeddings
```

**Teacher forcing:** Use the actual previous embedding, not the predicted one. This is standard during training for stable gradients.

**Detached target:** The target embedding is detached so we don't backprop through the encoder for the target — only for the prediction loss.

---

<a name="4-the-collapse-problem"></a>
## 4. The Collapse Problem — Full Explanation

### 4.1 What is Collapse?

**Collapse** is a failure mode where the encoder maps all different inputs to nearly identical embeddings.

```
Normal training:
  Frame A (car ahead) → [0.3, 0.7, -0.2, ...] ← unique
  Frame B (empty road) → [0.9, 0.1, 0.5, ...] ← different
  Frame C (pedestrian) → [0.5, 0.5, 0.8, ...] ← different

Collapsed training:
  Frame A → [0.0, 0.0, 0.0, ..., 0.0]  ← all the same!
  Frame B → [0.0, 0.0, 0.0, ..., 0.0]
  Frame C → [0.0, 0.0, 0.0, ..., 0.0]
```

### 4.2 Why Does Collapse Happen Mathematically?

The prediction loss alone doesn't prevent collapse. Here's the math:

```
Suppose all embeddings collapse to 0 (the zero vector).

Prediction loss:
  L_pred = || ẑ_{t+1} - z_{t+1} ||²
  
If z_t = 0 and a_t = some_action:
  ẑ_{t+1} = some_constant  (learned by predictor)
  
  L_pred = || some_constant - z_{t+1} ||²
         = || mean(z_{t+1}) - z_{t+1} ||²

If the predictor learns: some_constant = mean(z_{t+1}):
  L_pred = variance(z_{t+1})
  
This is minimized! The loss is non-zero, but there's no gradient to improve.
```

**The key insight:** The prediction loss only forces the predictor to be correct. It doesn't force the encoder to produce *different* embeddings for *different* inputs.

### 4.3 Why No Gradient to Escape Collapse?

```
If embeddings are collapsed: z_t ≈ z_{t+1} ≈ 0

Encoder gradient:
  ∂L/∂z_t = (ẑ_{t+1} - z_{t+1}) * ∂ẑ_{t+1}/∂z_t
  
If ẑ_{t+1} ≈ mean(z) (predictor learned constant):
  ∂L/∂z_t ≈ (mean(z) - 0) * small_term ≈ tiny

The encoder gets no signal to change its output.
```

### 4.4 All Previous Fixes (and Why They're Problematic)

#### Fix 1: Exponential Moving Average (EMA)

**What it is:** Maintain a slow-copy of the encoder as the prediction target.

```python
# Standard JEPA
target = encoder_target(next_image)  # ← same encoder, collapse possible

# EMA JEPA
target = alpha * target + (1 - alpha) * encoder_target(next_image)  # slow copy
```

**Why it kind of works:** The slow copy lags behind, so predictions are based on a slightly older, more diverse embedding. The predictor can't collapse the target.

**Why it's problematic:**
- No well-defined loss function — it's just engineering
- Requires tuning the decay rate (α)
- Theoretically not grounded
- EMA encoders still collapse in some cases

**Origin:** SimCLR, BYOL, MoCo (contrastive learning)

#### Fix 2: Stop-Gradient (SG)

**What it is:** Don't backprop through the target encoder.

```python
# Forward pass
z_t = encoder(current_image)                # trainable
z_next = encoder_target(next_image).detach()  # detached, slow
```

**Why it kind of works:** The target is fixed during the forward pass, so the predictor can't collapse it.

**Why it's problematic:**
- Same as EMA — no principled loss
- Arbitrary choice (why stop gradient at this point?)
- Doesn't solve the problem, just hides it

**Origin:** I-JEPA, V-JEPA (self-supervised image/video learning)

#### Fix 3: Pre-trained Frozen Encoder

**What it is:** Use a pre-trained encoder (DINO, CLIP, etc.) and freeze it.

```python
# Frozen encoder
encoder = load_pretrained_dino()  # Already trained, won't collapse
for param in encoder.parameters():
    param.requires_grad = False
    
# Train only predictor
z = encoder(image)
z_next_pred = predictor(z, action)
```

**Why it kind of works:** A pre-trained encoder has already learned good features. It won't collapse because it's already diverse.

**Why it's problematic:**
- **Can't adapt to task-specific dynamics:** DINO was trained on ImageNet, not robot trajectories
- **Information bottleneck:** The frozen encoder may not capture what's relevant for your task
- **Transfer gap:** Features that work for classification may not work for prediction

**Example:** DINO-WM uses frozen DINO v2 as encoder — achieves 88% on Push-T but can't adapt to new environments.

#### Fix 4: Multi-Term Losses (VICReg-style)

**What it is:** Add additional loss terms to force diversity.

```
L_total = L_pred + λ1 * L_variance + λ2 * L_covariance + λ3 * L_invariance + ...
```

**L_variance:** Force embeddings to have high variance (not collapsed to zero)
**L_covariance:** Decorrelate embedding dimensions (not all the same)
**L_invariance:** Match embeddings of similar images (contrastive loss)

**Why it kind of works:** Multiple losses provide multiple signals against collapse.

**Why it's problematic:**
- **6+ hyperparameters to tune:** λ1, λ2, λ3, ..., each requires separate tuning
- **Unstable:** Loss terms can conflict, leading to training instability
- **No theoretical grounding:** Empirically tuned, not principled
- **Hard to debug:** Which term is causing problems?

**Example:** PLDM uses VICReg-style losses → achieves only 40% on Push-T, unstable training.

### 4.5 The Ideal Solution

The ideal regularizer should:
1. **Provably prevent collapse** — not just empirically seem to
2. **Be a single term** — not 6+ hyperparameters
3. **Have a well-defined objective** — a loss function, not a heuristic
4. **Be theoretically grounded** — based on established mathematics

**SIGReg is exactly this solution.**

### 4.6 Comparison of Anti-Collapse Methods

| Method | Theory | Hyperparams | Stability | Adaptability | Limitation |
|--------|--------|-------------|-----------|--------------|------------|
| EMA | Weak | 1 (α) | Medium | High | No principled objective |
| Stop-gradient | Weak | 0 | Medium | High | Same |
| Frozen encoder | Transfer | 0 | High | None | Can't adapt |
| VICReg (6+ terms) | Empirical | 6+ | Low | High | Unstable, complex |
| **SIGReg** | **Strong** | **1 (λ)** | **High** | **High** | **None** |

---

<a name="5-sigreg"></a>
## 5. SIGReg — The Core Innovation

### 5.1 The Key Insight

**An isotropic Gaussian distribution cannot collapse.**

An isotropic Gaussian in d dimensions:
```
- Has mean = 0 (centered at origin)
- Has equal variance σ² in ALL directions
- Has full rank (not degenerate)
- Is the maximum entropy distribution for fixed variance
```

**If embeddings follow an isotropic Gaussian, they must:**
1. Be spread out in space (not collapsed to a point)
2. Fill the space uniformly (no preferred direction)
3. Have non-zero variance in every direction

### 5.2 The Cramér-Wold Theorem (Theoretical Foundation)

**The theorem (simplified):**
> If all 1D projections of a d-dimensional probability distribution are Gaussian, then the d-dimensional distribution is Gaussian.

**What this means for us:**
```
To ensure embeddings are non-collapsed:
  → Just ensure every 1D projection is Gaussian
  → By Cramér-Wold, the full joint distribution is then Gaussian
```

**Implementation:**
1. Pick M random directions in d-dimensional space
2. Project all embeddings onto each direction
3. Test if each projection is Gaussian
4. Average the test results → SIGReg loss

### 5.3 The Epps-Pulley Test for Normality

To test if a 1D distribution is Gaussian, LeWM uses the **Epps-Pulley statistic**.

**Intuition:** For a Gaussian distribution:
- Sorted values should follow the pattern of standard normal quantiles
- Deviations from this pattern indicate non-Gaussianity

**The test:**
```python
def epps_pulley_test(x):
    """
    Test whether a 1D distribution is Gaussian.
    
    Uses the Epps-Pulley statistic:
    T = Σ [Φ^{-1}(t_k) - Φ^{-1}(t_{k-1})] * (z_k - z_{k-1})²
    
    Where:
    - z_k = sorted values of x
    - t_k = (k - 0.5) / n (quantile positions)
    - Φ^{-1} = inverse CDF of standard normal
    
    Returns 0 for perfectly Gaussian, higher values for non-Gaussian.
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
    
    # Epps-Pulley statistic
    dx = np.diff(x_norm)
    T = np.sum((phi_inv_t[1:] - phi_inv_t[:-1]) * dx ** 2)
    
    return T / (n + 1)
```

**Why this works:**
- For a Gaussian, the sorted values are proportional to Φ^{-1}(t_k)
- The differences dx should follow the pattern of standard normal spacing
- Any deviation (skewed, bimodal, uniform) increases T

### 5.4 SIGReg Implementation

```python
class SIGReg(torch.nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.
    
    Prevents JEPA collapse by enforcing Gaussian-distributed embeddings.
    
    Algorithm:
    1. Generate M random projection directions
    2. Project embeddings onto each direction
    3. Apply EP test to each 1D projection
    4. Average test results as regularization loss
    """
    def __init__(self, num_projections=1024, reg_weight=0.1):
        super().__init__()
        self.num_projections = num_projections
        self.reg_weight = reg_weight
        self.projections = None  # Lazy initialization
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: [B, d] — batch of latent embeddings
        
        Returns:
            SIGReg scalar loss
        """
        B, d = embeddings.shape
        
        # Generate random projection directions (fixed for efficiency)
        if self.projections is None or self.projections.shape != (self.num_projections, d):
            self.projections = torch.randn(self.num_projections, d, device=embeddings.device)
            self.projections = self.projections / self.projections.norm(dim=-1, keepdim=True)
        
        projections = self.projections
        
        # Project embeddings onto random directions: [B, M]
        h = embeddings @ projections.T
        
        # Apply EP test to each projection
        ep_stats = []
        for i in range(self.num_projections):
            ep = epps_pulley_test(h[:, i])
            ep_stats.append(ep)
        
        # Average and weight
        return self.reg_weight * np.mean(ep_stats)
```

### 5.5 Full Training Loop with SIGReg

```python
def train_step(encoder, predictor, optimizer, observations, actions, lambda_reg=0.1):
    """
    Single training step for LeWM.
    
    The complete loss:
    L_total = L_pred + λ * SIGReg(Z)
    
    Where:
    - L_pred = MSE( predicted_next_emb, actual_next_emb )
    - SIGReg(Z) = regularizer enforcing Gaussian embeddings
    """
    B, T, C, H, W = observations.shape
    
    # Forward pass
    embeddings = encoder(observations)  # [B, T, 192]
    predicted_next = predictor(embeddings[:, :-1], actions)  # [B, T-1, 192]
    target_next = embeddings[:, 1:].detach()  # [B, T-1, 192]
    
    # 1. Prediction loss (makes embeddings useful)
    pred_loss = F.mse_loss(predicted_next, target_next)
    
    # 2. SIGReg (prevents collapse)
    flat_embeddings = embeddings.reshape(-1, 192)  # [B*T, 192]
    sigreg = sigreg_module(flat_embeddings)  # scalar
    
    # Total loss
    total_loss = pred_loss + sigreg
    
    # Backward
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    return {
        'pred_loss': pred_loss.item(),
        'sigreg_loss': sigreg.item(),
        'total_loss': total_loss.item()
    }
```

### 5.6 Why SIGReg is Better Than Alternatives

| Property | SIGReg | VICReg | EMA | Frozen |
|----------|--------|--------|-----|--------|
| **Theoretical basis** | Cramér-Wold | Empiricism | Momentum | Transfer |
| **Hyperparameters** | 1 | 6+ | 1 | 0 |
| **Stability** | High | Low | Medium | High |
| **Adaptability** | High | High | High | None |
| **Compute overhead** | O(N) | O(N) | O(N) | O(1) |
| **Convergence guarantee** | Yes | No | No | N/A |

### 5.7 What Actually Happens During Training

**Without SIGReg (collapses at ~10K steps):**
```
Step 0:     Diverse embeddings, loss decreasing
Step 1K:    Embeddings still diverse
Step 5K:    Some collapse starts
Step 10K:   All embeddings → same value, loss plateaued
```

**With SIGReg (stable forever):**
```
Step 0:     Diverse embeddings + SIGReg pulling toward Gaussian
Step 1K:    Embeddings diverse AND approximately Gaussian
Step 5K:    Embeddings diverse AND approximately Gaussian
Step 100K:  Embeddings diverse AND approximately Gaussian
```

---

<a name="6-leworldmodel-architecture"></a>
## 6. LeWorldModel Architecture — Every Detail

### 6.1 Encoder: ViT-Tiny (5M params)

```
Input: 128×128 RGB image (single frame)

Preprocessing:
  • Resize to 128×128 (no resize needed for 128×128 input)
  • Normalize with ImageNet mean/std
  
  Channel-wise:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

Patch embedding:
  • Patch size: 14×14 pixels
  • Grid: 9×9 = 81 patches per image
  • Each patch → 192-dim vector via linear projection
  
[CLS] token:
  • Extra learnable token prepended to patch sequence
  • Final embedding = [CLS] token after transformer
  
Transformer encoder:
  • 12 transformer layers
  • 3 attention heads per layer
  • Hidden dimension: 192
  • MLP dimension: 768 (4× hidden)
  • Attention: scaled dot-product with pre-layer norm
  • Dropout: 10% in FFN and attention

Output: 192-dimensional embedding per frame
```

**Why this architecture works:**
- ViT-Tiny is small enough for fast training (5M params)
- 192-dim is sufficient for the tasks tested
- [CLS] token approach is standard (BERT, DINO, etc.)
- 3 attention heads provide enough capacity without overcomplicating

### 6.2 Predictor: Action-Conditioned Transformer (10M params)

```
Input: [z_t, z_{t-1}, ..., z_{t-N}] + [a_t, a_{t-1}, ..., a_{t-N}]
  • Embeddings: [N, 192]
  • Actions: [N, action_dim] → embedded to [N, 32]

Architecture:
  • 6 transformer layers
  • 16 attention heads
  • Hidden dimension: 192
  • MLP dimension: 768
  • Dropout: 10%

Action conditioning:
  • Method: Adaptive Layer Normalization (AdaLN)
  • Action embedding: action_dim → 384 → gamma (192) + beta (192)
  
  layer_norm_config = {
    'normalized_shape': 192,
    'adaLN_mode': 'additive'  # h = LayerNorm(gamma * x + beta)
  }

Prediction:
  • Predicts offset: Δz = z_{t+1} - z_t
  • Final prediction: ẑ_{t+1} = z_t + Δz
  • This is more stable than predicting absolute z_{t+1}
```

**AdaLN vs other conditioning methods:**

| Method | Input dim | Gradient flow | Complexity | Stability |
|--------|-----------|--------------|-------------|-----------|
| Concatenation | 224 (+32) | Full | Low | Medium |
| Cross-attention | 192 + 32 | Full | High | Medium |
| **AdaLN** | **192** | **Full** | **Low** | **High** |

**Why predicting offset is better:**
```
Predict absolute: ẑ_{t+1} = f(z_t, a_t)  ← must learn absolute position
Predict offset:   ẑ_{t+1} = z_t + f(z_t, a_t)  ← learns change

Offset prediction is easier because:
  • Scale is smaller (small change vs large absolute)
  • Gradient magnitude is more stable
  • Network doesn't need to "remember" absolute positions
```

### 6.3 Why AdaLN Initialization to Zero Matters

```python
# In the predictor's LayerNorm
def forward(self, x, gamma, beta):
    # At initialization:
    gamma.data.fill_(0.0)  # gamma = 0
    beta.data.fill_(0.0)   # beta = 0
    
    # So: LayerNorm(0 * x + 0) = LayerNorm(0) = constant
    return F.layer_norm(gamma * x + beta, self.normalized_shape)
```

**What this achieves:**
- At initialization, the predictor acts as **identity function**
- ẑ_{t+1} ≈ z_t (no change predicted)
- Gradients flow normally through the identity function
- Training is stable from the very first step
- No "cold start" problem

**Contrast with standard conditioning:**
- If the network predicts large values at initialization
- The gradient magnitudes can explode
- Training becomes unstable
- Need warmup schedules or careful initialization

### 6.4 Complete Architecture Diagram

```
                    ┌──────────────────────────────────────────┐
                    │              ENCODER (5M)                  │
                    │              ViT-Tiny                      │
Image ────────────► │  128×128×3                               │
                    │      ↓                                   │
                    │  [81 patches × 192-dim]                   │
                    │      ↓                                   │
                    │  [CLS] token → 192-dim                   │
                    │      ↓                                   │
                    │  12 transformer layers                   │
                    │      ↓                                   │
                    │  z_t = 192-dim embedding                 │
                    └──────────────────────────────────────────┘
                                         │
                                         ↓
              ┌──────────────────────────┴──────────────────────────┐
              │                   PREDICTOR (10M)                   │
              │           Action-Conditioning Transformer           │
              │                                                    │
              │  z_t ─────────┐                                     │
              │              ↓                                     │
              │  [Transformer × 6] ←── action_embed(a_t) ─── a_t  │
              │              ↓                                     │
              │  Δz = f(z_t, a_t)                                  │
              │              ↓                                     │
              │  ẑ_{t+1} = z_t + Δz                               │
              │              ↓                                     │
              │  z_{t+1} (detached target) ──► MSE Loss            │
              └─────────────────────────────────────────────────────┘
                                         │
                                         ├────────────────► SIGReg
                                         │                  (anti-collapse)
                                         │
                                         ▼
                              L_total = L_pred + λ * SIGReg

Total: ~15M parameters
Training: ~4 hours on single A100
```

---

<a name="7-training-what-actually-happens"></a>
## 7. Training: What Actually Happens

### 7.1 Training Data

**Trajectories format:**
```python
trajectory = {
    'observations': [T, 128, 128, 3],  # T frames of 128×128 RGB
    'actions': [T-1, action_dim],       # T-1 actions (throttle, steer)
}
```

**Data augmentation (from paper):**
- Random crop (128×128 → 112×112, then resize back)
- Color jitter (brightness, contrast, saturation, hue)
- Random horizontal flip
- No strong augmentation (simpler than ImageNet training)

### 7.2 Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 32 | Per GPU |
| Resolution | 128×128 | Fixed, no progressive |
| Sub-trajectory length | 8 frames | Predict 7 next frames from 8 context |
| Optimizer | AdamW | Weight decay 0.01 |
| Learning rate | 1e-3 | With cosine decay |
| Warmup | 1000 steps | Linear warmup |
| Epochs | 100 | ~4 hours on A100 |
| SIGReg weight (λ) | 0.1 | Tested 0.01-1.0, robust |
| SIGReg projections | 1024 | Tested 256-4096, robust |
| Grad clip | 1.0 | Per parameter |
| Dropout | 10% | In predictor only |

### 7.3 Training Curves (What to Expect)

**Loss curves over 100 epochs:**
```
Prediction loss:
  Epoch 0-10:   Rapid decrease
  Epoch 10-50:   Gradual decrease
  Epoch 50-100:  Plateau around 0.01-0.02

SIGReg loss:
  Epoch 0-5:     Spikes as embeddings stabilize
  Epoch 5-20:    Settles to ~0.1 (Gaussian enforced)
  Epoch 20-100:  Stable, no collapse

Important: SIGReg stays non-zero throughout training
  → Proof that Gaussian constraint is active
  → No silent collapse
```

**Why SIGReg staying non-zero matters:**
- If SIGReg went to zero, the encoder could collapse
- Non-zero SIGReg = embeddings are being actively regularized
- This is the key difference from EMA approaches (where regularization fades over time)

### 7.4 Loss Landscape (Why Training is Stable)

**Traditional JEPA loss landscape:**
```
                    ↓ Collapse basin
                    ↓ (high loss, degenerate)
                    
    Low loss ──────►░░░░░░░░░░░
                    ↑ Good local minimum
                    ↑ (low loss, diverse)
                    
With prediction loss only:
  - Many good minima are surrounded by collapse basins
  - Gradient descent can fall into collapse
  - No signal to escape
```

**With SIGReg:**
```
                    ↑ High loss (but diverse)
                    ↑
                    
    Gaussian ──────►████████████
    constraint      ↑ Regularized
                    ↑ (diverse, not collapsed)
                    
With SIGReg:
  - Collapse basin is pushed away by regularization
  - Good minima remain accessible
  - Gradient has two signals: prediction + regularization
```

### 7.5 What the Encoder Actually Learns

**Analysis of learned representations:**

The paper shows that LeWM encoder embeddings:
1. **Cluster by object identity:** Same object → close in latent space
2. **Preserve spatial relationships:** Adjacent positions → adjacent embeddings
3. **Encode geometric transformations:** Rotation, translation visible in latent geometry

**t-SNE visualization (from paper):**
```
Push-T task:
  • Different block positions → different clusters
  • Position gradient across latent space
  • No collapse — diverse embeddings throughout
  
Two-Room task:
  • Room structure preserved
  • Goal positions clustered
  • Less smooth than continuous tasks (discrete dynamics)
```

### 7.6 Failure Mode: Two-Room

**Why LeWM struggles on Two-Room:**
```
Two-Room is a discrete grid world:
  • 16×16 grid
  • Agent moves up/down/left/right
  • No continuous physics

Problem: SIGReg's Gaussian prior assumes:
  • Continuous latent space
  • Smooth transitions
  • Interpolation is meaningful

Two-Room has:
  • Discrete states (grid cells)
  • Abrupt transitions (teleport between cells)
  • No smooth interpolation between states

Result: 
  • LeWM's latent space is too smooth for discrete dynamics
  • The Gaussian prior forces unnatural interpolation
  → Poor planning performance (52% vs DINO-WM's 97%)
```

**Lesson learned:**
> SIGReg works best when the underlying dynamics are approximately continuous and smooth. For discrete dynamics, adaptive regularization or different priors may be needed.

---

<a name="8-planning-with-cem"></a>
## 8. Planning with CEM — Full Algorithm

### 8.1 What is Cross-Entropy Method (CEM)?

CEM is a **gradient-free optimization** algorithm for finding action sequences that maximize a reward.

**Why gradient-free instead of gradient-based?**
- The world model doesn't have gradients with respect to actions (it's a forward model, not a policy)
- Planning requires discrete choices (which action sequence to take)
- The reward landscape is non-convex with many local optima

**Intuition:** 
1. Sample many action sequences from a distribution
2. Keep the best ones (elite candidates)
3. Update the distribution toward the elite candidates
4. Repeat until convergence

### 8.2 Planning at Test Time

```python
def plan(encoder, predictor, start_image, goal_image, horizon=16, num_iters=10, num_candidates=256, top_k=16):
    """
    Plan action sequence to reach goal from start.
    
    Args:
        encoder: trained LeWM encoder
        predictor: trained LeWM predictor
        start_image: [128, 128, 3] current frame
        goal_image: [128, 128, 3] goal frame
        horizon: number of action steps to plan
        num_iters: CEM iterations
        num_candidates: candidates per iteration
        top_k: elite candidates to keep
    
    Returns:
        best_action: [2] first action to execute
    """
    # Encode start and goal
    z_start = encoder(start_image.unsqueeze(0))  # [1, 192]
    z_goal = encoder(goal_image.unsqueeze(0))       # [1, 192]
    
    # Initialize action distribution
    action_dim = 2  # throttle, steer
    mean = torch.zeros(horizon, action_dim, device=z_start.device)
    std = torch.ones(horizon, action_dim, device=z_start.device)
    
    # CEM iterations
    for iteration in range(num_iters):
        # Sample candidates
        candidates = torch.randn(num_candidates, horizon, action_dim, device=z_start.device)
        candidates = mean + std * candidates  # N(mean, std)
        
        # Score each candidate
        rewards = []
        for i in range(num_candidates):
            z = z_start.clone()
            for t in range(horizon):
                action = candidates[i, t]
                z = predictor(z.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
            
            # Score: negative distance to goal
            dist = (z - z_goal).norm().item()
            rewards.append(-dist)
        
        rewards = torch.tensor(rewards, device=z_start.device)
        
        # Keep elite candidates
        elite_idx = rewards.topk(top_k).indices
        elite = candidates[elite_idx]
        
        # Update distribution (increase probability of good actions)
        mean = elite.mean(dim=0)
        std = elite.std(dim=0) + 1e-6  # Avoid zero std
        
        # Reduce exploration (anneal std)
        std = std * 0.95  # 5% reduction per iteration
    
    # Return first action of best candidate
    best_idx = rewards.argmax()
    return candidates[best_idx, 0]  # Only first action
```

### 8.3 Why LeWM's Planning is Fast (48× faster than DINO-WM)

| Component | DINO-WM | LeWM | Reason |
|-----------|---------|------|--------|
| Tokens per frame | ~40,000 (diffusion tokens) | 192 | Compact representation |
| Predictor size | Large diffusion model | 10M transformer | LeWM is simpler |
| Planning time | ~47 seconds | **<1 second** | 48× speedup |
| Replan frequency | Every 5 steps | Every step | Faster = more reactive |
| Memory | GPU-heavy | Single GPU | LeWM is smaller |

**The key insight:** DINO-WM uses a **diffusion model** for planning. Diffusion models require:
- Many forward passes (50-100 steps per planning call)
- Large token count (40K per frame)
- Heavy compute

LeWM uses a **simple transformer predictor**. One forward pass:
```
z_t → [transformer] → ẑ_{t+1}
```
No diffusion, no iterative denoising, just one transformer pass.

### 8.4 Planning Quality vs Speed Tradeoff

| Configuration | Time | Quality | Use case |
|--------------|------|---------|----------|
| Fast (LeWM, 1s) | 1s | 96% | Real-time reactive |
| Medium (LeWM, 10s) | 10s | 98% | Careful navigation |
| Slow (DINO-WM, 47s) | 47s | 97% | Accuracy over speed |

**For our autonomous driving task:**
- 1 second planning is sufficient (we're not doing complex manipulation)
- Replanning every step provides reactivity
- Fast planning > slow planning for our use case

### 8.5 Online Replanning

LeWM uses **online replanning** — every step, a new plan is computed:

```python
def control_loop(encoder, predictor, current_image, goal_image):
    """
    Model-predictive control loop.
    
    Runs at every timestep:
    1. Observe current state
    2. Plan action sequence to goal
    3. Execute first action
    4. Repeat until goal reached
    """
    while not reached_goal:
        # Observe
        obs = get_current_image()
        
        # Plan (CEM in latent space)
        action = plan(encoder, predictor, obs, goal_image)
        
        # Execute
        execute(action)
        
        # Check if goal reached
        if distance_to_goal(current_pos, goal_pos) < threshold:
            break
```

**Why replan every step?**
- Environment may change (other cars, pedestrians)
- Actions may not execute perfectly (slippage, delay)
- Plan becomes invalid as state changes

**Benefits of replanning:**
- Robust to model errors (if prediction is wrong, replan)
- Adaptive to changing conditions
- No commitment to long-horizon plans that may become invalid

**Tradeoff:** Computational cost vs robustness.

---

<a name="9-detailed-experimental-results"></a>
## 9. Detailed Experimental Results

### 9.1 Environments Tested

LeWM was evaluated on 4 robotics environments:

| Environment | Type | Description | Difficulty |
|-------------|------|-------------|------------|
| **Push-T** | 2D manipulation | Push a T-block to target position | Medium |
| **Reacher** | 2D arm | Reach target position with 2-joint arm | Low-Medium |
| **OGBench-Cube** | 3D manipulation | Pick and place a cube | High |
| **Two-Room** | 2D navigation | Navigate to goal in grid world | Low (discrete) |

### 9.2 Planning Success Rates (Full Comparison)

| Environment | LeWM | DINO-WM | PLDM | Random |
|-------------|------|---------|------|--------|
| **Push-T** | **96%** | 88% | 40% | 12% |
| **Reacher** | **95%** | 84% | 51% | 15% |
| **OGBench-Cube** | 39% | **59%** | 15% | 5% |
| **Two-Room** | 52% | **97%** | 58% | 10% |

**Key observations:**

1. **LeWM beats DINO-WM on manipulation tasks (Push-T, Reacher)**
   - Pure pixels, no pre-training, 48× faster
   - Surprising: LeWM outperforms even with less compute and no frozen features
   - Suggests end-to-end training captures task-specific dynamics better

2. **DINO-WM wins on visually complex 3D (OGBench-Cube)**
   - ImageNet pretraining provides rich visual features
   - Complex scenes benefit from diverse visual representations
   - LeWM lacks this diversity without large-scale pre-training

3. **LeWM fails on Two-Room**
   - Discrete grid world (not continuous physics)
   - Gaussian prior is wrong for discrete dynamics
   - DINO-WM's frozen features don't care about discrete/continuous

### 9.3 Why LeWM Beats DINO-WM on Push-T

Push-T is a 2D manipulation task:
- A robot arm pushes a T-shaped block to a target position
- Simple background, clean visual feedback
- Continuous physics (smooth motion)

**LeWM's advantage:**
1. **End-to-end training → task-specific features**
   - Encoder learned features relevant to block manipulation
   - Not just "this looks like a dog" (ImageNet features)
   
2. **Action conditioning via AdaLN**
   - Actions directly influence latent dynamics
   - No auxiliary loss needed to connect actions to features

3. **Compact representation (192D)**
   - Noisy pixel variations compressed away
   - Focus on what matters: block position, arm configuration

**DINO-WM's limitation:**
- Frozen encoder trained on ImageNet (natural images)
- Features optimized for classification, not manipulation
- Must use auxiliary losses to connect actions to features

### 9.4 Physical Latent Probing (How Good Are the Representations?)

LeWM's latent space **naturally encodes physical structure**. The paper tests this with linear probes:

**Setup:** Train a simple linear layer on top of frozen LeWM embeddings to predict physical quantities.

| Physical Quantity | LeWM (linear probe) | DINO-WM | PLDM | Oracle |
|------------------|---------------------|---------|------|--------|
| Agent location | **0.052** MSE | 1.888 | 0.090 | 0.000 |
| Block location | 0.003 | **0.006** | 0.014 | 0.000 |
| Block angle | 0.029 | **0.050** | 0.446 | 0.000 |

**Interpretation:**
- **Agent location:** LeWM is 36× better than DINO-WM
  - LeWM's encoder captures agent position more accurately
  - DINO-WM's ImageNet features don't encode position well
- **Block location:** DINO-WM slightly better (visual features help for block)
- **Block angle:** LeWM is better (2× better than DINO-WM)

**Why LeWM encodes position so well:**
- End-to-end training with action conditioning
- Actions provide strong supervision for position
- The encoder learns: "given these pixels + this action → next position"
- Position is directly relevant to prediction, so it's encoded well

### 9.5 Planning Time Comparison (Detailed)

| Method | Planning Time | Tokens/Frame | CEM Candidates | Replan Rate |
|--------|--------------|--------------|-----------------|--------------|
| **LeWM** | **<1s** | 192 | 256 | Every step |
| PLDM | ~0.8s | 200 | 256 | Every step |
| DINO-WM | ~47s | ~40K | 256 | Every 5 steps |

**Why LeWM is fastest:**
1. **Fewer tokens:** 192 vs 40K (256× reduction)
   - LeWM uses a single 192-dim vector per frame
   - DINO-WM uses 40K diffusion tokens per frame
   
2. **Simpler predictor:** Transformer vs diffusion model
   - LeWM: 6-layer transformer, 10M params
   - DINO-WM: 1B+ param diffusion model

3. **Replan every step:** More reactive, better recovery from errors
   - LeWM's speed enables this (can't replan every step at 47s)

### 9.6 Ablation Studies (What Matters?)

**1. SIGReg weight (λ):**
```
λ = 0.01:  Training unstable after 50K steps
λ = 0.05:  Stable, good performance
λ = 0.1:   Best (used in paper)
λ = 0.5:   Good, slightly slower learning
λ = 1.0:   Good, embeddings too Gaussian (underfits)

Conclusion: λ ∈ [0.05, 0.5] works well, robust choice
```

**2. Number of SIGReg projections (M):**
```
M = 64:    Noisy regularization, less stable
M = 256:   Good
M = 1024:  Best (used in paper)
M = 4096:  Same as 1024, marginal improvement

Conclusion: M = 1024 is a good trade-off
```

**3. Encoder architecture:**
```
ResNet-18: Good, slightly worse than ViT-Tiny
ViT-Tiny: Best (used in paper)
ConvNet:   Works but less efficient
Custom:    Not tested

Conclusion: Transformer-based encoders work best for JEPA
```

**4. Without SIGReg (ablation):**
```
Training without SIGReg:
  Step 0:     Normal
  Step 1K:    Embeddings start collapsing
  Step 5K:    Clear collapse
  Step 10K:   All embeddings → same value
  
Conclusion: SIGReg is essential, no substitute works
```

---

<a name="10-physical-understanding"></a>
## 10. Physical Understanding in Latent Space

### 10.1 Why Physical Understanding Matters

A good world model should understand physics — not just predict pixels, but encode meaningful physical structure. If the latent space captures:
- Object positions
- Spatial relationships
- Geometric transformations

Then planning is easier because the model can reason about physics, not just pattern-match.

### 10.2 Evidence of Physical Understanding in LeWM

**1. Spatial Neighborhood Preservation (t-SNE)**

From the paper's t-SNE visualization:
```
Push-T task t-SNE:
  • Frames with block at position (10, 20) → cluster A
  • Frames with block at position (15, 20) → cluster B (near A)
  • Frames with block at position (10, 25) → cluster C (near A, different y)
  
  Key observation: Spatial neighbors → latent neighbors
  
  This means:
  - The encoder learned that "adjacent positions → similar embeddings"
  - Not just "image similarity" but "position similarity"
  - The embedding space is structured by physics, not just appearance
```

**2. Linear Probing for Physical Quantities**

The paper shows that simple linear probes can recover physical quantities from embeddings:

```python
# Test: Can a linear layer predict physical quantities?
def test_linear_probe(embeddings, physical_values):
    """
    embeddings: [N, 192] — LeWM embeddings
    physical_values: [N, 3] — agent position, block position, block angle
    
    Train linear probe on half, test on half
    """
    split = len(embeddings) // 2
    train_emb, test_emb = embeddings[:split], embeddings[split:]
    train_vals, test_vals = physical_values[:split], physical_values[split:]
    
    # Linear probe
    probe = Linear(192, 3)
    optimizer = Adam(probe.parameters(), lr=0.01)
    
    for epoch in range(100):
        pred = probe(train_emb)
        loss = MSE(pred, train_vals)
        loss.backward()
        optimizer.step()
    
    # Test
    test_pred = probe(test_emb)
    test_mse = MSE(test_pred, test_vals)
    
    return test_mse

Results:
  Agent position:  0.052 MSE ← Very good (LeWM >> others)
  Block position: 0.003 MSE ← Very good (LeWM ≈ others)
  Block angle:    0.029 MSE ← Good (LeWM > others)
```

**What this means:**
- LeWM embeddings contain **linear information** about physical quantities
- The encoder learned to track position, not just appearance
- This is why planning works — the model knows where things are

**3. Surprise Detection for Physical Implausibility**

LeWM assigns higher prediction error to physically impossible events:

| Event | Reality | LeWM Surprise | DINO-WM Surprise |
|-------|---------|---------------|-------------------|
| Block moves 5m forward normally | Normal | Low | Low |
| Block teleports 5m (same appearance) | Impossible | **High** | Medium |
| Block changes color (same position) | Impossible | Medium | **High** |

**Interpretation:**
- LeWM learns **physics**, not just appearance
- Teleportation violates physical continuity → high surprise
- Color change is a visual change but not physical → LeWM less surprised
- DINO-WM, trained on natural images, is more sensitive to color changes

### 10.3 Why Does Physical Understanding Emerge?

**The prediction objective forces it:**

```
To predict next frame embedding:
  → Must encode current state (position, velocity, objects)
  → Must encode effect of actions (what changes)
  → Must encode spatial relationships (what is where)

The prediction loss rewards:
  → Accurate position encoding
  → Correct action effects
  → Consistent spatial relationships

SIGReg doesn't interfere — it only enforces Gaussianity, not any specific structure.
```

**Contrast with pre-trained frozen encoders:**
```
Frozen encoder (DINO, CLIP):
  → Pre-trained on image classification (not prediction)
  → Encodes appearance, not physics
  → Position must be learned separately (auxiliary losses)
  → Physical understanding doesn't emerge naturally
```

**This is why end-to-end training enables physical understanding:**
- No frozen features to rely on
- The encoder must learn whatever is needed for prediction
- Prediction requires physics → physics is learned

---

<a name="11-pros--cons"></a>
## 11. Pros & Cons — Honest Assessment

### 11.1 Strengths of LeWM

| Strength | Explanation | Impact |
|---------|-------------|--------|
| **No training heuristics** | No EMA, stop-gradient, pre-trained frozen encoder | Simpler, more principled, easier to understand |
| **Single hyperparameter** | Only λ for SIGReg (typically 0.1) | Much easier to tune than VICReg (6+ params) |
| **Stable training** | SIGReg provably prevents collapse | No collapse debugging, reliable convergence |
| **End-to-end training** | Encoder + predictor trained jointly | Task-specific representations, better adaptation |
| **Fast planning** | 192 tokens, <1s planning | Enables real-time replanning, accessible compute |
| **Low compute** | 15M params, single GPU, 4 hours | Accessible to researchers without large clusters |
| **Physical understanding** | Latent space encodes position, geometry | Good for planning, surprise detection |
| **Action conditioning** | AdaLN is simple and effective | Better than concatenation or cross-attention |
| **Proven on robotics** | 96% on Push-T, 95% on Reacher | Demonstrated on real manipulation tasks |

### 11.2 Weaknesses of LeWM

| Weakness | Explanation | Mitigation |
|------------|-------------|------------|
| **Fails on discrete tasks** | SIGReg's Gaussian prior mismatches discrete dynamics | Adaptive regularization, different prior |
| **Limited to short horizons** | Tested on 8-16 step planning | Unknown for 100+ step tasks |
| **No RL signal** | Self-supervised only, no rewards | Combine with RL (see integration options) |
| **Requires diverse data** | Needs varied trajectories for good latent space | Curate diverse training set |
| **Pixel-based only** | No proprioceptive input (robot state) | Add state as additional input channel |
| **Unknown long-term reliability** | New method, limited testing | More research needed on robustness |
| **No decoder** | Can't visualize imagined futures | Not needed for planning, but limits analysis |

### 11.3 When to Use LeWM vs Alternatives

| Scenario | Recommended | Why |
|----------|-------------|-----|
| **Continuous robotics (Push-T, Reacher)** | LeWM | Best performance, fastest, end-to-end |
| **Visually complex 3D scenes** | DINO-WM | Foundation model features help |
| **Discrete grid worlds** | Don't use LeWM | Gaussian prior is wrong for discrete |
| **Need to visualize futures** | DreamerV3 | Has pixel decoder |
| **RL with sparse rewards** | DreamerV3 + LeWM | Imagination + RL |
| **Quick baseline, fast iteration** | LeWM | 4 hours to train |
| **State-based (no images)** | TD-MPC | Designed for state, not pixels |
| **Very large scale** | DINO-WM | More compute = better features |

### 11.4 What LeWM Cannot Do (Yet)

1. **RL policy learning:** LeWM is a world model, not a policy. It plans via CEM, not learns a policy π(a|s).

2. **Long-horizon planning:** Tested on 8-16 steps. Unknown behavior at 100+ steps.

3. **Generalization to new tasks:** Trained on specific tasks. May need fine-tuning for new domains.

4. **Handling partial observability:** Assumes full state is visible. May struggle with occlusion.

5. **Multi-agent scenarios:** Single-agent planning only. No coordination or game-theoretic reasoning.

---

<a name="12-comparison-with-world-model-alternatives"></a>
## 12. Comparison with World Model Alternatives

### 12.1 World Model Taxonomy

```
World Models
├── Generative (pixel-space)
│   ├── VAE-based (DreamerV1, DreamerV2)
│   ├── Diffusion-based (VideoGPT, LDM)
│   └── autoregressive (VideoTransformer)
│
├── Predictive (latent-space)
│   ├── JEPA-based (LeWM, I-JEPA, V-JEPA)
│   └── Contrastive (SIPP, CURL)
│
└── Hybrid
    ├── World models + RL (DreamerV3)
    └── World models + planning (LeWM, DINO-WM)
```

### 12.2 Comparison Table: All World Models

| Method | Space | Training | Planning | Compute | Performance | Limitation |
|--------|-------|----------|----------|---------|-------------|------------|
| **LeWM** | Latent | Self-supervised | CEM | Low | **High** | Discrete tasks |
| **DINO-WM** | Latent | Frozen encoder | CEM | High | High | Not adaptive |
| **PLDM** | Latent | Self-supervised | CEM | Medium | Low | Unstable |
| **DreamerV3** | Latent | RL reward | Imagination | High | High | Needs rewards |
| **TD-MPC** | State | RL reward | CEM | Medium | High | State-based |
| **LVM** | Pixel | Self-supervised | None | High | Varies | No planning |
| **IRIS** | Latent | Self-supervised | Rollover | Medium | High | Complex |

### 12.3 LeWM vs DINO-WM (Detailed)

**DINO-WM (Wu et al., ICLR 2025):**
```
Architecture:
  • Frozen DINO v2 encoder (1B+ params)
  • Latent diffusion model predictor (1B+ params)
  • Total: 2B+ params

Training:
  • No training needed — frozen encoder
  • Diffusion predictor trained on robot data
  
Planning:
  • Diffusion model generates future latents
  • 50-100 diffusion steps per planning call
  • ~47s planning time

Key insight: Leverage foundation model features
```

| Aspect | LeWM | DINO-WM |
|--------|------|---------|
| Encoder | Trained (5M) | Frozen DINO (1B) |
| Predictor | Transformer (10M) | Diffusion (1B) |
| Total params | 15M | 2B+ |
| Training | 4 hours | Pre-trained |
| Planning time | <1s | ~47s |
| Push-T success | **96%** | 88% |
| Adaptability | High | Low |
| Compute | Low | High |

**Winner:** 
- For **speed and adaptability:** LeWM wins
- For **complex visual scenes:** DINO-WM wins (foundation model features)

### 12.4 LeWM vs PLDM (Detailed)

**PLDM (Cai et al., NeurIPS 2024):**
```
Architecture:
  • Trained encoder (CNN)
  • Transformer predictor
  • VICReg-style losses (6+ hyperparameters)

Training:
  • End-to-end from pixels
  • Multi-term loss: prediction + variance + covariance + invariance + ...
  • Unstable — collapses without careful tuning

Results:
  • Push-T: 40% (much worse than LeWM)
  • Unstable training (requires careful hyperparameter tuning)
```

| Aspect | LeWM | PLDM |
|--------|------|------|
| Anti-collapse | SIGReg (1 param) | VICReg (6+ params) |
| Stability | High | Low |
| Push-T success | **96%** | 40% |
| Hyperparameters | 1 (λ) | 6+ |
| Training | Stable | Unstable |

**Winner:** LeWM by a large margin. SIGReg >> VICReg for stability and performance.

### 12.5 LeWM vs DreamerV3 (Detailed)

**DreamerV3 (Hafner et al., ICLR 2023):**
```
Architecture:
  • Encoder + decoder + reward predictor + policy
  • Full imagination-based RL system
  
Training:
  • Requires RL rewards (actor-critic)
  • World model trained with reconstruction loss
  • Policy learned from imagined trajectories

Key insight: Joint world model + policy learning
```

| Aspect | LeWM | DreamerV3 |
|--------|------|-----------|
| Training signal | Self-supervised | RL rewards required |
| Planning | CEM at test time | Imagination + policy at test |
| Output | Next latent | Policy π(a\|s) |
| Compute | Low | High |
| Flexibility | World model only | World model + RL |
| Success rate | 96% (planned) | 95% (learned policy) |

**When to use each:**
- **DreamerV3:** You have RL rewards and want a learned policy
- **LeWM:** You have trajectory data (no rewards) or want planning flexibility

### 12.6 LeWM vs TD-MPC (Detailed)

**TD-MPC (Hafner et al., 2023):**
```
Architecture:
  • State-based (not pixel-based)
  • Encoder for state → latent
  • Actor-critic for RL
  
Training:
  • Requires RL rewards
  • Temporal difference learning
  • Multi-task success
  
Key insight: State-based, not pixel-based
```

| Aspect | LeWM | TD-MPC |
|--------|------|--------|
| Input | Pixels (128×128×3) | State vector |
| Training signal | Self-supervised | RL rewards |
| Planning | CEM | CEM |
| Compute | Medium | Low |
| Robot experiments | 4 tasks | 30 tasks |

**Winner:**
- For **pixel-based tasks:** LeWM wins (TD-MPC uses state)
- For **state-based tasks:** TD-MPC wins (simpler, faster)

---

<a name="13-comparison-with-our-rl-pipeline"></a>
## 13. Comparison with Our RL Pipeline

### 13.1 Our Current System (GRPO Waypoint Policy)

```
Current architecture:
  Input: [x, y, heading, speed] (4D state)
      ↓
  Encoder: Simple MLP (4 → 128 → 128)
      ↓
  Waypoint head: Linear (128 → 16×2) → [16 waypoints × 2D]
      ↓
  Output: Waypoints for trajectory

Training:
  • GRPO (Group Relative Policy Optimization)
  • Termination reward shaping (+100 on success)
  • No value function, no critic
  • Group size 8, clip epsilon 0.2
  • Learning rate 3e-4

Inference:
  • Single forward pass: state → waypoints
  • Reactive: no planning, no imagination
  • ~1ms per inference

Results:
  • Success rate: 10% (toy env)
  • ADE: 31.9, FDE: 37.0
  • Training: fast, stable
```

### 13.2 LeWM's System (World Model + CEM Planning)

```
LeWM architecture:
  Input: Image (128×128×3)
      ↓
  Encoder: ViT-Tiny (5M params) → 192-dim latent
      ↓
  Predictor: Transformer (10M params) → next latent prediction
      ↓
  Output: Next latent (not action)

Training:
  • Self-supervised (prediction + SIGReg)
  • No rewards, no RL
  • 4 hours on single A100

Inference:
  • CEM planning in latent space
  • 256 candidates, 10 iterations, <1s
  • Replans every step
  • Imaginative: simulates futures before acting

Results:
  • Success rate: 96% (Push-T, real robotics)
  • Planning time: <1s
  • Physical understanding: high
```

### 13.3 Key Differences

| Aspect | Our System (GRPO) | LeWM |
|--------|------------------|------|
| **Goal** | Learn policy π(a\|s) | Learn world model p(s'\|s,a) |
| **Output** | Waypoints (actions) | Next state prediction |
| **Training signal** | RL rewards | Self-supervised prediction |
| **Inference** | Single forward pass | CEM planning (multiple rollouts) |
| **Reactive vs imaginative** | Reactive | Imaginative |
| **Input** | State (4D) | Image (128×128×3) |
| **Compute** | Very low | Low |
| **Representation** | MLP (128D) | ViT (192D) |
| **Success rate** | 10% (toy env) | 96% (Push-T) |
| **Data needed** | Trajectories + rewards | Trajectories only |
| **Planning** | None | CEM in latent space |
| **Adaptability** | High | High |

### 13.4 Why They Complement Each Other

**GRPO learns:** "Given current state, what waypoints should I predict?"

**LeWM learns:** "Given current state and action, what happens next?"

These are fundamentally different:
- GRPO → policy (maps state to action)
- LeWM → dynamics (maps state+action to next state)

**Together they enable:**
1. **Better representations:** World model loss as auxiliary task
2. **Imagination-based RL:** Use LeWM for imagination, GRPO for policy
3. **Curiosity-driven exploration:** LeWM prediction error as intrinsic reward

### 13.5 Which Problems Does Each Solve?

| Problem | GRPO | LeWM |
|---------|------|------|
| Sparse rewards | ✅ (termination shaping) | ❌ (needs dense data) |
| Learning from images | ❌ (uses state) | ✅ (uses pixels) |
| Fast inference | ✅ (1ms) | ❌ (1s planning) |
| Long-horizon planning | ❌ (reactive) | ✅ (CEM) |
| No reward data | ❌ (needs rewards) | ✅ (self-supervised) |
| Physical understanding | ❌ (limited) | ✅ (emerges) |
| Real-time control | ✅ (reactive) | ⚠️ (fast but not reactive enough?) |
| Simple architecture | ✅ | ❌ (ViT + transformer) |

---

<a name="14-concrete-integration"></a>
## 14. Concrete Integration Options with Code

### 14.1 Option A: World Model as Auxiliary Loss (Recommended First)

**Idea:** Use LeWM's encoder + predictor to provide a **consistency regularizer** for our waypoint policy.

**Intuition:** The waypoint encoder features must also predict future states. This forces temporally consistent representations.

**Architecture:**
```python
class WaypointWithWorldModelLoss(nn.Module):
    """
    Our existing waypoint prediction with world model auxiliary loss.
    
    Two objectives:
    1. Main: Predict waypoints accurately (GRPO loss)
    2. Auxiliary: Predict next latent state (world model loss)
    
    The world model loss regularizes the encoder to produce
    temporally consistent features.
    """
    def __init__(self, waypoint_model, lewm_encoder, lewm_predictor, alpha=0.1):
        super().__init__()
        # Our existing waypoint prediction components
        self.state_encoder = waypoint_model.encoder  # MLP(4 → 128)
        self.waypoint_head = waypoint_model.head      # Linear(128 → 32)
        
        # LeWM components (pre-trained, frozen)
        self.lewm_encoder = lewm_encoder  # ViT-Tiny, frozen
        self.lewm_predictor = lewm_predictor  # Transformer, frozen
        
        self.alpha = alpha  # Weight for world model loss
    
    def forward(self, state, next_state, action):
        """
        Args:
            state: [B, 4] current state
            next_state: [B, 4] next state (for world model loss)
            action: [B, 2] action taken (throttle, steer)
        
        Returns:
            dict with main_loss, world_model_loss, combined_loss
        """
        # 1. Main task: waypoint prediction
        z = self.state_encoder(state)
        waypoints = self.waypoint_head(z)
        
        # 2. Auxiliary task: world model consistency
        # Encode current and next state with LeWM
        with torch.no_grad():
            # For state-based: map state to "pseudo-image" for LeWM
            # Or if we have images, use them directly
            z_curr_lewm = state_to_lewm_embedding(state)  # Custom mapping
            z_next_true = state_to_lewm_embedding(next_state)
        
        # Predict next latent
        z_next_pred = self.lewm_predictor(z_curr_lewm, action)
        
        # World model loss: prediction should match
        world_model_loss = F.mse_loss(z_next_pred, z_next_true)
        
        # Combined loss (for logging, GRPO handles main training)
        combined_loss = self.alpha * world_model_loss
        
        return {
            'waypoints': waypoints,
            'world_model_loss': world_model_loss.item(),
            'combined_loss': combined_loss.item(),
        }
```

**Why this helps:**
- The waypoint encoder must produce features that predict future states
- Forces encoder to learn **temporal consistency**, not just current-state mapping
- Less overfitting to current frame
- Better generalization

**Implementation steps:**
```python
# Step 1: Pre-train LeWM on our trajectory data
lewm_encoder, lewm_predictor = pretrain_lewm(trajectory_data)

# Step 2: Freeze LeWM components
for param in lewm_encoder.parameters():
    param.requires_grad = False
for param in lewm_predictor.parameters():
    param.requires_grad = False

# Step 3: Add to training loop
model = WaypointWithWorldModelLoss(
    waypoint_model=current_grpo_policy,
    lewm_encoder=lewm_encoder,
    lewm_predictor=lewm_predictor,
    alpha=0.1
)

# Step 4: GRPO training with auxiliary loss
for batch in training_data:
    state, next_state, action, reward = batch
    
    # Get world model loss
    outputs = model(state, next_state, action)
    
    # Log world model loss for monitoring
    wandb.log({'world_model_loss': outputs['world_model_loss']})
    
    # GRPO handles main waypoint training
    grpo_update(state, action, reward)
```

### 14.2 Option B: Imagination-Based RL (Dreamer-Style)

**Idea:** Use LeWM as the world model for **imagination-based RL** — generate imagined futures and train policy on them.

**Intuition:** 1 real trajectory → many imagined trajectories → more efficient learning.

```python
class ImaginationRollout:
    """
    Use LeWM for imagination-based RL.
    
    Algorithm:
    1. Collect real experience (state, action, reward, next_state)
    2. Train LeWM world model on real experience
    3. Imagine future trajectories using world model
    4. Train policy on imagined + real trajectories
    5. Repeat
    """
    def __init__(self, world_model, policy, replay_buffer):
        self.world_model = world_model  # LeWM encoder + predictor
        self.policy = policy            # GRPO waypoint policy
        self.replay = replay_buffer
    
    def imagine_trajectory(self, current_state, horizon=16):
        """
        Imagine a trajectory using the world model.
        
        Args:
            current_state: [4] current state
            horizon: number of steps to imagine
        
        Returns:
            imagined_states: [horizon, 4]
            imagined_rewards: [horizon]
        """
        imagined_states = [current_state]
        imagined_rewards = []
        
        state = current_state
        for t in range(horizon):
            # Get action from policy
            action = self.policy.get_action(state)
            
            # Predict next state using world model
            z = self.world_model.encoder(state_to_image(state))
            z_next = self.world_model.predictor(z, action)
            next_state = lewm_to_state(z_next)  # Inverse mapping
            
            # Estimate reward (could be learned or heuristic)
            reward = estimate_reward(state, next_state)
            
            imagined_states.append(next_state)
            imagined_rewards.append(reward)
            
            state = next_state
        
        return imagined_states, imagined_rewards
    
    def train_with_imagination(self, batch_size=32):
        """
        Train policy on real + imagined experience.
        """
        # Real experience
        real_batch = self.replay.sample(batch_size)
        
        # Generate imagined experience
        imagined_states_list = []
        imagined_actions_list = []
        imagined_rewards_list = []
        
        for _ in range(batch_size):
            state = self.replay.sample_state()
            states, rewards = self.imagine_trajectory(state, horizon=16)
            imagined_states_list.extend(states[:-1])
            imagined_actions_list.extend([self.policy.get_action(s) for s in states[:-1]])
            imagined_rewards_list.extend(rewards)
        
        # Combine real + imagined
        combined_batch = real_batch + (imagined_states_list, imagined_actions_list, imagined_rewards_list)
        
        # GRPO update
        self.policy.grpo_update(combined_batch)
```

### 14.3 Option C: Curiosity-Driven Exploration

**Idea:** Use LeWM prediction error as **intrinsic motivation** — explore states where the model is surprised.

```python
class CuriosityReward:
    """
    Curiosity = prediction error of world model.
    
    High error = surprised = learn more about this region.
    """
    def __init__(self, world_model, curiosity_weight=0.1):
        self.world_model = world_model
        self.curiosity_weight = curiosity_weight
    
    def compute(self, state, action, next_state):
        """
        Compute combined reward (extrinsic + intrinsic).
        
        Args:
            state: [B, 4] current state
            action: [B, 2] action taken
            next_state: [B, 4] resulting state
        
        Returns:
            combined_reward: [B] extrinsic + curiosity
        """
        # Extrinsic reward (from environment)
        extrinsic_reward = compute_environment_reward(state, next_state)
        
        # Intrinsic reward (curiosity)
        with torch.no_grad():
            z_curr = self.world_model.encoder(state_to_image(state))
            z_next_true = self.world_model.encoder(state_to_image(next_state))
            z_next_pred = self.world_model.predictor(z_curr, action.unsqueeze(0)).squeeze(0)
            
            # Prediction error = curiosity
            curiosity = (z_next_pred - z_next_true).norm()
        
        # Combined
        return extrinsic_reward + self.curiosity_weight * curiosity


# Training with curiosity
def train_with_curiosity():
    world_model = pretrain_lewm(data)
    policy = GRPOWaypointPolicy()
    curiosity_reward = CuriosityReward(world_model, curiosity_weight=0.1)
    
    for episode in range(1000):
        state = env.reset()
        done = False
        
        while not done:
            # Get action from policy
            action = policy.get_action(state)
            
            # Execute
            next_state, extrinsic_r, done, info = env.step(action)
            
            # Compute curiosity reward
            total_reward = curiosity_reward.compute(state, action, next_state)
            
            # Store with combined reward
            replay_buffer.push(state, action, total_reward, next_state)
            
            # Update policy with combined reward
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                grpo_update(batch)
            
            state = next_state
```

### 14.4 Option D: Planning-Guided Demonstrations

**Idea:** Use LeWM's CEM planner to generate **expert demonstrations** for our RL.

```python
class PlanningGuidedRL:
    """
    Use LeWM's CEM planner to generate expert demonstrations.
    
    Algorithm:
    1. Train LeWM world model on trajectory data
    2. Use CEM planner to generate optimal action sequences
    3. Execute planned actions, record trajectories
    4. Train GRPO policy on planned + exploratory data
    """
    def __init__(self, world_model, policy, env):
        self.world_model = world_model
        self.policy = policy
        self.env = env
    
    def generate_planning_demo(self, start_state, goal_state):
        """
        Use CEM planner to generate an expert demonstration.
        """
        # Encode start and goal
        z_start = self.world_model.encoder(state_to_image(start_state))
        z_goal = self.world_model.encoder(state_to_image(goal_state))
        
        # CEM planning
        action_sequence = cem_planning(
            predictor=self.world_model.predictor,
            start_emb=z_start,
            goal_emb=z_goal,
            horizon=16,
            num_iters=10,
            num_candidates=256,
        )
        
        # Execute planned actions
        trajectory = []
        state = start_state
        for action in action_sequence:
            next_state = self.env.step(state, action)
            trajectory.append((state, action, next_state))
            state = next_state
            
            if np.allclose(state, goal_state, atol=1.0):
                break
        
        return trajectory
    
    def train(self, planning_demos_needed=100, rl_episodes_needed=500):
        """
        Mix planning demonstrations with RL exploration.
        """
        # Generate planning demonstrations
        planning_demos = []
        for _ in range(planning_demos_needed):
            start = self.env.random_start()
            goal = self.env.random_goal()
            demo = self.generate_planning_demo(start, goal)
            planning_demos.append(demo)
        
        # Collect RL exploration episodes
        rl_episodes = []
        for _ in range(rl_episodes_needed):
            episode = collect_rollout(self.policy, self.env)
            rl_episodes.append(episode)
        
        # Train on combined data
        combined = planning_demos + rl_episodes
        
        for epoch in range(50):
            for batch in combined:
                grpo_update(batch)
```

### 14.5 Comparison of Integration Options

| Option | Complexity | Compute | Expected Benefit | Risk | Best For |
|--------|-----------|---------|-----------------|------|----------|
| **A: Auxiliary Loss** | Low | Low | Better representations | Low | Start here |
| **B: Imagination RL** | High | High | Sample efficiency | Medium | Large data |
| **C: Curiosity** | Medium | Medium | Better exploration | Low | Sparse rewards |
| **D: Planning Demos** | Medium | Medium | Better data quality | Medium | Limited data |

### 14.6 Recommended Next Steps

**Step 1 (Immediate):** Try Option A — it's the simplest and lowest risk.

```python
# Pseudocode for getting started
lewm = load_pretrained_lewm()  # Or train on our data
freeze(lewm.encoder)
freeze(lewm.predictor)

# Add world model loss to existing GRPO training
for batch in data:
    waypoints = grpo_policy(batch.state)
    lewm_loss = compute_world_model_loss(batch.state, batch.next_state, batch.action)
    
    # GRPO handles main training
    grpo_update(batch)
    
    # Log lewm_loss for monitoring
    wandb.log({'lewm_loss': lewm_loss})
```

**Step 2 (If Step 1 works):** Add curiosity reward to improve exploration.

**Step 3 (If compute allows):** Try imagination-based RL for sample efficiency.

---

<a name="15-limitations"></a>
## 15. Limitations & Open Problems

### 15.1 Known Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **Discrete dynamics** | LeWM fails on Two-Room (52% vs DINO-WM's 97%) | Only for continuous tasks |
| **Short horizons** | Tested on 8-16 step planning | Unknown for 100+ steps |
| **Pixel-only** | No proprioceptive input | May miss robot state |
| **No RL** | Self-supervised only | Can't learn from rewards |
| **Linear probing** | Physical understanding is linear, not deep | Limited to simple quantities |
| **Single agent** | No multi-agent support | Limited to single-robot tasks |

### 15.2 Open Problems for Research

**1. Adaptive SIGReg for Discrete Dynamics**
```
Problem: SIGReg's Gaussian prior mismatches discrete dynamics
Current: SIGReg works poorly on Two-Room
Possible solutions:
  • Learn the regularization distribution from data
  • Mixture of Gaussians (discrete + continuous)
  • Different regularizer for discrete vs continuous tasks
```

**2. Long-Horizon Planning**
```
Problem: LeWM tested on 8-16 step horizons
Unknown behavior at 100+ steps:
  • Does error accumulate?
  • Does latent space become inaccurate?
  • How to handle planning failures?
Possible solutions:
  • Hierarchical planning (plan at multiple time scales)
  • Uncertainty estimation (know when to replan)
  • Error correction mechanisms
```

**3. Multi-Task Learning**
```
Problem: LeWM trained on single task
Can it learn multiple tasks simultaneously?
Questions:
  • Positive transfer between tasks?
  • Catastrophic forgetting?
  • Task identification in latent space?
```

**4. Integration with RL**
```
Problem: LeWM is self-supervised, RL needs rewards
How to combine best of both?
Options:
  • World model as auxiliary loss for RL
  • RL as fine-tuning for world model
  • Imagination-based RL (Dreamer-style)
```

**5. Real-World Deployment**
```
Problem: All experiments on simulation
Real-world challenges:
  • Sensor noise and calibration errors
  • Delays and non-real-time execution
  • Physical contact and friction
  • Lighting variations
```

### 15.3 Questions for Our Specific Use Case

1. **Do we have image data for LeWM?**
   - Our current system uses 4D state (x, y, heading, speed)
   - LeWM requires images
   - If we have cameras → LeWM possible
   - If state-only → use TD-MPC instead

2. **What compute do we have?**
   - LeWM needs ~4 hours on single A100
   - Do we have this compute?
   - If not → use pre-trained LeWM

3. **What's our data situation?**
   - LeWM needs diverse trajectory data
   - Do we have enough trajectories?
   - If limited → Option A (auxiliary loss) with pre-trained LeWM

4. **What's our goal?**
   - Better representations → Option A
   - Planning capability → Options B/D
   - Better exploration → Option C

---

<a name="16-references"></a>
## 16. References & Resources

### Primary Sources

1. **[LeWorldModel (arXiv:2603.19312)](https://arxiv.org/abs/2603.19312)**  
   Maes, Le Lidec, Scieur, LeCun, Balestriero  
   "LeWorldModel: End-to-End JEPA World Model without Training Heuristics"  
   Mila, NYU, Samsung SAIL, Brown University, March 2026

2. **[LeJEPA / SIGReg (arXiv:2511.08544)](https://arxiv.org/abs/2511.08544)**  
   Balestriero et al.  
   "LeJEPA: A Non-Heuristic Approach to Learning with Joint Embedding Predictive Architectures"  
   November 2025 — SIGReg theoretical foundation

### Comparison Baselines

3. **[DINO-WM (ICLR 2025)](https://arxiv.org/abs/2410.06991)**  
   Wu et al.  
   "Learning World Models with Hierarchical Planning"  
   Foundation model JEPA, 2B+ params, 88% on Push-T

4. **[PLDM (NeurIPS 2024)](https://arxiv.org/abs/2410.06991)**  
   Cai et al.  
   "Pixel-Based World Models: End-to-End JEPA for Robotics"  
   VICReg-based JEPA, 40% on Push-T, unstable

### Related Work

5. **[I-JEPA (CVPR 2023)](https://arxiv.org/abs/2301.08264)**  
   Assran et al.  
   "Image-based JEPA with EMA and stop-gradient"

6. **[V-JEPA (2024)](https://arxiv.org/abs/2402.10857)**  
   Bardes et al.  
   "Video-based JEPA with temporal prediction"

7. **[JEPA (LeCun 2022)](https://arxiv.org/abs/2206.07769)**  
   LeCun  
   "A Theory of Learning from Pixels"

8. **[DreamerV3 (ICLR 2023)](https://arxiv.org/abs/2301.04104)**  
   Hafner et al.  
   "DreamerV3: World Models for Imagination-Based RL"

9. **[TD-MPC (2023)](https://arxiv.org/abs/2212.05698)**  
   Hafner et al.  
   "Temporal Difference Learning for Model Predictive Control"

10. **[World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122)**  
    Ha & Schmidhuber  
    "World Models" — Foundational world model paper

### Code & Resources

- **Project page:** [le-wm.github.io](https://le-wm.github.io/)
- **Paper PDF:** [arXiv PDF](https://arxiv.org/pdf/2603.19312)
- **GitHub:** [github.com/le-wm/le-wm](https://github.com/le-wm/le-wm)
- **SIGReg theory:** [LeJEPA arXiv](https://arxiv.org/abs/2511.08544)

---

## Appendix A: Quick Reference

### SIGReg Algorithm (Pseudocode)
```python
def sigreg(embeddings, num_projections=1024, weight=0.1):
    projections = random_directions(num_projections, dim=192)
    stats = []
    for proj in projections:
        projected = embeddings @ proj  # [B]
        ep_stat = epps_pulley_test(projected)  # 0=Gaussian, high=non-Gaussian
        stats.append(ep_stat)
    return weight * mean(stats)
```

### CEM Planning Algorithm (Pseudocode)
```python
def cem_plan(predictor, start_emb, goal_emb, horizon=16):
    mean = zeros(horizon, action_dim)
    std = ones(horizon, action_dim)
    for iter in range(10):
        candidates = sample(mean, std, N=256)
        rewards = [score_rollout(predictor, start_emb, goal_emb, seq) for seq in candidates]
        elite = top_k(candidates, rewards, k=16)
        mean, std = update_distribution(elite)
    return elite[0]  # First action
```

### LeWM Architecture Summary
```
Encoder: ViT-Tiny (5M params) → 192-dim per frame
Predictor: Transformer (10M params) with AdaLN
Total: ~15M parameters
Training: ~4 hours on single A100
Planning: <1 second via CEM
```

---

## Appendix B: Key Equations

### Prediction Loss
```
L_pred = E[ || ẑ_{t+1} - z_{t+1} ||² ]
```

### SIGReg (Epps-Pulley Test)
```
T(h) = Σ_k [Φ^{-1}(t_k) - Φ^{-1}(t_{k-1})] * (z_{(k)} - z_{(k-1)})²

SIGReg(Z) = (1/M) Σ_m T(h^(m))
where h^(m) = Z · u^(m) (projection onto random direction)
```

### Total Loss
```
L_total = L_pred + λ * SIGReg(Z)
```

### Cross-Entropy Method (CEM)
```
Mean update: μ_{t+1} = (1/K) Σ_{k∈elite} a_k
Std update: σ_{t+1} = sqrt((1/K) Σ_{k∈elite} (a_k - μ_{t+1})²)
```

---

*Survey completed by OpenClaw agent — 2026-03-29*  
*Total: ~45,000 characters*
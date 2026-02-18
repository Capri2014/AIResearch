# Deep Digest: ResAD & GoalFlow - Implementation Guide

**Date:** 2026-02-17  
**Status:** Complete Technical Deep Digest  
**Papers:** 
- ResAD: https://arxiv.org/abs/2510.08562
- GoalFlow: https://arxiv.org/abs/2503.05689

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [ResAD: Normalized Residual with Uncertainty](#2-resad-normalized-residual-with-uncertainty)
3. [GoalFlow: Trajectory Scoring & Selection](#3-goalflow-trajectory-scoring--selection)
4. [Combined Architecture](#4-combined-architecture)
5. [Implementation: ResAD Components](#5-implementation-resad-components)
6. [Implementation: GoalFlow Components](#6-implementation-goalflow-components)
7. [Integration with AR Decoder](#7-integration-with-ar-decoder)
8. [Training Scripts](#8-training-scripts)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Complete Pipeline Code](#10-complete-pipeline-code)

---

## 1. Executive Summary

### Key Innovations from Papers

| Paper | Core Innovation | Key Insight |
|-------|-----------------|-------------|
| **ResAD** | Normalized Residual Learning | Δ = (pred - sft) / uncertainty, not raw residual |
| **GoalFlow** | Goal-Driven Trajectory Scoring | Generate candidates, score and select best |

### Combined Approach

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Combined ResAD + GoalFlow Pipeline                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────────┐     ┌────────────────────────┐ │
│  │ SFT Model  │────►│  Residual Head  │────►│  Trajectory Scorer    │ │
│  │ (frozen)   │     │  + Uncertainty  │     │  + Selection         │ │
│  └─────────────┘     └─────────────────┘     └────────────────────────┘ │
│       │                    │                          │                   │
│       │                    ▼                          ▼                   │
│       │            ┌──────────────┐         ┌─────────────────┐        │
│       │            │ Δ = (y-ŷ)/σ  │         │ Score = f(s,a) │        │
│       │            │ Inertial Ref  │         │ Top-K Selection │        │
│       │            └──────────────┘         └─────────────────┘        │
│       │                    │                          │                   │
│       └────────────────────┼──────────────────────────┘                   │
│                            ▼                                             │
│                    ┌─────────────────────────────────┐                    │
│                    │  Final: y = ŷ + Δ × σ        │                    │
│                    │  With safety scoring           │                    │
│                    └─────────────────────────────────┘                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quick Comparison

| Aspect | ResAD | GoalFlow | Combined |
|--------|-------|----------|----------|
| **Core Idea** | Normalized residual | Trajectory scoring | Both |
| **Input** | SFT prediction + uncertainty | Multiple candidates | SFT + multi-candidate |
| **Output** | Corrected trajectory | Scored trajectory | Corrected + scored |
| **Use Case** | Single trajectory improvement | Best selection | Robust selection |

---

## 2. ResAD: Normalized Residual with Uncertainty

### 2.1 Key Formula

**Traditional Residual Learning:**
```
Δ = y - ŷ  (raw residual)
y_final = ŷ + Δ
```

**ResAD Normalized Residual:**
```
σ = uncertainty(prediction)  # Aleatoric uncertainty
Δ_normalized = (y - ŷ) / σ
y_final = ŷ + Δ_normalized × σ
```

### 2.2 Pseudo-Code: ResAD Algorithm

```
ALGORITHM: ResAD Training

INPUT:  SFT model f_θ (frozen), training data D = {(x, y)}
OUTPUT: Residual head g_φ with uncertainty estimation

1.  # Load frozen SFT model
2.  f_θ ← load_checkpoint("sft_model.pt")
3.  freeze(f_θ)
4.  
5.  # Initialize residual head
6.  g_φ ← ResidualHead(input_dim, hidden_dim, output_dim)
7.  σ_ψ ← UncertaintyHead(input_dim, hidden_dim)
8.  
9.  FOR epoch = 1 TO N_epochs:
10.     FOR batch (x, y) IN D:
11.         # Forward pass
12.         ŷ ← f_θ(x)                    # SFT prediction
13.         Δ ← g_φ(x, ŷ)                 # Raw residual prediction
14.         log_σ ← σ_ψ(x, ŷ)            # Log uncertainty
15.         σ ← exp(log_σ)               # Uncertainty > 0
16.         
17.         # Normalized residual
18.         Δ_norm ← (y - ŷ) / σ
19.         
20.         # Loss: NLL + KL divergence
21.         nll ← MSE(Δ_norm, Δ)          # Prediction loss
22.         kl ← KL(σ_prior, σ)           # Uncertainty regularization
23.         loss ← nll + β × kl
24.         
25.         # Update heads only
26.         φ, ψ ← optimizer(φ, ψ, loss)
27.        
28. RETURN g_φ, σ_ψ


ALGORITHM: ResAD Inference

INPUT:  SFT model f_θ, residual head g_φ, uncertainty head σ_ψ, input x
OUTPUT: Corrected prediction y_final with confidence

1.  # Forward pass
2.  ŷ ← f_θ(x)                    # SFT prediction
3.  Δ ← g_φ(x, ŷ)                 # Predicted residual
4.  log_σ ← σ_ψ(x, ŷ)             # Predicted uncertainty
5.  σ ← exp(log_σ)
6.  
7.  # Apply normalized residual
8.  Δ_norm ← Δ                    # g_φ predicts normalized directly
9.  y_final ← ŷ + Δ_norm × σ
10. 
11. # Confidence metrics
12. confidence ← 1.0 / σ
13. 
14. RETURN y_final, σ, confidence
```

### 2.3 Why Normalized Residual Works

```
Without Normalization (Traditional):
┌─────────────────────────────────────────┐
│  Easy samples:  Δ = 0.1                 │
│  Hard samples:  Δ = 1.0                 │
│                                         │
│  Problem: Both get same weight in MSE   │
│  Result: Model focuses on hard samples  │
│          and overfits easy samples      │
└─────────────────────────────────────────┘

With Normalization (ResAD):
┌─────────────────────────────────────────┐
│  Easy samples:  σ = 0.1,  Δ_norm = 1.0  │
│  Hard samples:  σ = 1.0,  Δ_norm = 1.0  │
│                                         │
│  Solution: Normalized residuals are     │
│  comparable. MSE treats them equally.   │
│  Result: Balanced learning              │
└─────────────────────────────────────────┘
```

### 2.4 Inertial Reference Frame

**Concept:** Use ego vehicle dynamics as a reference to make predictions more robust.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inertial Reference Frame                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Waypoints (map frame):                                      │
│  [x_map, y_map, θ_map]                                          │
│       ↓                                                          │
│  Transform to ego frame:                                          │
│  [x_ego, y_ego, θ_ego] relative to current pose                  │
│       ↓                                                          │
│  Predict residuals in ego frame                                   │
│       ↓                                                          │
│  Transform back to map frame                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. GoalFlow: Trajectory Scoring & Selection

### 3.1 Key Formula

**GoalFlow Architecture:**
```
1. Goal Conditioning: z_g = MLP(goal_features)
2. Flow Matching:   p_θ(τ | z_g) = flow(τ, z_g)
3. Scoring:         s(τ) = ScoreHead(τ, z_g)
```

### 3.2 Pseudo-Code: GoalFlow Algorithm

```
ALGORITHM: GoalFlow Training

INPUT:  Training trajectories D = {(τ, g, c)} where
        τ = trajectory [T, 3], g = goal [3], c = context [D]
OUTPUT: Flow model p_θ and scorer s_ψ

1.  # Initialize flow model and scorer
2.  p_θ ← ConditionalFlow(input_dim=3×T)
3.  s_ψ ← TrajectoryScorer(input_dim=3×T + goal_dim)
4.  
5.  FOR epoch = 1 TO N_epochs:
6.      FOR batch (τ, g, c) IN D:
7.          # Goal conditioning
8.          z_g ← MLP_goal(g)              # Goal embedding
9.          z_c ← MLP_context(c)           # Context embedding
10.         z ← concat(z_g, z_c)            # [goal_dim + context_dim]
11.         
12.         # Flow matching loss
13.         τ_flat ← flatten(τ)              # [3T]
14.         z_t ← p_θ.sample_t(z, t)        # Latent at time t
15.         flow_loss ← MSE(z_t, τ_flat)
16.         
17.         # Scoring loss
18.         τ_scores ← s_ψ(concat(τ_flat, z_g))
19.         score_loss ← BCE(τ_scores, ground_truth_scores)
20.         
21.         # Combined loss
22.         loss ← flow_loss + λ × score_loss
23.         
24.         # Update
25.         θ, ψ ← optimizer(θ, ψ, loss)
26.         
27. RETURN p_θ, s_ψ


ALGORITHM: GoalFlow Inference (Generation + Selection)

INPUT:  Context x, goal g, number of candidates K
OUTPUT: Best trajectory τ_best with score

1.  # Generate K candidate trajectories
2.  τ_candidates ← []
3.  FOR i = 1 TO K:
4.       τ_i ← p_θ.sample(goal=g)          # Sample from flow
5.       τ_candidates.append(τ_i)
6.  
7.  # Score each candidate
8.  scores ← []
9.  FOR τ IN τ_candidates:
10.        score ← s_ψ(concat(flatten(τ), goal_embedding(g)))
11.        scores.append(score)
12.  
13. # Select best
14. τ_best ← τ_candidates[argmax(scores)]
15. score_best ← max(scores)
16. 
17. RETURN τ_best, score_best, τ_candidates, scores
```

### 3.3 Trajectory Scoring Head Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Trajectory Scoring Head                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input:                                                          │
│  - Trajectory: τ [T, 3] (x, y, heading)                         │
│  - Goal: g [3] (target x, y, heading)                          │
│  - Context: c [D] (current state, traffic, etc.)                │
│                                                                  │
│  Architecture:                                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  τ_flat [3T] ──► Linear(3T, 256) ──► ReLU ──►        │   │
│  │                                                           │   │
│  │  goal [3]   ──► Linear(3, 64)   ──► ReLU ──►         │   │
│  │                                                           │   │
│  │  context [D] ──► Linear(D, 128) ──► ReLU ──►          │   │
│  │                                                           │   │
│  │  All concatenated: [256 + 64 + 128] = 448              │   │
│  │           ↓                                              │   │
│  │  Linear(448, 128) ──► ReLU ──► Linear(128, 1) ──►     │   │
│  │           ↓                                              │   │
│  │  Sigmoid ──► score [0, 1]                               │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Output:                                                         │
│  - score: [0, 1] - probability of good trajectory               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Multi-Candidate Generation Strategies

```
Strategy 1: Independent Sampling (Simple)
┌─────────────────────────────────────────┐
│  τ₁ ~ p_θ(·|g)                          │
│  τ₂ ~ p_θ(·|g)                          │
│  τ₃ ~ p_θ(·|g)                          │
│  ...                                     │
│  τ_K ~ p_θ(·|g)                         │
└─────────────────────────────────────────┘

Strategy 2: Denoising Diffusion (Higher Quality)
┌─────────────────────────────────────────┐
│  Start from noise ε ~ N(0, I)           │
│  Apply K diffusion steps                │
│  Get τ₁, τ₂, ..., τ_K                  │
└─────────────────────────────────────────┘

Strategy 3: Diversity Sampling (Ensembles)
┌─────────────────────────────────────────┐
│  Use multiple flow heads                │
│  Each produces diverse candidates       │
└─────────────────────────────────────────┘
```

---

## 4. Combined Architecture

### 4.1 Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Combined ResAD + GoalFlow Pipeline                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: SFT Prediction (Frozen)                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                  │    │
│  │  Input:  x = [images, state]                                   │    │
│  │      ↓                                                           │    │
│  │  ARDecoder                                                       │    │
│  │      ↓                                                           │    │
│  │  ŷ = [waypoints]  (T, 3)                                        │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  STAGE 2: ResAD Correction                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                  │    │
│  │  ŷ ──┬──► ResidualHead ──► Δ                                   │    │
│  │       │                                                          │    │
│  │       └──► UncertaintyHead ──► σ                                │    │
│  │                                                                  │    │
│  │  y_corr = ŷ + Δ × σ  (normalized residual)                      │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  STAGE 3: GoalFlow Scoring                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                  │    │
│  │  Generate K candidates: τ₁, τ₂, ..., τ_K                         │    │
│  │      ↓                                                           │    │
│  │  Score each: s(τ_i) = TrajectoryScorer(τ_i, goal)               │    │
│  │      ↓                                                           │    │
│  │  Select best: τ_best = argmax_i s(τ_i)                          │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  OUTPUT: τ_best, score_best, confidence, uncertainty                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Loss Functions

```
Combined Loss = L_sft + λ₁ × L_residual + λ₂ × L_scoring + λ₃ × L_uncertainty

Where:
  L_sft           = MSE(y_true, ŷ)              (SFT frozen, no backprop)
  L_residual      = MSE(Δ_true, Δ_pred)         (ResAD head)
  L_scoring       = BCE(scores_true, scores_pred) (Scorer head)
  L_uncertainty   = NLL(σ_pred, σ_true)          (Uncertainty head)
```

### 4.3 Inference Pipeline

```
INPUT:  Current state x, goal g, K candidates

1.  # SFT forward pass (frozen)
2.  ŷ ← ARDecoder(x)                      # [T, 3]
3.  
4.  # ResAD correction
5.  Δ ← ResidualHead(x, ŷ)                 # [T, 3]
6.  σ ← UncertaintyHead(x, ŷ)              # [T, 1]
7.  y_corr = ŷ + Δ × σ                     # [T, 3]
8.  
9.  # GoalFlow scoring (if enabled)
10. if K > 1:
11.     # Generate candidates
12.     τ_candidates = [y_corr]           # Start with ResAD correction
13.     FOR i = 2 TO K:
14.         τ_i ← FlowModel.sample(goal=g)  # Generate more
15.         τ_candidates.append(τ_i)
16.     
17.     # Score and select
18.     scores = [Scorer(τ, g) for τ in τ_candidates]
19.     best_idx = argmax(scores)
20.     τ_best = τ_candidates[best_idx]
21.     score_best = scores[best_idx]
22. else:
23.     τ_best = y_corr
24.     score_best = 1.0
25. 
26. # Output
27. RETURN τ_best, score_best, σ.mean(), confidence=1/σ.mean()
```

---

## 5. Implementation: ResAD Components

### 5.1 Uncertainty Head

```python
"""
ResAD Uncertainty Head
=========================
Predicts aleatoric uncertainty for each waypoint.
Outputs log(sigma) to ensure sigma > 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class UncertaintyHead(nn.Module):
    """
    Predicts uncertainty (aleatoric) for waypoint predictions.
    
    Architecture:
    - Takes SFT features + predictions as input
    - Outputs log(sigma) to ensure sigma > 0
    - Per-waypoint uncertainty estimation
    
    Usage:
        sigma_head = UncertaintyHead(
            feature_dim=256,
            waypoint_dim=3,
            hidden_dim=128,
        )
        
        log_sigma = sigma_head(features, waypoints)
        sigma = torch.exp(log_sigma)  # [B, T, 1]
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        waypoint_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.waypoint_dim = waypoint_dim
        self.hidden_dim = hidden_dim
        
        # Input: features + waypoints concatenated
        input_dim = feature_dim + waypoint_dim
        
        # Uncertainty estimation network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, waypoint_dim),  # Per-waypoint sigma
        )
    
    def forward(
        self,
        features: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [B, feature_dim] or [B, T, feature_dim]
            waypoints: [B, T, waypoint_dim]
            
        Returns:
            log_sigma: [B, T, waypoint_dim] (log uncertainty)
        """
        # Handle feature dimensions
        if features.dim() == 2:
            # [B, feature_dim] → [B, 1, feature_dim]
            features = features.unsqueeze(1)
        
        # Concatenate features and waypoints
        # [B, T, feature_dim] + [B, T, waypoint_dim] → [B, T, feature_dim + waypoint_dim]
        x = torch.cat([features, waypoints], dim=-1)
        
        # Forward through network
        log_sigma = self.net(x)  # [B, T, waypoint_dim]
        
        return log_sigma
    
    def loss(
        self,
        log_sigma: torch.Tensor,
        uncertainty_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        NLL loss for uncertainty estimation.
        
        The loss is: 0.5 * exp(-log_sigma) * (x - mean)^2 + 0.5 * log_sigma
        
        This is the negative log-likelihood of a Gaussian with learned variance.
        
        Args:
            log_sigma: [B, T, waypoint_dim] predicted log uncertainty
            uncertainty_target: [B, T, waypoint_dim] uncertainty of residual
            
        Returns:
            nll_loss: scalar tensor
        """
        # NLL of Gaussian
        sigma = torch.exp(log_sigma)
        nll = 0.5 * ((uncertainty_target ** 2) / sigma + log_sigma)
        
        return nll.mean()


class ResADResidualHead(nn.Module):
    """
    ResAD Residual Head with Inertial Reference.
    
    Predicts normalized residual: Δ_norm = (y - ŷ) / σ
    
    Usage:
        delta_head = ResADResidualHead(
            feature_dim=256,
            waypoint_dim=3,
            hidden_dim=128,
            use_inertial_ref=True,
        )
        
        delta_norm = delta_head(features, waypoints)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        waypoint_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_inertial_ref: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.waypoint_dim = waypoint_dim
        self.hidden_dim = hidden_dim
        self.use_inertial_ref = use_inertial_ref
        
        input_dim = feature_dim + waypoint_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, waypoint_dim),  # Predicts normalized residual
        )
    
    def forward(
        self,
        features: torch.Tensor,
        waypoints: torch.Tensor,
        ego_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        if self.use_inertial_ref and ego_state is not None:
            ego_features = ego_state.unsqueeze(1).expand(-1, waypoints.size(1), -1)
            waypoints_input = torch.cat([waypoints, ego_features], dim=-1)
        else:
            waypoints_input = waypoints
        
        x = torch.cat([features, waypoints_input], dim=-1)
        delta_norm = self.net(x)
        
        return delta_norm
    
    def apply_residual(
        self,
        waypoints: torch.Tensor,
        delta_norm: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """Apply normalized residual: y_final = ŷ + Δ_norm × σ"""
        if uncertainty.dim() == 3 and uncertainty.size(-1) == 1:
            uncertainty = uncertainty.expand_as(waypoints)
        
        corrected = waypoints + delta_norm * uncertainty
        return corrected


class ResADModule(nn.Module):
    """
    Complete ResAD Module combining residual head and uncertainty head.
    
    Usage:
        resad = ResADModule(
            feature_dim=256,
            waypoint_dim=3,
            hidden_dim=128,
            use_inertial_ref=True,
        )
        
        delta, sigma = resad(features, waypoints)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        waypoint_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_inertial_ref: bool = False,
    ):
        super().__init__()
        
        self.residual_head = ResADResidualHead(
            feature_dim=feature_dim,
            waypoint_dim=waypoint_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_inertial_ref=use_inertial_ref,
        )
        
        self.uncertainty_head = UncertaintyHead(
            feature_dim=feature_dim,
            waypoint_dim=waypoint_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        features: torch.Tensor,
        waypoints: torch.Tensor,
        ego_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        delta_norm = self.residual_head(features, waypoints, ego_state)
        log_sigma = self.uncertainty_head(features, waypoints)
        
        return delta_norm, log_sigma
    
    def apply(
        self,
        waypoints: torch.Tensor,
        delta_norm: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply residual correction."""
        sigma = torch.exp(log_sigma)
        corrected = waypoints + delta_norm * sigma
        
        return corrected, sigma
    
    def loss(
        self,
        delta_norm: torch.Tensor,
        log_sigma: torch.Tensor,
        target_residual: torch.Tensor,
        target_uncertainty: torch.Tensor,
        kl_weight: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """Compute ResAD loss."""
        sigma = torch.exp(log_sigma)
        
        # NLL loss
        nll = 0.5 * ((target_residual - delta_norm) ** 2 / sigma + log_sigma)
        nll_loss = nll.mean()
        
        # MSE loss
        mse_loss = F.mse_loss(delta_norm, target_residual)
        
        # KL divergence
        kl = 0.5 * (sigma - 1 - torch.log(sigma))
        kl_loss = kl.mean()
        
        total_loss = mse_loss + nll_loss + kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'nll_loss': nll_loss,
            'kl_loss': kl_loss,
            'sigma_mean': sigma.mean(),
        }
```

### 5.2 Inertial Reference Transform

```python
class InertialReferenceTransform(nn.Module):
    """
    Transform waypoints between map frame and ego (inertial) frame.
    """
    
    def __init__(self, waypoint_dim: int = 3):
        super().__init__()
        self.waypoint_dim = waypoint_dim
    
    def map_to_ego(
        self,
        waypoints: torch.Tensor,  # [B, T, 3]
        ego_pose: torch.Tensor,   # [B, 3] (x, y, heading)
    ) -> torch.Tensor:
        """Transform from map frame to ego frame."""
        B, T, _ = waypoints.shape
        
        ego_x = ego_pose[:, 0]
        ego_y = ego_pose[:, 1]
        ego_heading = ego_pose[:, 2]
        
        # Relative position
        rel_x = waypoints[:, :, 0] - ego_x.unsqueeze(1)
        rel_y = waypoints[:, :, 1] - ego_y.unsqueeze(1)
        
        # Rotate to ego frame
        cos_h = torch.cos(ego_heading)
        sin_h = torch.sin(ego_heading)
        
        ego_rel_x = rel_x * cos_h + rel_y * sin_h
        ego_rel_y = -rel_x * sin_h + rel_y * cos_h
        
        # Relative heading
        ego_heading_rel = waypoints[:, :, 2] - ego_heading.unsqueeze(1)
        ego_heading_rel = torch.atan2(
            torch.sin(ego_heading_rel),
            torch.cos(ego_heading_rel)
        )
        
        return torch.stack([ego_rel_x, ego_rel_y, ego_heading_rel], dim=-1)
    
    def ego_to_map(
        self,
        waypoints_ego: torch.Tensor,  # [B, T, 3]
        ego_pose: torch.Tensor,        # [B, 3]
    ) -> torch.Tensor:
        """Transform from ego frame to map frame."""
        B, T, _ = waypoints_ego.shape
        
        ego_x = ego_pose[:, 0]
        ego_y = ego_pose[:, 1]
        ego_heading = ego_pose[:, 2]
        
        cos_h = torch.cos(ego_heading)
        sin_h = torch.sin(ego_heading)
        
        # Rotate to map frame
        map_rel_x = waypoints_ego[:, :, 0] * cos_h - waypoints_ego[:, :, 1] * sin_h
        map_rel_y = waypoints_ego[:, :, 0] * sin_h + waypoints_ego[:, :, 1] * cos_h
        
        # Translate
        map_x = map_rel_x + ego_x.unsqueeze(1)
        map_y = map_rel_y + ego_y.unsqueeze(1)
        map_heading = waypoints_ego[:, :, 2] + ego_heading.unsqueeze(1)
        
        return torch.stack([map_x, map_y, map_heading], dim=-1)
```

---

## 6. Implementation: GoalFlow Components

### 6.1 Trajectory Scoring Head

```python
class TrajectoryScorer(nn.Module):
    """
    GoalFlow Trajectory Scoring Head.
    
    Scores each candidate trajectory.
    """
    
    def __init__(
        self,
        waypoint_dim: int = 3,
        goal_dim: int = 3,
        context_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Trajectory encoder
        self.traj_encoder = nn.Sequential(
            nn.Linear(waypoint_dim * 10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Context encoder
        if context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        
        # Scoring network
        combined_dim = hidden_dim + hidden_dim // 4
        if context_dim > 0:
            combined_dim += hidden_dim // 4
        
        self.scorer = nn.ModuleList()
        for i in range(num_layers - 1):
            self.scorer.append(nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            combined_dim = hidden_dim
        
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        trajectories: torch.Tensor,  # [B, K, T, waypoint_dim]
        goal: torch.Tensor,         # [B, goal_dim]
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        B, K, T, D = trajectories.shape
        
        # Encode trajectory
        traj_flat = trajectories.view(B * K, T * D)
        traj_feat = self.traj_encoder(traj_flat)  # [B*K, hidden_dim]
        
        # Encode goal
        goal_feat = self.goal_encoder(goal)  # [B, hidden_dim//4]
        goal_feat = goal_feat.unsqueeze(1).expand(-1, K, -1).contiguous()
        goal_feat = goal_feat.view(B * K, -1)  # [B*K, hidden_dim//4]
        
        # Combine
        if context is not None:
            ctx_feat = self.context_encoder(context)  # [B, hidden_dim//4]
            ctx_feat = ctx_feat.unsqueeze(1).expand(-1, K, -1).contiguous()
            ctx_feat = ctx_feat.view(B * K, -1)
            x = torch.cat([traj_feat, goal_feat, ctx_feat], dim=-1)
        else:
            x = torch.cat([traj_feat, goal_feat], dim=-1)
        
        # Score
        for layer in self.scorer:
            x = layer(x)
        
        scores = torch.sigmoid(self.output(x))  # [B*K, 1]
        
        return scores.view(B, K)


class ConditionalFlowModel(nn.Module):
    """
    Conditional Flow Model for trajectory generation.
    
    Generates trajectories conditioned on goal and context.
    """
    
    def __init__(
        self,
        input_dim: int = 30,  # T=10 * waypoint_dim=3
        condition_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        
        # Condition embedding
        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Flow layers
        self.flow = nn.ModuleList()
        for i in range(num_layers):
            self.flow.append(
                nn.Sequential(
                    nn.Linear(input_dim + hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, input_dim),
                )
            )
        
        # Prior (standard Gaussian)
        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_log_var', torch.zeros(1))
    
    def forward(
        self,
        x: torch.Tensor,      # [B, input_dim]
        condition: torch.Tensor,  # [B, condition_dim]
    ) -> torch.Tensor:
        """Forward through flow."""
        cond = self.condition_net(condition)
        
        for layer in self.flow:
            x = x + layer(torch.cat([x, cond], dim=-1))
        
        return x
    
    def sample(
        self,
        condition: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from flow model."""
        B = condition.size(0)
        
        # Sample from prior
        z = torch.randn(B * num_samples, self.input_dim, device=condition.device)
        
        # Transform through flow
        x = z
        cond = self.condition_net(condition).repeat(num_samples, 1)
        
        for layer in self.flow:
            x = x + layer(torch.cat([x, cond], dim=-1))
        
        return x.view(B, num_samples, -1)
    
    def log_prob(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability."""
        B = x.size(0)
        
        # Forward through flow
        z = x
        cond = self.condition_net(condition)
        
        for layer in self.flow:
            z = z + layer(torch.cat([z, cond], dim=-1))
        
        # Compute log prob under prior
        log_prob = -0.5 * (
            z.pow(2) + 
            self.prior_log_var.expand_as(z) + 
            math.log(2 * math.pi)
        ).sum(dim=-1)
        
        # Jacobian adjustment (simplified)
        return log_prob


class GoalFlowModule(nn.Module):
    """
    Complete GoalFlow module: Flow + Scoring.
    """
    
    def __init__(
        self,
        waypoint_dim: int = 3,
        num_waypoints: int = 10,
        goal_dim: int = 3,
        context_dim: int = 256,
        hidden_dim: int = 256,
        num_flow_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.waypoint_dim = waypoint_dim
        self.num_waypoints = num_waypoints
        self.input_dim = waypoint_dim * num_waypoints
        self.goal_dim = goal_dim
        self.context_dim = context_dim
        
        # Condition dimension = goal_dim + context_dim
        condition_dim = goal_dim + context_dim if context_dim > 0 else
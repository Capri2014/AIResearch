# Complete E2E Autonomous Driving Pipeline

**Date:** 2026-02-18  
**Status:** Complete Pipeline Documentation  
**Version:** 2.0

---

## Overview

This document describes the complete end-to-end training pipeline for autonomous driving, from raw data to simulation evaluation.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Waymo   │───→│  SSL    │───→│   SFT    │───→│  ResAD  │       │
│  │  Data   │    │ Pretrain│    │  Waypoint│    │ Residual│       │
│  └──────────┘    └──────────┘    │   BC     │    │ Learning│       │
│                                  └──────────┘    └──────────┘       │
│                                         │              │              │
│                                         │              ↓              │
│                                  ┌──────┴──────┐    ┌──────────┐    │
│                                  │    CoT     │──→│   RL     │    │
│                                  │  Reasoning │    │Refinement│    │
│                                  └─────────────┘    └──────────┘    │
│                                         │              │              │
│                                         ↓              ↓              │
│                                  ┌───────────────────────────┐           │
│                                  │   CARLA Simulation     │           │
│                                  │   + ScenarioRunner   │           │
│                                  └───────────────────────────┘           │
│                                         │                            │
│                                         ↓                            │
│                                  ┌───────────────────────────┐           │
│                                  │   Metrics & Eval      │           │
│                                  │   ADE / FDE / Safety │           │
│                                  └───────────────────────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Ingestion (Waymo Open Dataset)

### 1.1 Data Source

```
Input: Waymo Open Dataset
├── Format: TFRecord
├── Size: ~1.2 TB (full dataset)
├── Content per frame:
│   ├── Camera images (5 cameras)
│   ├── LiDAR point clouds
│   ├── 3D bounding boxes
│   ├── Camera calibration
│   └── Vehicle state (pose, velocity)
└── Frame rate: 10 Hz
```

### 1.2 Data Preprocessing

```
Preprocessing Pipeline:
├── Step 1: Scene extraction
│   ├── Extract 20-second driving segments
│   ├── Filter valid segments (no collisions)
│   └── Format: [T, H, W, C] tensors
│
├── Step 2: Feature extraction
│   ├── Perception features (detections, tracking)
│   ├── BEV representation (deprecated → JEPA/SSL)
│   └── Ego state (pose, velocity, acceleration)
│
└── Step 3: Waypoint extraction
    ├── Ground truth waypoints (10-20 steps)
    ├── Format: [T, 3] (x, y, heading)
    └── Timestamps aligned with frames
```

### 1.3 Data Format

```python
@dataclass
class DrivingEpisode:
    episode_id: str
    frames: List[Frame]
    
@dataclass  
class Frame:
    timestamp: float
    camera_images: np.ndarray  # [H, W, 3]
    lidar_points: np.ndarray      # [N, 4] (x, y, z, intensity)
    ego_state: EgoState          # (x, y, θ, v, a)
    waypoints: np.ndarray        # [T, 3] ground truth
    traffic_light_state: int     # 0=red, 1=yellow, 2=green
```

---

## Stage 2: Self-Supervised Pre-training (SSL)

### 2.1 Objective

```
Goal: Learn rich representations from unlabeled data

Method: Contrastive Learning + Temporal Consistency
├── Positive pairs: same scene, different views
├── Negative pairs: different scenes
└── Temporal: consecutive frames should be similar
```

### 2.2 Encoder Architecture

```
Architecture: JEPA (Joint Embedding Predictive Architecture)

┌─────────────────────────────────────────────┐
│           JEPA Encoder                       │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────┐                               │
│  │ Camera   │──→ [ConvNet / ViT]          │
│  │  Image   │                               │
│  └──────────┘                               │
│         ↓                                    │
│  ┌──────────┐                               │
│  │ Projector│──→ [MLP]                      │
│  └──────────┘                               │
│         ↓                                    │
│  ┌──────────┐                               │
│  │ Predictor│──→ [MLP]  (predict future)   │
│  └──────────┘                               │
│                                              │
└─────────────────────────────────────────────┘
```

### 2.3 Pre-training Code

```python
# training/pretrain/jepa_pretrain.py

class JEPAConfig:
    encoder: str = "resnet50"  # or "vit-b"
    hidden_dim: int = 256
    predictor_dim: int = 256
    temporal_window: int = 5
    temperature: float = 0.1
```

---

## Stage 3: Supervised Fine-Tuning (SFT)

### 3.1 Objective

```
Goal: Learn to predict waypoints from perception features

Input: Perception features (SSL encoder output)
Output: Waypoints ŷ ∈ ℝ^[T×3]

Loss: MSE(ŷ, y) + trajectory smoothness regularization
```

### 3.2 AR Decoder Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 AR Decoder                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: z ∈ ℝ^[D] (perception features)               │
│       ↓                                                 │
│  ┌─────────────────────────────────────┐              │
│  │   Positional Encoding              │              │
│  │   (sinusoidal or learned)        │              │
│  └─────────────────────────────────────┘              │
│       ↓                                                  │
│  ┌─────────────────────────────────────┐              │
│  │   Transformer Decoder              │              │
│  │   6 layers, 8 heads             │              │
│  │   causal attention mask           │              │
│  └─────────────────────────────────────┘              │
│       ↓                                                  │
│  ┌─────────────────────────────────────┐              │
│  │   Waypoint Head                   │              │
│  │   Linear(z) → [T×3]              │              │
│  └─────────────────────────────────────┘              │
│       ↓                                                  │
│  Output: ŷ ∈ ℝ^[T, 3] (x, y, heading)               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Training Command

```bash
python training/sft/train_waypoint_bc_cot.py \
    --config config/sft/waypoint_bc.yaml \
    --data waymo_preprocessed/ \
    --output out/sft_waypoint_bc/ \
    --epochs 10 \
    --batch 32
```

---

## Stage 4: Residual Learning (ResAD)

### 4.1 Objective

```
Goal: Learn correction on top of frozen SFT model

Formula: ŷ_final = ŷ + Δ
Where: Δ = (y - ŷ) / σ  (normalized residual)

Benefits:
├── Higher sample efficiency (delta is simpler than full prediction)
├── Uncertainty estimation (σ = aleatoric uncertainty)
├── Safe (frozen SFT anchors behavior)
└── Modular (can be added/removed independently)
```

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              ResAD Module                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐                                    │
│  │   Frozen SFT     │──→ ŷ (baseline waypoints)        │
│  │   Model         │                                    │
│  └──────────────────┘                                    │
│            ↓                                               │
│            ↓  (frozen, no gradient)                       │
│            ↓                                               │
│  ┌──────────────────┐                                    │
│  │   Uncertainty   │──→ σ (aleatoric uncertainty)       │
│  │   Head          │   (learned log(sigma))             │
│  └──────────────────┘                                    │
│            ↓                                               │
│  ┌──────────────────┐                                    │
│  │   Residual      │──→ Δ (normalized residual)         │
│  │   Head          │   Δ = (y - ŷ) / σ                 │
│  └──────────────────┘                                    │
│            ↓                                               │
│  ┌──────────────────┐                                    │
│  │   Combiner      │                                    │
│  │   ŷ_final =     │                                    │
│  │   ŷ + Δ × σ     │                                    │
│  └──────────────────┘                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Loss Function

```
ResAD Loss = NLL Loss + MSE Loss + KL Regularization

NLL Loss: 0.5 * exp(-log_sigma) * (Δ - Δ_target)² + 0.5 * log_sigma
MSE Loss: MSE(Δ, Δ_target)
KL Loss: KL(σ || 1)  # encourage σ ≈ 1
```

### 4.4 Training Command

```bash
python training/rl/resad_train.py \
    --sft-checkpoint out/sft_waypoint_bc/model.pt \
    --config config/rl/resad.yaml \
    --data waymo_preprocessed/ \
    --output out/rl/resad/
```

---

## Stage 5: Chain-of-Thought Reasoning (CoT)

### 5.1 Objective

```
Goal: Generate structured reasoning explaining driving decisions

Prompt → Reasoning → Answer (waypoints)

Example:
---
Input: "Intersection with oncoming traffic"
Reasoning: 
  1. I see vehicles approaching from the left
  2. The lead vehicle is decelerating
  3. Gap is not sufficient to turn
  4. I should wait for the gap
Answer: [waypoints showing vehicle staying in lane]
---
```

### 5.2 ARCoT Decoder

```
┌─────────────────────────────────────────────────────────────┐
│              ARCoT Decoder                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: [features, CoT prompt]                            │
│       ↓                                                     │
│  ┌─────────────────────────────────────┐                  │
│  │   CoT Prompt Encoder              │                  │
│  │   (learned or frozen LLM)        │                  │
│  └─────────────────────────────────────┘                  │
│       ↓                                                    │
│  ┌─────────────────────────────────────┐                  │
│  │   Joint Attention                 │                  │
│  │   (attend to features + prompt)   │                  │
│  └─────────────────────────────────────┘                  │
│       ↓                                                    │
│  ┌─────────────────────────────────────┐                  │
│  │   Dual Output Heads               │                  │
│  │   1. Reasoning text             │                  │
│  │   2. Waypoints                  │                  │
│  └─────────────────────────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 6: RL Refinement

### 6.1 Algorithms Implemented

| Algorithm | Type | Best For | Uncertainty |
|-----------|------|----------|-------------|
| **GRPO** | On-policy | Sparse rewards | No |
| **SAC** | Off-policy | Dense rewards | Entropy |
| **ResAD** | Supervised | Ground truth | σ (explicit) |

### 6.2 GRPO (Group Relative Policy Optimization)

```
Goal: Learn without value function via group-based advantages

Algorithm:
1. Sample K trajectories per state
2. Compute rewards: r₁, r₂, ..., r_K
3. Group relative advantage:
   Aᵢ = (rᵢ - mean(r)) / std(r)
4. Update policy to maximize Aᵢ × ratioᵢ
```

### 6.3 SAC (Soft Actor-Critic)

```
Goal: Maximize expected return + entropy (exploration)

Policy: π(a|s) = argmax E[Q(s,a) + α×H(π(·|s)]

Components:
├── Actor: Gaussian policy μ(s), σ(s)
├── Critic: Twin Q-networks Q₁, Q₂
├── Entropy coefficient: α (auto-tuned)
└── Replay buffer for off-policy learning
```

### 6.4 Integration Point

```
┌─────────────────────────────────────────────────────────────┐
│           RL Refinement Stage                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Frozen SFT model + waypoints                       │
│       ↓                                                    │
│  ┌──────────────────┐                                    │
│  │   Policy Head    │  (trainable)                      │
│  │   (GRPO/SAC)    │                                    │
│  └──────────────────┘                                    │
│       ↓                                                    │
│  Loss: PPO clipped / SAC αH                               │
│       ↓                                                    │
│  Output: Refined waypoints                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 7: Simulation Evaluation (CARLA)

### 7.1 Toy Environment (Fast Iteration)

```
training/rl/toy_waypoint_env.py

For rapid development without CARLA:
├── 2D navigation environment
├── Simple kinematics (bicycle model)
├── Collision detection
└── Progress metrics
```

### 7.2 CARLA Integration

```
┌─────────────────────────────────────────────────────────────┐
│           CARLA Evaluation Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────┐                  │
│  │   CarlaGymEnv                      │                  │
│  │   - Step/Reset interface           │                  │
│  │   - RGB camera + measurements     │                  │
│  │   - Reward shaping               │                  │
│  └─────────────────────────────────────┘                  │
│       ↓                                                      │
│  ┌─────────────────────────────────────┐                  │
│  │   ScenarioRunner                 │                  │
│  │   - Pre-defined scenarios        │                  │
│  │   - Success/fail criteria       │                  │
│  │   - Infraction detection        │                  │
│  └─────────────────────────────────────┘                  │
│       ↓                                                      │
│  ┌─────────────────────────────────────┐                  │
│  │   Metrics Computation             │                  │
│  │   - ADE / FDE                   │                  │
│  │   - Route completion             │                  │
│  │   - Safety violations           │                  │
│  └─────────────────────────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Evaluation Metrics

```python
@dataclass
class EvaluationMetrics:
    # Waypoint metrics
    ade: float  # Average Displacement Error
    fde: float  # Final Displacement Error
    
    # Safety metrics
    collision_rate: float
    lane_invasion_rate: float
    red_light_violations: int
    
    # Comfort metrics
    avg_jerk: float
    max_acceleration: float
    
    # Scenario-specific
    route_completion: float
    success_rate: float
```

---

## Complete Training Command

```bash
# Stage 1: Pre-train (SSL)
python training/pretrain/jepa_pretrain.py \
    --data waymo/ \
    --output out/pretrain/jepa/

# Stage 2: SFT (waypoint BC)
python training/sft/train_waypoint_bc_cot.py \
    --sft-encoder out/pretrain/jepa/encoder.pt \
    --data waymo_processed/ \
    --output out/sft/waypoint_bc/

# Stage 3: ResAD (optional, if ground truth available)
python training/rl/resad_train.py \
    --sft-checkpoint out/sft/waypoint_bc/model.pt \
    --data waymo_processed/ \
    --output out/rl/resad/

# Stage 4: CoT (optional, for reasoning)
python training/sft/train_waypoint_bc_cot.py \
    --use-cot \
    --sft-checkpoint out/sft/waypoint_bc/model.pt \
    --data waymo_processed/ \
    --output out/sft/waypoint_bc_cot/

# Stage 5: RL Refinement (choose one)
python training/rl/grpo_train.py \
    --sft-checkpoint out/sft/waypoint_bc/model.pt \
    --config config/rl/grpo.yaml \
    --output out/rl/grpo/

# OR
python training/rl/sac_train.py \
    --sft-checkpoint out/sft/waypoint_bc/model.pt \
    --config config/rl/sac.yaml \
    --output out/rl/sac/

# Stage 6: Evaluation
python training/rl/eval_toy_waypoint_env.py \
    --sft-checkpoint out/sft/waypoint_bc/model.pt \
    --rl-checkpoint out/rl/grpo/model.pt \
    --episodes 100
```

---

## File Structure

```
AIResearch-repo/
├── training/
│   ├── data/
│   │   ├── unified_dataset.py       # Data loading
│   │   └── waymo_converter.py       # TFRecord → PyTorch
│   ├── models/
│   │   ├── sft/
│   │   │   ├── ar_decoder.py        # AR Decoder
│   │   │   └── train_waypoint_bc_cot.py
│   │   └── rl/
│   │       ├── resad.py            # Residual learning
│   │       ├── grpo.py             # GRPO algorithm
│   │       ├── sac.py              # SAC algorithm
│   │       └── envs/
│   │           ├── carla_gym_env.py         # Fast training
│   │           └── carla_scenario_eval.py   # Evaluation
│   └── rl/
│       ├── train_rl_delta_waypoint.py
│       ├── resad_train.py
│       ├── grpo_train.py
│       └── sac_train.py
├── docs/
│   ├── pipeline/
│   │   └── complete-pipeline.md    # This document
│   └── roadmaps/
│       ├── personalized-e2ead-product-roadmap.md
│       └── engram-memory-mvp-roadmap.md
└── out/
    ├── pretrain/jepa/
    ├── sft/waypoint_bc/
    ├── rl/resad/
    ├── rl/grpo/
    └── rl/sac/
```

---

## Quick Reference: When to Use What

| Scenario | Use This Stage |
|----------|----------------|
| Fast iteration, no CARLA | Toy environment |
| Baseline waypoint prediction | SFT (AR Decoder) |
| Have ground truth, want uncertainty | SFT + ResAD |
| Sparse rewards, need exploration | GRPO |
| Dense rewards, sample efficiency | SAC |
| Need reasoning trace | + CoT |
| Realistic evaluation | CARLA + ScenarioRunner |
| Safety-critical scenarios | ScenarioRunner eval |

---

*Document updated: 2026-02-18*
*Version: 2.0 (complete pipeline)*

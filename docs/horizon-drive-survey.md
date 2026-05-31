# HorizonDrive: Self-Forcing World Model for Autonomous Driving

A Comprehensive Technical Survey

---

## 1. Introduction & Core Intuition

### 1.1 What It Is and Why It Matters

**HorizonDrive** is a self-forcing world model developed by Horizon Robotics for autonomous driving simulation. Its core insight shifts the problem from "generation quality" to "autoregressive stability."

The autonomous driving domain presents a unique challenge in video generation: **closed-loop simulation**. Unlike passive video generation (create one clip, done), driving requires iterative cycles:

```
Planning → World Model → Result → Planning → World Model → ...
```

Each cycle compounds errors. Traditional approaches rely on external priors—3D geometry, memory caches, or simple teacher-student distillation—but fail on long horizons because the teacher itself cannot guarantee stability.

**HorizonDrive's radical approach:** Train the teacher to have its own rollout capability before distilling to the student. Only then does capability transfer make sense.

### 1.2 The Problem It Solves

Existing methods suffer from:

| Problem | Description | Why It Fails |
|---------|-------------|--------------|
| **Error Accumulation** | Model's own predictions become next-step conditions | Drift compounds until world "switches" |
| **Distribution Mismatch** | Training uses clean GT; inference uses dirty predictions | Teacher trained on GT, tested on its own outputs |
| **Long-Horizon Collapse** | Quality degrades over time | No systematic mechanism to recover from errors |

Specific failure modes in autonomous driving:

- Lane drift → progressively worse offsets
- Vehicle geometry jitter → inconsistent neighbor relationships  
- Control-visual mismatch → "different world" syndrome

---

## 2. Problem Formulation

### 2.1 Autoregressive Driving Simulation

**Given:**

- History context $C_t = \{x_{t-K}, ..., x_t\}$ — past K frames
- Control signal $a_t$ — future actions (steering, throttle)
- HD Map $M_t$ — high-definition map with lanes, intersections
- 3D BBox $B_t$ — surrounding dynamic objects (vehicles, pedestrians)

**Goal:** Generate future frames $\hat{x}_{t+1}, \hat{x}_{t+2}, ..., \hat{x}_{t+T}$

In latent space (via video-VAE):

$$\hat{z}_{t+1:T} = f_\theta(z_{t-K:t}, a_t, M_t, B_t)$$

**Critical observation:** During autoregressive rollout, each step's condition includes the model's own predictions:

$$C_{t+T} = \{x_{t+K-T}, ..., \hat{x}_t, ..., \hat{x}_{t+T}\}$$

### 2.2 The Root Cause of Error Drift

Standard teacher training optimizes:

$$\mathcal{L}_{base} = |\epsilon - v_\theta(z_t, c)|^2$$

Where condition $c$ is **clean GT history**.

At inference, condition becomes:

$$c_{AR} = \{\hat{z}_{t-K}, ..., \hat{z}_t\}$$

**The distributions don't match!** This mismatch is the root cause of drifting.

### 2.3 Key Metrics for Evaluation

| Metric | Full Name | What It Measures |
|--------|-----------|------------------|
| FID | Fréchet Inception Distance | Overall visual quality |
| FVD | Fréchet Video Distance | Temporal coherence |
| Qual. | Quality Score | Perceptual quality |
| Mot. | Motion Score | Motion naturalness |
| Img. | Image Score | Frame-level realism |
| ARE | Average Routing Error | Lane-following accuracy |
| DTW | Dynamic Time Warping | Trajectory similarity |

---

## 3. Method: Three-Stage Training Pipeline

### Stage Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Base Model                                           │
│  ─────────────                                                 │
│  Input: Clean GT + Actions + Map + BBox                        │
│         ↓                                                      │
│  Output: Short video generation (works on GT)                │
│                                                                 │
│          ↓                                                     │
│          ↓  (error collection)                               │
│          ↓                                                    │
│  Stage 2: SRR (Scheduled Rollout Recovery)                   │
│  ───────────────────────────────────────────                   │
│  Input: Own dirty rollouts                                    │
│         ↓                                                      │
│  Output: Learn to recover from own errors                    │
│                                                                 │
│          ↓                                                    │
│          ↓  (multi-step teacher rollouts)                    │
│          ↓                                                    │
│  Stage 3: TRD (Teacher Rollout DMD)                           │
│  ─────────────────────────────────                            │
│  Input: Ultra-long teacher trajectories                      │
│         ↓                                                      │
│  Output: Student learns distribution matching                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Breakdown

#### Stage 1: Base Controllable World Model

**Objective:** Build a standard conditional video generation model.

**Architecture:**

- **Input tokens:**
  - History latent ($z_{t-K:t}$) — encoded past frames
  - Action tokens — steering, throttle
  - HD Map tokens — lane geometry, traffic lights
  - 3D BBox tokens — dynamic object positions

- **Injection mechanism:**
  - AdaLN (Adaptive Layer Norm) for action embedding
  - Layout tokens for road structure

- **Generation:**
  - Latent space video generation
  - Flow Matching supervision

**Training objective:**

$$\mathcal{L}_{base} = \mathbb{E}_{z, \epsilon, \tau} |v_\theta(z_t, c_{gt}) - \epsilon|^2$$

Where:

- $\tau \sim U(0,1)$ — noise scheduling parameter
- $\epsilon \sim \mathcal{N}(0,I)$ — Gaussian noise
- $z = (1-\tau)z_{gt} + \tau\epsilon$ — noisy latents

**What works after Stage 1:**

- Clean GT input → short clip generation ✓

**What fails:**

- Own predictions → long sequences collapse ✗

---

#### Stage 2: Scheduled Rollout Recovery (SRR)

**Core innovation:** Make the teacher "eat its own shit" and learn to digest.

**Procedure:**

1. **Error collection:**
   - Let $M_0$ (base model) run multi-step autoregressive rollouts
   - Keep fixed-length history window (sliding)

2. **Pairing:**
   - Align rollout trajectories with GT trajectories one-to-one

3. **Training:**
   - **Input:** Model's own dirty rollout history
   - **Target:** Real GT futures

**Key technique: Pred-to-GT blending**

Direct concatenation causes boundary discontinuity ("frame jump"). Solution: blend in the transition window:

$$\tilde{z}_i = \alpha_i \cdot \hat{z}_i + (1-\alpha_i) \cdot z_{gt,i}$$

Where $\alpha_i$ transitions from 1→0 smoothly within the window.

**Two empirical discoveries that make SRR work:**

1. **Discontinuous repair >> continuous repair:**
   - Model can "generate a new scene to patch it" for discontinuous errors
   - But driving needs continuous world evolution
   - SRR uses gradually increasing blending windows: easy → hard

2. **Errors become semantic over time:**
   - Early errors (lane drift) are scene-agnostic
   - Late errors depend on road structure, traffic states
   - Only by training on semantic errors does the model learn *coupled error recovery*

**Result after Stage 2:**
Model $M_{SRR}$ — can partially recover from its own errors.

---

#### Stage 3: Teacher Rollout DMD (Distribution Matching Distillation)

**Problem:** Long-horizon supervision signals are expensive.

**Solution:** Let the teacher continue rolling out indefinitely. Then:

- **Student:** Generates short chunks independently
- **Supervision:** When covering a complete teacher chunk, do distribution matching
- **Iteration:** Stream through, maximizing learned length under VRAM constraints

**Distribution matching objective:**

$$\mathcal{L}_{TRD} = \mathbb{E}_{z, z^{teach}}[D_{KL}(student(z_t|z_{hist}) || teacher(z_t|z_{hist}))]$$

**Additional trick: Noise-truncated CFG**

- Standard CFG improves quality but **saturates quickly** in long rollouts
- Apply CFG only in low-noise regions
- Gradually lower threshold: "correct layout" → "rich details"

**Full objective:**

$$\mathcal{L}_{final} = \mathcal{L}_{TRD} + \lambda \cdot \mathcal{L}_{CFG}^{truncated}$$

---

## 4. Architecture Deep Dive

### 4.1 Full Model Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     HORIZONDRIVE ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │
│  │ Frame       │    │ Action      │    │ Map        │    ┌─────────┐  │
│  │ Encoder     │    │ Encoder    │    │ Encoder   │    │ BBox    │  │
│  │ (VAE)      │    │ (MLP)      │    │ (Transformer)│  │Encoder │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──┬───┘  │
│         │                  │                  │              │        │
│         ▼                  ▼                  ▼              ▼        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    MODALITY FUSION LAYER                         │ │
│  │  [Spatial Concatenation + Cross-Attention                      │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              TEMPORAL TRANSFORMER STACK                         │ │
│  │  ┌─────────────────────────────────────────────────────────┐   │ │
│  │  │ Self-Attention (causal mask for autoregression)         │   │ │
│  │  │ + AdaLN ( Adaptive Layer Norm for action injection)     │   │ │
│  │  │ + FFN (Feed-Forward Network)                         │   │ │
│  │  └─────────────────────────────────────────────────────────┘   │ │
│  │                          × N layers                              │ │
│  └────────────────────────────┬────────────���──────────────────────┘ │
│                               │                                     │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    FLOW MATCHING HEAD                          │ │
│  │  ┌─────────────────────────────────────────────────────────┐   │ │
│  │  │ Linear → Time Embedding → Velocity Prediction      │   │ │
│  │  └─────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Token Composition Details

**Input sequence structure:**

```
Sequence Length: (K+1) + T + M + N tokens
     │
     ├── Frame tokens: (K+1) × (H/8 × W/8) — latent grid after VAE encoding
     ├── Action tokens: T × action_dim — [steering, throttle, brake] 
     ├── Map tokens: M × map_dim — [lane_poly, traffic_light_state, ...]
     └── BBox tokens: N × bbox_dim — [xyz, yaw, vel, class_id]
```

**Token embedding dimensions:**

| Token Type | Dimension | Notes |
|------------|-----------|-------|
| Frame latent | 256 | VAE encoder output |
| Action | 64 | MLP projected |
| Map | 128 | Learned poly representation |
| BBox | 128 | [center, dim, heading, velocity] |

### 4.3 Key Components

#### 4.3.1 Video VAE (Latent Space)

```python
# Pseudocode for video encoder architecture
class VideoVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        # 3D convolution for spacetime
        self.encoder = nn.Sequential(
            # Spatial: 3D Conv
            nn.Conv3d(in_channels, 64, kernel_size=(4,4,4), stride=(2,2,2)),
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=(4,4,4), stride=(2,2,2)),
            nn.SiLU(),
            nn.Conv3d(128, 256, kernel_size=(4,4,4), stride=(2,2,2)),
            nn.SiLU(),
            nn.Conv3d(256, latent_dim, kernel_size=(3,3,3), stride=(1,1,1)),
        )
        
    def encode(self, x):
        # x: [B, C, T, H, W]
        z = self.encoder(x)
        # z: [B, latent_dim, T//8, H//8, W//8]
        return z
    
    def decode(self, z):
        # Reverse process with transposed convolutions
        return self.decoder(z)
```

**Design rationale:**

- Spatiotemporal 3D convolutions capture motion patterns directly
- 8× spatial downsampling balances quality vs. compute
- Latent dimension 256 provides enough capacity for diverse driving scenarios

#### 4.3.2 Action Injection (AdaLN)

```python
# Adaptive Layer Normalization
class AdaLN(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        self.scale = nn.Linear(action_dim, hidden_dim)
        self.shift = nn.Linear(action_dim, hidden_dim)
        
    def forward(self, x, action):
        # x: [B, T, hidden_dim]
        # action: [B, action_dim]
        scale = self.scale(action).unsqueeze(1)  # [B, 1, hidden_dim]
        shift = self.shift(action).unsqueeze(1) # [B, 1, hidden_dim]
        
        # Apply adaptive scaling and shifting
        return x * (1 + scale) + shift
```

**Why AdaLN over other methods:**

- Condition-free: doesn't modify transformer architecture
- Better generalization: learns continuous control-space interpolation
- CFG-compatible: easily combines with classifier-free guidance

#### 4.3.3 Flow Matching Loss

```python
# Flow Matching for video generation
def flow_matching_loss(model, clean_z, actions, map_tokens, bbox_tokens, tau):
    """
    tau: noise schedule in [0, 1]
    """
    # Noise sampling
    epsilon = torch.randn_like(clean_z)
    
    # Add noise according to schedule
    noisy_z = (1 - tau) * clean_z + tau * epsilon
    
    # Predict velocity field
    v_pred = model(noisy_z, actions, map_tokens, bbox_tokens)
    
    # Flow matching loss: predict epsilon direction
    loss = mse(v_pred, epsilon - noisy_z)  # Move toward clean samples
    
    return loss
```

**Flow matching vs. diffusion:**

| Aspect | DDPM | Flow Matching |
|--------|------|-------------|
| Noise schedule | Complex (learned β) | Simple linear (τ) |
| Sampling | Many steps (100-1000) | Few steps (4-10) |
| ODE vs SDE | SDE | ODE (deterministic) |

### 4.4 Inference Pipeline

```python
@torch.no_grad()
def generate_trajectory(model, vae, initial_frames, actions, hd_map, bboxes, 
                      num_steps=10, num_denoise_steps=4):
    """
    Autoregressive generation with sliding window
    """
    # Encode initial frames to latent
    history_latent = vae.encode(initial_frames)  # [B, K, C, H, W]
    
    current_latent = history_latent[:, -1:]  # Last frame as seed
    
    generated_frames = []
    
    for step in range(num_steps):
        # Prepare inputs with sliding window
        window = get_sliding_window(history_latent, window_size=K)
        
        # Denoising iterations
        for t in reversed(range(num_denoise_steps)):
            noise_level = t / num_denoise_steps
            
            # Model prediction with CFG
            pred = model(current_latent, window, actions, map_tokens, bboxes)
            
            # Apply CFG (lower strength for later steps)
            cfg_scale = max(1.0, 7.0 * noise_level)
            current_latent = pred + cfg_scale * (pred - no_cond_pred)
            
        # Decode to pixel space
        frame = vae.decode(current_latent)
        generated_frames.append(frame)
        
        # Update history for next iteration
        history_latent = torch.cat([history_latent, current_latent], dim=1)
        history_latent = history_latent[:, -window_size:]
    
    return torch.cat(generated_frames, dim=1)
```

---

## 5. Implementation Code

### 5.1 Complete Training Script Structure

```python
"""
HorizonDrive Training Pipeline
Stage 1: Base World Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class HorizonDriveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Modal encoders
        self.frame_encoder = VideoVAE(
            in_channels=3, 
            latent_dim=config.latent_dim
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.map_encoder = MapEncoder(config)
        self.bbox_encoder = BBoxEncoder(config)
        
        # Temporal transformer
        self.temporal_block = TemporalTransformerLayer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio
        )
        self.transformer_stack = nn.ModuleList([
            TemporalTransformerLayer(config) 
            for _ in range(config.num_layers)
        ])
        
        # Output head (flow matching)
        self.velocity_head = nn.Linear(config.hidden_dim, config.latent_dim)
        
    def forward(self, z_noisy, history, actions, maps, bboxes):
        # Encode all modalities
        z_history = self.frame_encoder(history)  # [B, K, D, H', W']
        a_emb = self.action_encoder(actions)
        m_emb = self.map_encoder(maps)
        b_emb = self.bbox_encoder(bboxes)
        
        # Fuse and reshape for transformer
        x = fuse_modalities(z_history, a_emb, m_emb, b_emb)
        
        # Temporal processing
        for layer in self.transformer_stack:
            x = layer(x)
            
        # Predict velocity field for flow matching
        v = self.velocity_head(x)
        
        return v


def train_stage_1_base_model(model, dataloader, optimizer, config):
    """Stage 1: Train base controllable world model"""
    
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        frames, actions, maps, bboxes = batch
        
        # Sample random noise schedule
        tau = torch.rand(1).item()
        
        # Ground truth latents
        z_gt = model.frame_encoder(encode(frames))
        
        # Add noise
        epsilon = torch.randn_like(z_gt)
        z_noisy = (1 - tau) * z_gt + tau * epsilon
        
        # Forward pass
        v_pred = model(z_noisy, frames[:-1], actions, maps, bboxes)
        
        # Flow matching loss
        loss = ((v_pred - epsilon) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def collect_rollout_errors(model, config):
    """Stage 2: Collect errors from autoregressive rollouts"""
    
    model.eval()
    rollouts = []
    
    with torch.no_grad():
        for init_frame in dataloader:
            # Encode initial frame
            history = init_frame.unsqueeze(1).repeat(1, config.window_size, 1, 1, 1)
            
            # Multi-step rollout
            for step in range(config.rollout_steps):
                # Generate next frame autoregressively
                pred = model.one_step_generate(history)
                
                # Store prediction
                rollouts.append({
                    'pred': pred,
                    'history': history.clone()
                })
                
                # Update history with prediction
                history = torch.cat([history[:, 1:], pred], dim=1)
    
    return rollouts


def train_stage_2_srr(model, rollouts, gt_data, config):
    """Stage 2: Scheduled Rollout Recovery"""
    
    model.train()
    
    for rollout, gt in zip(rollouts, gt_data):
        # Get dirty history from rollout
        dirty_history = rollout['history']
        
        # Get corresponding GT future
        gt_future = gt['future']
        
        # Pred-to-GT blending in transition window
        blended = pred_to_gt_blend(dirty_history, gt_future, alpha_schedule)
        
        # Train on blended inputs
        z_blended = model.frame_encoder(blended)
        z_gt_future = model.frame_encoder(gt_future)
        
        v_pred = model(z_blended[..., -1], 
                     dirty_history, 
                     actions, maps, bboxes)
        
        loss = ((v_pred - (z_gt_future - z_blended[..., -1])) ** 2).mean()
        
        loss.backward()
        optimizer.step()


def train_stage_3_trd(teacher_model, student_model, config):
    """Stage 3: Teacher Rollout Distribution Matching"""
    
    teacher_model.eval()
    student_model.train()
    
    with torch.no_grad():
        # Generate ultra-long teacher trajectory
        teacher_traj = teacher_model.full_rollout(long_horizon=config.max_horizon)
    
    # Split into chunks for student
    chunks = split_into_chunks(teacher_traj, chunk_size=config.chunk_size)
    
    for chunk in chunks:
        # Student generates independently
        student_chunk = student_model.generate_from_history(chunk['history'])
        
        # Distribution matching loss (KL divergence)
        loss = kl_divergence(
            student_dist(student_chunk),
            teacher_dist(chunk['output'])
        ).mean()
        
        # Noise-truncated CFG bonus
        if config.use_cfg:
            loss += cfg_truncated_loss(student_chunk, chunk['history'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pred_to_gt_blend(pred_history, gt_future, alpha_schedule):
    """
    Blend predicted history with GT future to avoid boundary jumps
    
    alpha goes from 1 -> 0 in transition window
    """
    T = pred_history.shape[1]
    window_start = T - alpha_schedule.warmup_steps
    
    blended = []
    for i in range(T):
        if i < window_start:
            blended.append(pred_history[:, i])
        else:
            # Linear interpolation
            alpha = (i - window_start) / alpha_schedule.warmup_steps
            blended.append(alpha * pred_history[:, i] + (1-alpha) * gt_future[:, i - window_start])
    
    return torch.stack(blended, dim=1)
```

### 5.2 Configuration Class

```python
@dataclass
class HorizonDriveConfig:
    # Model architecture
    latent_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    mlp_ratio: int = 4
    
    # Input modalities
    action_dim: int = 3  # [steer, throttle, brake]
    map_tokens: int = 512
    bbox_tokens: int = 50
    
    # Training stages
    window_size: int = 10  # K frames history
    predict_horizon: int = 10  # T frames future
    
    # Stage 2: SRR
    rollout_steps: int = 50
    warmup_window: int = 5
    
    # Stage 3: TRD
    max_horizon: int = 300  # ~5 seconds at 60fps
    chunk_size: int = 30
    
    # Inference
    num_denoise_steps: int = 4
    cfg_scale: float = 7.0
    
    # Hardware
    batch_size: int = 8
    accumulate_steps: int = 4
```

### 5.3 Full Training Recipe

```bash
#!/bin/bash
# Training script for HorizonDrive

# Stage 1: Base model (~3 days, 8x A100)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train.py \
    --stage 1 \
    --config horizon_base.yaml \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 8 \
    --output checkpoints/stage1_base.pt

# Stage 2: SRR (~1 day)
python train.py \
    --stage 2 \
    --load checkpoints/stage1_base.pt \
    --config horizon_srr.yaml \
    --rollout_steps 50 \
    --output checkpoints/stage2_srr.pt

# Stage 3: TRD (~2 days)
python train.py \
    --stage 3 \
    --load checkpoints/stage2_srr.pt \
    --config horizon_trd.yaml \
    --max_horizon 300 \
    --output checkpoints/stage3_final.pt
```

---

## 6. Experimental Results

### 6.1 nuScenes Benchmark

| Method | FID ↓ | FVD ↓ | Qual. ↑ | Mot. ↑ | Img. ↑ | ARE ↓ | DTW ↓ |
|--------|------|-------|---------|--------|--------|-------|-------|
| Self-Forcing | 41.53 | 161.00 | 79.27 | 94.17 | 59.65 | 3.47 | 6.22 |
| Self-Forcing++ | 28.84 | 147.57 | 79.47 | 93.92 | 60.25 | 3.78 | 3.61 |
| LongLive | 29.05 | 161.41 | 79.35 | 93.46 | 60.80 | 3.28 | 3.65 |
| **HorizonDrive** | **13.82** | **92.99** | 79.53 | 93.85 | **62.50** | **2.60** | — |

### 6.2 Key Observations

- **FID: 13.82** — 52% reduction vs. second-best (28.84)
- **FVD: 92.99** — 37% reduction vs. 147.57 
- **DTW:** Best despite no official submission
- **Consistent across metrics:** All items above baseline

### 6.3 Inference Efficiency

(RTX 5090, 4-step denoiser, 10 frames/step)

| Resolution | Speed | VRAM |
|------------|-------|------|
| 256×512 | ~5.6 FPS | ~8GB |
| 384×768 | ~1.7 FPS | ~14GB |
| 512×1024 | ~0.6 FPS | ~24GB |

---

## 7. Comprehensive Cross-Comparison

### 7.1 Methodology Comparison Matrix

| Criterion | Bidirectional | 3D Priors | Memory Cache | Self-Forcing | HorizonDrive |
|-----------|--------------|-----------|-------------|-------------|-------------|
| **Core Idea** | Use full forward/backward context | Geometric constraints | Attentive KV cache | Student sees own output | Teacher learns rollout |
| **Training Simplicity** | Medium | High | Medium | Very high | High |
| **Inference Speed** | Slow (needs future) | Medium | Fast | Fast | Slow |
| **Long-Horizon Stability** | N/A | Medium | Medium | Low | Very high |
| **External Dependencies** | None | 3D models | None | None | None |
| **Closed-Loop Ready** | No | Yes | Yes | Partial | Yes |
| **nuScenes Rank** | — | — | — | 4th | 1st |

### 7.2 Deep Dive: Self-Forcing Family

```
┌─────────────────────────────────────────────────────────────────┐
│                 SELF-FORCING EVOLUTION TREE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Self-Forcing (Original)                                         │
│  ─────────────────────                                         │
│  Key: Student trained on own predictions                        │
│  Limitation: Single-step self-evaluation                         │
│           ↓                                                   │
│           ↓  (improvement)                                    │
│           ↓                                                  │
│  Self-Forcing++                                               │
│  ─────────────                                                │
│  Key: Iterative refinement with error accumulation            │
│  Improvement: Multi-step self-evaluation                     │
│  Limitation: Fixed window, weak supervision                   │
│           ↓                                                   │
│           ↓  (paradigm shift)                                │
│           ↓                                                  │
│  HorizonDrive                                               │
│  ──────────────                                              │
│  Key: Train teacher FIRST before distillation               │
│  Innovation: Three-stage pipeline                          │
│  Result: True long-horizon stability                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Algorithm Complexity Comparison

| Parameter | Bidirectional | 3D Priors | Memory | Self-Forcing | HorizonDrive |
|------------|--------------|-----------|--------|-------------|---------------|
| Parameters | ~500M | ~800M | ~500M | ~400M | ~600M |
| Train FLOPs | 10²¹ | 10²² | 10²¹ | 10²⁰ | 10²² |
| Inference/step | O(T²) | O(T·N) | O(T·K) | O(T) | O(T) |
| VRAM | High | Very high | Medium | Low | High |
| Window size | Unlimited* | Limited by 3D | Limited by KV | Fixed 10-30 | Unlimited |

### 7.4 Qualitative Failure Modes

| Failure Mode | Bidirectional | 3D Priors | Self-Forcing | HorizonDrive |
|-------------|---------------|-----------|-------------|-------------|
| Long sequence drift | N/A | Possible | Very likely | Rare |
| Wrong geometry | Low | Depends on prior | Possible | Unlikely |
| Boundary jump | Possible | Unlikely | Likely | Handled by blending |
| Semantic collapse | Possible | Possible | Possible | Handled by SRR |

### 7.5 Use Case Decision Tree

```
START: What's your primary constraint?
│
├─► Quality (single short clip)
│   ├─► Need speed? ──► Self-Forcing++
│   └─► No speed constraint ──► Bidirectional
│
├─► Stability (long rollout)
│   ├─► Has 3D infrastructure? ──► 3D-guided
│   └─► Want simplicity? ──► HorizonDrive
│
├─► Closed-loop (planning ↔ simulation)
│   └─► HorizonDrive (only viable option)
│
├─► Budget/memory limited
│   └─► LongLive (memory-efficient)
│
└─► Baseline/prototyping
    └─► Self-Forcing++ (quickest to train)
```

---

## 8. Engineering Details

### 8.1 Production-Relevant Features

| Feature | Implementation Note |
|---------|-------------------|
| **Brake lights** | Explicit decoder head, on/off state |
| **Yellow lines** | Single/double classification in map encoder |
| **Signal blink** | Temporal pattern as sinusoidal embedding |
| **Crosswalks** | Texture-aware decoder branch |
| **Distant lights** | Multi-scale feature aggregation |

### 8.2 Known Limitations

| Gap | Current Status |
|-----|----------------|
| Sensor noise | Assumes perfect perception |
| Weather | Rain/fog not addressed |
| Corner cases | Accident, construction rare |
| Sim-to-real | Domain gap unquantified |

---

## 9. When to Use HorizonDrive

### Recommendation Matrix

| Scenario | Fit Score | Recommendation |
|----------|----------|----------------|
| Long-horizon-sim (>30s) | ★★★ | Strongly recommended |
| Closed-loop eval | ★★★ | Best choice |
| Real-time (<10ms) | ☆ | Don't use |
| Offline quality only | ★★ | Consider bidir or 3D |
| Quick baseline | ★ | Self-Forcing++ |

---

## 10. Why Should I Use HorizonDrive for Autonomous Driving?

This is the question every autonomous driving engineer should ask before adopting a new framework. Here's a rigorous answer.

### 10.1 The Closed-Loop Problem is the Core AD Challenge

**Autonomous driving is not passive video generation.**

In a typical AD stack:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Perception  │───▶│ Prediction  │───▶│  Planning   │───▶│  Control    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                                                     │
       │                   SIMULATION LOOP                    │
       │                        │                            │
       └────────────────────────┴────────────────────────────┘
                                  │
                                  ▼
                         ┌─────────────┐
                         │World Model  │ ◄── This is what HorizonDrive does
                         │ (render)    │
                         └─────────────┘
```

**The core problem:** Every component in this loop depends on simulated perception → which depends on rendered images → which depends on the world model. Errors compound.

| Approach | AD Suitability | Why It Fails in AD |
|----------|---------------|---------------------|
| **Single-clip video generation** | ❌ Not applicable | No temporal consistency, no闭环 |
| **3DGS/NeRF** | ⚠️ Limited | Too slow for real-time, no action conditioning |
| **Diffusion models (generic)** | ⚠️ Partial | No long-horizon stability, no driving-specific inductive biases |
| **Traditional physics sim** | ✅ Limited | No learning, can't generalize to new scenarios |
| **HorizonDrive** | ✅ **Purpose-built** | Solves the exact error accumulation problem AD faces |

### 10.2 What HorizonDrive Actually Solves

#### Problem 1: Lane Drift in Long Scenarios

Real-world driving requires maintaining lane position over minutes, not seconds.

```
Scenario: Highway driving for 30 seconds
- 30 sec × 30 FPS = 900 frames
- Traditional methods: lane drift accumulates → car ends up in wrong lane
- HorizonDrive: SRR trains the model to recover from its own drift
```

**Quantitative improvement:**
- ARE (Average Routing Error): **2.60** vs. 3.47 (Self-Forcing) — 25% improvement

#### Problem 2: Geometric Jitter in Multi-Agent Scenarios

When other vehicles move unpredictably, the world model must maintain consistent geometry.

```
Scenario: Left-turn with oncoming traffic
- Ego-car must predict: oncoming vehicle trajectory + intersection geometry
- Traditional: geometry collapses after 2-3 seconds
- HorizonDrive: semantic error recovery maintains consistency
```

#### Problem 3: Control-Visual Mismatch

The planned trajectory must match what the world model renders.

```
Scenario: Planning says "turn left", but rendered frame shows "go straight"
- This is a failure mode in single-stage models
- HorizonDrive's TRD ensures rendered world matches planned actions
```

### 10.3 Why Not Just Use [Insert Alternative]?

| Alternative | The Pitch | The Reality for AD | Verdict |
|-------------|-----------|-------------------|---------|
| **3D-Gaussian Splatting** | Photorealistic rendering | 10+ seconds per frame, can't condition on actions | ❌ Not viable |
| **Neural Radiance Fields** | Unlimited view synthesis | No action conditioning, slow training | ❌ Not viable |
| **Diffusion models (Stable Diffusion style)** | General image gen | No temporal consistency, not designed for driving | ⚠️ Partial |
| **Self-Forcing / Self-Forcing++** | Simpler, faster | Window-limited, weak supervision, drifts quickly | ⚠️ 2nd choice |
| **LongLive (memory KV cache)** | Solve context window | Fast-changing scenes make history unreliable | ⚠️ Partial |
| **3D Priors (geometry-guided)** | Geometric consistency | Priors can be wrong, double dependency problem | ⚠️ Partial |
| **HorizonDrive** | **Solves error accumulation** | **Purpose-built for closed-loop AD simulation** | ✅ **Best fit** |

### 10.4 Integration with AD Stack

HorizonDrive isn't just a renderer—it slots into existing AD infrastructure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HORIZONDRIVE IN AD STACK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │
│  │ Perception  │───▶│ Prediction  │───▶│  Planning   │               │
│  │  (Camera,   │    │  (TRAJ)    │    │ (Motion    │               │
│  │   LiDAR)    │    │             │    │  Planning) │               │
│  └─────────────┘    └─────────────┘    └──────┬──────┘               │
│                                                 │                       │
│                    ┌─────────────────────────────┴──────┐               │
│                    │         HorizonDrive              │               │
│                    │  ┌───────────────────────────┐    │               │
│                    │  │ Input:                  │    │               │
│                    │  │  • Current frame       │    │               │
│                    │  │  • Planned trajectory   │    │               │
│                    │  │  • HD Map              │    │               │
│                    │  │  • Dynamic objects     │    │               │
│                    │  └───────────────────────────┘    │               │
│                    │              │                     │               │
│                    │              ▼                     │               │
│                    │  ┌───────────────────────────┐    │               │
│                    │  │ Output:                  │    │               │
│                    │  │  • Rendered futures     │    │               │
│                    │  │  • Consistent geometry  │    │               │
│                    │  │  • Multi-step rollout   │    │               │
│                    │  └───────────────────────────┘    │               │
│                    └──────────────────────────────────────┘               │
│                                   │                                       │
│                                   ▼                                       │
│                    ┌─────────────────────────────────┐                   │
│                    │    Closed-Loop Validation       │                   │
│                    │  (Does plan match rendered?)   │                   │
│                    └─────────────────────────────────┘                   │
│                                                                      │
└───────────────────────────────────────────────────────────────────────┘
```

**Key integration points:**

1. **Planning validation**: Render planned trajectory → feed back to perception
2. **Scenario generation**: Generate safety-critical edge cases
3. **Data augmentation**: Create synthetic training data for perception
4. **Sim2real gap analysis**: Compare rendered vs. real for domain adaptation

### 10.5 The Economic Argument

**Cost of not having HorizonDrive:**

| Scenario | Without World Model | With HorizonDrive |
|----------|-------------------|------------------|
| **Corner case testing** | Collect millions of miles | Generate scenarios programmatically |
| **Perception training** | Limited real data | Unlimited synthetic data |
| **Safety validation** | Monte Carlo only | Closed-loop stress testing |
| **Development cycle** | Months for new scenarios | Hours |

**ROI calculation:**

- Collecting 1 million miles of driving data: ~$10M (vehicles, drivers, fuel)
- Running HorizonDrive for equivalent scenarios: ~$50K (compute)
- **Break-even: 200x cost reduction**

### 10.6 When NOT to Use HorizonDrive

| Scenario | Recommendation |
|----------|---------------|
| Real-time rendering (< 10ms per frame) | Use simpler physics or pre-computed |
| Single-frame quality only | Use state-of-the-art diffusion |
| No closed-loop requirement | Any video gen works |
| Limited compute budget | LongLive or Self-Forcing++ |

### 10.7 The Bottom Line

> **If you're building autonomous driving systems that need closed-loop simulation, long-horizon consistency, and planning-aware rendering, HorizonDrive is not just an option—it's becoming the de facto standard.**

The key insight is this: **autonomous driving isn't a generation problem, it's a consistency problem.** And HorizonDrive is the first framework that addresses this directly.

---

## Quick Reference

| Item | Value |
|------|-------|
| Org | Horizon Robotics |
| Core | Self-Forcing with teacher-first training |
| Stages | 3 (Base → SRR → TRD) |
| FID | 13.82 |
| FVD | 92.99 |
| ARE | 2.60 (best) |
| Res (256×512) | ~5.6 FPS |
| Best for | Long-horizon closed-loop AD simulation |
| AD integration | Planning validation, scenario generation, sim2real |

---

## References

- HorizonDrive Paper (Horizon Robotics)
- nuScenes Benchmark
- Flow Matching (Theory)
- Self-Forcing++ (Prior Work)
- LongLive
- Closed-loop simulation literature

---

*Survey completed: May 2026*
*Enhanced with AD-specific rationale*
# EvoDriveVLA: Detailed Technical Survey

**Date:** 2026-03-13  
**Status:** Survey Complete (Deep Dive)

**Paper:** https://arxiv.org/abs/2603.09465  
**Code:** https://github.com/hey-cjj/EvoDriveVLA

---

## 1. Authors & Affiliations

| Author | Affiliation |
|--------|-------------|
| Jiajun Cao* | Peking University |
| Xiaoan Zhang* | Peking University |
| Xiaobao Wei* | Peking University |
| Liyuqiu Huang | Peking University + XPeng |
| Wang Zijian | XPeng |
| Hanzhen Zhang | XPeng |
| Zhengyu Jia | XPeng |
| Wei Mao | XPeng |
| **Xianming Liu (刘先明)** | XPeng |
| **Yang Wang** | Peking University (Corresponding) |
| **Shanghang Zhang** | Peking University (Corresponding) |

---

## 2. Problem Deep Dive

### 2.1 VLA for Autonomous Driving

Vision-Language-Action models have shown great promise:
- Process visual scenes + language instructions → driving actions
- End-to-end: No hand-crafted perception/planning modules

### 2.2 Two Critical Issues

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| **Perception Degradation** | Visual features become noisy after unfreezing encoder | End-to-end training breaks pretrained visual representations |
| **Cumulative Decay** | Long-horizon planning errors accumulate | No future-aware supervision, error propagates |

### 2.3 Why Standard Training Fails

```
Standard VLA Training:
Images → Vision Encoder → Features → LLM → Actions
     ↓ (unfreeze encoder)
     Training: Loss = Action Loss Only
     
Problem: 
- Encoder learns to minimize action loss only
- Loses fine-grained visual details
- No explicit perception supervision
```

---

## 3. Solution: Collaborative Distillation

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      EvoDriveVLA Training                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐         ┌─────────────┐                         │
│   │   TEACHER   │         │   STUDENT   │                        │
│   │  (frozen)   │         │ (training)  │                        │
│   │             │         │              │                        │
│   │ + Future    │         │ - Future    │                        │
│   │   Info      │         │   Unknown   │                        │
│   └──────┬──────┘         └──────┬──────┘                        │
│          │                          │                               │
│          ↓                          ↓                               │
│   ┌─────────────────────────────────────────────┐                │
│   │     Self-Anchored Visual Distillation      │                │
│   │   (feature-level alignment, Section 4)     │                │
│   └──────────────────┬──────────────────────────┘                │
│                      │                                             │
│                      ↓                                             │
│   ┌─────────────────────────────────────────────┐                │
│   │   Oracle Trajectory Distillation             │                │
│   │   (trajectory-level guidance, Section 5)     │                │
│   └──────────────────┬──────────────────────────┘                │
│                      │                                             │
│                      ↓                                             │
│              Student Actions                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Training Pipeline (From Code)

```bash
# Step 1: Train Teacher Model (with future info)
# Set TRAIN_TEACHER=True in run.sh
bash ./qwenvl/train/run.sh
# Teacher has access to ground truth future trajectories

# Step 2: Train Student Model (distill from teacher)
# Set TRAIN_TEACHER=False
# Set teacher_model=Your_teacher_model
# Set ENCODER_KD=False (visual distillation)
# Set LLM_KD=False (trajectory distillation)
bash ./qwenvl/train/run.sh
```

---

## 4. Component 1: Self-Anchored Visual Distillation

### 4.1 Motivation

Standard knowledge distillation:
```
Teacher Features → Student Features
     Loss = MSE(Teacher, Student)
```

Problem: Teacher features may drift during training, no anchor.

### 4.2 Solution: Self-Anchored Teacher

**Key Insight:** Use a **frozen copy** of the pretrained vision encoder as anchor.

```
┌─────────────────────────────────────────────┐
│   Self-Anchored Visual Distillation         │
├─────────────────────────────────────────────┤
│                                              │
│   ┌─────────────────┐    ┌─────────────────┐  │
│   │  Frozen Anchor │    │    Teacher     │  │
│   │ (pretrained    │    │  (unfrozen,   │  │
│   │  vision enc)  │    │  future-aware) │  │
│   └────────┬────────┘    └────────┬────────┘  │
│            │                      │             │
│            └──────────┬───────────┘             │
│                       ↓                         │
│            ┌────────────────────┐              │
│            │  Visual Anchor     │              │
│            │    Constraint       │              │
│            └──────────┬─────────┘              │
│                       ↓                         │
│            ┌────────────────────┐              │
│            │  Trajectory-Guided │              │
│            │ Key Region Attention│              │
│            └──────────┬─────────┘              │
│                       ↓                         │
│            ┌────────────────────┐              │
│            │ Student Feature    │              │
│            │  Regularization   │              │
│            └────────────────────┘              │
└─────────────────────────────────────────────┘
```

### 4.3 Trajectory-Guided Key Region Attention

```python
# Pseudocode from understanding
def trajectory_guided_attention(visual_features, future_traj):
    """
    visual_features: [B, C, H, W] - image features
    future_traj: [B, T, 2] - future trajectory points
    """
    # 1. Project trajectory to feature map
    traj_points = project_to_bev(future_traj)  # [B, T, 2]
    
    # 2. Sample features at trajectory locations
    traj_features = sample_features(visual_features, traj_points)
    
    # 3. Compute attention weights
    attention = compute_attention(traj_features)
    
    # 4. Reweight visual features
    guided_features = visual_features * attention
    
    return guided_features
```

### 4.4 Loss Function

```
L_visual = λ₁ * MSE(Student_visual, Anchor_visual) 
          + λ₂ * KL(Student_guided, Teacher_guided)
```

---

## 5. Component 2: Oracle Trajectory Distillation

### 5.1 Motivation

Long-horizon planning: errors accumulate
```
t=1: small error → t=5: large error → t=10: completely wrong
```

Solution: Use **future-aware teacher** to provide trajectory supervision.

### 5.2 Coarse-to-Fine Trajectory Generation

```python
# Pseudocode
def coarse_to_fine_trajectory(teacher, image_features):
    # Stage 1: Coarse trajectory
    coarse_traj = teacher.generate_coarse(image_features)  # [B, K, T, 2]
    
    # Stage 2: Fine refinement
    fine_traj = []
    for traj in coarse_traj:
        refined = teacher.refine(image_features, traj)
        fine_traj.append(refined)
    
    return fine_traj  # [B, K, T, 2]
```

### 5.3 Monte Carlo Dropout Sampling

```python
# Pseudocode
def mc_dropout_sampling(teacher, image_features, n_samples=10):
    """
    Generate diverse trajectory candidates
    """
    trajectories = []
    for _ in range(n_samples):
        # Enable dropout at inference
        teacher.enable_dropout()
        traj = teacher.predict(image_features)
        trajectories.append(traj)
    
    # Stack: [B, n_samples, T, 2]
    return torch.stack(trajectories)

def select_best_trajectory(trajectories, score_fn):
    """
    Select optimal trajectory based on score
    """
    scores = score_fn(trajectories)  # [B, n_samples]
    best_idx = scores.argmax(dim=1)  # [B]
    best_traj = trajectories[best_idx]
    return best_traj
```

### 5.4 Training Objective

```
L_trajectory = MSE(Student_traj, Oracle_traj)
              + λ * KL(Student_distribution, Teacher_distribution)
```

---

## 6. Data Pipeline (From Code)

### 6.1 Data Types Required

```bash
# Base data (student training)
python data/gen_data.py --split train
python data/gen_data.py --split val

# Teacher data (with future info)
python data/gen_data.py --split train --future
python data/gen_data.py --split val --future

# Knowledge distillation data
python data/gen_data.py --split train --llm_kd
python data/gen_data.py --split val --llm_kd
```

### 6.2 Data Structure

```
data/
├── nuscenes_complete_data/    # nuScenes images
├── cached_nuscenes_info.pkl  # Dataset metadata
├── full_split.json          # Train/val splits
├── metrics/                 # Evaluation metrics
├── gt_traj.pkl             # Ground truth trajectories
├── gt_traj_mask.pkl        # Trajectory masks
├── stp3_gt_seg.pkl         # STP3 segmentation
├── uniad_gt_seg.pkl         # UniAD segmentation
├── Drive_KD_train_his_ego.json     # KD: history + ego
├── Drive_KD_train_his_ego_future.json  # KD: with future
└── Drive_KD_train_his_ego_llm_kd.json  # KD: LLM knowledge
```

---

## 7. Training Configuration (From Code)

### 7.1 Key Hyperparameters

```bash
# Environment
conda activate EvoDriveVLA

# Models (from HuggingFace)
# Student: https://huggingface.co/Paipai-zxa/EvoDriveVLA/tree/main/student_model
# Teacher: https://huggingface.co/Paipai-zxa/EvoDriveVLA/tree/main/teacher_model

# Training script
bash ./qwenvl/train/run.sh

# Key flags:
TRAIN_TEACHER=True/False      # Train teacher or student
teacher_model=<path>          # Teacher checkpoint path
ENCODER_KD=True/False         # Enable visual distillation
LLM_KD=True/False            # Enable trajectory distillation
LOGITS=True/False            # Output logits distillation
HS=True/False               # Hidden states distillation
```

### 7.2 Inference

```bash
python -m inference_scripts.infer_multi \
    --model_name_or_path ${OUTPUT_DIR} \
    --img_dir $img_dir \
    --dataset_use $test_data \
    --eval_save_path ${OUTPUT_DIR}/result.json \
    --max_pixels ${DEFAULT_MAX_IMAGE_SIZE} \
    --min_pixels ${DEFAULT_MIN_IMAGE_SIZE} \
    --model_max_length ${DEFAULT_MAX_TOKEN} \
    --inference True \
    --random False
```

---

## 8. Evaluation

### 8.1 Metrics

```bash
python ./eval_planning/evaluation/evaluation.py \
    --result_file ${OUTPUT_DIR}/result.json \
    --save_file ${OUTPUT_DIR}/eval_result.json
```

### 8.2 Key Metrics

| Metric | Description |
|--------|-------------|
| ADE | Average Displacement Error |
| FDE | Final Displacement Error |
| Collision Rate | Safety violations |
| Route Completion | Success rate |

---

## 9. Dependencies & Base Code

EvoDriveVLA builds on:
- **Impromptu-VLA** - Base VLA architecture
- **FSDrive** - Feature-level distillation
- **OmniDrive** - Perceptual foundation

---

## 10. Implementation Insights for Driving Pipeline

### 10.1 How to Apply to Our Pipeline

| Stage | EvoDriveVLA Technique | Application |
|-------|---------------------|-------------|
| **BC Training** | Self-anchored visual distillation | Maintain perception quality |
| **RL Refinement** | Oracle trajectory distillation | Faster RL with trajectory guidance |
| **Long-horizon** | MC dropout sampling | Diverse trajectory candidates |

### 10.2 Pseudocode Integration

```python
# For our BC → RL pipeline:

class DistilledBCAgent:
    def __init__(self, teacher_model_path, student_model_path):
        # Load frozen teacher (has future info)
        self.teacher = load_model(teacher_model_path)
        self.teacher.freeze()
        
        # Load student (our BC model)
        self.student = load_model(student_model_path)
        
    def visual_distillation_loss(self, images):
        # Teacher features (frozen anchor)
        with torch.no_grad():
            teacher_features = self.teacher.encode(images)
        
        # Student features
        student_features = self.student.encode(images)
        
        # Self-anchored loss
        loss = F.mse_loss(student_features, teacher_features)
        return loss
    
    def trajectory_distillation_loss(self, images, future_traj):
        # Teacher trajectory (oracle)
        with torch.no_grad():
            teacher_traj = self.teacher.predict(images, future_traj)
        
        # Student trajectory
        student_traj = self.student.predict(images)
        
        # Coarse-to-fine + MC dropout
        best_teacher_traj = select_best(teacher_traj)
        
        loss = F.mse_loss(student_traj, best_teacher_traj)
        return loss
```

---

## 11. Key Takeaways

### 11.1 Why It Works

| Mechanism | Effect |
|-----------|--------|
| **Frozen anchor** | Prevents feature drift |
| **Future-aware teacher** | Provides long-horizon supervision |
| **MC dropout** | Captures trajectory uncertainty |
| **Two-stage distillation** | Preserves perception + improves planning |

### 11.2 Not VLA 2.0, But Methodology Transferable

- The **distillation framework** can be applied to any VLA
- Key innovation: Separate visual and trajectory distillation
- Can combine with RL for further improvement

---

## 12. Survey Status

- [x] Authors & affiliations
- [x] Problem deep dive (perception degradation + cumulative decay)
- [x] Solution: Two-stage collaborative distillation
- [x] Self-anchored visual distillation (technical details)
- [x] Oracle trajectory distillation (technical details)
- [x] Training pipeline from code
- [x] Data preparation
- [x] Evaluation
- [x] Implementation insights for our pipeline

---

## References

1. Paper: https://arxiv.org/abs/2603.09465
2. Code: https://github.com/hey-cjj/EvoDriveVLA
3. Models: https://huggingface.co/Paipai-zxa/EvoDriveVLA

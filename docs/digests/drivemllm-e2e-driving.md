# DriveMLM: LLM-Based End-to-End Autonomous Driving

**Visual Intelligence 2025 | LLM-Augmented | Modular E2E | Closed-Loop**

**Paper:** [arXiv:2312.09245](https://arxiv.org/abs/2312.09245) | **Code:** [github.com/hustvl/DriveMLM](https://github.com/hustvl/DriveMLM) | **Data:** Custom Decision-State Dataset

---

## TL;DR

DriveMLM brings **large language models (LLMs)** into the behavioral planning loop for autonomous driving, achieving **+3.2 pts on CARLA Town05 Long** over Tesla Autopilot and **+4.7 pts** over Apollo. Key innovation: bridges LLM language decisions with vehicle control commands via a standardized **decision-state** interface — enabling plug-and-play integration with existing modular AD systems while maintaining end-to-end differentiable training.

This is distinct from VADv2 (VLM reasoning) — DriveMLM uses a full LLM for behavioral planning, not just semantic understanding.

---

## 1. System Decomposition

### What IS End-to-End
```
Multi-Modal Input (Cam/Lidar) → MLLM Encoder → LLM Behavior Planner → Decision States → Trajectory
        ↑                                                   ↑
    Raw sensor                                   Learned decision states
    (camera + lidar)                             (speed, lane,, yield etc.)
```

### What IS Modular (Not End-to-End)
- **Motion Planning Module:** Off-the-shelf planner generates reference trajectories (not learned)
- **Control Layer:** PID/controller tracks planned trajectory (rule-based)
- **Perception:** Frozen perception backbone (not fine-tuned with LLM)
- **HD Map:** Optional map prior (not learned end-to-end)

### Core Architecture

| Component | Type | Notes |
|-----------|------|-------|
| **Multi-Modal Encoder** | ResNet + PointPillars | Camera + LiDAR feature extraction |
| **LLM Backbone** | Qwen-VL / LLaVA-7B | Frozen or fine-tuned |
| **Decision-State Interface** | Standardized token set | Bridges language → control |
| **Trajectory Decoder** | MLP | Maps decision states to waypoints |
| **Control Adapter** | Rule-based | Converts decisions to steer/throttle |

**Key difference from UniAD/VAD:** Full LLM for planning, not just perception. LLM outputs behavioral decisions (e.g., "slow down", "change lane") that map to trajectories.

---

## 2. Inputs & Outputs

### Inputs
| Input | Shape | Temporal Context |
|-------|-------|------------------|
| **6 surround cameras** | 6×H×W×3 | Current frame only |
| **LiDAR point clouds** | N×4 (x,y,z,intensity) | Current frame only |
| **Navigation command** | Text ("turn left at intersection") | High-level intent |
| **Driving rules** | Text prompts | "Yield to pedestrians", "Follow traffic lights" |
| **User commands** | Text ("take me to the mall") | Destination/route |

### Decision-State Outputs (Key Innovation)
| State | Format | Description |
|-------|--------|-------------|
| **Speed** | Discrete tokens | accelerate / maintain / decelerate / stop |
| **Lane action** | Discrete tokens | keep_lane / lane_left / lane_right |
| **Yield** | Binary | yield / not_yield |
| **Traffic behavior** | Discrete | follow_lane / stop_at_light / stop_at_sign |
| **Explanation** | Natural language | Reasoning for decision |

### Trajectory Outputs
| Output | Format | Planning Horizon |
|--------|--------|-----------------|
| **Future waypoints** | T×2 coordinates (e.g., 40 frames @ 10Hz = 4s) | 2-4 seconds |
| **Control signals** | (steer, throttle, brake) | Direct control |

### Temporal Handling
- **Current frame only:** No explicit temporal modeling in LLM
- **Implicit temporal:** Can prompt LLM with "the car in front is slowing down" for context
- **Future work:** Add temporal attention or memory tokens

---

## 3. Training Objectives

### Primary: Decision-State Classification Loss

```
L_decision = Σ_i CrossEntropy(pred_state_i, gt_state_i)
```

Where states are: speed, lane, yield, traffic_behavior.

### Multi-Modal Alignment Loss

```
L_align = cosine_similarity(visual_features, language_features)
```

Aligns visual encoder outputs with LLM embedding space.

### Trajectory Regression Loss

```
L_traj = ||pred_waypoints - gt_waypoints||²
```

Maps decision states to waypoint coordinates.

### Explanation Generation Loss (Optional)

```
L_explain = -Σ_t log P(token_t | visual, decision, previous_tokens)
```

Trains LLM to generate natural language explanations.

### Total Loss

```
L_total = λ₁·L_decision + λ₂·L_align + λ₃·L_traj + λ₄·L_explain
```

| Loss Component | Type | Weight (λ) |
|----------------|------|------------|
| **L_decision** | Cross-entropy | 1.0 |
| **L_align** | Cosine similarity | 0.5 |
| **L_traj** | MSE | 1.0 |
| **L_explain** | NLL | 0.2 |

---

## 4. Evaluation Protocol & Metrics

### Benchmarks
| Dataset | Type | Notes |
|---------|------|-------|
| **CARLA Town05 Long** | Simulator | 10km route, diverse scenarios |
| **nuScenes** | Real-world | Perception + planning |
| **Custom Decision Dataset** | Synthetic | 50K clips with state annotations |

### Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| **Driving Score (DS)** | CARLA official metric | Higher = better |
| **Route Completion (RC)** | % of route completed | Higher = better |
| **Infraction Score (IS)** | Penalty for violations | Higher = fewer infractions |
| **Decision Accuracy** | State prediction accuracy | Higher = better |
| **BLEU-4 (explanation)** | Language quality | Higher = more fluent |

### Results
| System | Town05 Long (DS) | Improvement |
|--------|------------------|-------------|
| **DriveMLM + Autopilot** | 72.1 | +3.2 vs baseline |
| **DriveMLM + Apollo** | 68.4 | +4.7 vs baseline |
| **Baseline Autopilot** | 68.9 | — |
| **Baseline Apollo** | 63.7 | — |

---

## 5. Tesla/Ashok Claims Mapping

### What Maps Well

| Tesla Claim | DriveMLM Approach | Match? |
|-------------|------------------|--------|
| **Camera-first** | Camera + LiDAR (not camera-only) | Partial |
| **Long-tail handling** | LLM can reason about rare scenarios via prompting | ✅ Strong |
| **End-to-end learning** | LLM → decision states → trajectory (differentiable) | ✅ Strong |
| **Regression testing** | CARLA simulator enables scenario replay | ✅ Strong |
| **Shadow mode** | Can run in parallel with baseline system | ✅ Strong |

### What Doesn't Map

| Tesla Claim | Gap in DriveMLM |
|-------------|-----------------|
| **Pure camera input** | Uses LiDAR (not camera-only) |
| **No HD maps** | Can use HD map priors |
| **Real-time inference** | LLM inference slower than dedicated planners |
| **Online learning** | Static model, no online adaptation |

---

## 6. What to Borrow for AIResearch

### ✅ Directly Applicable

1. **Decision-State Interface**
   - Standardized token set bridging LLM → control
   - Enables interpretable planning with language explanations
   - Location: `planning/decision_states.py`

2. **LLM Planning Framework**
   - Prompt engineering for driving scenarios
   - Multi-modal input handling (camera + LiDAR)
   - Location: `llm_planner/` 

3. **Data Engine Pipeline**
   - Automated decision-state annotation from logs
   - Explanation generation for training
   - Location: `data_engine/`

4. **Evaluation Harness**
   - CARLA closed-loop evaluation
   - Decision accuracy + trajectory metrics
   - Location: `evaluation/`

### 🔄 Needs Adaptation

- **LLM model size:** 7B+ models needed — consider distillation for on-board
- **Inference speed:** LLM is bottleneck — need quantization/distillation
- **Temporal modeling:** Add memory tokens for multi-frame context

### 📋 Implementation Path

```
Month 1-2: Decision-state token set + LLM planning interface
Month 2-3: Data engine for annotation + CARLA integration  
Month 3-4: Fine-tuning on decision dataset + evaluation
```

---

## 7. Citations & Links

### Primary Citation

```bibtex
@article{wang2024drivemlm,
  title={DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving},
  author={Wang, Wenhai and others},
  journal={Visual Intelligence},
  volume={3},
  number={22},
  year={2025},
  publisher={Springer}
}
```

### Related Papers

- **UniAD** (CVPR 2023): Unified perception-prediction-planning — [arXiv:2212.10111](https://arxiv.org/abs/2212.10111)
- **VAD** (ICCV 2023): Vectorized autonomous driving — [arXiv:2303.12077](https://arxiv.org/abs/2303.12077)
- **VADv2** (ICLR 2026): VLM-augmented E2E — [arXiv:2503.00123](https://arxiv.org/abs/2503.00123)
- **DiffusionDrive** (CVPR 2025): Truncated diffusion for planning — [arXiv:2411.15139](https://arxiv.org/abs/2411.15139)

### Resources

- **Code:** https://github.com/hustvl/DriveMLM
- **Paper:** https://arxiv.org/abs/2312.09245
- **Project Page:** (referenced in paper)

---

## Summary

DriveMLM demonstrates that **LLMs can serve as behavioral planners** in autonomous driving systems, bridging the gap between language-level reasoning and vehicle control. The key innovation is the **decision-state interface** that standardizes how LLM outputs map to driving actions. While not as camera-centric as Tesla's approach, it offers superior reasoning capabilities for complex edge cases.

**For AIResearch:** The decision-state interface and LLM planning framework are directly applicable to our waypoint-based pipeline. Consider distilling the LLM for real-time inference.

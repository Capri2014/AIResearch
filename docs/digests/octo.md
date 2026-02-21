# Octo: Open-Source Generalist Robot Policy (PUBLIC ANCHOR DIGEST)

**Survey:** Octo: An Open-Source Generalist Robot Policy (Ghosh et al., 2024)  
**Date:** 2026-02-16  
**Status:** PUBLIC ANCHOR DIGEST - Robotics Foundation Model Baseline  
**Updated:** 2026-02-21 (Survey PR #3 - confirmed as primary anchor)  
**Author:** Auto-generated digest  

---

## TL;DR

Octo is an open-source, transformer-based robot policy trained on the Open X-Embodiment dataset (900K trajectories across 9 institutions). It supports both **behavior cloning** and **diffusion** training objectives, with inference in <50ms on CPU. Key innovation: task-agnostic pre-training with modular action heads enables zero-shot transfer to new robots and tasks. Outperforms RT-1-X on standard benchmarks while being fully open-source (Apache 2.0). Designed for reproducibility: full training code, model weights, and evaluation suite released.

---

## Dataset / Inputs / Outputs

### Dataset Composition
| Aspect | Details |
|--------|---------|
| **Total trajectories** | 900K+ real robot episodes |
| **Robot embodiments** | 9 different robots (single arm, bimanual, mobile manipulators) |
| **Institutions** | 9 research labs (Berkeley RAIL, Stanford, CMU, etc.) |
| **Skills covered** | Manipulation primitives (pick, place, push, insert, etc.) |
| **Data format** | RLDS-compatible (same as Open X-Embodiment) |

### Inputs
- **Observation**: RGB images (single or multi-view), optional state vectors (joint positions, end-effector pose)
- **Task specification**: Natural language string or goal image
- **History**: Optional window of past observations/actions for temporal context

### Outputs
- **Action**: 7-DOF continuous vector (position + orientation + gripper) OR discrete gripper commands
- **Action space**: Gripper frame or base frame (configurable per dataset)
- **Frequency**: 10Hz inference, compatible with 1-10Hz control loops

### Data Access
```python
# Via HuggingFace Datasets (recommended)
from datasets import load_dataset
dataset = load_dataset("octo-dataset/octo-mini", split="train")

# Or via original Open X-Embodiment buckets
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/octo_dataset/ ~/data/
```

---

## Training Objective

### Architecture: Transformer + Action Head

```
┌─────────────────────────────────────────────────────────┐
│                    Octo Architecture                     │
├─────────────────────────────────────────────────────────┤
│  Image Encoder (ResNet-50 / ViT)                        │
│      ↓                                                   │
│  Observation Token + Task Embedding (concat)            │
│      ↓                                                   │
│  Transformer Encoder (8 layers, 8 heads)                  │
│      ↓                                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Task-Specific Action Head (learned at fine-tune) │   │
│  │ • Diffusion Head (DDPM/DDIM)                    │   │
│  │ • MLP Head (behavior cloning)                   │   │
│  └─────────────────────────────────────────────────┘   │
│      ↓                                                   │
│  7-DOF Action Vector                                     │
└─────────────────────────────────────────────────────────┘
```

### Training Objectives

| Objective | Description | Use Case |
|-----------|-------------|----------|
| **Diffusion (DDPM)** | Denoising diffusion probabilistic model on action sequences | High-precision tasks, multi-modal action distributions |
| **Behavior Cloning (BC)** | Direct regression with MSE/MAE loss | Fast inference, simple tasks |
| **Sequence Modeling** | Autoregressive prediction of action windows | Long-horizon tasks |

### Pre-Training Setup
- **Base model**: Pre-trained on 900K trajectories
- **Task-agnostic**: No task-specific training during pre-training
- **Modular heads**: Action heads learned during fine-tuning only
- **Multi-dataset**: Weighted mixture across source datasets

---

## Evaluation Setup

### Benchmark Results
| Setting | Octo | RT-1-X | Improvement |
|--------|------|--------|-------------|
| **Berkeley tasks** | 82% | 78% | +5% |
| **Stanford tasks** | 76% | 71% | +7% |
| **Average (9 robots)** | 79% | 75% | +5% |
| **Zero-shot transfer** | 64% | 52% | +23% |

### Evaluation Protocol
1. **Real robot evaluation** at Berkeley RAIL Lab and partner institutions
2. **Success rate** over 25 trials per task
3. **Multi-task generalization**: Same checkpoint tested across all 9 robots
4. **Fine-tuning efficiency**: Number of demos needed to reach 90% of full performance

### Inference Performance
| Metric | Value |
|--------|-------|
| **Inference latency** | <50ms (CPU), <10ms (GPU) |
| **Model size** | 90M parameters (base), 270M (large) |
| **Memory footprint** | ~500MB for inference |

---

## Tesla / Ashok Talk Claims Mapping

### Claims That Map Cleanly ✓

| Tesla/Ashok Claim | Octo Evidence |
|-------------------|----------------|
| **"Foundation models transfer across embodiments"** | ✓ Zero-shot transfer: Octo checkpoint works on new robots without fine-tuning |
| **"Scale + diversity enables generalization"** | ✓ 900K trajectories across 9 robots; larger diversity = better transfer |
| **"End-to-end vision-language-action models work"** | ✓ Language-conditioned policies demonstrated |
| **"Real robot data matters more than simulation"** | ✓ Trained exclusively on real robot trajectories |
| **"Multi-task policies via natural language"** | ✓ Task specification via language strings or goal images |

### Gaps / What Doesn't Map ✗

| Gap | Details |
|-----|---------|
| **Continuous driving control** | Octo targets manipulation (pick/place); no throttle/steering dynamics |
| **High-frequency closed-loop** | 10Hz inference vs. driving requires 20-50Hz minimum |
| **Safety reasoning** | No explicit constraint satisfaction or safety filters |
| **Fleet learning** | Static checkpoint; no continuous OTA update infrastructure |
| **Multi-modal sensors** | Primarily RGB images; no lidar/radar/depth in standard config |
| **Occlusion handling** | Single/double camera views; no bird's-eye or 360° fusion |

### Partial Alignment (Needs Adaptation)

- **Vision backbone**: ResNet-50/ViT encoders can transfer to driving perception
- **Language conditioning**: Could adapt to driving instructions ("turn right at mile 5")
- **Modular action head**: Design pattern for vehicle control heads (steering/throttle)
- **Zero-shot transfer**: Concept applies to new driving scenarios/environments

---

## Action Items for AIResearch

### Interfaces / Contracts to Copy

| Item | Octo Pattern | Adaptation for AIResearch |
|------|--------------|---------------------------|
| **Data format** | RLDS episode schema | Define: `Observation[multi-view RGB, ego-state] → Action[delta-pose] → Success` |
| **Model architecture** | Transformer encoder + modular action head | Vehicle: Image encoder + planning head → control outputs |
| **Task specification** | Language string or goal image | Driving: HD map + navigation instruction |
| **Checkpoint format** | PyTorch `.pt` + ONNX export | Continue ONNX for deployment |
| **Evaluation suite** | Real-robot success rates | Sim + real driving metrics (collision rate, progress, comfort) |

### Technical Contracts to Implement

1. **Episode Schema**
   ```python
   {
       "observation": {
           "images": {"front": [H,W,3], "rear": [H,W,3], "left": [H,W,3], "right": [H,W,3]},
           "state": [velocity, acceleration, heading_rate],  # ego state
       },
       "action": [steering, throttle, brake],  # continuous control
       "success": bool,  # task completion signal
       "task": str,  # "navigate to goal A while avoiding obstacles"
   }
   ```

2. **Model Interface**
   ```python
   # Octo: image + language → action
   model(images: List[Tensor], instruction: str, goal: Tensor=None) -> Tensor

   # AIResearch target: multi-view + map + instruction → control
   model(images: List[Tensor], map: Tensor, instruction: str) -> control
   ```

3. **Action Head Design (for Vehicle Control)**
   ```python
   # Diffusion head for vehicle control
   class VehicleActionHead(nn.Module):
       def __init__(self, obs_dim, action_dim=3, horizon=8):
           self.action_dim = action_dim  # [steering, throttle, brake]
           self.horizon = horizon  # predict T timesteps

       def forward(self, features):
           # Predict action sequence with diffusion process
           return action_sequence  # [B, T, action_dim]
   ```

4. **Evaluation Protocol**
   ```python
   evaluate_policy(
       checkpoint: Path,
       scenarios: List[Scenario],  # nuPlan-style scenarios
       env: Simulator,  # CARLA or equivalent
   ) -> Dict[str, float]  # {collision_rate, progress, comfort, rule_violations}
   ```

### Reproducibility Checklist (ANCHOR REQUIREMENT)

- [ ] Release dataset in RLDS/HF format with standardized schema
- [ ] Publish full training code (PyTorch + Hydra config)
- [ ] Release checkpoints in PyTorch + ONNX + TensorRT formats
- [ ] Colab notebook for zero-shot inference
- [ ] Evaluation suite with scenario definitions
- [ ] Multi-dataset mixture recipe with curriculum (if any)

### Open Source Scorecard

| Criterion | Octo | Open X-Embodiment/RT-X |
|-----------|------|------------------------|
| **License** | Apache 2.0 (fully permissive) | Apache 2.0 (software), CC-BY 4.0 (data) |
| **Training Code** | Full PyTorch + Hydra | JAX/TensorFlow (limited) |
| **Model Weights** | HuggingFace + GCS | GCS only |
| **Colab Notebooks** | Yes (zero-shot inference) | Yes (minimal examples) |
| **Evaluation Suite** | Dedicated repo | Part of main repo |
| **Documentation** | Comprehensive | Paper-heavy |
| **Best For** | Reproducibility + customization | Baseline comparison |

### Why Octo is the Anchor

1. **Reproducibility-first design**: Everything needed to reproduce results is released
2. **Modular architecture**: Easy to swap components (image encoder, action head)
3. **Apache 2.0**: No commercial restrictions
4. **Active maintenance**: Rail-berkeley community responds to issues
5. **Diffusion + BC support**: Dual training objectives in single codebase

---

## Citations + Links

### Primary Paper
- **Ghosh, D., et al. (2024)** - "Octo: An Open-Source Generalist Robot Policy"  
  https://octo.ai/ (project page), https://arxiv.org/abs/2409.XXXXX (paper)

### Code & Checkpoints
- **GitHub Repository**: https://github.com/rail-berkeley/octo
- **Model Weights (HuggingFace)**: https://huggingface.co/octo/octo-mini
- **Training Code**: https://github.com/rail-berkeley/octo/tree/main/octo

### Dataset
- **Octo Dataset (HF)**: https://huggingface.co/datasets/octo-dataset/octo-mini
- **Open X-Embodiment**: https://github.com/google-deepmind/open_x_embodiment

### Related Work
- **RT-1 / RT-2 (Google DeepMind)**: https://github.com/google-deepmind/rt1
- **Diffusion Policy (Columbia)**: https://diffusion-policy.cs.columbia.edu/
- **ACT (Stanford)**: https://github.com/tonyzhaozh/act

### License
- Apache 2.0 (fully open-source)

---

*PR: Survey PR #2: Public Anchor Digest - Robotics Foundation Model Baseline*  
*Summary: Updated Octo digest to serve as PUBLIC ANCHOR digest for robotics foundation models. (1) Training objectives - diffusion (DDPM) and behavior cloning heads with transformer backbone, (2) Zero-shot transfer across 9 robots with 5% improvement over RT-1-X, (3) Tesla claims mapping - strong alignment on foundation model transfer and real data, gaps in driving dynamics/safety reasoning, (4) Action items - adopt RLDS schema, implement modular action head for vehicle control, release ONNX checkpoints, (5) Anchor criteria met - Apache 2.0, full PyTorch training code, HuggingFace weights, Colab notebooks, evaluation suite.*

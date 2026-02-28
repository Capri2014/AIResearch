# Open X-Embodiment & RT-X: Public Anchor Digest — Robotics Foundation Model Baseline

**Survey:** Open X-Embodiment: Robotic Learning Datasets and RT-X Models (Open X-Embodiment Collaboration, 2023-2025)  
**Date:** 2026-02-28  
**Status:** PUBLIC ANCHOR DIGEST - Robotics Foundation Model Baseline  
**Author:** Auto-generated digest  

---

## TL;DR

Open X-Embodiment is the **largest open-source robotic manipulation dataset** to date (1M+ real robot trajectories across 22 robot embodiments from 21 institutions). It provides standardized RLDS data formats and two trained models: **RT-1-X** (efficient 35M-param Transformer) and **RT-2-X** (vision-language-action model with 55B parameters). Key finding: **cross-embodiment transfer works** — models trained on diverse robots outperform single-robot baselines by **50% in low-data regimes**, demonstrating that scale + diversity enables positive transfer across robot morphologies.

**Why this is the anchor:** Google DeepMind's official release includes full dataset in standardized format, model checkpoints (JAX + TensorFlow), and inference Colabs. Octo builds on this with better open-source training code, but RT-X is the foundational release that enabled the entire ecosystem.

---

## Dataset / Inputs / Outputs

### Dataset Composition

| Aspect | Details |
|--------|---------|
| **Total trajectories** | 1M+ real robot episodes |
| **Robot embodiments** | 22 different robots (single arm, bi-manual, quadrupeds, mobile manipulators) |
| **Institutions** | 21 research labs globally (Berkeley, Stanford, CMU, ETH, etc.) |
| **Skills covered** | 527 distinct manipulation skills |
| **Tasks** | 160,266 task instances |
| **Data sources** | 60 existing robot datasets pooled and standardized |

### Standardized RLDS Format

- **Observation Space:**
  - RGB images from workspace cameras (320x320, single view)
  - No wrist/hand cameras, no depth in current release
  - Optional state vectors (joint positions, end-effector pose)

- **Action Space:**
  - 7-DOF continuous vector: `[x, y, z, roll, pitch, yaw, gripper]`
  - Represented in gripper frame (absolute or delta modes)
  - Supports both position and velocity control modes

- **Task Specification:**
  - Natural language strings describing the goal
  - Example: "pick up the red cup", "move pepper to tray"

### Data Access

```python
# Via TensorFlow Datasets (recommended)
import tensorflow_datasets as tfds
ds = tfds.load('oxe_dataset_name', split='train')

# Manual download via GS bucket
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/{dataset_name} ~/tensorflow_datasets/

# Dataset spreadsheet with metadata
# https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/
```

---

## Training Objective

### RT-1-X Architecture (Efficient Transformer)

| Component | Details |
|-----------|---------|
| **Model size** | ~35M parameters |
| **Image encoder** | EfficientNet-B0 + TokenLearner |
| **Language encoder** | Universal Sentence Encoder |
| **Architecture** | Transformer encoder (8 layers, 6 heads) |
| **Output** | 7-DOF action head (MLP) |
| **Inference frequency** | 3Hz (333ms per step) |

**Training:** Supervised behavior cloning with MSE loss on action prediction.

### RT-2-X Architecture (Vision-Language-Action)

| Component | Details |
|-----------|---------|
| **Model size** | 55B parameters (VLA co-fine-tuned) |
| **Base model** | PaLI-X or ViT-Ego |
| **Action representation** | Actions as natural language tokens |
| **Training** | Joint fine-tuning on robotics data + VLM objectives |
| **Emergent capabilities** | Spatial reasoning, preposition understanding |

**Key insight:** By representing actions as text tokens ("move to x=0.1, y=0.2..."), RT-2-X inherits reasoning capabilities from pre-trained VLMs.

### Training Mixture

- **Multi-dataset training:** Weighted sampling across 60 source datasets
- **No explicit curriculum:** All data mixed from start
- **Standardization:** RLDS format ensures consistent observation/action spaces

---

## Evaluation Setup

### RT-1-X Evaluation (In-Distribution)

| Setting | Result |
|---------|--------|
| **Evaluation sites** | 6 academic labs (Berkeley RAIL, Freiburg AiS, NYU CILVR, Stanford IRIS, USC CLVR, Berkeley AUTOLab) |
| **Small-data regime (<500 examples)** | RT-1-X outperforms single-dataset baselines by **50%** |
| **Zero-shot transfer** | Works on new robots without fine-tuning |

### RT-2-X Evaluation (Emergent Skills)

| Setting | Result |
|---------|--------|
| **Emergent capabilities** | Novel spatial reasoning tasks not seen in training |
| **Preposition understanding** | Distinguishes "move apple ON cloth" vs "move apple NEAR cloth" |
| **Comparison vs RT-2** | RT-2-X outperforms RT-2 by **3x** on emergent skill benchmarks |
| **Real robot demos** | Successfully generalizes to unseen tasks in academic labs |

### Evaluation Protocol

- Real robot execution (not simulation) at partner institutions
- 3Hz control frequency
- Success rate over N trials per task (typically 20-50 trials)
- Multi-embodiment transfer: same checkpoint tested on different robots

---

## Tesla / Ashok Talk Claims Mapping

### Claims That Map Cleanly ✓

| Tesla/Ashok Claim | RT-X Evidence |
|-------------------|----------------|
| **"Foundation models transfer across embodiments"** | ✓ RT-1-X improves on new robots without any fine-tuning; positive transfer demonstrated across 22 embodiments |
| **"Scale + diversity enables generalization"** | ✓ 1M+ trajectories across diverse robots; 50% small-data improvement directly from diversity |
| **"End-to-end vision-language-action models work"** | ✓ RT-2-X treats actions as language tokens; shows emergent reasoning and spatial understanding |
| **"Real robot data matters more than simulation"** | ✓ Entire dataset is real robot trajectories; no sim-to-real gap in training |
| **"Multi-task policies via natural language"** | ✓ Task specification via language strings; RT-2-X shows compositional understanding |

### Gaps / What Doesn't Map ✗

| Gap | Details |
|-----|---------|
| **Continuous control for driving** | RT-X targets discrete manipulation (pick/place); no continuous throttle/steering dynamics |
| **High-frequency closed-loop** | 3Hz policy (333ms) vs autonomous driving requires 20-50Hz minimum |
| **Safety reasoning / constraint satisfaction** | No explicit safety filters or collision avoidance |
| **Fleet learning / OTA updates** | Static checkpoint; no continuous learning infrastructure |
| **Multi-modal sensors** | Primarily RGB images; no lidar/radar/depth in standard config |
| **Occlusion handling** | Single/double camera views; no bird's-eye or 360° fusion |
| **Long-horizon planning** | Short action horizons (~1-2s); driving requires longer horizons |

### Partial Alignment (Needs Adaptation)

| Component | RT-X Pattern | Adaptation for Driving |
|-----------|-------------|------------------------|
| **Language-conditioned policies** | Task via language strings | Driving commands: "turn left at mile 5" |
| **Vision backbone reuse** | EfficientNet/RN50 encoders | Transfer to driving perception |
| **Action head design** | 7-DOF gripper control | Vehicle: steering, throttle, brake |
| **Positive transfer** | Cross-robot generalization | Cross-environment/scene transfer |
| **Diffusion for actions** | RT-2 uses token-based actions | Could model uncertainty in driving |

---

## Action Items for AIResearch

### Interfaces / Contracts to Copy

| Item | RT-X Pattern | Adaptation for AIResearch |
|------|--------------|---------------------------|
| **Data format** | RLDS episode schema | Define: `Observation[multi-view RGB, ego-state] → Action[delta-pose] → Success` |
| **Model architecture** | Transformer encoder + action head | Vehicle: Image encoder + planning head → control outputs |
| **Task specification** | Language string | Driving: HD map + navigation instruction |
| **Checkpoint format** | TensorFlow SavedModel + JAX | Export ONNX for deployment |
| **Evaluation suite** | Multi-lab real-robot success rates | Sim + real driving metrics |
| **Dataset mixture** | Weighted sampling across 60 datasets | Define scenario mixture: highway/urban/intersection |

### Technical Contracts to Implement

**1. Standardized Episode Schema**
```python
{
    "observation": {
        "image": {"front": [H,W,3], "rear": [H,W,3]},  # RGB cameras
        "state": [velocity, acceleration, heading],   # ego state
    },
    "action": [steering, throttle, brake],  # vehicle control
    "success": bool,  # task completion
    "task": str,      # "navigate to goal A avoiding obstacles"
}
```

**2. Model Interface**
```python
# RT-X: image + language → action
model(image: Tensor, task: str) -> Tensor[7]

# AIResearch target: multi-view + map + instruction → control
model(images: List[Tensor], map: Tensor, instruction: str) -> control
```

**3. Action Head for Vehicle Control**
```python
class VehicleActionHead(nn.Module):
    def __init__(self, obs_dim, action_dim=3, horizon=8):
        self.action_dim = action_dim  # [steering, throttle, brake]
        self.horizon = horizon

    def forward(self, features):
        # Predict action sequence (diffusion or MLP)
        return action_sequence  # [B, T, action_dim]
```

### Reproducibility Checklist (ANCHOR REQUIREMENT)

- [x] Dataset in RLDS format (via TFDS)
- [x] Model checkpoints (JAX + TensorFlow)
- [x] Colab notebooks for inference
- [x] Dataset spreadsheet with metadata
- [ ] Full training code (limited - JAX/TF only)
- [ ] ONNX export for deployment

### Open Source Scorecard

| Criterion | Open X-Embodiment/RT-X | Octo |
|-----------|------------------------|------|
| **License** | Apache 2.0 (software), CC-BY 4.0 (data) | Apache 2.0 |
| **Training Code** | JAX/TensorFlow (limited) | Full PyTorch + Hydra |
| **Model Weights** | GCS only | HuggingFace + GCS |
| **Colab Notebooks** | Yes | Yes |
| **Baseline for** | Foundation dataset release | Reproducibility |

### Why This is the Anchor

1. **Foundational dataset:** 1M+ trajectories across 22 robots enabled the entire field
2. **Official release:** Google DeepMind's official models and data
3. **Standardized format:** RLDS schema adopted by community (Octo, etc.)
4. **Cross-embodiment proof:** Demonstrates transfer learning across robot morphologies
5. **Emergent capabilities:** RT-2-X shows VLA reasoning transfers to robotics

---

## Citations + Links

### Primary Paper
- **Open X-Embodiment Collaboration (2023-2025)** - "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"  
  https://robotics-transformer-x.github.io/  
  https://arxiv.org/abs/2310.08864

### Code & Checkpoints
- **GitHub Repository**: https://github.com/google-deepmind/open_x_embodiment
- **RT-1-X JAX Checkpoint**: `gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax/`
- **RT-1-X TF Inference Colab**: https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb
- **Dataset Colab**: https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb

### Dataset Resources
- **Dataset Spreadsheet**: https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/
- **RLDS Format Spec**: https://github.com/google-research/rlds

### Related Work
- **RT-1**: Brohan et al., "RT-1: Robotics Transformer for Real-World Action at Scale" (2022)
- **RT-2**: Brohan et al., "RT-2: Vision-Language-Action Models" (2023)
- **Octo**: Ghosh et al., "Octo: An Open-Source Generalist Robot Policy" (2024) - builds on Open X-Embodiment

### License
- Software: Apache 2.0
- Dataset: CC-BY 4.0

---

*PR: Survey PR #3 (Public Anchor Digest - Robotics Foundation Model Baseline)*  
*Summary: Updated Open X-Embodiment digest as anchor baseline covering: (1) 1M+ trajectories across 22 robot embodiments, standardized RLDS format, (2) RT-1-X (35M Transformer) + RT-2-X (55B VLA) architectures, (3) 50% improvement over single-robot baselines, RT-2-X shows 3x emergent skill gains, (4) Tesla/Ashok mapping: cross-embodiment transfer proven, VLA reasoning works, gaps: no driving control, 3Hz too slow, no safety constraints, (5) Action items: adapt 7-DOF to vehicle control, implement ONNX export, replicate multi-site evaluation.*

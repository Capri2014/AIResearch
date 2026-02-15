# Open X-Embodiment & RT-X Digest

**Survey:** Open X-Embodiment: Robotic Learning Datasets and RT-X Models (Open X-Embodiment Collaboration, 2023-2025)  
**Date:** 2026-02-15  
**Author:** Auto-generated digest  

---

## TL;DR

Open X-Embodiment is the largest open-source robotic manipulation dataset to date (1M+ real robot trajectories across 22 robot embodiments from 21 institutions). It provides standardized data formats and two trained models: **RT-1-X** (efficient Transformer) and **RT-2-X** (vision-language model co-fine-tuned to output actions as natural language tokens). Key finding: cross-embodiment transfer works—models trained on diverse robots outperform single-robot baselines by 50% in low-data regimes, demonstrating that scaling + diversity enables positive transfer across robot morphologies.

---

## Dataset / Inputs / Outputs

### Dataset Composition
| Aspect | Details |
|--------|---------|
| **Total trajectories** | 1M+ real robot episodes |
| **Robot embodiments** | 22 different robots (single arm, bi-manual, quadrupeds) |
| **Institutions** | 21 research labs globally |
| **Skills covered** | 527 distinct manipulation skills |
| **Tasks** | 160,266 task instances |
| **Data sources** | 60 existing robot datasets pooled and standardized |

### Standardized Format
- **RLDS (Reinforcement Learning Datasets) episode format**: Each episode is a sequence of (observation, action, reward) tuples
- **Observations**: RGB images from workspace cameras (note: no wrist/hand cameras, no depth in current release)
- **Actions**: 7D vector (x, y, z, roll, pitch, yaw, gripper opening) in gripper frame; absolute or delta modes supported
- **Task specification**: Natural language strings describing the goal

### Access
```python
# Via TensorFlow Datasets
import tensorflow_datasets as tfds
ds = tfds.load('oxe_dataset_name', split='train')

# Manual download via GS bucket
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/{dataset_name} ~/tensorflow_datasets/
```

Dataset spreadsheet with metadata: https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/

---

## Training Objective

### RT-1-X Architecture
- **Type**: Efficient Transformer-based policy (following RT-1 architecture)
- **Inputs**: Single RGB image (320x320) + task string embedded via TokenLearner
- **Outputs**: 7-DOF continuous action vector
- **Training**: Supervised behavior cloning on the mixed dataset
- **Key design**: Image tokenizer + Transformer encoder + action head

### RT-2-X Architecture
- **Type**: Vision-Language Model (VLM) co-fine-tuned to output robot actions
- **Inputs**: RGB image + natural language task description
- **Outputs**: Actions represented as natural language tokens (e.g., "move to position x=0.1...")
- **Training**: Joint fine-tuning on robotics data + VLM pretraining objectives
- **Key insight**: Emergent capabilities from VLM-scale model (55B parameters)

### Training Mixture
- Multi-dataset training with weighted sampling across 22 embodiments
- Curriculum: no explicit curriculum, all data mixed from start
- Environment: Standardized across datasets via RLDS format

---

## Evaluation Setup

### RT-1-X Evaluation
| Setting | Result |
|---------|--------|
| **In-distribution skills** | Evaluated at 6 academic labs (Berkeley RAIL, Freiburg AiS, NYU CILVR, Stanford IRIS, USC CLVR, Berkeley AUTOLab) |
| **Small-data regime (<500 examples)** | RT-1-X outperforms single-dataset baselines by **50%** |
| **Success rates** | Published per-lab, per-task breakdown in paper |

### RT-2-X Evaluation (Emergent Skills)
| Setting | Result |
|---------|--------|
| **Emergent capabilities** | Novel spatial reasoning tasks not seen in training |
| **Preposition understanding** | Distinguishes "move apple ON cloth" vs "move apple NEAR cloth" |
| **Comparison** | RT-2-X outperforms RT-2 by **3x** on emergent skill benchmarks |

### Evaluation Protocol
- Real robot execution (not simulation) at partner institutions
- 3Hz control frequency
- Success rate over N trials per task
- Multi-embodiment transfer: same checkpoint tested on different robots

---

## Tesla / Ashok Talk Claims Mapping

### Claims That Map Cleanly ✓

| Tesla/Ashok Claim | RT-X Evidence |
|-------------------|----------------|
| **"Foundation models transfer across embodiments"** | ✓ RT-1-X improves on new robots without any fine-tuning on that robot; positive transfer demonstrated across 22 embodiments |
| **"Scale + diversity enables generalization"** | ✓ 1M+ trajectories across diverse robots; small-data gains (50% improvement) directly from diversity |
| **"End-to-end vision-language-action models work"** | ✓ RT-2-X treats actions as language tokens; shows emergent reasoning |
| **"Real robot data matters more than simulation"** | ✓ Entire dataset is real robot trajectories; no sim-to-real gap in training |
| **"Multi-task policies via natural language"** | ✓ Task specification via language strings; RT-2-X shows compositional understanding |

### Gaps / What Doesn't Map ✗

| Gap | Details |
|-----|---------|
| **Continuous control for driving** | RT-X targets discrete manipulation (pick/place); no continuous throttle/steering dynamics |
| **High-frequency perception** | 3Hz policy (333ms) vs autonomous driving requires 10-20Hz minimum |
| **Safety constraints** | No explicit safety reasoning or constraint satisfaction in RT-X |
| **Fleet learning at scale** | RT-X is a static checkpoint; Tesla's claims involve continuous over-the-air updates |
| **Occlusion handling** | Single camera view only; no bird's-eye or multi-view fusion |

### Partial Alignment (Needs Adaptation)

- **Language-conditioned policies**: RT-2-X shows this works for manipulation; could adapt to driving commands ("turn left at intersection")
- **Vision backbone reuse**: EfficientNet/RN50 image encoders could transfer; action heads need redesign for vehicle dynamics
- **Positive transfer claim**: Holds in robotics; likely holds in autonomous driving (similar reasoning for cross-environment generalization)

---

## Action Items for AIResearch

### Interfaces / Contracts to Copy

| Item | RT-X Pattern | Adaptation for AIResearch |
|------|--------------|---------------------------|
| **Data format** | RLDS episode format (TFDS-native) | Define similar schema: `Observation[image, state] → Action[delta] → Reward` |
| **Action representation** | 7-DOF gripper frame vector | Vehicle: `[steering, throttle, brake]` or delta pose in ego frame |
| **Task specification** | Natural language strings | Driving instructions: "turn right at Main St", "follow lead vehicle" |
| **Checkpoint format** | TensorFlow SavedModel + JAX params | Export ONNX or TorchScript for inference deployment |
| **Evaluation protocol** | Multi-lab real-robot evaluation | Multi-site (sim + real) autonomous driving benchmarks |
| **Dataset mixture** | Weighted sampling across 60 source datasets | Define driving scenario mixture: highway, urban, intersection, etc. |

### Technical Contracts to Implement

1. **Standardized Episode Schema**
   ```python
   # Mimic RLDS structure
   {
       "observation": {
           "image": {"bytes": [H, W, 3]},  # RGB
           "state": [14],  # joint positions / ego state
       },
       "action": [7],  # [x, y, z, roll, pitch, yaw, gripper]
       "reward": float,  # sparse binary success
       "task": str,      # "pick up the red cup"
   }
   ```

2. **Model Interface**
   ```python
   # RT-1-X signature: image + language → action
   model(image: Tensor[1, 320, 320, 3], task: str) -> Tensor[1, 7]
   
   # AIResearch target: multi-view images + instruction → control
   model(images: List[Tensor], instruction: str) -> control_signal
   ```

3. **Evaluation Hook**
   ```python
   # Standard success metric per task
   evaluate_policy(
       checkpoint: Path,
       tasks: List[str],  # language-specified tasks
       env: GymEnv,        # real or sim
   ) -> Dict[str, float]  # success_rates per task
   ```

### Reproducibility Checklist

- [ ] Publish dataset in RLDS-compatible format (TFDS or equivalent)
- [ ] Release model checkpoints (ONNX + training code)
- [ ] Colab notebook for inference (following RT-X example)
- [ ] Evaluation protocol documentation (success rate definition, trial count)
- [ ] Multi-dataset mixture recipe (weights, curriculum if any)

---

## Citations + Links

### Primary Paper
- **Open X-Embodiment Collaboration (2023)** - "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"  
  https://arxiv.org/abs/2310.08864 (v9, May 2025)

### Code & Checkpoints
- **GitHub Repository**: https://github.com/google-deepmind/open_x_embodiment
- **RT-1-X JAX Checkpoint**: `gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax/`
- **RT-1-X TensorFlow Inference Colab**: https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb
- **Dataset Colab**: https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb

### Dataset Resources
- **Dataset Spreadsheet (metadata + citations)**: https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/
- **Data Format Spec (RLDS)**: https://github.com/google-research/rlds

### Related Work
- **RT-1 (original)**: Brohan et al., "RT-1: Robotics Transformer for Real-World Action at Scale" (2022)
- **RT-2 (VLA model)**: Brohan et al., "RT-2: Vision-Language-Action Models" (2023)

### License
- Software: Apache 2.0
- Dataset materials: CC-BY 4.0

---

*PR: Survey PR #3: Open X-Embodiment & RT-X Digest*  
*Summary: Created comprehensive digest for Open X-Embodiment (1M+ trajectories, 22 robots) covering dataset format (RLDS), training objectives (RT-1-X Transformer, RT-2-X VLA), real-robot evaluation (6 labs, 50% small-data gain), and Tesla/Ashok claims mapping. Identified action items for AIResearch: standardize episode schema, adapt 7-DOF action vector to vehicle controls, replicate multi-site evaluation protocol, and release checkpoints in ONNX format.*

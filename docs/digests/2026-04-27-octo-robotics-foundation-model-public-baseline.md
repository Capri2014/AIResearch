# Octo: Robotics Foundation Model — Public Anchor Digest

**Survey PR #3 (12:00pm PT)** — Public anchor digest for Tesla/Ashok "robotics foundation model" claims.

**Reference Talk:** Ashok Elluswamy — "Building Foundational Models for Robotics at Tesla" (Tesla AI Day / S3 / AI Symposium)  
**Sources:** Octo paper (https://arxiv.org/abs/2405.12213) | Code (https://github.com/octo-models/octo) | Project (https://octo-models.github.io/) | Checkpoints (https://github.com/octo-models/octo/releases)

---

## TL;DR (3 bullets)

- **Octo** is the most reproducible open-source robotics foundation model: full training/inference code + pretrained checkpoints (800k trajectories) + adapter-based finetuning for new embodiments.
- **Diffusion policy** head outputs action distributions (not discrete tokens); handles 7D end-effector actions across 22+ robot embodiments; supports language OR goal-image instruction.
- **Maps to Tesla/Ashok claims:** ✅ transformer backbone, ✅ cross-embodiment transfer, ✅ efficient finetuning / adaptation. ❌ no fleet data engine, ❌ no world simulator, ❌ no vehicle dynamics.

---

## Dataset / Inputs / Outputs

### Pretraining Dataset (Open X-Embodiment)

| Aspect | Details |
|--------|---------|
| **Scale** | 800k real robot trajectories from Open X-Embodiment |
| **Format** | RLDS (Robot Learning Dataset Format) — episode sequences |
| **Coverage** | 22+ robot embodiments, 60+ datasets from 34 labs worldwide |
| **Annotations** | Language instructions for subset; goal images for goal-conditioned variants |
| **Download** | `pip install rlds` + Octo data preparation scripts |
| **Data mix** | Weighted mixture per Octo config; diverse embodiments weighted higher |

### Model Inputs

| Input | Description |
|-------|-------------|
| **RGB image** | Workspace camera (variable resolution, typically 224×224 or 256×256) |
| **Instruction** | Text string OR goal image (dual-modality conditioning) |
| **Proprioception** | Joint positions / end-effector pose when available in dataset |
| **History** | Multi-frame temporal stacking (typically 2–4 frames) |

### Model Outputs (Action Contract)

| Output | Specification |
|--------|----------------|
| **Action space** | 7D end-effector command (x, y, z, roll, pitch, yaw, gripper open/close) |
| **Representation** | Absolute OR delta (configurable per robot) |
| **Action head** | Modular tokenization adapts to new embodiments |
| **Prediction mode** | Diffusion denoising → full action distribution (not point estimate) |
| **Action chunking** | 4 actions predicted per forward pass for temporal consistency |

---

## Training Objective (BC / diffusion / etc.)

### Architecture
- **Encoder:** Vision Transformer (ViT-B/L) processing RGB observations
- ** backbone:** Transformer encoder processing encoded images + instruction tokens
- **Head:** Diffusion transformer decoder predicting action residuals

### Objective
- **Denoising Diffusion Probabilistic Model (DDPM)** over continuous action space
- **Supervision:** Behavior cloning via diffusion — predict action noise at each diffusion step
- **Key advantage over RT-X:** Outputs full action distribution, enabling risk-aware / uncertainty-qualified deployment

### Finetuning
- **Adapter-based:** Lightweight adapter modules for new robot embodiments
- **Compute:** Few hours on single A100 GPU (80GB)
- **Data requirement:** 100–500 demonstrations for effective transfer

---

## Evaluation Setup

### Multi-Robot Transfer (Real Robot)
- **Platforms:** 9+ different robot platforms (varying degrees-of-freedom, sensors, morphologies)
- **Protocol:** Pretrain on mixed data → finetune on target robot with small dataset → evaluate success rate
- **Result:** Finetuned Octo outperforms training from scratch; effective cross-embodiment transfer demonstrated

### Ablations
- **Language vs goal-image:** Both modalities work; goal images slightly more robust for shape/position tasks
- **Diffusion vs discrete:** Diffusion head captures action uncertainty; better for novel situations

### Limitations (from paper)
- Short-horizon manipulation tasks (seconds to minutes)
- No long-horizon autonomy or multi-task chaining
- Limited to workspace manipulation (not navigation, mobile manipulation)

---

## Tesla/Ashok Talk Mapping

### What Maps Cleanly ✅

| Tesla/Ashok Claim | Octo Alignment |
|------------------|---------------|
| "Data-driven foundation model" | Pretrained on 800k trajectories from diverse robots |
| "Transformer backbone" | ViT encoder + transformer diffusion decoder |
| "Cross-embodiment transfer" | Tested on 9+ robot platforms with adapter finetuning |
| "Efficient adaptation" | Few hours finetuning on single A100; 100–500 demos |
| "Language/instruction conditioning" | Supports text OR goal-image instruction |

### What Doesn't Map ❌

| Tesla/Ashok Claim | Octo Gap |
|-------------------|----------|
| "Fleet data engine" | No continuous data collection / curation pipeline |
| "World simulator" | No implicit world model; pure policy, no rollouts |
| "Vehicle dynamics / driving" | Manipulation domain; 7D gripper actions, not vehicle control |
| "Humanoid scale / fleet deployment" | Laboratory manipulation; no real-world deployment |
| "Long-horizon autonomy" | Short-horizon skills only; no task chaining |

---

## Action Items for AIResearch

### Interfaces / Contracts to Copy

1. **Data contract (RLDS format):** Adopt RLDS episode schema for unified robot data handling
   ```python
   # Example: observation/action dict structure
   {
     "observation": {
       "image": {"rgb": [H, W, 3]},
       "state": {...}  # proprioception
     },
     "action": {"world_vector": [3], "rotational_vector": [3], "gripper": [1]}
   }
   ```

2. **Action contract (7D end-effector):** Standardize on 7D action space (x, y, z, roll, pitch, yaw, gripper)
   - Consider mapping to vehicle control (steering, throttle, brake) as derived space

3. **Diffusion policy head:** Integrate diffusion-based action generation for uncertainty modeling
   - Enables risk-aware planning: sample from action distribution

4. **Adapter-based finetuning:** Implement lightweight adapters for new embodiments / tasks
   - Preserves pretrained backbone; efficient for domain adaptation

5. **Multi-modality instruction:** Support language AND goal-image conditioning
   - Language for explicit commands; goal images for implicitdesired states

---

## Citations + Links

### Primary Sources
1. **Octo Paper:** https://arxiv.org/abs/2405.12213
2. **Octo Project:** https://octo-models.github.io/
3. **Octo Code:** https://github.com/octo-models/octo
4. **Octo Checkpoints:** https://github.com/octo-models/octo/releases
5. **Open X-Embodiment Dataset:** https://github.com/google-deepmind/open_x_embodiment
6. **RLDS Format:** https://github.com/google-research/rlds

### Related Work
- **RT-X (Google DeepMind):** https://arxiv.org/abs/2310.08864 — predecessor dataset effort
- **RT-1/RT-2 (Google):** Vision-language models for robot manipulation
- **Diffusion Policy (Berkeley):** Actuated diffusion for robot control

---

## Summary

- **Octo** is the most reproducible open-source robotics foundation model with full code + pretrained checkpoints
- **Diffusion-based** action modeling enables uncertainty-aware deployment; tested on 9+ robot platforms
- **Maps to Tesla/Ashok:** transformer backbone, cross-embodiment transfer, efficient finetuning all aligned
- **Gaps:** no fleet data engine, no world simulator, manipulation domain (not vehicle control)
- **AIResearch action items:** copy RLDS data contract, 7D action space, diffusion policy head, adapter finetuning
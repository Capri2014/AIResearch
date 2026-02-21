# Robotics Foundation Models Survey (Octo + Open X-Embodiment)

**Date:** 2026-02-21  
**Surveyed by:** Agent (pipeline)  
**Sources:** 
- Octo: https://octo-models.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/

## TL;DR

**Octo** is an open-source generalist robot policy built on the **Open X-Embodiment** dataset. It achieves strong zero-shot performance across multiple robots and can be efficiently finetuned to new tasks. Trained on **800k robot episodes** from 25 datasets.

## Key Insights

### 1. Open X-Embodiment Dataset
- **800k robot episodes** from multiple institutions
- 25 diverse datasets with different:
  - Robot embodiments (arms, bi-manual, mobile manipulators)
  - Sensors (with/without wrist cameras, force-torque)
  - Labels (language instructions, goal images)
- Collaborative effort across 20+ labs

### 2. Octo Model

**Architecture:**
- Transformer-based diffusion policy
- Two versions: Octo-Small (27M) and Octo-Base (93M parameters)
- Supports: language instructions, goal images, observation history
- Multi-modal action distribution via diffusion decoding

**Key Features:**
- Flexible task/observation definitions
- Quick finetuning to new action spaces
- Works with different camera configs

### 3. Results

**Zero-shot (on WidowX UR5):**
| Model | Success Rate |
|-------|-------------|
| RT-1-X | 0.60 |
| RT-2-X (55B) | 0.85 |
| **Octo** | **0.80** |

**Finetuning (100 demos):**
| Baseline | Avg Success |
|----------|-------------|
| From Scratch | 0.20 |
| VC-1 | 0.15 |
| **Octo** | **0.72** |

**Key finding:** Finetuning Octo outperforms next best by **52%**

### 4. Why It Matters

- **Generalist policy** - One model, many robots
- **Efficient finetuning** - 100 demos = good performance
- **Flexible** - Supports new observations (force-torque), new action spaces (joint position)

## Architecture

```
Input: [image_0, image_1, ...] + [language goal] → 
  Transformer + Diffusion Decoding →
    Action prediction (e.g., [gripper_x, gripper_y, gripper_z, gripper_open])
```

## Relevance to Autonomous Driving

### Transfer Learning Potential

1. **Policy architecture** - Diffusion policy works for continuous control
2. **Pretraining** - Large heterogeneous data → better generalization
3. **Finetuning** - Efficient adaptation to new domains

### For Our Waypoint Pipeline

```
Waypoint prediction can use similar architecture:
  - Observation: camera images
  - Goal: target destination
  - Action: waypoint sequence (diffusion)
```

## Implementation

**GitHub:** https://github.com/octo-models/octo

```python
# Quick usage example
from octo.model import OctoModel

# Load pretrained model
model = OctoModel.load_pretrained("octo-base")

# Inference
observations = {"image": current_image, "goal_image": goal_image}
actions = model.predict_action(observations)
```

## Action Items

- [ ] Try Octo for waypoint prediction task
- [ ] Explore: diffusion policy for driving
- [ ] Consider: pretrained vision encoder from Octo
- [ ] Study: finetuning recipe for our domain

## Related Reading

- RT-1 / RT-2 (Google Robotics)
- Open X-Embodiment (CoRL 2023)
- Diffusion Policy (Chi et al.)
- VC-1 (Visual Representations for RL)

# World Models & Learned Simulators Survey (DreamerV3)

**Date:** 2026-02-21  
**Surveyed by:** Agent (pipeline)  
**Sources:** 
- DreamerV3: https://github.com/danijar/dreamerv3
- Paper: https://arxiv.org/pdf/2301.04104

## TL;DR

**DreamerV3** is a scalable reinforcement learning algorithm that learns a world model from experiences and uses it to train a policy from imagined trajectories. Published in Nature (2025), it masters diverse domains with **fixed hyperparameters** - no per-task tuning needed.

## Key Insights

### 1. World Model Architecture

```
Observation (image) → Encoder → Latent State (z)
    ↓
Latent State + Action → World Model (Recurrent) → 
    ├── Next latent state prediction
    └── Reward prediction
    ↓
Imagined Trajectories → Actor-Critic Training
```

**Three components:**
- **World Model:** Predicts future latents and rewards
- **Actor:** Learns policy from imagined trajectories  
- **Critic:** Values states for advantage estimation

### 2. Key Innovations

1. **Symmetrical recurrent state** - Latent state as input to both world model and actor/critic
2. **Fixed hyperparameters** - Works across all domains without tuning
3. **Scalable** - Larger models = better performance + data efficiency

### 3. Results

**Performance across domains:**
- Atari games
- DeepMind Control Suite
- Minecraft
- Robotics (real + sim)

| Metric | DreamerV3 | Previous Best |
|--------|----------|---------------|
| Domain Coverage | Diverse | Limited |
| Hyperparameter Tuning | None needed | Significant |
| Data Efficiency | High | Medium |

### 4. Why Fixed Hyperparameters Matter

- Reduces expert knowledge requirements
- Lower computational resources needed
- More robust to distribution shift
- Better generalization

### 5. Implementation

**GitHub:** https://github.com/danijar/dreamerv3

```python
# Training example
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer \
  --configs crafter \
  --run.train_ratio 32
```

**Requirements:** Python 3.11+, JAX, GPU

## Relevance to Autonomous Driving

### For Driving World Model

```
Input: (current_frame, action) → World Model → 
  ├── Future scene prediction (latent)
  └── Reward prediction (collision, off-road, progress)
      ↓
  Imagined rollouts → Policy improvement
```

### Potential Applications

1. **Learned simulator** - Replace physics simulation with learned model
2. **Policy training** - Train in imagined world, deploy in real
3. **Data augmentation** - Generate diverse scenarios
4. **Closed-loop testing** - Safe exploration in simulation

### Connection to Our Work

```
Our pipeline          → DreamerV3 component
SFT Waypoint BC     → Policy (actor)
RL Delta Head       → Policy improvement
Toy Environment     → World model training
```

## Action Items

- [ ] Study DreamerV3 implementation in detail
- [ ] Consider: world model for driving scenes
- [ ] Explore: GAIA-2 style video prediction
- [ ] Build prototype: simple world model for waypoint prediction

## Related Reading

- DreamerV1/V2 (Hafner et al.)
- PlaNet (Hafner et al.)
- World Models (Ha & Schmidhuber)
- GAIA-2 (Google's video generation model)

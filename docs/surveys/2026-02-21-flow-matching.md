# Flow Matching for Generative Modeling Survey

**Date:** 2026-02-21  
**Surveyed by:** Agent (pipeline)  
**Sources:** Research papers on Flow Matching, Rectified Flow, Consistency Models

## TL;DR

**Flow Matching** is a paradigm for training diffusion models that can be simpler and faster than traditional score-based diffusion. It's particularly relevant for **action-conditioned video generation** (world simulators).

## Key Insights

### 1. What is Flow Matching?

Instead of learning scores (∇log p(x)), flow matching learns a **vector field** that pushes samples toward the data distribution.

**Key equation:**
```
dx/dt = v_t(x)  where v_t is the learned velocity field
```

### 2. Flow Matching vs Score-Based Diffusion

| Aspect | Score-Based Diffusion | Flow Matching |
|--------|----------------------|---------------|
| Objective | Score (∇log p) | Velocity field |
| ODE solving | Requires many steps | Fewer steps possible |
| Training | Multiple noise levels | Simpler training |
| Sampling | Slower (more steps) | Faster |

### 3. Key Variants

#### A. Original Flow Matching (Liu et al., 2022)
- Learns velocity field v_t(x)
- Theoretical guarantees on distribution matching

#### B. Rectified Flow (Liu, 2023)
- Refines flow with "reflow" operation
- Straightens trajectories for faster sampling
- Achieves high quality in fewer steps

#### C. Consistency Models (Song et al., 2023)
- Consistency = points on same trajectory map to same endpoint
- Single-step or few-step generation
- Can distill from pretrained diffusion models

#### D. EDM (Karras et al., 2022)
- Elucidated diffusion training
- Better noise scheduling
- Foundation for many modern implementations

### 4. Why Relevant for World Simulators?

- **Action-conditioned video**: Given current frame + action → predict future frames
- Flow matching can generate video more efficiently
- Fewer sampling steps = faster real-time simulation

### 5. Implementation Resources

| Project | Language | Notes |
|---------|----------|-------|
| https://github.com/NVlabs/edm | Python | Official EDM implementation |
| https://github.com/ermaot/Reflow | Python | Rectified Flow |
| https://github.com/openai/consistency_models | Python | Consistency Models |

## Architecture for World Simulator

```
Input: (current_frame, action) → 
  Video Diffusion (Flow Matching) → 
    Future frames prediction
    
# Training objective: learn velocity field
min || v_t(x_t, c) - v_target(x_1, x_0) ||^2
```

## Relevance to Our Work

### Potential Applications
1. **World model for driving** - Generate future scenes given current state + actions
2. **Data augmentation** - Generate diverse scenarios
3. **Faster simulation** - Use consistency models for real-time rollout

### Implementation Path
1. Start with EDM framework (well-documented)
2. Add action conditioning (concatenate action embedding)
3. Train on driving video data (nuScenes, Waymo)

## Action Items

- [ ] Review EDM paper (Karras et al., 2022)
- [ ] Explore rectified flow for faster sampling
- [ ] Consider: distill diffusion to consistency model for real-time
- [ ] Build prototype: action-conditioned video generation

## Citations

- Flow Matching (Liu et al., 2022)
- Rectified Flow (Liu, 2023)
- Consistency Models (Song et al., 2023)
- EDM (Karras et al., 2022)

# Papers for Series Architecture (World Model → Policy)

**Date:** 2026-02-16

## Core Series / World Model Papers

| Paper | Year | Key Idea | Series? |
|-------|------|----------|---------|
| **GAIA-1** (Wayve) | 2023 | World model + policy in series | ✅ Yes |
| **GAIA-2** | 2024 | Action-conditioned video prediction | ✅ Yes |
| **Drive-JEPA** | 2024 | JEPA for driving, predict future embeddings | ✅ Yes |
| **DriveGPT4** | 2024 | LLM-based driving with reasoning | ✅ Yes |
| **UniAD** (CVPR 2023) | 2023 | Planning-oriented perception, predict-then-plan | ✅ Yes |
| **VAD** | 2022 | Vectorized autonomous driving | Partial |

## Model-Based RL (Foundation)

| Paper | Year | Key Idea |
|-------|------|----------|
| **Dreamer** (DeepMind) | 2020 | World model → latent dynamics → policy |
| **DreamerV2** | 2021 | Improved world model with discrete latents |
| **DreamerV3** | 2023 | Scaling world models |
| **PlaNet** | 2019 | Latent dynamics model for planning |
| **SRN** | 2020 | Scene representation networks |

## JEPA-Based (Meta)

| Paper | Year | Key Idea |
|-------|------|----------|
| **JEPA** (Meta) | 2022 | Joint embedding predictive architecture |
| **I-JEPA** | 2023 | Image JEPA |
| **V-JEPA** | 2024 | Video JEPA |
| **Drive-JEPA** | 2024 | JEPA for autonomous driving |

## Driving-Specific (Series)

| Paper | Year | Key Idea |
|-------|------|----------|
| **ST-P3** | 2023 | Spatiotemporal predictive planning |
| **交通事故** | - | - |
| **TrafficGPT** | 2024 | LLMs for traffic understanding |

## Recommended Reading Order

### For Series Architecture:
1. **GAIA-1** (Wayve) - Best intro to driving world model
2. **Dreamer** - Foundational world model paper
3. **Drive-JEPA** - JEPA for driving
4. **UniAD** - Planning-oriented

### Key Insights:
- **GAIA-1**: World model predicts video → policy acts
- **Dreamer**: Latent world model → RSSM → policy
- **Drive-JEPA**: JEPA encoder → predictor → policy
- **UniAD**: Perception → Prediction → Planning (series)

## Our Implementation (Series Mode)

To implement series in our pipeline:

```python
config = UnifiedConfig(
    architecture="series",  # NEW: series vs parallel
    use_world_model=True,
    use_vla_as_policy=True,
)

# Forward pass in series:
# 1. World Model predicts future latent
future_latent = world_model.predict(current_obs, action)
# 2. Policy/VLA uses predicted latent
trajectory = vla(future_latent)
```

## Files Referenced

- This document: `docs/papers/series-architecture-papers.md`

# DreamZero Survey - World Action Models

**Date:** 2026-02-21  
**Surveyed by:** Agent (pipeline)  
**Source:** https://dreamzero0.github.io/DreamZero.pdf

## TL;DR

DreamZero introduces **World Action Models (WAM)** - a paradigm shift from Vision-Language-Action (VLA) models. WAMs jointly predict future world states (video) and actions using a pretrained video diffusion backbone, achieving **2x better generalization** than state-of-the-art VLAs.

## Key Insights

### 1. Problem: VLA Limitations
- VLAs excel at semantic generalization but struggle with **physical motion generalization** to unseen environments
- Tend to overfit to dominant training behaviors (e.g., default to pick-and-place)
- Rely on repetitive demonstrations from homogeneous data

### 2. Solution: World Action Model
- **Core idea:** Jointly model future world states (video) + actions
- Video serves as dense representation of world dynamics
- Learn physical dynamics from heterogeneous robot data without repetitive demos
- Built on **pretrained video diffusion backbone** (14B parameters)

### 3. Key Results
| Metric | DreamZero | Best VLA |
|--------|-----------|----------|
| Zero-shot task progress (unseen) | 62.2% | 27.4% |
| Unseen verbs (DROID) | 49% | 25-32% |
| Cross-embodiment (human video) | +42% improvement | - |
| New robot adaptation | 30 min data | - |

### 4. Real-Time Deployment
- 38x speedup via model/system optimizations
- **7Hz closed-loop control** with 14B model
- Key: autoregressive video diffusion + caching strategies

### 5. Cross-Embodiment Transfer
- Video-only demos from humans/other robots → 42% improvement on unseen tasks
- Only 10-20 min of data needed
- Adapts to new robot (YAM) with just 30 min play data while retaining zero-shot generalization

## Architecture

```
Input: observation (image sequence) → Video Diffusion Backbone → 
  ├── Future video prediction (world state)
  └── Action prediction
      │
      ▼
final_waypoints = world_model(obs) + action_head(obs)
```

## Relevance to Our Work

### Potential Applications
1. **World model for driving** - Use video diffusion to predict future scenes
2. **Action model integration** - Joint video + waypoint prediction
3. **Cross-domain transfer** - Human driving videos → robot policy
4. **Real-time inference** - 7Hz optimization techniques applicable

### Implementation Considerations
- 14B model requires significant compute
- Video diffusion backbone pretraining is critical
- Need heterogeneous data for diverse skill learning

## Action Items

- [ ] Explore video diffusion backbones (DiT, Sora-like architectures)
- [ ] Consider: joint world state + waypoint prediction for driving
- [ ] Study: real-time inference optimization techniques
- [ ] Reference: DROID dataset for heterogeneous robot data

## Citations

- DreamZero paper: https://dreamzero0.github.io/DreamZero.pdf
- Project page: https://dreamzero0.github.io/
- DROID dataset: https://droid-dataset.github.io/

## Related Reading

- World Models (Ha & Schmidhuber)
- DiT (Scalable Diffusion Models with Transformers)
- GAIA-2 (Google's video generation)

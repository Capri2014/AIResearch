# SceneDiffuser: Diffusion for Trajectory Prediction

**Date:** 2026-03-03  
**Status:** Survey Complete

## Paper

- **Title:** SceneDiffuser: Scene-Conditioned Diffusion Model for Multi-Agent Trajectory Prediction
- **Venue:** 2023
- **Paper:** https://arxiv.org/abs/2305.12754

## Core Idea

Diffusion model for realistic multi-agent trajectory prediction:
- **Diffusion process**: Add noise to trajectories, learn to denoise
- **Scene conditioning**: Map and agent history condition the denoising
- **Multi-modal**: Naturally captures diverse futures
- **Physically plausible**: Diffusion learns realistic distributions

## Architecture

```
Input: Agent history + Scene context
        ↓
Scene Encoder
        ↓
Diffusion Process (T steps)
├── t=0: Random noise trajectory
├── t=1: Slightly denoised
├── ...
└── t=T: Clean trajectory prediction
        ↓
Output: Realistic future trajectory
```

## Key Innovations

### 1. Diffusion for Trajectories

- Treat trajectory prediction as denoising problem
- Forward process: Add Gaussian noise to GT trajectory
- Reverse process: Learn to denoise, conditioned on scene

### 2. Scene-Conditioned Denoising

- At each denoising step, attend to scene context
- Map features + agent history → condition the network
- Ensures physically plausible outputs

### 3. Multi-Modal Output

- Diffusion naturally models multi-modality
- Sample multiple times → diverse futures
- Each sample is a valid trajectory

## Why It Matters

1. **Multi-modality**: Diffusion naturally captures diverse futures
2. **Realism**: Learns real trajectory distributions, not just modes
3. **Flexibility**: Can condition on anything (map, goals, semantics)

## Comparison

| Method | Multi-modality | Realism | Efficiency |
|--------|---------------|---------|------------|
| **SceneDiffuser** | Excellent (sampling) | High | Slow (T steps) |
| **MultiPath++** | Good (K anchors) | Medium | Fast |
| **MTR** | Good (K queries) | Medium | Medium |
| **QCNet** | Good (sampling) | Medium | Fast |

## Implementation Insights

```python
class SceneDiffuser(nn.Module):
    def __init__(self, T=20):
        self.T = T  # Number of diffusion steps
        self.scene_encoder = TransformerEncoder(...)
        self.denoiser = DenoisingNetwork(...)
    
    def forward(self, history, map_features):
        # Encode scene
        scene_emb = self.scene_encoder(history, map_features)
        
        # Start from noise
        x_t = torch.randn_like(target_trajectory)
        
        # Reverse diffusion (denoise)
        for t in reversed(range(self.T)):
            x_t = self.denoiser(x_t, scene_emb, t)
        
        return x_t
    
    def forward_process(self, x_0, t):
        # Add noise: q(x_t | x_0)
        noise = torch.randn_like(x_0)
        return sqrt_alphas[t] * x_0 + sqrt_one_minus_alphas[t] * noise
```

## Training

```python
# Diffusion training (simplified)
for trajectory, map_data in dataloader:
    t = random.randint(0, T)  # Random timestep
    noise = torch.randn_like(trajectory)
    noisy_trajectory = forward_process(trajectory, t)
    
    # Denoise
    predicted_noise = model(noisy_trajectory, map_data, t)
    
    # MSE loss
    loss = mse_loss(predicted_noise, noise)
```

## Pros/Cons

| Pros | Cons |
|------|------|
| Natural multi-modality | Slow inference (T steps) |
| High realism | Memory for many agents |
| Flexible conditioning | Needs careful tuning |

## References

- Paper: https://arxiv.org/abs/2305.12754

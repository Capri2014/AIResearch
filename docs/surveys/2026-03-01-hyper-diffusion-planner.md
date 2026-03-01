# Hyper Diffusion Planner (HDP): Diffusion Models for E2E Autonomous Driving

**Paper**: [Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving](https://arxiv.org/pdf/2602.22801v1.pdf)  
**Project Page**: [Hyper-Diffusion-Planner](https://zhengyinan-air.github.io/Hyper-Diffusion-Planner/)  
**Authors**: Tsinghua AIR (智能产业研究院) + Xiaomi Auto  
**Date**: Feb 2026 (arXiv:2602.22801v1)

---

## TL;DR

HDP achieves **10x performance improvement** over baselines through:
1. **τ₀-pred + τ₀-loss**: Custom loss for low-dimensional trajectory data
2. **Hybrid trajectory representation**: Combines waypoint + velocity prediction
3. **Data scaling**: 70M real-world frames enable multimodal generation
4. **RL post-training**: Safety reinforcement on top of imitation learning

**Key insight**: Diffusion models work for driving, but need careful design for low-dimensional trajectory data vs high-dimensional image pixels.

---

## 1. Problem: Diffusion Models for Driving?

### Why the skepticism?
- Diffusion models excel at image generation (high-dimensional, pixels)
- Trajectories are low-dimensional (few coordinate points)
- Need real-time inference (10-20 Hz)
- Must guarantee safety (not just "looking good")

### What HDP achieves
- 200 km real-world driving tests
- 10x improvement over baselines
- No complex rule post-processing

---

## 2. System Architecture

```
Sensors (Camera/Lidar) → Perception (BEV) → HDP (Diffusion Decoder) → Trajectory
```

### HDP Input
- **BEV features**: Surrounding vehicles, pedestrians, lane markings
- **Navigation**: Route instructions
- **Noisy trajectory**: Random noise + clean trajectory tokens

### HDP Output
- Future trajectory (waypoints over time)

### Process
1. **Add noise**: Gradually add noise to clean trajectory
2. **Learn denoise**: Train decoder to predict clean trajectory
3. **DDIM sampling**: Generate from pure noise

---

## 3. Key Design: Loss Space for Trajectories

### Three Prediction Targets
| Target | Description |
|--------|-------------|
| **ε-pred** (noise) | Predict the added noise (standard in images) |
| **v-pred** (velocity) | Predict "flow velocity" |
| **τ₀-pred** (data) | Directly predict clean trajectory |

### Three Loss Functions
| Loss | Description |
|------|-------------|
| **ε-loss** | MSE(predicted_noise, true_noise) |
| **v-loss** | MSE(predicted_v, computed_v) |
| **τ₀-loss** | MSE(predicted_trajectory, true_trajectory) |

### Experiment Results (9 combinations)

**Finding**: τ₀-pred + τ₀-loss is best for trajectories!

| Combination | Convergence | Trajectory Quality |
|------------|-------------|-------------------|
| τ₀-pred + τ₀-loss | ✅ Fastest, most stable | ✅ Smoothest |
| ε-pred + ε-loss | ❌ Slow, unstable | ❌ High-frequency jitter |
| v-pred + v-loss | ⚠️ Medium | ⚠️ Medium |

### Why τ₀-pred works better
- Trajectories lie on a **low-dimensional manifold**
- Directly predicting on-manifold points is easier
- Predicting high-dimensional noise/off-manifold is like "solving calculus in elementary school"

---

## 4. Key Design: Hybrid Trajectory Representation

### Two representations
| Representation | Pros | Cons |
|--------------|------|------|
| **Waypoints** (positions) | Good global geometry (curves) | Velocity jitters |
| **Velocity** | Smooth, comfortable | Less accurate global position |

### HDP Solution: Hybrid Supervision
```
Loss = α * τ₀-loss(velocity) + β * τ₀-loss(integrated_waypoints)
```
- Backbone outputs velocity (stable to train)
- Loss computed on both velocity AND integrated waypoints
- Mathematically valid as diffusion loss

### Results
- Hybrid > Waypoint-only > Velocity-only
- Combines global geometry + local smoothness

---

## 5. Data Scaling: The Multimodal Secret

### The mode collapse problem
Previous work: "Diffusion models collapse to single mode for driving"

### HDP's finding: **It's a data problem!**

| Data Size | Trajectory Diversity | Behavior |
|----------|---------------------|----------|
| 100k frames (NAVSIM size) | Low | Mode collapse |
| 1M frames | Medium | Some diversity |
| 10M frames | High | Clear multimodal |
| 70M frames | ✅ Highest | Rich multi-modal |

### Key insight
- With enough data, diffusion models naturally learn multimodal driving behaviors
- 20% improvement from 10M → 70M frames
- Confirms diffusion models have strong scaling properties

---

## 6. Safety: RL Post-Training

### Problem with pure imitation learning
- IL mimics human data but doesn't explicitly avoid collisions
- Rare danger scenarios missing from training data

### HDP's solution: RL post-training
1. Define **safety reward**: Distance to obstacles
2. Use **PPO/GRPO** to fine-tune the diffusion model
3. Encourage trajectories that maintain safety margin

### Result
- Explicit safety constraints on top of learned behavior
- Smooth transition from IL → RL

---

## 6b. RL + Diffusion: Weighted Regression

### The challenge
- How to efficiently apply RL to diffusion models?
- Standard RL requires sampling many trajectories

### HDP's elegant solution: Weighted Regression
```
Loss = Σ w_i * L_IL(θ)
where w_i = f(safety_reward_i)
```

### How it works
1. **Compute safety reward**: Distance to obstacles
2. **Weight trajectories**: Higher reward → higher weight in loss
3. **Fine-tune**: Multiply IL loss by reward-based weight
4. **Result**: Model learns to prefer safer trajectories

### Key properties
- **Low compute overhead**: No additional RL sampling needed
- **Compatible with hybrid loss**: Naturally combines with τ₀-loss
- **Effective**: +improvement in real-world tests

### Why it works
- Diffusion model already generates diverse trajectories
- RL post-training selects among existing samples
- Weighted regression = implicit preference learning

### Comparison with standard RL

| Approach | Compute | Stability | Our Status |
|----------|---------|----------|------------|
| Standard PPO | High | Medium | We use this |
| HDP Weighted | Low | High | Consider adopting |
| Re-parameterization | Medium | Medium | Alternative |

---

## 7. Comparison with Our Pipeline

### Our pipeline
```
Waymo episodes → SSL pretrain → Waypoint BC → RL refinement → CARLA eval
```

### HDP's approach
```
Perception BEV → Diffusion Decoder → Trajectory
         ↓
   RL post-training (optional)
```

### Where HDP fits
| Component | Our Implementation | HDP |
|-----------|-------------------|-----|
| Trajectory prediction | Waypoint BC + RL delta | Diffusion decoder |
| Loss function | MSE on waypoints | τ₀-pred + τ₀-loss |
| Safety | RL reward shaping | RL post-training |
| Multi-modal | Not explicit | Natural with scale |

### What we can adopt
1. **Use τ₀-loss** instead of MSE for waypoint training
2. **Hybrid representation** (velocity + waypoints)
3. **Scale data** - more scenarios = more multimodal
4. **RL post-training** for safety

---

## 8. Technical Details

### Model Architecture
- **Transformer-based** diffusion decoder
- Cross-attention: trajectory tokens ↔ perception tokens
- 70M parameters (similar to ours)

### Training
- **DDPM/DDIM** sampling
- 100-200 epochs
- Batch size: 256

### Inference
- **20-30 steps** DDIM (faster than 1000 steps)
- ~10 Hz on GPU (Orin-level)

---

## 9. Results Summary

| Metric | Baseline | HDP | Improvement |
|--------|---------|-----|-------------|
| Collision rate | - | - | **10x reduction** |
| Comfort score | - | - | **Higher** |
| Planning success | - | - | **10x** |
| Real-world km | 0 | 200 | ✅ |

### Ablation Studies
- τ₀-loss: +15% vs ε-loss
- Hybrid repr: +8% vs waypoint only
- 70M data: +20% vs 10M

---

## 10. Related Work Comparison

| Method | Approach | Multi-modal | Real Testing |
|--------|---------|------------|-------------|
| **HDP** | Diffusion + τ₀-loss | ✅ With scale | ✅ 200km |
| **Diffusion Planner** | Standard diffusion | ❌ | ❌ |
| **UniAD** | Attention-based | ❌ | Partial |
| **VLAW** | VLA + World Model | ✅ | ✅ |

---

## 11. Action Items for AIResearch

### Immediate (This Week)
1. **Adopt τ₀-loss**: Modify waypoint BC training to use τ₀-pred + τ₀-loss
   - Instead of: `MSE(pred_waypoints, gt_waypoints)`
   - Use: Direct trajectory prediction loss

### Short-Term (This Month)
2. **Hybrid representation**: Add velocity prediction head + hybrid loss
   - Output: (waypoints, velocities)
   - Loss: weighted combination

3. **Expand data**: Use more scenarios for multimodal behavior

### Medium-Term (This Quarter)
4. **Weighted RL post-training**: Use HDP's weighted regression approach
   - Instead of full PPO, use: `Loss = Σ w_i * L_IL` where `w_i = f(reward)`
   - Lower compute, more stable than standard RL
   - Naturally compatible with τ₀-loss

5. **Diffusion decoder**: Consider replacing MLP with transformer diffusion

---

## 12. Key Takeaways

1. **Diffusion works for driving** - but need custom loss for low-dim data
2. **τ₀-pred + τ₀-loss** is the key innovation
3. **Data scale matters** - 70M frames enables natural multimodality
4. **Hybrid representation** combines best of both worlds
5. **RL for safety** - post-training adds explicit safety

---

## Citations

- Ninan et al., "Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving", arXiv:2602.22801, 2026
- HDP Project: https://zhengyinan-air.github.io/Hyper-Diffusion-Planner/

### Related Reading
- NAVSIM: "Learning from pixels in the loop"
- UniAD: "Planning-oriented autonomous driving"
- VLAW: "VLA × World Model Co-evolution" (our other survey)

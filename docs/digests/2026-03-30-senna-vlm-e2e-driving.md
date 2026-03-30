# Senna & Senna-2: VLM-E2E Dual-System Driving Policy — Digest

Source: [Senna (arXiv:2410.22313)](https://arxiv.org/abs/2410.22313), [Senna-2 (arXiv:2603.11219)](https://arxiv.org/abs/2603.11219), [GitHub](https://github.com/hustvl/Senna)

## TL;DR (5 bullets)
- **Hybrid Architecture**: Senna combines a Large Vision-Language Model (Senna-VLM) for high-level semantic reasoning with a separate end-to-end planner (Senna-E2E) for precise trajectory prediction — dual-system, not fully monolithic.
- **Decoupled Planning**: LVLM handles "what to do" (lane changes, turns, speed adjustments in natural language); E2E model handles "how to drive" (precise waypoints/trajectories).
- **Three-Stage Training**: Mix pre-training on DriveX → driving fine-tuning → planning fine-tuning; Senna-2 adds explicit VLM-E2E alignment stages.
- **SOTA Planning**: Achieves 27.12% reduction in average planning error and 33.33% collision rate reduction on nuScenes vs. non-pretrained baselines.
- **Senna-2 Alignment**: New consistency-oriented training with decision adapter + hierarchical RL in 3DGS environments; achieves 19.3% F1 improvement in dual-system consistency.

## Problem
- **UniAD Limitations**: While UniAD unified perception-prediction-planning in a single transformer, it lacks semantic reasoning capability for complex, rare edge cases — limited commonsense understanding of traffic scenes.
- **LVLM Trajectory Prediction Pitfalls**: Prior work (e.g., DriveGPT, some VLM-based approaches) directly predicts trajectories from LVLMs, but LVLMs are not suited for precise numerical outputs; suboptimal results.
- **The Core Gap**: Need both semantic reasoning (complex scenarios, intent prediction) AND precise low-level control (trajectory smoothness, collision avoidance) — but these require different model capabilities.

## Method (by section)

### System Decomposition: True vs. Modular E2E

**Senna (v1)**:
```
[Multi-view Cameras] → [Senna-VLM (LVLM)] → [High-level Decision: "turn left"]
                                        ↓
                      [Decision Token] → [Senna-E2E (E2E Planner)] → [Trajectory]
```

- **Senna-VLM**: Based on LLaVA-v1.6-34b (vicuna-7b-v1.5 base), processes 6-view camera input with multi-image encoding. Generates:
  - Scene description (natural language)
  - Meta-action prediction (lane change left/right, accelerate, decelerate, keep lane, turn)
  - Planning explanation
- **Senna-E2E**: End-to-end trajectory predictor that takes camera features + decision token from VLM → outputs future waypoints (typical: 2-second horizon, 0.5s intervals).
- **Not truly end-to-end**: The VLM and E2E are separate modules with explicit interface (decision token). This is a hybrid system, not a monolithic E2E transformer.

**Senna-2 (March 2026)** — Evolution:
- **Decision Adapter**: New module that converts VLM's implicit embeddings into guidance for E2E policy (not just explicit decision tokens).
- **Three-Stage Consistency Training**:
  1. **Driving Pre-training**: Preliminary decision-making + planning
  2. **Open-loop Alignment**: Align VLM and E2E in simulation
  3. **Closed-loop Alignment**: Bottom-up Hierarchical RL in 3DGS environments

### Inputs/Outputs + Temporal Context

**Inputs**:
- 6-view camera images (front, front-left, front-right, rear, rear-left, rear-right)
- Optional: multi-view prompts for spatial reasoning (e.g., "what's in front of me?")
- Token per image: 128 tokens (from LLaVA encoding)

**Outputs**:
- **Senna-VLM**: Natural language scene description + meta-action (6 classes: lane change L/R, turn L/R, accelerate, decelerate, keep lane)
- **Senna-E2E**: Trajectory coordinates (x, y, heading) for future timesteps

**Temporal Context**:
- Multi-image encoding captures temporal context across frames
- DriveX dataset provides large-scale temporal sequences
- No explicit recurrent state mentioned; temporal modeling via multi-image prompt

### Training Objectives

**Senna (Three-Stage)**:
1. **Stage 1 — Mix Pre-training**: 
   - General image-text alignment + driving-specific QA generation
   - Uses LLaVA pretrained weights
   - Pre-train on DriveX (large-scale, 200K+ clips, proprietary)
2. **Stage 2 — Driving Fine-tuning**:
   - Fine-tune on nuScenes for scene understanding
   - Planning-oriented Q&A generation (LLaVA-v1.6-34b generates scene descriptions + explanations)
3. **Stage 3 — Planning Fine-tuning**:
   - Full-parameter fine-tuning for meta-action prediction
   - Loss: Cross-entropy for action classification

**Senna-2 (Consistency-Oriented)**:
- Adds decision adapter training + hierarchical RL alignment
- **Bottom-up HRL**: Learns to adjust trajectories when VLM decision and E2E output misalign
- Loss includes: trajectory regression (L1/L2) + decision consistency loss + RL safety reward

### Eval Protocol + Metrics + Datasets

**Datasets**:
- **nuScenes**: Primary benchmark (20K samples)
- **DriveX**: Large-scale pre-training data (proprietary, 200K+ clips)

**Metrics**:
| Metric | Description |
|--------|-------------|
| Average Planning Error (APE) | Mean L2 distance between predicted and ground truth trajectory |
| Collision Rate | % of scenarios where predicted trajectory collides with objects |
| Meta-action Accuracy | Top-1 accuracy of high-level decision prediction |
| FDE (Final Displacement Error) | L2 error at final timestep |
| AF-CR (Acceleration Failure - Collision Rate) | Closed-loop safety metric from Senna-2 |

**Results (nuScenes)**:
- **Senna**: 27.12% APE reduction, 33.33% collision rate reduction vs. baseline (no pretraining)
- **Senna-2**: 5.7% FDE reduction (open-loop), 30.6% AF-CR reduction (closed-loop), 19.3% F1 improvement in dual-system consistency

## Key takeaways

### What Maps to Tesla/Ashok Claims

| Tesla Claim | Senna Approach | Match? |
|-------------|----------------|--------|
| **Camera-first** | 6-view camera only, no LiDAR | ✅ Strong match |
| **Long-tail handling** | VLM provides semantic reasoning for rare scenarios | ✅ Addresses this |
| **Regression testing** | Closed-loop eval in 3DGS (Senna-2) | ⚠️ Partial — no mention of fleet-level regression |
| **End-to-end learning** | Hybrid VLM+E2E, not truly E2E | ❌ Modular interface |
| **Scalability with data** | Pre-training on DriveX shows data scaling benefit | ✅ |

### What Doesn't Map

- **No mention of shadow mode / disengagement tracking**: Tesla emphasizes ops metrics; Senna focuses on benchmark metrics.
- **No online mapping / vectorization**: Uses nuScenes static maps, not learned online mapping.
- **No mention of corner case mining / data engine**: Pure model training pipeline, no active data collection described.
- **No vehicle-specific calibration**: Generic nuScenes model, not tailored to specific vehicle dynamics.

### What to Borrow for AIResearch

1. **Waypoint Head Architecture**: The E2E trajectory prediction head (outputting future waypoints) is directly applicable to AIResearch's planning stack. The decision token → waypoint mapping is clean and interpretable.
2. **Planning-Oriented QA Generation**: Using LLaVA to generate scene descriptions + planning explanations for data augmentation is a clever trick — could apply to AIResearch's sim-to-real pipeline.
3. **Eval Harness**: The meta-action accuracy + APE + collision rate triplet is a solid baseline evaluation protocol. Add closed-loop 3DGS evaluation for more robust testing.
4. **Dual-System Consistency Loss**: Senna-2's approach of explicitly training alignment between high-level decisions and low-level trajectories is valuable — could prevent "semantic drift" in longer-horizon planning.
5. **3DGS Closed-Loop Eval**: The bottom-up HRL in 3DGS environments (from Senna-2) is a strong sim evaluation approach — more realistic than CARLA for vision-based methods.

## Action items for this repo
- [ ] Consider integrating decision adapter concept into AIResearch E2E pipeline
- [ ] Explore planning-oriented QA data generation using VLM for sim data augmentation
- [ ] Adopt meta-action + APE + collision rate as standard eval triplet
- [ ] Investigate 3DGS-based closed-loop evaluation for vision-centric planning
- [ ] Track Senna-2 open-source release for potential fine-tuning experiments

## Citations

- **Senna Paper** — "Senna: Bridging Large Vision-Language Models and End-to-End Autonomous Driving" — [arXiv:2410.22313](https://arxiv.org/abs/2410.22313)
- **Senna-2 Paper** — "Senna-2: Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning" — [arXiv:2603.11219](https://arxiv.org/abs/2603.11219)
- **Code & Models** — [GitHub: hustvl/Senna](https://github.com/hustvl/Senna)
- **Hugging Face** — [Senna-7B](https://huggingface.co/rb93dett/Senna)
- **LLaVA Foundation** — [LLaVA-v1.6-34b](https://huggingface.co/liuhaotian/llava-v1.6-34b)

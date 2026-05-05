# LMDrive: Closed-Loop End-to-End Driving with Large Language Models — Digest

Source: https://arxiv.org/abs/2312.07488 (CVPR 2024)

## TL;DR (5 bullets)
- **Method**: Language-guided E2E driving framework combining vision encoder + LLM (Vicuna/LLaVA) for multi-modal reasoning
- **Inputs**: Multi-view cameras (4 views) + LiDAR + natural language navigation instructions + "notice" instructions (human feedback)
- **Training**: Two-stage — (1) vision encoder pretraining onDriving data, (2) instruction finetuning with language alignment
- **Eval**: CARLA LangAuto benchmark (Tiny/Short/Long), ~64K clips, reports Driving Score (DS) and route completion
- **Key insight**: LLM enables "reasoning" over long-tail events via language instructions; closed-loop eval in simulator

## Problem
- Existing E2E AD models mimic driving patterns without reasoning → fail on long-tail, adversarial scenarios
- Modular approaches (perception → planning → control) are interpretable but lose E2E gradient flow
- **Gap**: How to inject commonsense reasoning into E2E policies while staying in the loop?

## Method (by section)

### Architecture
```
Multi-view cameras → Vision Encoder (ResNet26/50) → Visual Tokens
     ↓
Multi-view LiDAR → Point Cloud Encoder → BEV Tokens
     ↓
[Visual Tokens] + [Language Instruction] → LLM (Vicuna-7B / LLaVA-1.5-7B) → Action Predictions
```

### Two-Stage Training
1. **Vision Encoder Pretraining**: Freeze LLM, train visual encoder + Q-Former to produce language-aligned visual tokens
2. **Instruction Finetuning**: Full E2E fine-tune with paired (sensor data, instruction, control signal) tuples

### Instruction Types
- **Navigation**: "turn left", "change lane", "go straight" — from navigation software
- **Notice**: "pedestrian ahead", " construction zone", "adversarial weather" — human-provided context
- **Misleading** (optional): Test robustness to incorrect instructions

## Data / Training
- **Dataset**: ~64K clips (CARLA 0.9.10.1), 8 towns, multi-weather, 2-20s per clip
- **Collection**: Rule-based expert autopilots generate (sensor, instruction, action) triplets
- **Frequency**: ~10Hz, multi-view cameras (400x1200) + LiDAR (180° FOV)
- **Storage**: Organized in CARLA leaderboard format with 3D bboxes, affordances, measurements

## Evaluation
- **Benchmark**: CARLA LangAuto (Tiny/Short/Long) — language-guided navigation routes
- **Metrics**:
  - **Driving Score (DS)**: Composite of route completion × infraction penalty
  - **Route Completion (%)**: Fraction of route completed
  - **Infraction Rate**: Collisions, red-light violations, off-road
- **Results** (LLaVA-1.5-7B backbone):
  | Benchmark | DS (LMDrive) | DS (baseline) |
  |-----------|-------------|--------------|
  | LangAuto-Short | 50.6 | ~33 |
  | LangAuto-Long | 36.2 | ~21 |
- **Ablation**: Notice instructions improve DS by ~15% on adversarial scenarios

## Key takeaways
- **Pros**:
  - Open code (GitHub: opendilab/LMDrive) + dataset on HuggingFace
  - Closed-loop eval in CARLA — realistic perception-action feedback
  - Language instruction interface enables human-in-the-loop feedback ("notice")
  - Multi-modal fusion (camera + LiDAR) with LLM reasoning
- **Cons**:
  - **Simulator gap**: CARLA ≠ real-world (perfect sensors, no latency)
  - **LLM latency**: 7B model at 10Hz is impractical for real-time driving
  - **No temporal reasoning**: Processes frames in chunks, no long-horizon planning
- **Maps to Tesla/Ashok claims**:
  - ✅ **Camera-first**: Yes — primary input is multi-view cameras
  - ✅ **Long-tail handling**: Notice instructions address adversarial cases
  - ❌ **Regression testing**: No — only CARLA simulation benchmarks
  - ��� **Real-world closed-loop**: Not validated on real data / vehicles
  - ⚠️ **Waypoints**: Outputs control signals directly (throttle/steering), not waypoint heads

## Action items for this repo
- [ ] Adopt LLM-style "notice" tokens as context conditioning for waypoint head
- [ ] Extract LMDrive's evaluation harness (CARLA LangAuto) as baseline for closed-loop testing
- [ ] Replace LLM with lightweight language encoder for real-time inference
- [ ] Add "misleading instruction" ablation to test robustness

## Citations
- **LMDrive Paper** — "LMDrive: Closed-Loop End-to-End Driving with Large Language Models" (CVPR 2024) — https://arxiv.org/abs/2312.07488
- **Code** — OpenDILab/LMDrive: https://github.com/opendilab/LMDrive
- **Dataset** — HuggingFace LMDrive: https://huggingface.co/datasets/OpenDILabCommunity/LMDrive
- **Related** — InterFuser (predecessor, multi-modal fusion): https://github.com/opendilab/InterFuser
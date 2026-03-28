# DiffusionDrive — Truncated Diffusion for Real-Time End-to-End Autonomous Driving

Source: https://arxiv.org/abs/2411.15139 (CVPR 2025 Highlight)

## TL;DR (5 bullets)
- **Architecture**: Truncated diffusion policy for end-to-end planning — 10x faster than vanilla diffusion (45 FPS), learns multimodal trajectory distributions directly from human demonstrations.
- **Key insight**: Truncates diffusion process early (fewer denoising steps) while preserving multimodal planning capability; combines BEV encoder with diffusion decoder.
- **Results**: 88.1 PDMS on NAVSIM (3.5 higher than prior art), 0.57m L2 @ 3s on nuScenes open-loop, SOTA closed-loop on CARLA Town05.
- **Tesla/Ashok alignment**: Camera-only input (multi-view), learns from human demonstrations at scale, but lacks explicit long-tail regression testing and failsafe wrappers described in Tesla's approach.
- **For AIResearch**: Strong candidate for waypoint head evaluation — explicit diffusion-based trajectory sampling aligns with waypoint prediction; easy to integrate with nuScenes/NAVSIM eval harness.

## Problem
- **Core challenge**: Diffusion models for robotic manipulation showed promise for multimodal behavior but were too slow for real-time driving (10-1000 denoising steps, 1-2 FPS).
- **Why it matters**: Real-world driving requires 10+ FPS, multimodal planning (multiple valid trajectories), and robustness to dynamic agents.

## Method (by section)

### System Decomposition (E2E vs Modular)
- **True E2E**: Multi-view camera → BEV encoder → Truncated diffusion decoder → Trajectory sampling → Control (steer/throttle)
- **Modular elements**: Uses ResNet-34/50 backbone (pretrained), BEV feature extraction — similar to VAD/VADv2 but replaces deterministic head with diffusion
- **Key difference from UniAD**: UniAD uses query-based transformer with staged detection/prediction/planning; DiffusionDrive uses single-stage diffusion with truncated denoising

### Inputs/Outputs + Temporal Context
- **Inputs**: Multi-view camera images (6 cameras typical), optional past N frames for temporal context
- **Outputs**: Probabilistic trajectory distribution — samples K trajectories, selects best or aggregates
- **Temporal**: Processes sequential frames; temporal aggregation in BEV encoder (not explicit memory bank like UniAD)

### Training Objective
- **Diffusion loss**: Denoising objective — predict noise added to ground-truth trajectory at random timesteps
- **Imitation learning baseline**: Trains on human driving data (nuScenes, NAVSIM, CARLA)
- **No RL/世界模型**: Pure imitation learning from demonstrations; no online interaction or world model

### Architecture Details
- **Truncated diffusion**: Only 2-4 denoising steps (vs 10-1000 in robotics diffusion policies)
- **Conditional generation**: BEV features as condition for diffusion process
- **Trajectory parameterization**: Waypoint sequence (e.g., 8 points @ 0.5s intervals)

## Data / Training
- **Datasets**: nuScenes (1000 scenes), NAVSIM (large-scale), CARLA (closed-loop)
- **Scale**: Not explicitly stated; uses pre-trained backbones (ResNet-34/50 on ImageNet)
- **Training**: End-to-end supervised learning on human trajectories

## Evaluation

### Metrics + Protocols
- **NAVSIM PDMS** (Planning Driver Model Score): Primary metric — combines L2 error, collision rate, progress
- **nuScenes open-loop**: L2 displacement @ 1/2/3s, collision rate
- **CARLA closed-loop**: Route completion, driving score, collision rate

### Results
| Benchmark | Metric | DiffusionDrive | Prior Best |
|-----------|--------|---------------|------------|
| NAVSIM | PDMS | **88.1** | 84.6 (VADv2) |
| nuScenes | L2 @ 3s | **0.57m** | 0.70m (VAD-Base) |
| CARLA Town05 | Route Comp | **87.3%** | 75.2% (VADv2) |

### Key Finding
- 64% higher mode diversity score than vanilla diffusion — captures multiple valid driving behaviors (lane change vs keep lane vs car-following)

## Key takeaways
1. **Diffusion works for driving** if truncated — 2-4 steps gives real-time performance while preserving multimodal output
2. **Imitation learning scales**: No RL needed to achieve SOTA; data-driven trajectory distribution captures human variance
3. **Still camera-only**: No LiDAR, no map — but relies on BEV features which may benefit from depth supervision
4. **Gap to Tesla**: No safety wrapper/failsafe, no long-tail regression testing, no explicit world model for simulation
5. **Fast inference**: 45 FPS on single GPU — suitable for real deployment

## What maps to Tesla/Ashok claims and what doesn't

### Aligns:
- **Camera-first**: Multi-view camera only, no LiDAR
- **Regression testing**: Can be evaluated on large-scale datasets (NAVSIM has 20k+ scenarios)
- **End-to-end learning**: Single differentiable pipeline from pixels to trajectory
- **Human demonstrations**: Learns from human driving data at scale

### Doesn't align:
- **Long-tail handling**: No explicit mechanism for rare scenarios; relies on dataset coverage
- **Failsafe/rule-based wrapper**: Pure learning — no hybrid system
- **Online adaptation**: No online learning or continuous improvement from fleet data
- **Explicit world model**: No predictive world model for safety verification

## Action items for this repo
- [ ] Integrate DiffusionDrive as waypoint head candidate in AIResearch eval harness
- [ ] Benchmark on nuScenes planning task with L2 + collision metrics
- [ ] Compare mode diversity vs deterministic heads (VAD, UniAD)
- [ ] Consider truncated diffusion for multi-modal planning in AIResearch stack

## Citations
- **DiffusionDrive paper** — https://arxiv.org/abs/2411.15139
- **Code & models** — https://github.com/hustvl/DiffusionDrive, https://huggingface.co/hustvl/DiffusionDrive
- **NAVSIM benchmark** — https://github.com/autonomousvision/navsim
- **VADv2 (prior best)** — https://arxiv.org/abs/2402.13243 (ICLR 2026)
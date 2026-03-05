# DriveLM: Driving with Graph Visual Question Answering — Digest

Source: https://github.com/OpenDriveLab/DriveLM (arXiv: https://arxiv.org/abs/2312.14150, ECCV 2024 Oral)

## TL;DR (5 bullets)
- DriveLM is a **VLM-based end-to-end driving policy** that uses **Graph VQA** to jointly perform perception, prediction, and planning — replacing modular stacks with language-driven reasoning across tasks.
- Instantiates **DriveLM-Data** (nuScenes + CARLA) with graph-structured reasoning annotations linking perception → prediction → planning; proposes **DriveLM-Agent** baseline using BLIP-2 + GPteacher.
- Training combines **imitation learning** (behavior cloning on expert trajectories) with **graph-based VQA supervision** — language reasoning provides intermediate supervision without discrete planning labels.
- Evaluation on **nuScenes planning** (L2 error, collision rate) and **CARLA Leaderboard 2.0** — shows competitive planning performance vs UniAD while enabling interpretable language reasoning.
- Clean mapping to Tesla/Ashok: **camera-first (VLM uses images), interpretable reasoning chain, language as intermediate representation**; gaps: **no closed-loop RL, no long-tail regression testing, limited real-time inference claims**.

## Problem

Modular autonomous driving stacks (perception → prediction → planning) suffer from **error propagation** and **lack of interpretability**. End-to-end models like UniAD address this but still lack **semantic reasoning** — they can't explain *why* a decision was made. DriveLM asks: can we use **Vision-Language Models (VLMs)** to provide interpretable, reasoning-driven E2E driving?

## System Decomposition

### What is truly end-to-end vs modular
- **End-to-end**: Single VLM (BLIP-2 backbone) processes camera images → outputs trajectory + language reasoning.
- **Modular aspects**: Uses **Graph VQA** structure — perception, prediction, and planning are separate *questions* in a graph, not fully fused. Each node (Q: "What objects?" → A: detection) feeds into next (Q: "Where will they go?" → A: prediction) → planning (Q: "What trajectory?" → A: trajectory).
- **Comparison to UniAD**: UniAD uses query-based transformer for all tasks with hierarchical task queries; DriveLM uses language as the unifying interface between tasks.

### Inputs / Outputs + Temporal Context
- **Inputs**: 6 surrounding camera images (nuScenes setup), optional HD map as auxiliary input.
- **Temporal**: Processes current frame; reasoning chain provides implicit temporal context via graph dependencies (detection → prediction → planning).
- **Outputs**: 
  1. **Language answers**: Perceptual ("What objects?") and predictive ("Where will they go?") descriptions.
  2. **Trajectory**: Planned ego trajectory as waypoints (x, y) in world coordinates.
- **Conditioning**: Uses **GPteacher** (grounding prompt teacher) to generate question-answer pairs from perception labels.

## Training Objective(s)

### Primary: Imitation Learning
- **Behavior cloning**: Train VLM to predict expert trajectory given camera input + language prompts.
- Loss: L2 regression on future waypoints.

### Secondary: Graph VQA Supervision
- **Language reasoning loss**: Train VLM to answer graph-structured questions about perception and prediction.
- Provides intermediate supervision without discrete planning labels.
- Enables interpretability: model must verbalize what it perceives and predicts.

### Distillation Component
- **GPteacher**: Frozen perception models (detector, tracker) generate QA pairs from ground-truth labels → used as supervision for VLM reasoning.

## Evaluation Protocol + Metrics + Datasets

### Datasets
- **DriveLM-nuScenes**: nuScenes with Graph VQA annotations (perception → prediction → planning reasoning chains).
- **DriveLM-CARLA**: CARLA simulator with language annotations for closed-loop evaluation.

### Metrics
- **Planning**: L2 error (m) at 1s/2s/3s horizons, collision rate (%).
- **Language**: BLEU, CIDEr for QA quality (but impact on driving unclear).
- **CARLA**: Driving score, route completion, infraction score (Leaderboard 2.0).

### Results (nuScenes planning)
| Method | L2@3s (m) | Collision (%) |
|--------|-----------|----------------|
| ST-P3 | 2.09 | 0.65 |
| UniAD | 1.53 | 0.53 |
| **DriveLM-Agent** | 1.78 | 0.61 |

DriveLM is competitive but slightly below UniAD on pure planning metrics — tradeoff for interpretability.

## Mapping to Tesla/Ashok Claims

### What maps cleanly
- **Camera-first**: VLM processes raw camera images; no LiDAR requirement.
- **Interpretable reasoning**: Language outputs explain perception and prediction — aligns with "AI explains its decisions" narrative.
- **Unified representation**: Language as intermediate representation between perception → planning (similar to "token" concept in Tesla's FP8 work).
- **Efficient pretraining**: Uses pretrained BLIP-2 (frozen visual encoder) — reduces training compute.

### What doesn't map
- **Long-tail regression testing**: No dedicated long-tail benchmark; nuScenes is curated, not adversarial.
- **Real-time inference**: VLM inference is slow (~100ms+ per frame); no mention of optimization for onboard deployment.
- **RL /闭环**: No reinforcement learning or closed-loop training — purely imitation learning.
- **Fleet-scale data**: Trained on nuScenes (1k scenes), not Tesla's millions of miles.
- **Safety-critical planning**: Collision rate still 0.6% — not ready for safety-critical deployment.

## What to Borrow for AIResearch (esp. waypoint head + eval harness)

1. **Graph VQA structure**: Adopt the reasoning graph (perception → prediction → planning) as interpretable intermediate supervision for waypoint heads.
2. **GPteacher pipeline**: Use frozen perception models to generate QA pairs — can bootstrap reasoning without manual annotation.
3. **nuScenes planning benchmark**: Use DriveLM's evaluation protocol (L2 error + collision) as baseline for waypoint head comparison.
4. **CARLA Leaderboard integration**: DriveLM-CARLA provides closed-loop eval — critical for testing planning robustness.
5. **Language as safety layer**: Use VLM outputs as explainability layer on top of learned waypoint policy — can flag uncertain predictions.

## Action Items for AIResearch
- [ ] Implement Graph VQA wrapper around existing waypoint head; evaluate if language reasoning improves planning confidence.
- [ ] Run waypoint head on nuScenes using DriveLM eval protocol for apples-to-apples comparison.
- [ ] Explore BLIP-2 / LLaVA backbone for real-time inference (current VLM too slow for 10Hz planning).
- [ ] Add closed-loop CARLA evaluation to existing sim harness.

## Citations
- **DriveLM Paper** — "DriveLM: Driving with Graph Visual Question Answering" (ECCV 2024 Oral) — https://arxiv.org/abs/2312.14150
- **DriveLM GitHub** — Code, data, and challenge: https://github.com/OpenDriveLab/DriveLM
- **nuScenes** — 1000-scene multimodal dataset: https://www.nuscenes.org/nuscenes
- **CARLA Leaderboard 2.0** — Closed-loop driving benchmark: https://carla.org/leaderboard/
- **BLIP-2** — Pretrained VLM backbone used: https://salesforce.blip2.github.io/

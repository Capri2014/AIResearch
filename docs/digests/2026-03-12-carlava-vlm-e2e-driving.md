# CarLLaVA — Vision Language Models for Camera-Only Closed-Loop Driving

**Date:** 2026-03-12  
**Status:** Survey Complete  
**Source:** Wayve + University of Tübingen, arXiv:2406.10165, CARLA AD Challenge 2024 (1st Place)

---

## TL;DR (5 bullets)

- **CarLLaVA** is a Vision Language Model (VLM) for end-to-end camera-only driving that wins the CARLA AD Challenge 2.0 sensor track — beating prior SOTA by **458%**
- Uses **LLaVA-NeXT vision encoder** (CLIP ViT-L-336px) pretrained on internet-scale VLM data — no expensive LiDAR or BEV/depth/segmentation labels required
- Proposes **semi-disentangled output**: time-conditioned waypoints (longitudinal) + space-conditioned path waypoints (lateral) with separate PID controllers
- **Efficient training recipe**: data bucketing to oversample "interesting" scenarios (acceleration, steering, hazards) — reduces compute by filtering trivial straight-driving data
- From **Wayve** (real-world E2E driving company) — bridges gap between large VLMs and closed-loop driving performance

---

## Problem

End-to-end autonomous driving faces several key challenges:

1. **Sensor cost**: Most SOTA CARLA methods rely on LiDAR + expensive auxiliary labels (BEV semantics, depth, segmentation) — hard to transfer to real world
2. **Representation trade-offs**: Waypoints work well for longitudinal control but poorly for lateral control (turns); direct control predictions sacrifice collision avoidance
3. **Trivial data dominance**: Most driving is boring straight-line travel — training on full datasets wastes compute on easy samples
4. **Foundation model gap**: LLMs/VLMs show reasoning capabilities but are rarely evaluated in closed-loop driving — mostly open-loop qualitative analysis

---

## Method

### Architecture Overview

```
Multi-view Cameras → CLIP ViT-L-336px Encoder → Adapter → LLaMA Decoder → Semi-Disentangled Heads → PID Controllers → Control Commands
                         ↑                                    ↑
              (LLaVA-NeXT pretrained)              Learnable queries + target/speed tokens
```

### Core Innovations

| Component | Description |
|-----------|-------------|
| **HD-Vision Encoder** | LLaVA-NeXT (CLIP ViT-L-336px) — highest-res CLIP model; uses "anyres" technique to split high-res images into multiple 336×336 patches |
| **VLM Backbone** | LLaMA decoder (350M–1.3B params) — leverages internet-scale VLM pretraining |
| **Semi-Disentangled Output** | Path waypoints (space-conditioned) for lateral control + time-conditioned waypoints for longitudinal control |
| **Data Bucketing** | Oversample interesting scenarios (hazards, turns, stops) vs trivial straight driving |

### System Decomposition

**Truly End-to-End:**
- Single model: camera pixels → LLaMA transformer → waypoints → PID controllers → steering/throttle/brake
- No auxiliary labels required: no BEV, depth, segmentation, or LiDAR
- Pure imitation learning from expert trajectories

**What could be considered "modular-ish":**
- PID controllers for lateral/longitudinal decoupling (but learned, not hand-tuned)
- Separate heads for path vs waypoints — but both trained end-to-end
- Target point encoding (navigation) as input — standard for navigation-aware driving

### Training Objective

- **Imitation Learning**: Mean Squared Error (MSE) between predicted and expert waypoints
- **Language modeling loss** (optional): For driving explanation generation — auto-regressive LM loss on tokenized explanations
- **Data bucketing**: Sampling strategy that oversamples challenging scenarios rather than uniform sampling

### Inputs/Outputs

| Input | Details |
|-------|---------|
| **Cameras** | Front view (primary), optional rear view for C2T1 variant |
| **Resolution** | High-res split into 336×336 patches via anyres (critical for distant traffic lights, pedestrians) |
| **Navigation** | Next 2 target points (ego-coordinate frame) |
| **Ego State** | Vehicle speed |

| Output | Details |
|--------|---------|
| **Path Waypoints** | Space-conditioned, 4 waypoints for lateral control |
| **Time Waypoints** | Time-conditioned, 4 waypoints for longitudinal control |
| **Control** | Steering, throttle, brake via PID controllers |
| **Language (optional)** | Driving explanation auto-generated |

---

## Evaluation

### Benchmark: CARLA Leaderboard 2.0

| Model | Sensors | Aux Labels | DS ↑ | RC ↑ | IS ↑ |
|-------|---------|-------------|------|------|------|
| **CarLLaVA (ours)** | Camera only | None | **6.87** | **18.08** | **0.42** |
| TF++ | Camera | Depth, OD, SS, BS | 5.56 | 11.82 | 0.47 |
| Zero-shot TF++ | Camera | Depth, OD, SS, BS | 0.58 | 8.53 | 0.38 |
| CaRINA hybrid | Camera | IS, OD | 1.23 | 9.56 | 0.31 |
| Kyber-E2E | L+C+R+M | IS, OD | 3.47 | 8.48 | 0.50 |
| Sensor (privileged) | LiDAR | Privilege | 0.25 | 15.20 | 0.10 |

*DS = Drive Score, RC = Route Completion, IS = Infraction Score*

### Key Results

- **458% improvement** over prior SOTA on driving score
- **32.6% improvement** over best concurrent submission
- **Camera-only** without any auxiliary labels (BEV, depth, segmentation)
- Demonstrates transfer of VLM pretraining to driving task

### Ablations (Table 2)

| Configuration | DS |
|---------------|-----|
| CarLLaVA default | **6.87** |
| - Path waypoints | 4.49 |
| - VLM pretraining (random init) | 0.45 |
| ResNet-34 backbone | 2.71 |
| Lower resolution (1300) | 3.93 |

**Key findings:**
- Path waypoints + semidisentangled representation critical (+2.38 DS)
- VLM pretraining massive impact (0.45 → 6.87)
- Resolution matters: 2100 optimal, 336px base resolution insufficient for distant objects

---

## Mapping to Tesla/Ashok Claims

### What Aligns ✓

| Tesla Claim | CarLLaVA Evidence |
|-------------|------------------|
| **Camera-first** | Camera-only, no LiDAR — matches Tesla's 8-camera philosophy |
| **End-to-end, no intermediate labels** | No BEV/depth/segmentation — pure camera → control |
| **Fleet data + interesting data mining** | Data bucketing strategy samples challenging scenarios, filters trivial data |
| **Foundation model leverage** | Uses LLaVA-NeXT (internet-scale VLM pretraining) — aligns with "foundational models for robotics" |
| **Inference engineering solvable** | Paper notes "recent works showed this is a solvable engineering problem" — same pragmatic view as Tesla |

### What Doesn't Align ✗

| Gap | Details |
|-----|---------|
| **Scale** | CARLA data (2.9M samples) vs Tesla fleet (500 years/day) — orders of magnitude gap |
| **Temporal context** | C1T2 uses single prior frame — Tesla emphasizes long context (~2B tokens) |
| **Long-tail evaluation** | No world-model simulator for regression testing — Tesla's learned simulator not replicated |
| **Real-world deployment** | CARLA simulation only — no on-road deployment demonstrated |
| **Multi-camera** | Single front view (C1T1) primary — Tesla uses 8 cameras with full surround |

---

## What to Borrow for AIResearch

### High Priority

1. **VLM pretraining transfer**: Use CLIP/ViT encoders pretrained on internet-scale data rather than training from scratch
   - Architecture: LLaVA-NeXT vision encoder → LLaMA decoder
   - Rationale: Massive improvement (0.45 → 6.87 DS) from VLM pretraining alone

2. **Semi-disentangled waypoint heads**: Separate path (lateral) + time-waypoints (longitudinal) with independent PID controllers
   - Rationale: Addresses representation trade-off in prior works
   - Implementation: Two learnable query sets → two MLP heads → separate PID

3. **Data bucketing for efficient training**: Create buckets for acceleration, steering, hazards, stop signs, etc.; oversample interesting scenarios
   - Rationale: Reduces compute waste on trivial straight-driving data
   - Implementation: Filter by acceleration thresholds, steering angle, object proximity

4. **High-resolution anyres**: Split high-res input into 336×336 patches for distant object detection
   - Rationale: Default CLIP resolution insufficient for traffic lights, pedestrians at distance

### Medium Priority

5. **Waypoint autoregression**: Direct prediction from transformer output features (vs GRU/sequential)
   - Simpler architecture, avoids GRU complexity

6. **Language explanation generation**: Joint training with driving explanations (optional LM loss)
   - Enables interpretability, could support reasoning traces

### Lower Priority (Future Work)

- **Temporal modeling**: Extend beyond single prior frame (C1T2 is start)
- **Multi-camera surround**: Full 360° coverage like Tesla's 8 cameras
- **World model integration**: Add learned simulator for long-tail evaluation

---

## Citations + Links

### Primary Paper
- **CarLLaVA: Vision Language Models for Camera-Only Closed-Loop Driving**
- arXiv:2406.10165 (June 2024)
- Authors: Katrin Renz*, Long Chen, Ana-Maria Marcu, Jan Hünermann, Benoit Hanotte, Alice Karnsund, Jamie Shotton, Elahe Arani, Oleg Sinavski (Wayve, University of Tübingen)
- **Winner: CARLA Autonomous Driving Challenge 2.0 (Sensor Track)**

### Key References

| Paper | Relevance |
|-------|-----------|
| LLaVA-NeXT (2024) | Vision encoder backbone |
| TCP (2023) | Prior camera-only SOTA on CARLA |
| TF++ (2022) | Camera-only with auxiliary labels |
| DriveGPT4 (2023) | LLM for driving (open-loop) |
| DriveMLM (2023) | MLLM for closed-loop driving |
| UniAD (2023) | Unified E2E (BEV-centric) |

### Links
- Paper: https://arxiv.org/abs/2406.10165
- Video: https://youtu.be/E1nsEgcHRuc
- CARLA Leaderboard: https://leaderboard.carla.org/

---

## Summary

**CarLLaVA** demonstrates that VLM pretraining transfers effectively to closed-loop driving, achieving SOTA on CARLA with camera-only input and no auxiliary labels. Key innovations are the semi-disentangled waypoint representation and data bucketing for efficient training. It aligns with Tesla's camera-first, foundation-model direction but lacks the scale, temporal context, and real-world deployment. For AIResearch, the VLM pretraining + high-res anyres + waypoint architecture are the most actionable takeaways.

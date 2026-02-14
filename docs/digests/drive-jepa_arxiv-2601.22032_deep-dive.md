# Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving — Digest

Source:
- Paper (arXiv v1): https://arxiv.org/abs/2601.22032v1 (HTML: https://arxiv.org/html/2601.22032v1 , PDF: https://arxiv.org/pdf/2601.22032v1.pdf)
- Code: https://github.com/linhanwang/Drive-JEPA

## TL;DR (5 bullets)
- Drive-JEPA targets two bottlenecks in end-to-end driving: (1) **video SSL/world-model pretraining that actually transfers to planning**, and (2) **multimodality** despite logs providing only a single human future.
- It adapts **V-JEPA** (predict future latents from masked views with EMA teacher / stop-grad) to driving video, yielding a **planning-aligned ViT encoder** trained efficiently at scale. (Method §3.1–3.2)
- Planning uses a **proposal-centric** trajectory generator: waypoint-anchored proposals are **iteratively refined** via waypoint-anchored deformable attention over BEV features. (Method §3.3)
- Multimodality is injected by **distilling simulator-scored pseudo-teacher trajectories** (selected from a clustered vocabulary) into the proposal distribution (“MTD”). (Method §3.4)
- A **momentum-aware selection** step re-scores candidate trajectories using comfort/temporal-consistency to reduce frame-to-frame jitter. (Method §3.5)

## Problem
- Video-generative world models can be expensive and may optimize irrelevant pixel details; latent dynamics objectives often don’t show clear scaling benefits for driving planning. (Intro §1)
- Driving futures are inherently multimodal, but each logged scene often provides **one** human trajectory → diversity collapses under pure BC supervision. (Intro §1)

## Method (by section)

### 3.1 Preliminary: End-to-end driving + V-JEPA recap
- E2E planner predicts future waypoints from multi-view images + ego status (command/speed/accel). (Method §3.1 “End-to-end Autonomous Driving”)
- V-JEPA: encoder + predictor minimize L1 between predicted masked target latents and stop-grad EMA target latents; loss only on masked positions; EMA stabilizes and mitigates collapse. (Eq. 1; Method §3.1 “V-JEPA”)

Cite: https://arxiv.org/html/2601.22032v1#S3.SS1

### 3.2 Driving Video Pretraining (planning-aligned video SSL)
- Initialize from V-JEPA 2 weights, then curate a **driving video dataset** from CoVLA, DrivingDojo, OpenScene.
- Pretrain on **front-view** 8-frame clips at 2 Hz, 512×256.
- Claim: can scale pretraining to **208 hours** efficiently due to latent prediction + collapse prevention. (Method §3.2)

Cite: https://arxiv.org/html/2601.22032v1#S3.SS2

Downstream (perception-free) planner described as a simple transformer decoder with learnable waypoint queries attending to encoder features (Eq. for decoder → MLP to waypoints). (Method §3.2)

### 3.3 Waypoint-anchored Proposal Generation (proposal-centric planner)
- Maintain Np proposals, each is an M-waypoint trajectory.
- Iteratively refine proposal queries; decode to waypoints each iteration; use predicted waypoints as anchors to sample/aggregate BEV features, updated via Waypoint-anchored Deformable Attention (WADA). (Method §3.3)
- Base guidance uses a min-over-proposals trajectory loss with discounted supervision over refinement iterations. (Eq. for \mathcal{L}_{traj} in §3.3)

Cite: https://arxiv.org/html/2601.22032v1#S3.SS3

### 3.4 Multimodal Trajectory Distillation (MTD)
- Build a trajectory vocabulary by clustering >100k trajectories; select 8192 centers.
- For each scene, score vocabulary trajectories with a rule-based simulator (NAVSIM v2-style EPDMS scoring) and select a set of high-quality multimodal pseudo-teachers \mathcal{P}_t.
- Train proposals against both the human trajectory and the pseudo-teachers via a min-over-proposals formulation (Eq. 2). This prevents mode collapse and encourages multimodal proposal distributions.

Cite: https://arxiv.org/html/2601.22032v1#S3.SS4

### 3.5 Momentum-aware Trajectory Selection
- Score proposals with an MLP head trained with BCE using simulator-derived supervision.
- Add a comfort/consistency term by comparing current proposals to the previously selected trajectory; re-weight scores (example: S ← (7S + S_c)/8) to reduce frame-to-frame distortion (jitter).

Cite: https://arxiv.org/html/2601.22032v1#S3.SS5

### 3.6 Losses
- Total loss combines: trajectory distillation, scoring, and lightweight auxiliary tasks (proposal-centric mapping + collision prediction).

Cite: https://arxiv.org/html/2601.22032v1#S3.SS6

## Data / Training
- Pretraining: 8 H800 GPUs, 50 epochs (paper).
- Planner training: 2× A30 GPUs, 20 epochs, batch size 64.
- Proposal count: Np=32.

Cite: https://arxiv.org/html/2601.22032v1#S4.SS2

Repo notes:
- The released code targets NAVSIM and uses a cache-heavy pipeline (feature caches + metric caches).

Cite: https://github.com/linhanwang/Drive-JEPA#training

## Evaluation
- Benchmarks: NAVSIM v1/v2, Bench2Drive.
- Metrics:
  - NAVSIM v1: PDMS (defined as product/aggregation over collision, drivable area compliance, TTC, ego progress, comfort). (Eq. 3 in §4.1)
  - NAVSIM v2: EPDMS extends with additional rule-compliance and comfort terms.
- Headline results (abstract): 93.3 PDMS (v1), 87.8 EPDMS (v2).

Cite: https://arxiv.org/abs/2601.22032v1

## Key takeaways
- **Planning-aligned** video SSL matters: V-JEPA’s latent prediction + collapse prevention is a plausible “scale-friendly” pretrain recipe versus pixel reconstruction.
- Proposal-centric planning + distillation from sim-scored pseudo-teachers is a clean way to inject multimodality beyond single-future logs.
- Comfort/jitter is real: diversity can hurt temporal consistency; a simple momentum-aware adjustment can help.

## Action items for this repo
1) **Pretrain objective**: add a JEPA-style masked latent prediction objective option (beyond contrastive) on our episodes backend.
2) **Proposal-centric waypoint head**: implement a simple “K-proposals + scorer” head for our spec (**20×(x,y)** ego-frame), keeping it behind a flag.
3) **Sim-distillation analogue**: in CARLA, generate K feasible trajectories per scene (rule-based filters) and distill into the proposal set (minimum-over-proposals loss + optional score supervision).
4) **Jitter metric**: add a temporal consistency metric to `metrics.json` (e.g., mean L2 change between consecutive predicted waypoint sequences) to quantify comfort-like stability.

## Citations
- Drive-JEPA paper (HTML): https://arxiv.org/html/2601.22032v1
- Drive-JEPA abstract + links: https://arxiv.org/abs/2601.22032v1
- Code / scripts / cache workflow: https://github.com/linhanwang/Drive-JEPA

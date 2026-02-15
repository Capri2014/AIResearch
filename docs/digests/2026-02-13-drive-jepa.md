# Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving — Digest

Source:
- Paper (arXiv v1): https://arxiv.org/abs/2601.22032v1 (PDF: https://arxiv.org/pdf/2601.22032v1.pdf)
- Code: https://github.com/linhanwang/Drive-JEPA

## TL;DR (5 bullets)
- Combines **video self-supervised pretraining (V-JEPA style)** with **trajectory learning** for end-to-end driving.
- Key motivation: pure “world model” style pretraining often yields **limited downstream planning gains**, and driving is **multimodal** but logs provide **one human trajectory** per scene.
- Approach: pretrain a **ViT encoder** on driving video with a JEPA predictive objective, then train a planner with **multimodal trajectory distillation** using **simulator-generated diverse trajectories** + human trajectories.
- Uses a **proposal-centric planner**: generate multiple trajectory proposals and learn a distribution over them.
- Reports strong NAVSIM results (paper abstract): **+3 PDMS** over prior methods in perception-free with a simple decoder; full method reaches **93.3 PDMS (v1)** and **87.8 EPDMS (v2)**.

## Problem
- End-to-end driving wants representations that transfer to planning.
- Video SSL can learn general scene features, but doesn’t necessarily align with planning.
- Driving ambiguity: each logged scene usually has only one demonstrated future; learning multimodal behavior is hard without additional supervision.

## Method (by section)
### 1) Driving video pretraining (V-JEPA-style)
- Adapt a **Video JEPA** objective to driving video.
- Pretrain a **ViT encoder** to produce predictive representations intended to be useful for planning.

What I’d want to extract next (from the PDF) once we parse the full text cleanly:
- What are the exact inputs (multi-camera? single?) and temporal context length?
- What is predicted (future latent tokens? masked patches?) and what losses are used?
- Any driving-specific augmentations / masking strategy?

### 2) Proposal-centric planning head
- Given pretrained visual features, generate **multiple** waypoint/trajectory proposals.
- Learn to score/select proposals rather than regress a single future.

### 3) Multimodal Trajectory Distillation (MTD)
- Distill **diverse simulator-generated trajectories** along with human trajectories.
- Intended effect: prevent mode collapse (repo visualization shows proposals become multimodal with MTD).

### 4) Momentum-aware trajectory selection
- Select final trajectory accounting for “cross-frame comfort” / stability.

## Data / Training
- Repo targets **NAVSIM v1/v2**, with a cache-first workflow (compute and store features / metric caches).
- Repo provides scripts for: feature caching, training, evaluation.

Source: GitHub README (Drive-JEPA) https://github.com/linhanwang/Drive-JEPA

## Evaluation
- Dataset/benchmark: NAVSIM.
- Metrics mentioned (paper abstract / repo): PDMS (v1), EPDMS (v2).

Source: arXiv abstract https://arxiv.org/abs/2601.22032v1

## Key takeaways
- If your end goal is **planning**, pure video SSL should be made **planning-aligned** (objective + decoder).
- Multimodality can be injected via **sim-generated proposals** + distillation, rather than only relying on logged single futures.
- Proposal + selection is a pragmatic route: easier to maintain diversity than direct regression.

## Action items for this repo
- [ ] Add a short note in our pipeline docs about **proposal-based waypoint heads** (vs single-regression waypoint BC).
- [ ] Consider adding an optional **“proposal generator + scorer”** baseline on top of our 20-waypoint spec.
- [ ] For SSL: evaluate whether our contrastive objective should be complemented with a **JEPA-style predictive** loss (masked spatiotemporal prediction in latent space).
- [ ] Identify the minimal “sim-trajectory distillation” analogue we can do with **CARLA ScenarioRunner**: generate K feasible trajectories per scene, distill a distribution or rank.

## Citations / anchored claims
- Motivation + headline results (abstract): https://arxiv.org/abs/2601.22032v1
- Pipeline description + training/eval scripts + cache workflow: https://github.com/linhanwang/Drive-JEPA

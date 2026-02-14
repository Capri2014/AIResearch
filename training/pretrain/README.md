# Pre-train (skeleton)

Pre-training is where you learn general-purpose representations before action-labeled fine-tuning.

## Typical objectives (examples)
- Vision-language: contrastive / captioning / masked modeling
- Video: masked video modeling / temporal contrast / future prediction
- World-model-ish: latent dynamics (optional)

## What we keep in this repo (for now)
We do **not** attempt to reproduce large-scale pretraining here.

Instead we define:
- **Backbone interface** (what gets loaded as "pretrained")
- **Dataset contract** (what a pretraining batch looks like)
- A stub script showing where pretrain artifacts would be written

## Repo scaffolding
- `training/pretrain/run_pretrain_stub.py` — dependency-free placeholder
- `training/pretrain/batch_contract.md` — defines the minimum batch dict format
- `training/pretrain/dataloader_episodes.py` — PyTorch episodes-backed dataloader (per-frame)
- `training/pretrain/train_ssl_stub_torch.py` — PyTorch SSL stub (placeholder objective)
- `training/pretrain/dataloader_temporal_pairs.py` — episodes-backed dataset yielding *(t, t+Δt)* frame pairs for temporal SSL
- `training/pretrain/train_ssl_temporal_contrastive_v0.py` — minimal temporal InfoNCE stub (anchor=t, positive=t+Δt)

## Why temporal contrastive is important (driving)
Multi-camera contrast (same timestamp, different views) is useful, but it does **not** teach invariance to motion, lighting changes, and small viewpoint shifts over time.

Temporal contrastive (t vs t+Δt) is a cheap, high-signal objective for driving video:
- **Stability:** encourages embeddings to change smoothly under normal ego motion.
- **Dynamics-aware features:** learns representations that track objects/lanes across frames.
- **Better transfer to waypoint policies:** waypoint prediction depends heavily on short-horizon temporal continuity.

In this repo we keep it minimal on purpose:
- start with a single camera (default `front`) so the objective is easy to debug
- keep the batch contract consistent (stacked tensors + validity masks)
- later combine temporal positives with multi-camera positives (multi-positive InfoNCE)

For a deeper implementation walk-through, see:
- `docs/temporal_ssl_infonce.md`

## Outputs
- A checkpoint (weights) for a backbone encoder
- Optionally a config file describing normalization/tokenization

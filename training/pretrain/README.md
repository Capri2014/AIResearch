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

## Outputs
- A checkpoint (weights) for a backbone encoder
- Optionally a config file describing normalization/tokenization

# Octo: Public Anchor for Robotics Foundation Model — Digest

Source: https://arxiv.org/abs/2405.12213 (project: https://octo-models.github.io/ ; code: https://github.com/octo-models/octo)

## TL;DR (3 bullets)
- **Octo** is the most reproducible open-source robotics foundation model: full training/finetuning code + pretrained checkpoints + 800k trajectory pretraining from Open X-Embodiment.
- **Diffusion policy** head outputs action distributions (not discrete tokens), handles 7D end-effector actions across 22+ robot embodiments; language OR goal-image instruction modality.
- Best fit for Tesla/Ashok "foundation model" narrative: transformer backbone, cross-embodiment transfer, efficient finetuning. Key gap: no fleet data engine, no world simulator, no vehicle dynamics.

## Problem
The Tesla/Ashok talk claims a "foundation model for robotics" but relies on internal fleet data. What publicly available baseline can the team benchmark against? Octo is the leading candidate: open code, open weights, proven transfer across robots.

## Dataset / Inputs / Outputs
### Pretraining Dataset
- **800k trajectories** from Open X-Embodiment (RLDS format), spanning 22+ robot embodiments.
- Language annotations available for subset; goal images for goal-conditioned variants.

### Model Inputs
- **RGB image** (workspace camera, variable resolution).
- **Instruction**: text string OR goal image (dual modality support).
- **Proprioception**: joint positions / end-effector pose when available.
- **History**: multi-frame temporal stacking supported.

### Model Outputs (Action Contract)
- **7D end-effector**: x, y, z, roll, pitch, yaw, gripper (absolute or delta).
- **Modular action head**: adapts to new robots via tokenization.
- **Diffusion output**: full action distribution (not point estimate).

## Training Objective
- **Transformer + Diffusion**: ViT encoder → transformer decoder with diffusion head.
- **Objective**: Denoising diffusion over continuous action space (imitation learning via diffusion).
- **Finetuning**: adapter-based, few hours on single A100.

## Evaluation Setup
- **9 robot platforms** evaluated with finetuning (100-500 demos per robot).
- **Metrics**: success rate; finetuned Octo outperforms training from scratch.
- **Positive scaling**: larger/more diverse pretrain → better transfer.

## What Maps to Tesla/Ashok Claims vs What Doesn't
### Maps Cleanly
- ✅ Foundation model pretraining on diverse data
- ✅ Diffusion-based action modeling (expressive distributions)
- ✅ Cross-embodiment transfer (9 robots proven)
- ✅ Efficient finetuning (few hours, consumer GPU)

### Doesn't Map
- ❌ Fleet data engine / active data mining
- ❌ World simulator / video generation
- ❌ End-to-end driving stack
- ❌ Real-time fleet deployment

## Action Items for AIResearch
- [ ] **Adopt diffusion policy head** over discrete BC for safety-critical control.
- [ ] **Standardize action contract**: 7D end-effector + modular head design.
- [ ] **Integrate RLDS loader** for dataset ingestion pipeline.
- [ ] **Build adapter finetuning pipeline** for new embodiments.

## Citations + Links
- Paper: https://arxiv.org/abs/2405.12213
- Code: https://github.com/octo-models/octo
- Website: https://octo-models.github.io/
- Open X-Embodiment: https://github.com/google-deepmind/open_x_embodiment
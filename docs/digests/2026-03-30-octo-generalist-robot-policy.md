# Octo: An Open-Source Generalist Robot Policy — Digest

Source: https://arxiv.org/abs/2405.12213 (project: https://octo-models.github.io/ ; code: https://github.com/octo-models/octo ; models: https://huggingface.co/rail-berkeley)

## TL;DR (5 bullets)
- **Octo** is an **open-source transformer-based diffusion policy** pretrained on **800k trajectories** from the **Open X-Embodiment** dataset — the largest open robotics dataset to date.
- Provides **pretrained base (93M) and small (27M) models** on HuggingFace with full training/finetuning code; reproducibility is strong with ~8-14h pretraining on TPUv4-128.
- Supports **multiple observation modalities** (RGB, language, goal images) and can be **finetuned to new action spaces** in hours on a single consumer GPU.
- Evaluation across **9 robotic platforms** demonstrates positive transfer; finetuning on new robots beats training from scratch, especially in low-data regimes.
- **Maps to Tesla/Ashok claims**: strong on **unified data contracts, cross-embodiment transfer, modular architecture, diffusion-based action generation**; gaps remain on **humanoid/full-body, factory-scale, long-horizon autonomy, fleet learning**.

## Problem
Robotics learning historically suffers from **siloed per-robot policies** and limited data sharing. The paper asks: can we build a **generalist pretrained policy** that (1) ingests diverse datasets, (2) transfers across robot morphologies, and (3) finetunes efficiently to new setups with minimal data?

Key differentiator from RT-X: Octo is **open-source**, uses **diffusion** rather than discrete action token prediction, and emphasizes **modular architecture** for easy finetuning to new action spaces.

## Dataset / Inputs / Outputs
### Pretraining data
- **Source:** Open X-Embodiment dataset — 800k trajectories from the full OXE pool (~1M+ available, Octo uses a filtered "magic soup" mix).
- **Total size:** ~1.2TB preprocessed; original dataset release is ~1M+ trajectories across 60 datasets, 22 robot embodiments.
- **Format:** RLDS episode format (sequence of episodes/steps with obs/action/metadata).

### Model inputs
- **Visual:** Multiple RGB cameras (workspace view, optional wrist camera) — supports arbitrary camera configs via tokenization.
- **Language:** Text instruction / task string (e.g., "pick up the spoon").
- **Goal images:** Alternative to language — can condition on a goal image rather than text.
- **History window:** 2 timesteps (current + previous observation); configurable.

### Model outputs (action contract)
- **Action chunking:** Pretrained with **action chunk size of 4** — predicts next 4 actions at once; can execute all or use receding horizon (execute first, resample).
- **Action space:** Variable by finetuning — defaults to 7D end-effector (position + rotation + gripper) but **modular readout heads** support new action spaces.
- **Modality:** **Continuous-valued diffusion** — not discrete tokens like RT-2-X.

## Training objective (BC / diffusion / etc.)
This is a **transformer-based diffusion policy**:
- **Diffusion objective:** Denoising diffusion over action sequences; uses standard DDPM-style training.
- **Transformer backbone:** Transformer encoder-decoder processing multimodal tokens (visual, language, proprio).
- **Tokenization:** Images tokenized via CNN/ResNet; language via transformer encoder; actions tokenized for diffusion.
- **Finetuning modes:** Three options — `head_only`, `head_mlp_only`, and `full` (full transformer finetuning).

**Comparison to RT-X:** RT-1-X uses discrete action token prediction (cross-entropy); RT-2-X co-finetunes a VLM with action tokens. Octo uses continuous diffusion, which is arguably more natural for continuous robot control and handles multimodality better.

## Evaluation setup
### Multi-platform transfer (9 robots)
- Tested on **9 different robot platforms** — WidowX, Franka, xArm, UR5, and others.
- Finetuning with **small target-domain datasets** (a few hundred trajectories) shows strong positive transfer vs. training from scratch.
- Reported: Octo finetuning outperforms training from scratch, especially in low-data regimes (consistent with RT-X findings).

### Reproducibility
- **Training:** Full training code provided; pretraining on 800k trajectories takes ~8h (small) to ~14h (base) on TPUv4-128.
- **Finetuning:** Works on single NVIDIA 4090 in hours.
- **Inference:** ~13 it/sec (base) / ~17 it/sec (small) on 4090.
- **Data loader:** Standalone PyTorch dataloader for Open X-Embodiment provided; dataset preprocessing script available.

### Gaps
- Evaluation is primarily **simulation / lab robot** — not factory floor or real-world long-horizon.
- Real-world success rates not fully quantified in publicly available metrics.

## What maps cleanly to Tesla / Ashok talk claims vs what doesn’t
### Maps cleanly
- **"Data is the moat"**: Octo builds on Open X-Embodiment's standardized RLDS format — the contract is the key artifact.
- **Cross-embodiment transfer**: Positive transfer demonstrated across 9 robot platforms; supports the "one model, many robots" thesis.
- **Diffusion for actions**: Diffusion-based continuous action generation maps to Tesla's interest in generative models for control.
- **Modular architecture**: Supports new observation/action spaces via finetuning — matches "adapt to new tasks quickly" narrative.
- **Language + goal conditioning**: Supports both text instructions and goal images — flexible instruction interface.
- **Open source**: Full code, pretrained models, data loaders — reproducibility is strong.

### Doesn’t map (or not demonstrated)
- **Humanoid / full-body control**: Octo focuses on **manipulation** (arm + gripper); no locomotion, balance, or whole-body control.
- **Long-horizon autonomy**: Evaluations are short-horizon, single-task; no multi-hour autonomy or deep planning demonstrated.
- **Factory-scale deployment**: Lab setups; no demonstration of robustness to grease, clutter, safety constraints, or continuous operation.
- **Fleet learning**: Pretraining on static dataset; not a continuously learning system receiving data from a fleet.
- **End-to-end sensing-to-control**: Modular tokenization pipeline; not a fully monolithic end-to-end differentiable stack.

## Action items for AIResearch (interfaces / contracts to copy)
- [ ] **Adopt RLDS episode format** as canonical on-disk schema for manipulation data — matches Octo's data pipeline and Open X-Embodiment standard.
- [ ] **Build modular tokenization layer** — Octo's approach separates visual, language, and action tokenizers; enables swapping in new sensors/action spaces without retraining from scratch.
- [ ] **Use diffusion over discrete tokens** for action generation — continuous diffusion handles continuous control more naturally than discrete action token classification.
- [ ] **Implement action chunking (size=4)** — predict multiple future actions; enables receding horizon control and smoother execution.
- [ ] **Design for multi-platform transfer from day 1** — architecture should support arbitrary observation/action spaces via configurable tokenizers and readout heads.
- [ ] **Provide standalone data loader** — Octo ships a standalone PyTorch dataloader for Open X-Embodiment; mirror this for internal datasets to enable external researchers to plug in.
- [ ] **Benchmark with "finetune from Octo" vs "train from scratch"** — quantify transfer gains in low-data regimes; make it a standard baseline.

## Citations / links
- Paper (arXiv): https://arxiv.org/abs/2405.12213
- Project website: https://octo-models.github.io/
- GitHub repo (training/finetuning code): https://github.com/octo-models/octo
- Pretrained models (HuggingFace): https://huggingface.co/rail-berkeley
- Inference Colab: https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz
- Open X-Embodiment dataset: https://robotics-transformer-x.github.io/
- RLDS format reference: https://github.com/google-research/rlds
- BibTeX citation included in repo:
```
@article{octo_2023,
  title={Octo: An Open-Source Generalist Robot Policy},
  author = {{Octo Model Team} and Dibya Ghosh and Homer Walke and Karl Pertsch and ...},
  booktitle = {Proceedings of Robotics: Science and Systems},
  address = {Delft, Netherlands},
  year = {2024},
}
```
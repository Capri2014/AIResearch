# Octo: Open-Source Generalist Robot Policy — Digest

Source: https://octo-models.github.io/ | Code: https://github.com/octo-models/octo | Paper: arXiv (to appear in RSS 2024)

## TL;DR (5 bullets)
- **Octo** is a **transformer-based diffusion policy** pretrained on **800k robot episodes** from Open X-Embodiment; two sizes: **Octo-Small (27M)** and **Octo-Base (93M)** params.
- Fully **open-source** (MIT license) with full training/finetuning scripts, HuggingFace checkpoints, and Colab notebooks — the most reproducible generalist robot policy available.
- Supports **flexible observation/action spaces**: multiple RGB cameras, language instructions or goal images, can finetune to new proprioceptive inputs (force-torque) and action spaces (joint position vs end-effector).
- Zero-shot on WidowX: **0.50/0.70/0.80** vs RT-1-X (0.20/0.35/0.60); matches RT-2-X (55B params) while being 600× smaller. Finetuning on 100 demos: **72% avg** vs next best 52% improvement.
- Clean mapping to Tesla/Ashok: diffusion-based action generation, modular architecture for new embodiments, unified data contract from Open X; gaps: still manipulation-focused, not humanoid/full-body, no long-horizon autonomy demonstrated.

## Problem
Robot learning has been stuck on siloed policies per robot/task. The field needs a generalist policy that (1) pretrains on diverse data, (2) adapts quickly to new robots/tasks with minimal finetuning, and (3) provides an open, reproducible baseline for the research community.

## Dataset / Inputs / Outputs
### Pretraining Data
- **800k trajectories** from 25 datasets in Open X-Embodiment (heterogeneous: 22 robot embodiments, varied sensors, mixed language labels).
- Total dataset size ~1.2TB after preprocessing.
- Data loading via **RLDS** format + custom Open X-Embodiment (OXE) dataloader.

### Model inputs (flexible)
- **Multiple RGB cameras** (workspace + optional wrist)
- **Language instructions** OR **goal images** (task specification)
- **Observation history** (temporal context)
- **Proprioception** (varies by robot; supports adding new modalities at finetune time)

### Model outputs (action contract)
- **Diffusion-based action generation**: denoise a noisy action sequence rather than predict single next action.
- **Action dimensions**: configurable at finetune time (default: 7D end-effector, but supports joint position control, etc.).
- Supports both **absolute** and **delta** action spaces.

## Training objective (BC / diffusion / etc.)
- **Diffusion policy objective**: train a transformer to denoise action sequences via iterative refinement.
- Training uses **classifier-free guidance** (10% drop of task conditioning) for improved robustness.
- **Architecture**: Transformer backbone with modular tokenizers for images/text; attention over observation history.
- **Finetuning modes**: head_only, head_mlp_only, full (configurable which layers to freeze).
- Training compute: TPUv4-128 pod, 8h for Octo-S, 14h for Octo-B.

## Evaluation setup
### Zero-shot ( WidowX UR5, 3 tasks)
| Task | RT-1-X | RT-2-X | Octo |
|------|--------|--------|------|
| WidowX | 0.20 | 0.50 | **0.50** |
| Bridge | 0.35 | — | **0.70** |
| RT-1 | 0.60 | 0.85 | **0.80** |

### Finetuning (100 demos per task, 6 setups)
| Setup | From Scratch | VC-1 | Octo |
|-------|--------------|------|------|
| CMU Baking | 0.25 | 0.30 | **0.50** |
| Stanford Coffee | 0.45 | 0.00 | **0.75** |
| Berkeley Peg Insert* | 0.10 | 0.05 | **0.70** |
| Berkeley Pick-Up† | 0.00 | 0.00 | **0.60** |
| Berkeley Bimanual† | 0.20 | 0.50 | **0.80** |
| Berkeley Coke | 0.20 | 0.10 | **1.00** |
| **Average** | 0.20 | 0.15 | **0.72** |

*New observation input (force-torque) | †New action space (joint position)

### Reproducibility
- ✅ Full training code (JAX)
- ✅ Finetuning scripts with config files
- ✅ HuggingFace model hub (rail-berkeley/octo-small-1.5, octo-base-1.5)
- ✅ Colab notebooks for inference/finetuning
- ✅ Standard eval envs (Gym interface)
- ⚠️ Real robot eval requires hardware access (WidowX, etc.)

## What maps cleanly to Tesla / Ashok talk claims vs what doesn’t
### Maps cleanly
- **"Foundation model for robotics"**: Octo is the closest real instantiation — pretrained on diverse data, generalizes across embodiments, adapts via finetuning.
- **Diffusion for action**: Tesla/Ashok mentioned diffusion-style generation; Octo uses diffusion objectives (no other open generalist does).
- **Modular architecture**: supports adding new observations (force-torque) and action spaces (joint control) — matches "interfaces to copy" narrative.
- **Unified data contract**: leverages Open X-Embodiment's RLDS schema; demonstrates heterogeneous data works.

### Doesn’t map (or not demonstrated)
- **Humanoid / full-body control**: All eval on robot arms; no locomotion, balance, or whole-body contact.
- **Long-horizon autonomy**: Tasks are short-horizon (seconds to minutes), not hours of continuous operation.
- **Factory-grade robustness**: Lab evaluation, not factory floor distribution (grease, occlusion, safety).
- **Fleet-scale closed-loop**: Pretraining + finetuning recipe, not a deployed continuous learning system.
- **End-to-end sensing-to-control**: Requires explicit task specification (language/goal image); no autonomous task discovery.

## Action items for AIResearch (interfaces / contracts to copy)
- [ ] **Adopt diffusion policy objective** for next-gen manipulation policies (Octo's architecture is the reference implementation).
- [ ] **Use RLDS/OXE dataloader** for heterogeneous robot data — mirrors Open X's data contract.
- [ ] **Design modular observation tokenizers** — Octo's key innovation is swapping in new cameras/proprio at finetune time.
- [ ] **Default to language + goal-image conditioning** — both supported; goal images gave +25% on WidowX.
- [ ] **Create a "golden finetuning recipe"** — Octo shows 100-demos achieves strong transfer; document the config (head_mlp_only, lr=1e-4, etc.).
- [ ] **Benchmark on new action spaces** — Octo demonstrates joint-position control; consider this for humanoid arms.

## Citations / links
- Project site: https://octo-models.github.io/
- GitHub (MIT): https://github.com/octo-models/octo
- HuggingFace checkpoints: https://huggingface.co/rail-berkeley
- Colab inference: https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz
- BibTeX (RSS 2024): `@inproceedings{octo_2023, title={Octo: An Open-Source Generalist Robot Policy}, ...}`
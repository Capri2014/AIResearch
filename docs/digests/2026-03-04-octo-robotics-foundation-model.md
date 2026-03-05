# Octo: An Open-Source Generalist Robot Policy — Digest

Source: https://octo-models.github.io/ (arXiv: https://arxiv.org/abs/2405.12213 ; code: https://github.com/octo-models/octo)

## TL;DR (5 bullets)
- Octo is an **open-source transformer-based diffusion policy** pretrained on **800k episodes** from the Open X-Embodiment dataset, with **27M (Small)** and **93M (Base)** parameter variants — far more reproducible than RT-2-X's 55B.
- Supports **flexible observation/action spaces**: language instructions OR goal images, multiple camera views, proprioception, and can adapt to new action spaces (joint position, delta controls) via finetuning.
- Training objective is **diffusion**: denoising model predicts action sequences; outputs actions via **diffusion decoding** (not discrete token prediction like RT-2-X).
- Zero-shot matches RT-2-X on WidowX tasks; finetuning yields **52% average improvement** over next-best baseline across 6 real robot setups with only ~100 demonstrations.
- Clean mapping to Tesla/Ashok: **unified data contracts, cross-embodiment transfer, efficient finetuning**; gaps: **not humanoid-scale, no long-horizon autonomy demos, factory-specific sim-to-real not addressed**.

## Problem

Can we build an open-source, generalist robot policy that (1) works across many robot embodiments, (2) supports flexible observation/action configurations, and (3) can be efficiently finetuned to new tasks with minimal data? Octo answers by pretraining a diffusion transformer on the Open X-Embodiment mixture and designing for modularity.

## Dataset / Inputs / Outputs

### Dataset (training)
- **800k robot episodes** from **25 datasets** within the Open X-Embodiment collection.
- Heterogeneous: various robot embodiments, scenes, sensors (with/without wrist cameras), and labels (with/without language).
- Standardized via RLDS episode format.

### Model inputs
- **Language instruction**: natural language task description (e.g., "pick up the cube").
- **Goal image**: alternative conditioning via target image (Octo uniquely supports this; outperforms language on WidowX by 25%).
- **Observation history**: multiple camera views + proprioceptive state.
- **Flexible tokenization**: designed to accommodate new observation types (e.g., force-torque) and action spaces.

### Model outputs (action contract)
- **7D end-effector actions**: x, y, z, roll, pitch, yaw, gripper open/close (or deltas/rates).
- Supports **reparameterization** for different action spaces: joint position control, end-effector delta, etc.
- Actions are generated via **diffusion decoding** (multiple denoising steps).

## Training objective

**Diffusion policy** (not behavior cloning):
- Transformer backbone (decoder-only) processes observations and language/goal conditioning.
- Trains as a **conditional denoising diffusion probabilistic model (DDPM)**: given noisy action sequences, predict noise to remove; during inference, iteratively denoise to produce actions.
- Uses **action chunking**: predicts sequences of future actions (not single-step).
- Same finetuning recipe works across all evaluation tasks — no task-specific architecture changes.

## Evaluation setup

### Zero-shot (out-of-the-box)
- Evaluated on **9 real robot setups across 4 institutions** (WidowX BridgeV2, Stanford Coffee, Berkeley Peg Insert, Berkeley Pick-Up, Berkeley Bimanual, Berkeley Coke, etc.).
- With language conditioning: **0.50 / 0.70 / 0.80** on WidowX tasks (vs RT-1-X's 0.20/0.35/0.60 and RT-2-X's 0.50/—/0.85).
- With **goal image conditioning**: 25% higher average success than language on WidowX.

### Finetuning
- Finetuned on ~100 target demonstrations per task.
- Same hyperparameter/config for all tasks.
- Results (average success rate):
  - From scratch: 20%
  - VC-1 (prior SOTA pretrained): 15%
  - **Octo finetuned: 72%** (52% improvement over next-best).

### Novel adaptation tests
- **New observations**: force-torque inputs for Berkeley Peg Insert → worked without architecture changes.
- **New action space**: joint position control for Berkeley Pick-Up → handled via same finetuning recipe.
- **New embodiments**: Berkeley Bi-Manual, Berkeley Coke → worked.

## Mapping to Tesla/Ashok Claims

### What maps cleanly
- **Foundation model pretraining on diverse robot data**: Octo shows positive transfer from large-scale pretraining (800k episodes, 25 datasets).
- **Unified action/observation contracts**: 7D end-effector standardized; flexible obs tokenization.
- **Efficient finetuning**: 100 demos yields strong policies — matches the "fast adaptation" narrative.
- **Cross-embodiment transfer**: single policy works across WidowX, bi-manual, different robots.

### What doesn't map (gaps)
- **Scale**: 93M params is tiny vs Tesla's fleet-scale aspirations; no evidence of **billion-parameter** real-time inference.
- **Long-horizon autonomy**: eval tasks are short-horizon (pick-place, insert); no complex multi-stage routines.
- **Humanoid/full-body control**: only arm-level manipulation; no legged locomotion or bimanual coordination beyond dual-arm.
- **Factory/safety**: no sim-to-real on Tesla-specific hardware; no compliance with industrial safety standards.
- **End-to-end sensory-motor**: still uses modular perception → action; no continuous sensor fusion beyond cameras + proprio.

## Action Items for AIResearch

1. **Adopt Octo's action contract**: standardize 7D end-effector format (x,y,z,roll,pitch,yaw,gripper) as internal API — simplifies benchmarking and model swapping.
2. **Build diffusion policy infrastructure**: Octo's code is open; replicate training pipeline on Tesla internal data mixture to establish baseline.
3. **Explore goal-image conditioning**: Octo shows goal images beat language by 25% — relevant for Tesla's "demonstration learning" use case; prototype as alternative to teleop language prompts.
4. **Finetuning recipe as contract**: Octo's finetuning config should become the "standard adapter" — any new task/robot should be finetuned from Octo rather than trained from scratch.
5. **Add force/torque to obs space**: Octo supports new observations without architecture changes; prioritize force-feedback integration for insertion tasks.

## Citations

```
@article{octo_2024,
  title={Octo: An Open-Source Generalist Robot Policy},
  author = {Octo Model Team and Dibya Ghosne and Homer Walke and Karl Pertsch and Kevin Black and Oier Mees and Sudeep Dasari and Joey Hejna and Charles Xu and Jianlan Luo and Tobias Kreiman and You Liang Tan and Lawrence Yunliang Chen and Pannag Sanketi and Quan Vuong and Ted Xiao and Dorsa Sadigh and Chelsea Finn and Sergey Levine},
  booktitle = {Proceedings of Robotics: Science and Systems},
  address = {Delft, Netherlands},
  year = {2024}
}
```

- Project site: https://octo-models.github.io/
- GitHub: https://github.com/octo-models/octo
- Paper: https://arxiv.org/abs/2405.12213
- Open X-Embodiment dataset: https://github.com/google-deepmind/open_x_embodiment

---

*Digest created: 2026-03-04*
*Updated for Survey PR #3: 2026-03-05*

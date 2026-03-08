# Octo: An Open-Source Generalist Robot Policy — Digest

Source: https://octo-models.github.io/ | Paper: https://proceedings.mlr.press/v1/2024a/fu4c3a12.html | Code: https://github.com/octo-models/octo

## TL;DR (5 bullets)
- **Octo** is a **transformer-based diffusion policy** trained on **800k robot trajectories** from the Open X-Embodiment dataset — making it the most reproducible open robotics foundation model baseline available today.
- Model variants: **Octo-Base** (93M params, 13 it/sec on 4090) and **Octo-Small** (27M params, 17 it/sec). MIT-licensed, with **pretrained checkpoints on HuggingFace** (`rail-berkeley/octo-base-1.5`, `rail-berkeley/octo-small-1.5`).
- Supports **multiple RGB camera inputs**, **language commands** or **goal images** as instructions, and **action chunking** (predicts next 4 actions at once).
- Out-of-the-box **modular cross-attention** allows finetuning to new observation/action spaces with small datasets — key for domain adaptation.
- Clean mapping to Tesla/Ashok: diffusion objective is more aligned with "world model" thinking than BC; modular architecture supports new embodiments; what doesn't map: **no fleet data loop**, **no 3D scene reasoning**, **no end-to-end driving/manipulation unity**.

## Problem
Robot learning has historically been siloed: separate policies per robot, task, and environment. The field lacks a "foundation model" that (1) pretrains on diverse data, (2) generalizes across embodiments, and (3) finetunes efficiently to new domains with modest data. Octo addresses this by:

1. Training a **generalist policy** on a large, heterogeneous dataset (800k trajectories from Open X-Embodiment).
2. Using a **modular transformer architecture** that can ingest arbitrary observation/action tokens and adapt via finetuning.
3. Releasing **fully reproducible code** — pretrained weights, training scripts, Colab examples, and evaluation harnesses.

## Dataset / Inputs / Outputs

### Pretraining data
- **Source**: Open X-Embodiment dataset (60 datasets, 22 robot embodiments, 1M+ trajectories pooled from 34 labs). Octo uses a processed subset of **800k trajectories**.
- **Format**: RLDS episode format, loaded via the `rlds_dataset_mod` package or Octo's built-in Open X-Embodiment (OXE) dataloader.
- **Scale**: ~1.2TB pre-processed (requires significant storage but fully downloadable).

### Model inputs
- **Vision**: Multiple RGB camera streams (workspace + optional wrist). Octo tokenizes images via ViT tokenizers.
- **Language**: Text instructions (e.g., "pick up the spoon", "open the drawer"). The model was augmented with GPT-3.5 rephrasings of language instructions for robustness.
- **Goal images**: Alternative to language — can condition on a goal image rather than text.
- **Observation history**: Window size of 2 (current + previous observation); `timestep_pad_mask` handles missing history at episode start.

### Model outputs (action contract)
- **Action space**: Variable-length action vectors depending on the target robot. Octo uses **action chunking** (predicts the next 4 actions per forward pass).
- **Representation**: Continuous-valued action tokens (diffusion decoder outputs), not discretized.
- **Finetuning flexibility**: The readout head can be swapped to accommodate new action dimensions — key for adapting to new robot embodiments.

## Training objective (BC / diffusion / etc.)

**Octo is a diffusion policy, NOT behavior cloning.**

- **Objective**: Denoising diffusion probabilistic model (DDPM)-style training. The model learns to reverse a noise process to generate action sequences from latent action tokens.
- **Why diffusion?**: Compared to behavior cloning (RT-X style), diffusion policies have shown stronger performance on long-horizon tasks, better handling of multimodal action distributions, and more stable training.
- **Architecture**:
  - **Tokenizers**: Image tokens (ViT), language tokens (BERT-style), proprio tokens.
  - **Transformer backbone**: Token sequences → transformer → cross-attention between modalities.
  - **Diffusion head**: Outputs action tokens via a diffusion decoding process.
- **Training details**:
  - Octo-Base: ~14 hours on TPUv4-128
  - Octo-Small: ~8 hours on TPUv4-128
  - Uses JAX (Flax/Haiku).

## Evaluation setup

### Zero-shot / few-shot evaluation
- Octo provides **Colab examples** for zero-shot inference and finetuning:
  - [01_inference_pretrained.ipynb](https://githubtocolab.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb)
  - [02_finetune_new_observation_action.py](examples/02_finetune_new_observation_action.py)
- Pretrained models evaluated on **in-distribution** tasks from the Open X-Embodiment mixture.

### Finetuning evaluation
- **Simulated environments**: Gym wrapper interface for rollout evaluation (`examples/03_eval_finetuned.py`).
- **Real robot**: WidowX evaluation example (`examples/04_eval_finetuned_on_robot.py`).
- **Benchmark modes**: Three finetuning modes — `head_only`, `head_mlp_only`, `full` — with options for `image_conditioned`, `language_conditioned`, or `multimodal` tasks.

### Reproducibility strengths
- ✅ Checkpoints on HuggingFace (`hf://rail-berkeley/octo-base-1.5`)
- ✅ Training config + script (`scripts/train.py`, `scripts/configs/octo_pretrain_config.py`)
- ✅ Finetuning scripts + examples
- ✅ Open X-Embodiment dataloader integrated
- ⚠️ Requires significant compute (TPU) for full pretraining; JAX ecosystem may have a learning curve

### Reproducibility gaps
- No fleet data engine (obviously — this is research code)
- No built-in simulator integration for closed-loop eval at scale
- Real-robot evaluation still requires hardware access

## What maps cleanly to Tesla / Ashok talk claims vs what doesn't

### Maps cleanly
- **"Foundation model for robotics"**: Octo is literally this — a pretrained transformer backbone that can finetune to new embodiments/tasks.
- **Diffusion objective aligns with world modeling**: Ashok mentions learning "next state / next video" prediction; diffusion policies model action distributions similarly (denoising = planning).
- **Modular attention for new sensors/embodiments**: Octo's cross-attention between visual/language/proprio tokens mirrors the talk's emphasis on "one network" that absorbs heterogeneous inputs.
- **Language as API**: Task strings / goal images as the instruction interface is directly aligned with "language as the API" narrative.
- **Action chunking (temporal abstraction)**: Predicting 4 actions at once addresses "control frequency" concerns (Ashok mentions 36 Hz).

### Doesn't map (or is not demonstrated)
- **Fleet data loop**: Octo is a one-time pretraining; no continuous data collection, interesting-data mining, or fleet-scale refinement.
- **3D geometric reasoning**: No Gaussian splatting or explicit 3D scene representation — pure image-based policy.
- **End-to-end driving + manipulation**: Octo focuses on manipulation; no driving domain, no camera-to-control across vehicle platforms.
- **Full-body humanoid control**: Action space is arm/EEF focused; no balance, locomotion, or whole-body dynamics.
- **Real-time safety gating**: No explicit safety layer, collision checking, or runtime verification built in.
- **Long-horizon autonomy**: Evaluations are short-horizon manipulation tasks; no hours-long autonomy demonstrated.

## Action items for AIResearch (interfaces / contracts to copy)

- [ ] **Adopt diffusion policy objective** for future manipulation policies — better long-horizon stability than BC.
- [ ] **Use action chunking** (Octo uses chunk size 4) to reduce control frequency and improve temporal coherence.
- [ ] **Mirror Octo's tokenization schema** for observations: ViT for images, BERT for language, linear for proprio. This creates a modular contract that new sensors can plug into.
- [ ] **Implement the finetuning modes** (`head_only`, `head_mlp_only`, `full`) as a standard playbook for domain adaptation — useful for rapidly deploying to new robots/tasks.
- [ ] **Add goal-image conditioning** as an alternative to language — useful for vision-based task specification without verbal annotations.
- [ ] **Create a "golden loader" Colab** similar to Octo's examples that: (1) loads a pretrained policy, (2) visualizes an episode, (3) runs inference, (4) overlays predicted actions. This accelerates experimentation.
- [ ] **Explore JAX/Flax** for large-scale transformer training if not already — Octo's training pipeline is well-structured and reproducible.

## Citations / links

- **Project page**: https://octo-models.github.io/
- **Paper (RSS 2024)**: https://proceedings.mlr.press/v1/2024a/fu4c3a12.html
- **GitHub repo**: https://github.com/octo-models/octo
- **HuggingFace checkpoints**: https://huggingface.co/rail-berkeley
  - Octo-Base: https://huggingface.co/rail-berkeley/octo-base
  - Octo-Small: https://huggingface.co/rail-berkeley/octo-small
- **Inference Colab**: https://githubtocolab.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb
- **Open X-Embodiment dataset**: https://robotics-transformer-x.github.io/
- **RLDS format**: https://github.com/google-research/rlds
- **Diffusion Policy background**: https://arxiv.org/abs/2304.13705 (Diffusion Policy: Visuomotor Policy Learning via Action Diffusion)

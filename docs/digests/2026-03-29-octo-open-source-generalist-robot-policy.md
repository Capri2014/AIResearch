# Octo: An Open-Source Generalist Robot Policy — public anchor digest

Source paper: https://arxiv.org/abs/2405.12213  
Project site: https://octo-models.github.io/  
Code + checkpoints + examples: https://github.com/octo-models/octo

## TL;DR

- **Pick Octo over RT-X as the public anchor**: it is the stronger reproducibility baseline because it ships the full stack — paper, code, training loop, finetuning loop, checkpoints, dataloaders, and example evaluation scripts.
- Octo is a **transformer-based diffusion policy** pretrained on **800k robot trajectories** curated from **25 Open X-Embodiment datasets** (drawn from the broader ~1.5M-episode Open X pool).
- Inputs are a flexible mix of **language instructions, goal images, multi-camera RGB observations, observation history, and optional downstream proprioception**; outputs are **continuous chunked actions**, with the pretrained OXE setup centered on end-effector-style manipulation control.
- Evaluation is unusually practical for an open robotics FM: **9 real robot setups across 4 institutions**, with both **zero-shot multi-robot control** and **~100-demo finetuning** to new observations, new action spaces, and new embodiments.
- The clean takeaway for AIResearch is less “copy this exact manipulation policy” and more “copy these contracts”: **modality-keyed observation dictionaries, pad masks for missing sensors, task/action separation, chunked action decoding, and adapter-friendly finetuning paths**.

## Why Octo is the better public anchor than RT-X

RT-X/Open X-Embodiment matters as the data standardization story, but **Octo is the better baseline to anchor implementation work** because it is the more complete public package:

- **open training code** (`scripts/train.py`)
- **open finetuning code** (`scripts/finetune.py`)
- **open checkpoints** on Hugging Face
- **open examples** for inference, finetuning, gym rollout, and real-robot eval
- **open Open-X dataloading path** including a standalone PyTorch loader

That makes Octo the closest thing to a copyable “robot foundation model baseline” rather than just a paper + dataset release.

## Dataset / inputs / outputs

### Dataset

Octo is pretrained on a curated subset of **25 datasets** from **Open X-Embodiment**.

Key facts from the paper/project/repo:
- **800k robot trajectories** used for Octo pretraining.
- The broader **Open X-Embodiment** pool is described as roughly **1.5M robot episodes**.
- The Octo mixture keeps datasets with:
  - **image observations**
  - **delta end-effector control**
  - reasonably diverse, non-niche behaviors
- The mixture is heterogeneous across:
  - robot embodiments
  - camera setups
  - language availability
  - tasks, scenes, and labs

Important preprocessing / interface choices:
- missing camera channels are **zero-padded**
- gripper conventions are aligned so **+1 = open** and **0 = closed**
- training uses a curated/reweighted mixture rather than a naive flat union
- the repo documents an Open-X download / preprocessing path, with the processed pretraining data at roughly **~1.2 TB**

### Inputs

Octo’s input contract is one of the most useful parts to copy.

**Task inputs**
- **language instruction**
- **goal image**

**Observation inputs**
- **primary RGB camera**
- optional **wrist RGB camera**
- **observation history** (the released setup uses a history window of 2)
- optional downstream modalities during finetuning, e.g. **force-torque / proprio**

**Implementation-level contract worth noting**
- observations are represented as a **dictionary keyed by modality/channel**, e.g. `image_primary`, `image_wrist`, `proprio`
- `timestep_pad_mask` says which timesteps in the history window are valid
- `pad_mask_dict` says which modalities are present or absent for a given example

This is a very clean way to support heterogeneous datasets without turning every new sensor combination into a custom model fork.

### Outputs

- Octo predicts **continuous actions** using a diffusion head.
- The released pretrained models use **action chunking** and predict the **next 4 actions at once**.
- In the OXE-style pretrained setup, this is effectively the standard manipulation contract around **end-effector motion + gripper command**.
- During finetuning, Octo can swap to **new action spaces** by replacing / adding a lightweight action head while keeping most pretrained weights.

For AIResearch, the important lesson is not the exact 7D manipulation action space; it is the **chunked action interface with adapter-friendly output heads**.

## Training objective

Octo is best understood as **behavior cloning / imitation learning with a diffusion action decoder**.

More specifically:
- backbone: **transformer policy** over tokenized task + observation inputs
- action decoder: **conditional diffusion head**
- training loss: **standard DDPM denoising objective** on continuous actions
- output form: **chunked future actions** rather than only next-step regression

Why this matters:
- the paper explicitly positions diffusion decoding as better suited than plain **MSE regression** or **discretized action classification** for modeling **multi-modal continuous action distributions**
- only **one transformer forward pass** is needed per action prediction; the multi-step denoising happens in the smaller action head

Finetuning story:
- Octo is designed to finetune to **new observations**, **new action spaces**, and **new robot morphologies**
- the abstract/project site emphasize this can be done within **a few hours on consumer GPUs / standard GPUs**, which is part of why Octo is a practical public baseline rather than just a scale demo

## Evaluation setup

Octo is evaluated on **9 real robot setups across 4 institutions**.

The evaluation has two useful regimes:

### 1) Zero-shot / out-of-the-box control
Goal: can one pretrained policy directly control multiple robots in environments from the pretraining distribution?

Project-site headline results:

| Model | WidowX | UR5 | RT-1 Robot |
|---|---:|---:|---:|
| RT-1-X | 0.20 | 0.35 | 0.60 |
| RT-2-X | 0.50 | — | 0.85 |
| **Octo** | **0.50** | **0.70** | **0.80** |

Notes:
- Octo **outperforms RT-1-X** in out-of-the-box language-conditioned control.
- It is **competitive with RT-2-X** while being vastly smaller and actually open.
- Octo also supports **goal image conditioning**, and the project site reports **~25% higher average success on WidowX tasks** with goal images than with language alone.

### 2) Small-data finetuning
Goal: does pretraining give a better initialization for new tasks, sensors, and action spaces?

Project-site headline results (each task uses **~100 target demonstrations**):

| Method | CMU Baking | Stanford Coffee | Berkeley Peg Insert* | Berkeley Pick-Up† | Berkeley Bimanual† | Berkeley Coke | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| From Scratch | 0.25 | 0.45 | 0.10 | 0.00 | 0.20 | 0.20 | 0.20 |
| VC-1 | 0.30 | 0.00 | 0.05 | 0.00 | 0.50 | 0.10 | 0.15 |
| **Octo** | **0.50** | **0.75** | **0.70** | **0.60** | **0.80** | **1.00** | **0.72** |

\* new observation input (force-torque)  
† new action space (joint position control)

What this establishes:
- Octo is not just a frozen zero-shot policy.
- It works as a **strong initialization** for adaptation to:
  - new sensors
  - new control spaces
  - new embodiments
  - longer-horizon manipulation tasks

### Reproducibility reality check

What is genuinely reproducible:
- loading pretrained checkpoints
- running inference
- reproducing the model/data contracts
- reproducing the published pretraining / finetuning pipeline
- adapting the model to a new gym or robot environment

What is less reproducible:
- exact real-robot benchmark replication across 9 setups, because hardware, calibration, environments, and teleop pipelines are institution-specific

Still, compared with most robotics FM papers, **Octo is unusually reproducible**.

## What maps cleanly to Tesla / Ashok talk claims vs what does not

Reference Tesla digest in this repo: `docs/digests/2026-02-12-tesla-foundational-models.md`

### Maps cleanly

**1) One shared policy over heterogeneous robot data**
- Tesla/Ashok’s public framing is “foundational models for robotics.”
- Octo is a real public example of **one pretrained policy over many embodiments/tasks** instead of one policy per robot.

**2) Camera-first end-to-end control**
- Octo is fundamentally a **pixels/task → actions** policy.
- It is not modular in the classic perception/planning/control-stack sense.

**3) Standardized data/IO contracts matter more than individual model tricks**
- The biggest transferable idea is not “use diffusion” by itself.
- It is the combination of **shared observation keys, task fields, action contracts, pad masks, and finetuning adapters** that lets one model span many datasets.

**4) Pretrain broad, adapt cheaply**
- Octo supports the claim that broad pretraining can produce a useful initialization that adapts with **small target-domain data**.

### Does not map cleanly

**1) No world simulator / learned environment model**
- Tesla/Ashok emphasizes simulator / generated video / regression testing.
- Octo is a policy paper, not a learned world-model or simulator paper.

**2) No 3D geometric latent like Gaussian splats**
- Tesla’s public talk leans on explicit 3D scene reasoning.
- Octo is image-token based, not a public 3D scene representation stack.

**3) No fleet-scale long-tail data engine**
- Tesla’s moat is selective mining from an enormous deployed fleet.
- Octo uses a pooled research dataset; it is broad, but not a live data engine.

**4) No full-body humanoid / driving / factory autonomy claim**
- Octo is primarily about **manipulation**.
- It does not demonstrate full-body locomotion, whole-body contacts, driving, or deployment-grade safety-critical operation.

**5) Limited evidence on long-horizon autonomy**
- There are some longer manipulation tasks in finetuning, but this is still far from “general robot operating system” territory.

## Action items for AIResearch (interfaces / contracts to copy)

### 1) Copy the observation dictionary contract
Use a modality-keyed observation structure rather than a monolithic tensor blob.

Recommended shape:
- `observation.image_front`
- `observation.image_left`
- `observation.image_right`
- `observation.state`
- `observation.route` or `observation.goal`
- `observation.timestep_pad_mask`
- `observation.pad_mask_dict`

Why:
- makes missing sensors explicit
- keeps multi-camera support composable
- avoids architecture forks for every sensor configuration

### 2) Separate task spec from observation spec
Octo keeps **task** distinct from **observation**. AIResearch should do the same.

For driving-like work, that likely means keeping route / nav instruction / scenario intent in a distinct `task` structure, not burying it inside image tensors or ad hoc metadata.

### 3) Use chunked action prediction as a first-class interface
Even if AIResearch does not use diffusion, it should copy the **action chunk** abstraction:
- predict `N` future controls / waypoints
- allow receding-horizon execution
- keep the execution policy separate from the training target format

This cleanly supports smoothing, ensembling, replanning, and future world-model integration.

### 4) Preserve adapter-friendly input/output heads
Octo’s best systems lesson is that new sensors and new action spaces should mostly require:
- a new tokenizer / adapter
- a new readout head
- minimal disturbance to the pretrained backbone

That is the right contract to copy for adding new cameras, BEV channels, proprio, force, or different control targets.

### 5) Make masks and missing-modality behavior explicit in the schema
Do not rely on implicit conventions.

Document:
- what happens when a camera is absent
- what happens at the start of a sequence
- how padding is represented
- how action padding is represented

Octo’s `timestep_pad_mask` and `pad_mask_dict` are worth copying almost directly.

### 6) Ship a “golden path” reproducibility stack
Octo is compelling because it ships:
- checkpoint loading
- example inference
- example finetuning
- example eval env integration
- standalone dataloader examples

AIResearch should mirror that pattern for any foundation-model-like baseline it adopts.

## Bottom line

If the question is “what public baseline is closest to a **reproducible robotics foundation model starter kit**?”, the answer is **Octo**, not RT-X.

Octo does **not** prove Tesla-style fleet robotics, humanoid autonomy, world simulation, or safety-critical deployment. But it **does** prove that a single open policy can be pretrained over heterogeneous robot data and then adapted through clean interfaces to new sensors, new action spaces, and new embodiments. That makes it the right public anchor for this repo’s Survey PR #3 slot.

## Citations + links

- **Octo paper (arXiv / RSS 2024)** — abstract, architecture, training mixture, diffusion objective, and evaluation framing: https://arxiv.org/abs/2405.12213
- **Octo HTML paper view** — useful for section-level details on data curation, block-masked transformer design, and DDPM action head: https://arxiv.org/html/2405.12213v2
- **Octo project site** — headline numbers for zero-shot and finetuning eval, plus model sizes and high-level framing: https://octo-models.github.io/
- **Octo GitHub repo** — training script, finetuning script, model code, dataloaders, example notebooks, and eval integration: https://github.com/octo-models/octo
- **Raw README** — concise source for install, checkpoints, action chunking, pad-mask semantics, and reproducibility hooks: https://raw.githubusercontent.com/octo-models/octo/main/README.md
- **Open X-Embodiment / RT-X project** — the underlying cross-lab dataset standardization effort that Octo builds on: https://robotics-transformer-x.github.io/
- **Open X-Embodiment code/data hub** — RLDS-format dataset access and loading references: https://github.com/google-deepmind/open_x_embodiment

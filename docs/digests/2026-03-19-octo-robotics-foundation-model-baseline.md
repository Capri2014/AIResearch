# Octo: Robotics Foundation Model Baseline — Public Anchor (Survey PR #3)

Source: https://octo-models.github.io/ (paper: https://arxiv.org/abs/2405.12213 ; code: https://github.com/octo-models/octo)

## TL;DR (3 bullets)
- **Octo** selected as public anchor — best open-code reproducibility (pip-installable, HF Hub weights, Colab notebooks) vs RT-X.
- **Transformer + Diffusion** architecture on **800k episodes** (Open X-Embodiment subset); achieves **0.80 zero-shot** WidowX success (competitive with RT-2-X's 0.85 at 1/60th parameters).
- **Action contract**: 7D continuous end-effector actions via diffusion; supports custom action spaces — clean mapping to driving (steer, throttle, brake).

## TL;DR Extended
- Octo is the most reproducible open robotics foundation model — ships with pip install, HuggingFace Hub weights, Google Colab inference/finetuning notebooks, and full training recipe.
- Chosen over RT-X because: (1) code is actually runnable out-of-box, (2) supports arbitrary action spaces via config, (3) diffusion objective aligns with Tesla's reported output stochasticity, (4) handles multi-camera inputs natively.
- Single model across 25 robot embodiments — directly tests "one foundational network" thesis from Tesla/Ashok.

## Dataset / Inputs / Outputs
### Dataset
- **800k episodes** from 25 datasets, 25 robot embodiments (WidowX, UR5, Franka, bi-manual, mobile manipulators)
- Subset of Open X-Embodiment released in RLDS format
- Each episode: sequence of (observation, action, reward, terminal)

### Inputs
- **RGB images**: 1-4 cameras per episode (configurable)
- **Language instruction**: free-form task string ("pick up the apple", "push the block left")
- **Goal image**: optional future desired state (goal-conditioned)
- **Observation history**: temporal context (default: 2-4 frames)

### Outputs
- **Continuous 7D action vector**: x, y, z, roll, pitch, yaw, gripper (end-effector frame)
- **Diffusion decoding**: outputs sequence of future actions (not single-step)
- **Custom action spaces**: configurable via hydra config — can remap to driving (steer, throttle, brake)

## Training Objective
### Architecture
- **Tokenization**: observation/action tokens via ViT/MLP encoders
- **Backbone**: Transformer (8-layer, 512 hidden, 8 heads)
- **Objective**: Denoising Diffusion Probabilistic Model (DDPM)
  - Forward process: corrupt actions with Gaussian noise over T steps
  - Reverse process: learn to denoise via MSE on noise prediction
  - At inference: iterative denoising (50-100 steps) → action sequence

### Why Diffusion?
1. **Multi-modal action handling**: heterogeneous data (absolute/relative, position/velocity) creates multi-modal output distributions; diffusion captures this better than discrete classification
2. **Sequence prediction**: outputs future action horizon (8-16 steps), not just next-step
3. **Alignment with Tesla claims**: Ashok mentioned output stochasticity — diffusion is explicitly stochastic
4. **Scalability**: same objective works across embodiments without architectural changes

### Training Details
- Batch size: 256
- Learning rate: 1e-4 with warmup
- Hardware: 64 A100s for base model (paper reports training costs)
- Checkpoints: `octo-base-32k` (32k gradient steps), `octo-large-80k`

## Evaluation Setup
### Zero-Shot Transfer
| Model | Params | WidowX Success | Notes |
|-------|--------|-----------------|-------|
| Octo | 35M | 0.80 | 800k episodes, diffusion |
| RT-2-X | 2B | 0.85 | VLM co-tuning |
| RT-1-X | 80M | 0.60 | Discrete action tokens |

### Finetuning
- **100 demos**: 0.72 success (52% better than next best method)
- **500 demos**: approaches in-distribution performance
- Real-robot evaluation (not simulation-only)
- Tasks: picking, placing, pushing, manipulation in clutter

### Evaluation Tasks
- Short-horizon manipulation (< 30 seconds per task)
- Language-conditioned (task string → action)
- Goal-conditioned (goal image → action)
- Cross-embodiment transfer (train on mix, eval on held-out robot)

## What Maps to Tesla/Ashok Claims vs What Doesn't

### Maps Cleanly ✅
| Tesla/Ashok Claim | Octo Alignment |
|-------------------|----------------|
| "One foundational network across robots" | Single model, 25 embodiments — directly tests this |
| "Fleet data + pretraining" | 800k episodes from diverse labs shows transfer works |
| "Language as API" | Task string conditioning explicit and functional |
| "Efficient fine-tuning" | 100 demos = strong performance mirrors data efficiency |
| "End-to-end, camera-first" | RGB observations, no depth required by default |
| "Diffusion / stochastic outputs" | Diffusion objective aligns with output stochasticity |
| "Multi-camera perception" | Supports 1-4 cameras, configurable |

### Doesn't Map ❌
| Gap | Details |
|-----|---------|
| **Humanoid / full-body** | Manipulation only (7D end-effector), no locomotion/balance |
| **Long-horizon autonomy** | Short-horizon evaluations (seconds per task) |
| **Real-time fleet deployment** | Research model, no continuous learning loop |
| **3D scene understanding** | 2D image-to-action, no explicit geometric reasoning |
| **Video generation** | Outputs actions, not world model video generation |
| **Real-time control (36 Hz)** | Diffusion inference is iterative (not yet real-time) |
| **Factory/harsh environments** | Lab-tested, not factory-floor distribution |

## Comparison: Octo vs RT-X

| Dimension | Octo | RT-X (RT-1-X / RT-2-X) |
|-----------|------|------------------------|
| **Code reproducibility** | ✅ pip install, Colab, HF Hub | ⚠️ Partial (dataset, some checkpoints) |
| **Action space** | Customizable via config | Fixed 7D |
| **Objective** | Diffusion | BC (RT-1-X), VLM co-tune (RT-2-X) |
| **Parameters** | 35M (base) | 80M (RT-1-X), 2B (RT-2-X) |
| **Zero-shot WidowX** | 0.80 | 0.60 (RT-1-X), 0.85 (RT-2-X) |
| **Finetuning** | ✅ Full recipe | ⚠️ Limited public recipes |
| **Multi-camera** | ✅ 1-4 cameras | ❌ Single camera (default) |

**Decision**: Octo chosen for reproducibility + diffusion alignment + configurability.

## Action Items for AIResearch (Interfaces/Contracts to Copy)

### High Priority
- [ ] **Adopt diffusion policy** for waypoint/action prediction — aligns with Octo architecture and Tesla output stochasticity
- [ ] **Use Octo as pretraining backbone**: load `octo-base-32k`, replace action head with driving-specific outputs
- [ ] **Mirror RLDS episode schema** for on-disk data format — enables direct loading of Open X-Embodiment

### Medium Priority
- [ ] **Implement goal-conditioned driving**: use "goal frame" = future desired position (Octo supports goal images)
- [ ] **HuggingFace Hub integration**: Octo weights are HF-native; our models could follow same loading pattern
- [ ] **Test finetuning**: Octo + 100–500 driving demos to measure transfer vs scratch training

### Low Priority / Exploratory
- [ ] **Define unified action contract**: 7D (or similar) that maps driving (steer, throttle) ↔ manipulation (EEF)
- [ ] **Multi-camera setup**: leverage Octo's 1-4 camera support for surround view
- [ ] **Explore diffusion inference speedup**: knowledge distillation or fewer denoising steps for real-time

## Contract Summary (for implementation)

```
Observation:
  - images: (B, T, N_cam, H, W, 3)  # T=history, N_cam=cameras
  - task: str                       # language instruction
  - goal: (B, H, W, 3) optional     # goal image

Action:
  - continuous: (B, T, 7)            # 7D end-effector
  - or custom: (B, T, A)            # configurable

Training:
  - objective: diffusion (DDPM)
  - loss: MSE on noise prediction
  - inference: 50-100 iterative denoising steps
```

## Citations / Links
- Project site: https://octo-models.github.io/
- Paper (arXiv): https://arxiv.org/abs/2405.12213
- GitHub: https://github.com/octo-models/octo
- HuggingFace Hub: https://huggingface.co/octo-models
- Open X-Embodiment: https://github.com/google-deepmind/open_x_embodiment
- Diffusion Policy (Chi et al.): https://diffusionpolicy.cs.cmu.edu/
- Colab (inference): https://colab.research.google.com/github/octo-models/octo/blob/main/octo/colab/inference_demo.ipynb
- Colab (finetuning): https://colab.research.google.com/github/octo-models/octo/blob/main/octo/colab/finetuning_demo.ipynb

## PR Summary

- **PR**: [Survey PR #3] Octo Robotics Foundation Model Baseline
- **Choice**: Octo over RT-X (better reproducibility, diffusion objective, configurable)
- **Key insight**: 35M params achieves 0.80 zero-shot (vs RT-2-X's 0.85 at 2B params) — validates efficient pre-training + fine-tuning
- **Action items**: Adopt diffusion policy, use Octo backbone, mirror RLDS schema, test transfer on driving data

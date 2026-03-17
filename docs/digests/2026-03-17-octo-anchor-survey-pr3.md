# Octo: Robotics Foundation Model Baseline — Public Anchor

Source: https://octo-models.github.io/ (paper: https://arxiv.org/abs/2405.12213 ; code: https://github.com/octo-models/octo)

## TL;DR (3 bullets)
- **Octo** selected as public anchor — best open-code reproducibility (pip-installable, HF Hub weights, Colab notebooks) vs RT-X.
- **Transformer + Diffusion** architecture on **800k episodes** (Open X-Embodiment subset); achieves **0.80 zero-shot** WidowX success (competitive with RT-2-X's 0.85 at 1/60th parameters).
- **Action contract**: 7D continuous end-effector actions via diffusion; supports custom action spaces — clean mapping to driving (steer, throttle, brake).

## Dataset / Inputs / Outputs
- **800k episodes** from 25 datasets, 25 robot embodiments (WidowX, UR5, Franka, bi-manual, mobile manipulators)
- **Inputs**: RGB images (1-4 cameras), language instruction string, optional goal image, observation history
- **Outputs**: Continuous 7D action vector (x, y, z, roll, pitch, yaw, gripper) via diffusion decoding; supports custom action spaces

## Training Objective
- **Diffusion policy**: Tokenize observations/actions → Transformer backbone → denoising diffusion process → iterative action refinement
- Multi-step action prediction (outputs sequence of future actions)
- Better handling of multi-modal action distributions vs discrete token classification (RT-1-X) or VLM co-tuning (RT-2-X)

## Evaluation Setup
- **Zero-shot**: 0.80 success on WidowX UR5 (vs RT-2-X 0.85, RT-1-X 0.60)
- **Finetuning**: 0.72 success with **100 demos** (52% better than next best method)
- Real-robot evaluation (not simulation-only)
- Tasks: picking, placing, pushing, manipulation in clutter

## What Maps to Tesla/Ashok Claims vs What Doesn't

### Maps Cleanly ✅
- "One foundational network across robots" — Octo IS this (single model, multiple embodiments)
- "Fleet data + pretraining" — 800k episodes from diverse labs shows transfer works
- "Language as API" — Task string conditioning explicit and functional
- "Efficient fine-tuning" — 100 demos = strong performance mirrors data efficiency goals
- "End-to-end, camera-first" — RGB observations, no depth required by default
- Diffusion objective — aligns with Tesla's reported output stochasticity

### Doesn't Map ❌
- **Humanoid / full-body** — Octo is manipulation only (7D end-effector), no locomotion/balance
- **Long-horizon autonomy** — Short-horizon evaluations (seconds per task)
- **Real-time fleet deployment** — Research model, no continuous learning loop
- **3D scene understanding** — 2D image-to-action, no explicit geometric reasoning
- **Video generation** — Outputs actions, not world model video generation

## Action Items for AIResearch (Interfaces/Contracts to Copy)
- [ ] **Adopt diffusion policy** for waypoint/action prediction — aligns with Octo architecture and Tesla output stochasticity
- [ ] **Use Octo as pretraining backbone**: load `octo-base-32k`, replace action head with driving-specific outputs
- [ ] **Mirror RLDS episode schema** for on-disk data format — enables direct loading of Open X-Embodiment
- [ ] **Implement goal-conditioned driving**: use "goal frame" = future desired position (Octo supports goal images)
- [ ] **HuggingFace Hub integration**: Octo weights are HF-native; our models could follow same loading pattern
- [ ] **Test finetuning**: Octo + 100–500 driving demos to measure transfer vs scratch training
- [ ] **Define unified action contract**: 7D (or similar) that maps driving (steer, throttle) ↔ manipulation (EEF)

## Citations / Links
- Project site: https://octo-models.github.io/
- Paper (arXiv): https://arxiv.org/abs/2405.12213
- GitHub: https://github.com/octo-models/octo
- HuggingFace Hub: https://huggingface.co/octo-models
- Open X-Embodiment: https://github.com/google-deepmind/open_x_embodiment
- Diffusion Policy (Chi et al.): https://diffusionpolicy.cs.cmu.edu/
- Colab (inference): https://colab.research.google.com/github/octo-models/octo/blob/main/octo/colab/inference_demo.ipynb
- Colab (finetuning): https://colab.research.google.com/github/octo-models/octo/blob/main/octo/colab/finetuning_demo.ipynb

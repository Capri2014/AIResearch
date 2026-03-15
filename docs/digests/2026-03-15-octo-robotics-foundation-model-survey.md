# Octo: Robotics Foundation Model Baseline — Survey Selection

Source: https://octo-models.github.io/ (paper: https://arxiv.org/abs/2405.12213 ; code: https://github.com/octo-models/octo)

## TL;DR (3 bullets)
- **Octo** selected as the public anchor for robotics foundation models due to **best open-code reproducibility** (pip-installable, HF Hub weights, Colab notebooks).
- **Transformer + Diffusion** architecture, trained on **800k episodes** from Open X-Embodiment, achieving **0.80 zero-shot** success on WidowX (competitive with RT-2-X's 0.85 at 1/60th parameters).
- **Action contract**: 7D end-effector continuous actions via diffusion decoding; supports custom action spaces for driving (steer, throttle, brake).

## Selection Rationale
Compared to RT-X (RT-1-X / RT-2-X):
| Aspect | Octo | RT-X |
|--------|------|------|
| Architecture | Transformer + Diffusion | Transformer / VLM co-tune |
| Open weights | ✅ HF Hub (27M/93M) | ✅ RT-1-X only |
| pip-installable | ✅ | ❌ |
| Colab notebooks | ✅ Inference + Finetuning | ❌ |
| Zero-shot (WidowX) | 0.80 | RT-1-X: 0.60; RT-2-X: 0.85 |
| Finetuning (100 demos) | **0.72** (52% better) | Not reported |

**Octo wins** for reproducibility: pip-installable, full inference/finetuning APIs, HF-native weights.

## Dataset / Inputs / Outputs
- **800k episodes** from 25 datasets (Open X-Embodiment subset)
- **Inputs**: RGB images (1-4 cameras), language instruction string, optional goal image, observation history
- **Outputs**: Continuous 7D action vector (x, y, z, roll, pitch, yaw, gripper) via diffusion decoding; supports custom action spaces

## Training Objective
- **Diffusion policy**: Tokenize observations/actions → Transformer backbone → denoising diffusion process → iterative action refinement
- Multi-step action prediction (outputs sequence of future actions)
- Better handling of multi-modal action distributions vs discrete token classification

## Evaluation Setup
- **Zero-shot**: 0.80 success on WidowX UR5 (vs RT-2-X 0.85, RT-1-X 0.60)
- **Finetuning**: 0.72 success with 100 demos (52% better than next best)
- Real-robot evaluation (not simulation-only)
- Tasks: picking, placing, pushing, manipulation in clutter

## Tesla / Ashok Claims Mapping

### Maps Cleanly
- ✅ "One foundational network across robots" — Octo is literally this
- ✅ "Fleet data + pretraining" — 800k episodes from diverse labs shows transfer works
- ✅ "Language as API" — Task string conditioning explicit
- ✅ "Efficient fine-tuning" — 100 demos = strong performance
- ✅ "End-to-end, camera-first" — RGB observations, no depth required
- ✅ Diffusion objective — Aligns with Tesla's output stochasticity

### Doesn't Map
- ❌ Humanoid / full-body control — Octo is manipulation only (7D EEF)
- ❌ Long-horizon autonomy — Short-horizon evaluations (seconds)
- ❌ Real-time fleet deployment — Research model, no continuous learning loop
- ❌ 3D scene understanding / Gaussian Splatting — 2D image-to-action only
- ❌ Multi-camera video generation — Outputs actions, not video

## Action Items for AIResearch
- [ ] Adopt diffusion policy for waypoint/action prediction
- [ ] Use Octo as pretraining backbone — load `octo-base`, replace action head with driving outputs
- [ ] Mirror RLDS episode schema for on-disk data format
- [ ] Implement goal-conditioned driving (goal image = future desired position)
- [ ] Explore HuggingFace Hub integration for model weights
- [ ] Test finetuning with 100–500 driving demonstrations
- [ ] Define unified action contract (7D or similar) mapping driving ↔ manipulation

## Citations / Links
- Project: https://octo-models.github.io/
- Paper: https://arxiv.org/abs/2405.12213
- GitHub: https://github.com/octo-models/octo
- HF Hub: https://huggingface.co/octo-models
- Open X-Embodiment: https://github.com/google-deepmind/open_x_embodiment
- Diffusion Policy: https://diffusionpolicy.cs.cmu.edu/
- Colab (inference): https://colab.research.google.com/github/octo-models/octo/blob/main/octo/colab/inference_demo.ipynb
- Colab (finetuning): https://colab.research.google.com/github/octo-models/octo/blob/main/octo/colab/finetuning_demo.ipynb

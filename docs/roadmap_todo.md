# Roadmap / TODO (living)

This file is a lightweight queue of high-ROI items to read, digest, or implement.
It is intentionally not exhaustive.

## Survey / reading queue (high ROI)

### Robotics foundation models (Tesla)
- **"Building Foundational Models for Robotics at Tesla" — Ashok Elluswamy**
  - Digest: `docs/digests/tesla_foundational_models_robotics_elluswamy.md`
  - Follow-ups (public anchors; turn into separate digests + PRs):
    - ~~3D Gaussian Splatting (Kerbl et al., 2023)~~ ✅ Done: survey/2026-02-21-3d-gaussian-splatting.md
    - ~~Robotics foundation model baseline (Octo or Open X-Embodiment / RT-X)~~ ✅ Done: survey/2026-02-21-robotics-foundation-models.md
    - ~~Latest end-to-end driving stack (post-UniAD)~~ ✅ Done: survey/2026-02-21-e2e-driving-post-uniad.md
    - ~~World models / learned simulators (DreamerV3)~~ ✅ Done: survey/2026-02-21-world-models-dreamerv3.md
    - ~~**DreamZero** — end-to-end driving (PDF: https://dreamzero0.github.io/DreamZero.pdf)~~ ✅ Done: survey/2026-02-21-dreamzero.md
    - ~~**GitHub open source robotic arm projects**~~ ✅ Done: survey/2026-02-21-github-robotic-arms.md
    - ~~**Contingency Planning in Autonomous Driving** — "Why stuck at last 1%"~~ ✅ Done: survey/2026-02-21-contingency-planning.md

### Generative modeling: Flow Matching
- ~~**Flow Matching / Rectified Flow / Consistency-style training**~~ ✅ Done: survey/2026-02-21-flow-matching.md
  - Questions to answer in a digest:
    - How Flow Matching relates to diffusion (score matching) and why it can be simpler/faster
    - Which variants are most stable in practice (rectified flow, consistency, EDM-style)
    - What open implementations are cleanest and most reusable
    - How this could apply to action-conditioned video generation ("world simulator" direction)
  - Deliverable: one digest under `docs/digests/` with citations + action items for this repo.

### Post-SFT RL (policy refinement)
- **RL after imitation (BC/SFT) for driving/robotics policies**
  - TODO: survey recent (2023–2026) papers + repos on refining an imitation policy with RL, especially:
    - online RL / sim RL with imitation initialization
    - offline RL / RLHF-style preference learning for embodied policies (if applicable)
    - residual RL (policy = BC + delta)
    - safe RL / constraint-based RL relevant to long-tail safety
  - Questions to answer in a digest:
    - What is the best "minimal" RL setup that consistently improves over BC in practice?
    - What reward shaping / proxy metrics are used when real-world rewards are sparse?
    - What evaluation protocols are standard (closed-loop metrics, regressions, safety counters)?
    - Which open repos are most reusable for our waypoint-policy + sim-eval stack?
  - Deliverable: one digest under `docs/digests/` with citations + action items.

## Quick Wins (High Impact, Low Effort)

These are tasks that can be completed in 1-2 days and provide immediate value.

### ML Infrastructure
- [ ] **Add MLflow logging to existing training scripts** (1 day)
  - Location: `training/sft/train_waypoint_bc_cot.py`, `training/rl/*_train.py`
  - Track: loss curves, metrics, hyperparameters, artifacts
  - Deliverable: MLflow integration with config-driven logging

- [ ] **Dockerize training environment** (1 day)
  - Location: `Dockerfile.training`, `docker-compose.yml`
  - Include: CUDA, PyTorch, CARLA dependencies
  - Deliverable: Reproducible container for training

### Model Deployment
- [ ] **Export AR Decoder to ONNX** (1 day)
  - Location: `training/models/sft/ar_decoder.py`
  - Test: Inference with ONNX Runtime, FP16 optimization
  - Deliverable: ONNX export script + validation test

### Data & Evaluation
- [ ] **Add scenario coverage tracking to evaluation** (2 days)
  - Location: `training/rl/envs/carla_scenario_eval.py`
  - Track: Scenario types, success/fail rates, edge cases
  - Deliverable: Coverage report per evaluation run

- [ ] **Create benchmark dataset from nuScenes** (2 days)
  - Location: `training/data/nuscenes_converter.py`
  - Extract: Key frames, waypoints, scenario tags
  - Deliverable: Processed nuScenes subset for testing

## Priority Order
1. Dockerize training environment (prerequisite for all)
2. Add MLflow logging (improves experiment tracking)
3. Export AR Decoder to ONNX (deployment prep)
4. Create nuScenes benchmark (data prep)
5. Scenario coverage tracking (evaluation polish)

# Roadmap / TODO (living)

This file is a lightweight queue of high-ROI items to read, digest, or implement.
It is intentionally not exhaustive.

## Survey / reading queue (high ROI)

### Robotics foundation models (Tesla)
- **"Building Foundational Models for Robotics at Tesla" — Ashok Elluswamy**
  - Digest: `docs/digests/tesla_foundational_models_robotics_elluswamy.md`
  - Follow-ups (public anchors; turn into separate digests + PRs):
    - 3D Gaussian Splatting (Kerbl et al., 2023) + a reference implementation
    - Robotics foundation model baseline (Octo or Open X-Embodiment / RT-X)
    - Latest end-to-end driving stack (post-UniAD)
    - World models / learned simulators (DreamerV3 and/or driving-focused world model)

### Generative modeling: Flow Matching
- **Flow Matching / Rectified Flow / Consistency-style training**
  - TODO: survey the most useful papers + repos for *practical* flow-matching training (especially for video / world-model-ish generation).
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

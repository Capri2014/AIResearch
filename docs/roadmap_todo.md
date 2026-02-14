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

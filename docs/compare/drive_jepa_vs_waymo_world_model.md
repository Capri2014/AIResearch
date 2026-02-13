# Compare: Drive-JEPA vs Waymo world model

## One-line framing
- **Drive-JEPA**: planning-first: video SSL encoder + proposal-based planner + multimodal trajectory distillation for E2E driving.
- **Waymo world model**: likely world-model-first: learn a generative/predictive model of driving scenes for simulation/forecasting (need exact source to be precise).

## Side-by-side (current best-effort; will tighten after Waymo link)
- Goal / problem setting
  - Drive-JEPA: improve end-to-end planning; address multimodality with sim trajectories.
  - Waymo: model the world dynamics / scene evolution for driving.
- Inputs
  - Drive-JEPA: driving video (paper uses V-JEPA-style); repo focuses on NAVSIM.
  - Waymo: TBD (depends on post; could be camera, map, object state, etc.).
- Outputs
  - Drive-JEPA: planned trajectory / waypoints (via proposal selection).
  - Waymo: predicted future frames/states (world model), potentially usable for planning.
- Core modeling idea
  - Drive-JEPA: JEPA predictive representation + proposal-centric planning head.
  - Waymo: generative/predictive model of environment.
- Pretraining
  - Drive-JEPA: self-supervised video JEPA.
  - Waymo: likely large-scale pretraining on logged fleet data.
- Distillation / fine-tuning
  - Drive-JEPA: multimodal distillation from simulator-generated diverse trajectories + human.
  - Waymo: TBD.
- World model vs policy separation
  - Drive-JEPA: representation + policy/planner coupled toward planning.
  - Waymo: world model can be separate module; policy may be downstream.
- Evaluation setup
  - Drive-JEPA: NAVSIM; PDMS/EPDMS.
  - Waymo: TBD.

## What we should copy (repo-impacting)
- Proposal-based waypoint planning (generate K, then score/select) as a strong baseline.
- A clean story for injecting multimodality when logs are single-future: sim trajectories + distillation.
- Evaluate “planning-aligned” SSL objectives (JEPA-style prediction in latent space) rather than generic contrastive-only.

## What we should avoid
- Overfitting to a benchmark’s cache/feature pipeline too early (we want a minimal, debuggable episodes backend first).

## Action items for this repo
- [ ] Add a new doc: `docs/digests/drive_jepa.md` (done in this branch).
- [ ] Once Waymo source is provided, fill in this compare doc with concrete citations.
- [ ] Add an experimental “K-proposal + scorer” head for our **20-waypoint** spec (keep behind a flag).

## Sources
- Drive-JEPA paper: https://arxiv.org/abs/2601.22032v1
- Drive-JEPA code: https://github.com/linhanwang/Drive-JEPA

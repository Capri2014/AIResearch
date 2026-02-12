# AIResearch — Physical AI for Autonomous Driving + Robotics (Sim/Data + Large Models)

This repo is a **starter skeleton** for two tightly-coupled tracks:

1) **Large models for autonomy**: policies that map (vision, state, optional language) → actions for driving + robots.
2) **Simulation & data**: using simulation to generate training data and run repeatable evaluation suites.

> Status: initial scaffold commit. The goal is to make the project structure “real” first, then iterate.

---

## Why simulation?
Simulation is our multiplier:
- **Training**: generate large-scale rollouts, rare edge cases, and perfect labels.
- **Evaluation**: run deterministic scenario suites for regression testing and safety gating.
- **Sim→Real**: use domain randomization + system ID to bridge gaps.

See: `docs/sim_training_eval.md`.

---

## Repo map
- `docs/` — summaries + references
- `sim/` — simulation env adapters, scenarios, evaluation
  - `sim/driving_carla/` — CARLA-first driving sim scaffold
  - `sim/robotics/` — robotics sim scaffold (Isaac and MuJoCo tracks)
- `models/` — unified policy interface + stubs
- `demos/` — “run it” entrypoints (thin wrappers)
- `data/` — dataset schemas + tooling stubs
- `results/` — metrics outputs (tracked or gitignored later)

---

## Quickstart (placeholder)
This scaffold does **not** install CARLA/Isaac/MuJoCo yet. Start by reading:

- `docs/overview.md`
- `sim/driving_carla/README.md`
- `sim/robotics/README.md`

When you’re ready, we’ll add:
- a minimal CARLA docker-compose workflow
- a minimal scenario runner + eval that outputs JSON metrics
- a small dataset format + recorder

---

## Design principles
- **One policy interface** across domains (`models/policy_interface.py`)
- **Scenario-driven evaluation** (YAML/JSON scenario specs)
- **Episode recorder** writes unified dataset format (`sim/common/recorder.py`)
- Prefer **reproducible metrics** over flashy demos.

---

## TODO (roadmap)

### Agent Swarm (reusable framework)
- Build a reusable **agent swarm framework** for this repo (task router + planner + worker agents + aggregator):
  - supports “research → implement → eval” parallel lanes
  - emits structured artifacts (notes, citations, code PRs, eval reports)
  - standard prompt templates + checklists for consistency

### Robotics VLA base (Xiaomi)
- Review and integrate Xiaomi’s robotics **VLA base** ideas into our roadmap:
  - source: https://mp.weixin.qq.com/s/FEMJBekqPHpMpE60b8kV4A
  - extract: model interface (obs/action), data recipe, training stages (pretrain → BC/SFT → RL), and evaluation protocol
  - decide: what we can reuse for our driving-first plan + later robotics track

---

## Maintenance
This repository is maintained with help from **ClawBot (OpenClaw)**.

- I will open PRs from `feature/*` branches.
- You review and merge into `main`.
- No secrets are committed to the repo; credentials live only on the VPS.

## License
Add a license in `LICENSE` (MIT/Apache-2.0 are common).
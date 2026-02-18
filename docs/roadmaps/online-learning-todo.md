# Online Learning Follow-up Roadmap (ToDo)

**Date:** 2026-02-18

This roadmap captures the follow-up work after we implement:
- Train with CARLA Gym env (fast iteration)
- Evaluate with ScenarioRunner (realistic metrics)

The goal here is to add **online learning** capability (continual improvement during live interaction), without breaking the existing offline/batch pipeline.

---

## Phase 0 — Definitions & Interfaces (1–2 days)

- [ ] Define *online learning* precisely for our stack:
  - on-policy (GRPO/PPO-style): update after each rollout window
  - off-policy (SAC-style): update continuously from replay
- [ ] Standardize env API:
  - `reset() -> obs, info`
  - `step(action) -> obs, reward, done, info`
  - `info` must include: `progress`, `infractions`, `collision`, `route_completion`
- [ ] Standardize policy API:
  - `act(obs) -> action, logp, extra`
  - optional `update(batch)`

---

## Phase 1 — Minimal Online Loop (Toy Env) (2–4 days)

- [ ] Add a minimal online loop for SAC (easiest):
  - collect 1 step
  - push to replay
  - update every step after warmup
- [ ] Add online loop for GRPO:
  - rollout window (e.g., 256 steps)
  - group sampling per state
  - update after window
- [ ] Logging:
  - online reward curve
  - policy entropy
  - KL drift vs initial checkpoint

---

## Phase 2 — Online Learning in CARLA Gym (5–10 days)

- [ ] Add asynchronous stepping + batched envs (vectorized) for throughput
- [ ] Safety constraints for online learning:
  - action clipping + safety layer
  - early termination on infractions
  - rollback policy checkpoint on instability
- [ ] Replay buffer persistence across runs (SAC)
- [ ] Curriculum:
  - start with straight driving
  - then lane change
  - then intersections

---

## Phase 3 — ScenarioRunner Online Fine-tuning (10–20 days)

- [ ] Use ScenarioRunner episodes as online rollouts (slower but higher-quality signals)
- [ ] Reward shaping derived from infractions and completion
- [ ] Domain randomization:
  - weather
  - traffic density
  - pedestrian density

---

## Phase 4 — Hybrid Online/Offline (Ongoing)

- [ ] Periodic offline replay distillation:
  - store good online episodes
  - distill into SFT or delta-head
- [ ] Maintain frozen SFT baseline; online only updates delta/residual heads
- [ ] Add evaluation gate:
  - only promote online policy if ScenarioRunner metrics improve significantly

---

## Risks / Things to Watch

- Instability (policy collapse, Q explosion)
- Reward hacking
- Catastrophic forgetting (overfitting to current town/scenario)
- Regression safety: always compare against frozen baseline

---

## Deliverables Checklist

- [ ] `training/rl/online/` package
- [ ] `OnlineTrainer` base class
- [ ] `OnlineSACTrainer`, `OnlineGRPOTrainer`
- [ ] `CARLAOnlineRunner` (handles stepping, logging, checkpointing)
- [ ] Documentation + reproducible configs

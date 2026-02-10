# Alpamayo‑R1 — follow‑up notes (toy-policy lens)

This note continues the repo’s Alpamayo‑R1 exploration from a **"what can we copy in a small, runnable way"** angle.

It intentionally focuses on the *policy output parameterization* + *rollout/eval plumbing* we can implement in this repo.

## 1) Action parameterization → rollout (why accel + curvature is useful)
A common compact trajectory predictor setup is:

- Policy outputs per-step action parameters over a horizon `t=0..T-1`:
  - acceleration `a[t]` (m/s²)
  - curvature `κ[t]` (1/m)  (≈ steering; yaw rate is `v*κ`)
- A simple kinematic model rolls these forward to produce a trajectory `(x,y,yaw,v)`.

One unicycle-style discretization:

- `x_{t+1} = x_t + v_t cos(yaw_t) dt`
- `y_{t+1} = y_t + v_t sin(yaw_t) dt`
- `yaw_{t+1} = yaw_t + v_t κ_t dt`
- `v_{t+1} = max(0, v_t + a_t dt)`

Why this is a good "first step":
- Forces **physical consistency** (no teleporting waypoints).
- Makes it easy to compute control-relevant metrics (smoothness, constraints).
- Lets us separate concerns:
  - policy head → predicts `a, κ`
  - integrator → produces trajectory
  - evaluator → computes metrics + checks

## 2) Minimal metrics worth emitting early
Even in a toy demo (no map, no collisions), emitting a `metrics.json` artifact is valuable.

Suggested "starter" metrics:
- **max_abs_accel**, **max_abs_curvature** (command magnitude)
- **min_speed**, **final_speed** (sanity)
- **max_abs_jerk**: jerk is `Δa/dt` (smoothness proxy)
- **max_abs_curvature_rate**: `Δκ/dt` (steering smoothness proxy)

These mirror the kinds of things you later care about in real autonomy:
- comfort (jerk)
- actuator feasibility (rates)
- stability / constraint violations

## 3) What to implement next in this repo
Near-term:
- Make the toy demo write `metrics.json`.
- Add a tiny CLI so we can run without plotting (useful for CI).

Longer-term (still small):
- Add a "scenario" input (e.g., LEFT/RIGHT/STRAIGHT) that changes the synthetic action profile.
- Add a batch runner that aggregates mean/std metrics over N seeds.

## References
- Alpamayo‑R1 model card: https://huggingface.co/nvidia/Alpamayo-R1-10B
- Community distillation repo: https://github.com/mu-hashmi/alpamayo-r1-distilled
- Paper (as linked by the distillation repo): https://arxiv.org/abs/2511.00088

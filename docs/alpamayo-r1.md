# Alpamayo‑R1 (my understanding) — summary + hands‑on demo plan

## What it is (high level)
**Alpamayo‑R1** (by NVIDIA) is described as a **vision‑language‑action (VLA)** model for **autonomous driving**: it consumes perception inputs (and optionally language/task context) and outputs **driving actions / trajectories**.

A practical way to think about it:
- It is not “just a VLM that talks about driving.”
- It is a model (or model+head) that produces **control‑relevant outputs** (actions/trajectories) that can be executed or evaluated.

## Why people care
Driving is a physical control problem with hard constraints:
- actions must be **physically plausible** (no teleporting)
- safety metrics matter (collision, offroad, jerk)
- you need robust behavior under distribution shift (weather, occlusion, sensor noise)

Alpamayo‑R1 is interesting because it frames autonomy as:
- **large model policy** → predicts actions in a structured form
- and emphasizes **evaluation + data pipelines** (datasets, labels, metrics)

## A key idea worth copying for hands‑on learning
A commonly used trick (also used in the public distilled repo) is to predict **action parameters** (e.g., acceleration + steering curvature) and then integrate a simple vehicle model (a **unicycle/bicycle kinematics** approximation) to produce trajectories.

Why that’s useful:
- training is often more stable than predicting raw future XY waypoints directly
- the resulting trajectories tend to be **physically consistent**

## What we will demo in this repo (no GPU required)
We’ll build a toy “Alpamayo‑style” pipeline:

1) Define a **student** that predicts a short horizon of:
   - acceleration `a[t]`
   - curvature `kappa[t]` (≈ steering)
2) Integrate with a unicycle model to get a trajectory:
   - (x, y, yaw, v) over time
3) Compute simple metrics:
   - max curvature, max accel
   - trajectory smoothness proxy
   - sanity checks (speed non‑negative)

This will **not** reproduce Alpamayo‑R1 performance; it’s a conceptual demo so you can understand the modeling choices.

## Practical next steps (real model later)
Once the toy demo is in place, we can add a “real” path:
- plug in a pretrained teacher model (Alpamayo‑R1 or similar)
- generate teacher labels on a dataset
- distill into a small student

## References
- HuggingFace model page (Alpamayo‑R1): https://huggingface.co/nvidia/Alpamayo-R1-10B
- Distillation repo (community): https://github.com/mu-hashmi/alpamayo-r1-distilled
- Paper (as linked by the distillation repo): https://arxiv.org/abs/2511.00088

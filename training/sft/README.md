# SFT / IL (supervised fine-tuning / imitation learning)

SFT/IL covers supervised training that maps observations (and optionally language/task context) to actions/trajectories.

## Common variants
- **Behavior cloning (BC):** learn from logged actions (human or stack)
- **Teacher distillation:** learn to match a stronger teacher policy

## What we provide here
- A **toy runnable BC** example (simple and fast, not meant to be state of the art)
- A **distillation stub** that defines the teacher/student interface

Downstream, this should connect to:
- rollout / evaluation (produce `metrics.json`)
- dataset schema (`data/schema.md`)

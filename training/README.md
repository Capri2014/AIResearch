# Training

This folder is organized by the common autonomy / VLA recipe:

1) **Pre-train**: learn general representations (vision/video/language) from broad data
2) **SFT / IL**: supervised fine-tuning / imitation learning to predict actions or trajectories
3) **RL**: optimize task reward + constraints (offline RL or online RL in simulation)

Each subfolder contains:
- a short `README.md` describing the approach and expected inputs/outputs
- a minimal script or stub to clarify wiring (even if not runnable at scale)

See also: `data/schema.md`.

# Waypoint BC (supervised fine-tuning)

This repo’s first supervised baseline is a simple **waypoint behavior cloning** model.

## Waypoint spec (v1)
- Horizon: **2.0s**
- Rate: **10Hz**
- Steps: **20** waypoints
- Coordinates: **ego-frame XY meters**

So the model output is an array shaped `(20, 2)` per sample.

## Baseline: pure-python ridge regression

This baseline exists to validate the **end-to-end contract** quickly without heavy ML deps.

Run:
```bash
python3 -m training.sft.train_waypoint_bc_np --episodes-glob "out/episodes/**/*.json"
```

Outputs (typical):
- a lightweight checkpoint under `out/` (see script output)

## Notes
- This baseline is intentionally simple; it’s not expected to be SOTA.
- Its value is:
  - debugging labels
  - confirming data plumbing
  - providing a yardstick for later PyTorch models

## Next step (PyTorch BC)
Once the encoder pretrain is stable, the next BC upgrade is:
- a PyTorch waypoint head on top of the pretrained encoder
- train/eval with consistent masks (`image_valid_by_cam`)

## Relevant source files
- Baseline training
  - `training/sft/train_waypoint_bc_np.py`
- Baseline model
  - `models/waypoint_policy_ridge.py`
- Waypoint extraction (Waymo → episodes)
  - `data/waymo/waypoint_extraction.py`
- Demo episodes
  - `demos/waymo_contract_demo/run.py`

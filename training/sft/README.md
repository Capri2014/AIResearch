# SFT / imitation learning

This folder contains supervised fine-tuning (SFT) and imitation learning scripts.

## Waypoint BC (NumPy baseline)

A dependency-light baseline to validate dataset contracts:

1) Generate a stub Waymo episode:
```bash
python -m data.waymo.convert --out-dir out/episodes/waymo_stub
```

2) Train the NumPy-only waypoint BC model:
```bash
python -m training.sft.train_waypoint_bc_np --episodes-glob "out/episodes/**/*.json"
```

Outputs:
- `out/sft_waypoint_bc_np/model.json`

You can load it via `models/waypoint_policy_ridge.py`.

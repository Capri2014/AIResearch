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

## Waypoint BC (PyTorch, image-conditioned)

A minimal image-conditioned waypoint head on top of `TinyMultiCamEncoder`:

```bash
python -m training.sft.train_waypoint_bc_torch_v0 \
  --episodes-glob "out/episodes/**/*.json" \
  --cam front \
  --batch-size 32 \
  --num-steps 200
```

Optionally initialize the encoder from temporal SSL pretrain:

```bash
python -m training.sft.train_waypoint_bc_torch_v0 \
  --episodes-glob "out/episodes/**/*.json" \
  --pretrained-encoder out/pretrain_temporal_contrastive_v0/encoder.pt
```

Outputs:
- `out/sft_waypoint_bc_torch_v0/model.pt`
- `out/sft_waypoint_bc_torch_v0/train_metrics.json`

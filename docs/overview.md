# AIResearch — Overview

This repo is building a **driving-first Physical AI pipeline**.

Goal (v1):
- **Waymo multi-camera** logged data → normalize into an **episode shard contract**
- **Self-supervised pretraining** of a multi-camera encoder (PyTorch)
- **Waypoint behavior cloning** (20 waypoints, 2s @ 10Hz) fine-tune
- **CARLA ScenarioRunner** evaluation that writes `metrics.json`

## Pipeline (end-to-end)

1) **Ingest / normalize**
- Input: Waymo Open Dataset style sources (eventually TFRecords)
- Output: `episode.json` files that match `data/schema/episode.json`

2) **Pretrain (SSL)**
- Input: episodes backend (debuggable, no TF dependency required)
- Output: `encoder.pt` (PyTorch state dict)

3) **Fine-tune (BC)**
- Input: episodes + extracted waypoint targets
- Output: a policy that predicts **20 (x,y) waypoints** in ego frame

4) **Evaluate (CARLA ScenarioRunner)**
- Input: a policy checkpoint + CARLA ScenarioRunner route/scenario config
- Output: `metrics.json` matching `data/schema/metrics.json`

## Key repo contracts

- Episodes: `data/schema/episode.json`
- Metrics: `data/schema/metrics.json`
- Pretrain batch contract: `training/pretrain/batch_contract.md`

## How to run (minimal)

### 0) Make synthetic episodes (contract demo)
This generates episodes without Waymo dependencies, to validate the training/eval plumbing.

```bash
python3 -m demos.waymo_contract_demo.run
```

You should get something like:
- `out/episodes/.../*.json`

### 1) SSL pretrain smoke run
```bash
python3 -m training.pretrain.train_ssl_contrastive_v0 \
  --episodes-glob "out/episodes/**/*.json" \
  --num-steps 10 \
  --batch-size 16 \
  --device cuda \
  --num-workers 4
```

### 2) Waypoint BC baseline (pure-python)
```bash
python3 -m training.sft.train_waypoint_bc_np --episodes-glob "out/episodes/**/*.json"
```

### 3) ScenarioRunner eval stub
```bash
python3 -m sim.driving.carla_srunner.run_srunner_eval --out-dir out/srunner_eval
```

## Repo navigation

- Waymo → episodes contract
  - `data/waymo/convert.py`
  - `data/waymo/validate_episode.py`
  - `data/waymo/waypoint_extraction.py`

- Pretrain
  - `training/pretrain/dataloader_episodes.py`
  - `training/pretrain/train_ssl_contrastive_v0.py`
  - `models/encoders/tiny_multicam_encoder.py`

- BC
  - `training/sft/train_waypoint_bc_np.py`
  - `models/waypoint_policy_ridge.py`

- CARLA ScenarioRunner eval
  - `sim/driving/carla_srunner/run_srunner_eval.py`

## Notes / constraints
- This is **driving-first**: evaluation is via **ScenarioRunner**.
- Optional dependencies are guarded. If `torch` / `PIL` aren’t installed, scripts should error clearly.

# Pretrain environment notes (PyTorch)

These pretraining scripts assume a **separate training environment** (GPU box / workstation) with PyTorch installed.

## Minimal install

1) Install PyTorch (pick the right CUDA build for your machine):
- https://pytorch.org/get-started/locally/

2) Install repo pretrain deps:
```bash
pip install -r training/pretrain/requirements.txt
```

## Quick sanity checks

```bash
python -c "import torch; import PIL; print('torch', torch.__version__); print('PIL ok')"
```

## Running the contrastive SSL v0

1) Generate an episode shard (contract mode):
```bash
python -m data.waymo.convert --out-dir out/episodes/waymo_stub
```

2) Run SSL (requires real images on disk; contract mode uses placeholders, so this is mainly for wiring):
```bash
python -m training.pretrain.train_ssl_contrastive_v0 --episodes-glob 'out/episodes/**/*.json'
```

In practice you’ll use TFRecord conversion to write real JPEGs and then point `--episodes-glob` at those.

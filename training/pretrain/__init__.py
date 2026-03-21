"""SSL Pretraining module.

This module provides self-supervised pretraining for driving perception.

Key components:
- Waymo SSL dataset: training/pretrain/waymo_ssl_dataset.py
- Temporal contrastive training: training/pretrain/train_waymo_ssl.py
- MoCo SSL training: training/pretrain/moco_waymo_ssl.py

Usage:
    # Run SSL pretraining (vanilla temporal contrastive)
    python -m training.pretrain.train_waymo_ssl \
        --episode-dir /path/to/episodes \
        --batch-size 32 \
        --num-steps 1000

    # Run MoCo SSL pretraining (momentum contrast)
    python -m training.pretrain.moco_waymo_ssl \
        --episode-dir /path/to/episodes \
        --batch-size 32 \
        --num-steps 10000 \
        --moco-m 0.999 \
        --queue-size 65536
"""

# Import lazily to avoid circular imports
def __getattr__(name):
    if name == "WaymoTemporalPairDataset":
        from training.pretrain.waymo_ssl_dataset import WaymoTemporalPairDataset
        return WaymoTemporalPairDataset
    elif name == "create_waymo_ssl_dataloader":
        from training.pretrain.waymo_ssl_dataset import create_waymo_ssl_dataloader
        return create_waymo_ssl_dataloader
    elif name == "collate_temporal_pairs":
        from training.pretrain.waymo_ssl_dataset import collate_temporal_pairs
        return collate_temporal_pairs
    elif name == "MoCoEncoder":
        from training.pretrain.moco_waymo_ssl import MoCoEncoder
        return MoCoEncoder
    elif name == "MoCoQueue":
        from training.pretrain.moco_waymo_ssl import MoCoQueue
        return MoCoQueue
    elif name == "load_moco_checkpoint":
        from training.pretrain.moco_waymo_ssl import load_moco_checkpoint
        return load_moco_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WaymoTemporalPairDataset",
    "create_waymo_ssl_dataloader",
    "collate_temporal_pairs",
    "MoCoEncoder",
    "MoCoQueue",
    "load_moco_checkpoint",
]

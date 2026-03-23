"""SSL Pretraining module.

This module provides self-supervised pretraining for driving perception.

Key components:
- Waymo SSL dataset: training/pretrain/waymo_ssl_dataset.py
- Temporal contrastive training: training/pretrain/train_waymo_ssl.py
- MoCo SSL training: training/pretrain/moco_waymo_ssl.py
- SimCLR SSL training: training/pretrain/simclr_waymo_ssl.py
- BEV SSL training: training/pretrain/bev_ssl_pretrain.py (temporal + cross-modal)

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

    # Run SimCLR SSL pretraining (simple contrastive)
    python -m training.pretrain.simclr_waymo_ssl \
        --episode-dir /path/to/episodes \
        --batch-size 64 \
        --num-steps 10000 \
        --temperature 0.07

    # Run BEV SSL pretraining (temporal + cross-modal alignment)
    python -m training.pretrain.bev_ssl_pretrain_aug \
        --episode-dir /path/to/episodes \
        --batch-size 32 \
        --num-epochs 100 \
        --output-dir out/bev_ssl \
        --use-bev-augmentations
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
    elif name == "SimCLRModel":
        from training.pretrain.simclr_waymo_ssl import SimCLRModel
        return SimCLRModel
    elif name == "SimCLRConfig":
        from training.pretrain.simclr_waymo_ssl import SimCLRConfig
        return SimCLRConfig
    elif name == "simclr_loss":
        from training.pretrain.simclr_waymo_ssl import simclr_loss
        return simclr_loss
    elif name == "BEVEncoder":
        from training.pretrain.bev_encoder import BEVEncoder
        return BEVEncoder
    elif name == "BEVConfig":
        from training.pretrain.bev_encoder import BEVConfig
        return BEVConfig
    elif name == "create_bev_encoder":
        from training.pretrain.bev_encoder import create_bev_encoder
        return create_bev_encoder
    elif name == "BEVSSLConfig":
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLConfig
        return BEVSSLConfig
    elif name == "BEVSSLTrainer":
        from training.pretrain.bev_ssl_pretrain_aug import BEVSSLTrainer
        return BEVSSLTrainer
    elif name == "bev_ssl_training_loop":
        from training.pretrain.bev_ssl_pretrain_aug import bev_ssl_training_loop
        return bev_ssl_training_loop
    elif name == "AugmentationPipeline":
        from training.pretrain.bev_ssl_pretrain_aug import AugmentationPipeline
        return AugmentationPipeline
    elif name == "BEVAugmentationConfig":
        from training.pretrain.bev_augmentations import BEVAugmentationConfig
        return BEVAugmentationConfig
    elif name == "BEVAugmentation":
        from training.pretrain.bev_augmentations import BEVAugmentation
        return BEVAugmentation
    elif name == "build_bev_augmentation":
        from training.pretrain.bev_augmentations import build_bev_augmentation
        return build_bev_augmentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WaymoTemporalPairDataset",
    "create_waymo_ssl_dataloader",
    "collate_temporal_pairs",
    "MoCoEncoder",
    "MoCoQueue",
    "load_moco_checkpoint",
    "SimCLRModel",
    "SimCLRConfig",
    "simclr_loss",
    "BEVEncoder",
    "BEVConfig",
    "create_bev_encoder",
    "BEVSSLConfig",
    "BEVSSLTrainer",
    "bev_ssl_training_loop",
    "AugmentationPipeline",
    "BEVAugmentationConfig",
    "BEVAugmentation",
    "build_bev_augmentation",
]

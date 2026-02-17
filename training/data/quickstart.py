"""
QuickStart: Unified Dataset Interface

Examples for using Waymo, Alpamayo, and nuScenes with a common interface.

Usage:
    python training/data/quickstart.py --dataset waymo --data-root /data/waymo
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.data.unified_dataset import (
    UnifiedDataset,
    DatasetRegistry,
    DrivingState,
    UnifiedSample,
)


def quickstart_waymo():
    """Example: Using Waymo dataset."""
    print("=" * 60)
    print("Waymo Dataset QuickStart")
    print("=" * 60)
    
    # Create dataset
    dataset = UnifiedDataset.from_config({
        "name": "waymo",
        "data_root": "/data/waymo",  # Update this path
        "split": "train",
        "cot": {
            "enabled": True,
        },
    })
    
    print(f"Dataset: {dataset.adapter.name}")
    print(f"Split: {dataset.split}")
    print(f"Episodes: {len(dataset.sample_ids)}")
    
    # Iterate over samples
    print("\nSample iteration:")
    for i, sample in enumerate(dataset):
        if i >= 2:  # Show first 2 samples
            break
        
        print(f"\n  Sample {i}: {sample.sample_id}")
        print(f"    Timestamp: {sample.timestamp:.3f}s")
        print(f"    State: speed={sample.state.ego_speed:.2f} m/s, "
              f"heading={np.degrees(sample.state.ego_heading):.1f}°")
        print(f"    Waypoints shape: {sample.expert_trajectory.shape if sample.expert_trajectory is not None else 'None'}")
        print(f"    Images: {list(sample.images.keys())}")
        
        if sample.cot_trace:
            print(f"    CoT: {sample.cot_trace.perception[:50]}...")


def quickstart_alpamayo():
    """Example: Using Alpamayo dataset."""
    print("=" * 60)
    print("Alpamayo-R1 Dataset QuickStart")
    print("=" * 60)
    
    dataset = UnifiedDataset.from_config({
        "name": "alpamayo",
        "data_root": "/data/alpamayo",
        "split": "train",
    })
    
    print(f"Dataset: {dataset.adapter.name}")
    print(f"Samples: {len(dataset.sample_ids)}")
    
    for i, sample in enumerate(dataset):
        if i >= 2:
            break
        
        print(f"\n  Sample {i}: {sample.sample_id}")
        print(f"    State: speed={sample.state.ego_speed:.2f} m/s")
        print(f"    Waypoints: {sample.expert_trajectory.shape if sample.expert_trajectory is not None else 'None'}")


def quickstart_nuscenes():
    """Example: Using nuScenes dataset."""
    print("=" * 60)
    print("nuScenes Dataset QuickStart")
    print("=" * 60)
    
    dataset = UnifiedDataset.from_config({
        "name": "nuscenes",
        "data_root": "/data/nuscenes",
        "version": "v1.0-trainval",
        "split": "train",
    })
    
    print(f"Dataset: {dataset.adapter.name}")
    print(f"Samples: {len(dataset.sample_ids)}")
    
    for i, sample in enumerate(dataset):
        if i >= 2:
            break
        
        print(f"\n  Sample {i}: {sample.sample_id}")
        print(f"    Objects: {len(sample.objects)}")


def compare_datasets():
    """Compare different datasets with unified interface."""
    print("=" * 60)
    print("Dataset Comparison")
    print("=" * 60)
    
    # List available datasets
    print("\nAvailable datasets:")
    for name in DatasetRegistry.list_datasets():
        print(f"  - {name}")
    
    # Check if datasets are configured
    print("\nDataset configuration check:")
    for name in DatasetRegistry.list_datasets():
        config = {"name": name, "data_root": f"/data/{name}"}
        try:
            adapter_class = DatasetRegistry.get(name)
            # Don't actually create - just show it would work
            print(f"  {name}: Ready to use")
        except Exception as e:
            print(f"  {name}: {e}")


def training_integration():
    """Show how to integrate with training pipeline."""
    print("=" * 60)
    print("Training Integration")
    print("=" * 60)
    
    print("""
# Usage with PyTorch DataLoader:

from torch.utils.data import DataLoader
from training.data.unified_dataset import UnifiedDataset

# Create unified dataset
dataset = UnifiedDataset.from_config({
    "name": "waymo",
    "data_root": "/data/waymo",
    "split": "train",
})

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

# Training loop
for batch in dataloader:
    # batch is a dict with:
    # - states: [B, state_dim]
    # - images: [B, C, H, W] or paths
    # - waypoints: [B, T, 3]
    # - cot_traces: list of CoTTrace
    ...


# collate_fn for UnifiedDataset:
def collate_fn(batch):
    import torch
    return {
        'states': torch.stack([s.get_state_tensor() for s in batch]),
        'waypoints': torch.stack([torch.from_numpy(s.get_waypoints()) for s in batch]),
        'episode_ids': [s.episode_id for s in batch],
        'timestamps': torch.tensor([s.timestamp for s in batch]),
    }
""")


def main():
    parser = argparse.ArgumentParser(description="Unified Dataset QuickStart")
    parser.add_argument("--dataset", type=str, default="list",
                        choices=["list", "waymo", "alpamayo", "nuscenes", "compare", "train"])
    parser.add_argument("--data-root", type=str, default="/data")
    args = parser.parse_args()
    
    if args.dataset == "list":
        DatasetRegistry.register_all()
        compare_datasets()
    elif args.dataset == "waymo":
        quickstart_waymo()
    elif args.dataset == "alpamayo":
        quickstart_alpamayo()
    elif args.dataset == "nuscenes":
        quickstart_nuscenes()
    elif args.dataset == "compare":
        compare_datasets()
    elif args.dataset == "train":
        training_integration()


if __name__ == "__main__":
    import numpy as np
    main()

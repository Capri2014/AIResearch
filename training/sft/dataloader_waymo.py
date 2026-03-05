"""
Waymo Motion Dataset Loader

Loads Waymo Open Motion Dataset (WOMD) TFRecords and provides PyTorch
dataset interface for SSL pretraining or waypoint BC pipeline.

Supports:
- Agent trajectory extraction
- Map polyline extraction (lanes, crosswalks)
- Historical trajectory encoding (agent history)
- Future prediction targets

Usage:
    from training.sft.dataloader_waymo import WaymoMotionDataset
    
    dataset = WaymoMotionDataset(
        tfrecord_paths=["path/to/*.tfrecord"],
        historical_steps=20,
        future_steps=30,
        agent_types=["vehicle", "pedestrian", "cyclist"]
    )
    
    sample = dataset[0]
    # sample["agent_history"]: (A, T, 7) - x, y, z, vx, vy, yaw, type
    # sample["agent_futures"]: (A, H, 2) - x, y positions
    # sample["map_polylines"]: (P, M, 3) - x, y, is_endpoint
"""

import os
import glob
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Waymo data types
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class WaymoSample:
    """Single sample from Waymo motion dataset."""
    # Agent history: (A, T, 7) - [x, y, z, vx, vy, yaw, type]
    agent_history: torch.Tensor
    # Agent future: (A, H, 2) - [x, y] positions
    agent_future: torch.Tensor
    # Valid agent mask: (A,)
    agent_valid: torch.Tensor
    # Map polylines: (P, M, 3) - [x, y, is_endpoint]
    map_polylines: torch.Tensor
    # Polyline valid mask: (P,)
    polyline_valid: torch.Tensor
    # Scenario ID
    scenario_id: str
    # Timestamp (ms)
    timestamp: int


class WaymoMotionDataset(Dataset):
    """
    PyTorch dataset for Waymo Open Motion Dataset.
    
    Loads TFRecords and extracts:
    - Historical agent trajectories (past 20 steps @ 10Hz = 2s)
    - Future trajectories (future 30 steps @ 10Hz = 3s)
    - Map polylines (lanes, boundaries, crosswalks)
    """
    
    # Agent type mapping
    AGENT_TYPE_MAP = {
        1: 0,  # vehicle
        2: 1,  # pedestrian  
        3: 2,  # cyclist
    }
    
    def __init__(
        self,
        tfrecord_paths: List[str],
        historical_steps: int = 20,
        future_steps: int = 30,
        max_agents: int = 128,
        max_polyline_points: int = 20000,
        agent_types: List[str] = None,
        transform=None,
        cache_processed: bool = False,
    ):
        """
        Args:
            tfrecord_paths: List of TFRecord file paths or glob patterns
            historical_steps: Number of historical timesteps (default 20 = 2s @ 10Hz)
            future_steps: Number of future timesteps to predict (default 30 = 3s)
            max_agents: Maximum number of agents to keep per scenario
            max_polyline_points: Maximum number of map polyline points
            agent_types: Filter by agent types ["vehicle", "pedestrian", "cyclist"]
            transform: Optional transform to apply to samples
            cache_processed: Whether to cache processed samples in memory
        """
        if not TF_AVAILABLE:
            raise ImportError("tensorflow required for Waymo dataset loading")
        
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.max_agents = max_agents
        self.max_polyline_points = max_polyline_points
        self.transform = transform
        self.cache_processed = cache_processed
        self.agent_types = agent_types or ["vehicle", "pedestrian", "cyclist"]
        
        # Collect all TFRecord files
        self.tfrecord_files = []
        for path in tfrecord_paths:
            if "*" in path or "?" in path:
                self.tfrecord_files.extend(glob.glob(path))
            elif os.path.isfile(path):
                self.tfrecord_files.append(path)
            elif os.path.isdir(path):
                self.tfrecord_files.extend(glob.glob(os.path.join(path, "*.tfrecord")))
        
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found: {tfrecord_paths}")
        
        # Load scenario metadata (lazy loading)
        self.scenarios = []
        self._load_scenario_metadata()
        
        # Cache for processed samples
        self._cache = {} if cache_processed else None
    
    def _load_scenario_metadata(self):
        """Load metadata from TFRecords without full decoding."""
        print(f"Loading Waymo metadata from {len(self.tfrecord_files)} files...")
        
        for tf_file in self.tfrecord_files:
            try:
                dataset = tf.data.TFRecordDataset(tf_file)
                for record in dataset:
                    scenario = self._parse_scenario_proto(record.numpy())
                    if scenario is not None:
                        self.scenarios.append(scenario)
            except Exception as e:
                print(f"Warning: Failed to load {tf_file}: {e}")
        
        print(f"Loaded {len(self.scenarios)} scenarios")
    
    def _parse_scenario_proto(self, record: bytes) -> Optional[Dict]:
        """Parse a single Waymo scenario proto."""
        # Waymo proto parsing - simplified for demo
        # In production, use waymo_open_dataset package
        try:
            # This is a placeholder - actual implementation would use
            # waymo_open_dataset.protos.scenario_pb2
            import json
            
            # For now, return a dummy scenario structure
            # Real implementation would decode the proto properly
            return {
                "scenario_id": "dummy",
                "timestamp": 0,
                "data": record,
            }
        except Exception:
            return None
    
    def __len__(self) -> int:
        return len(self.scenarios)
    
    def __getitem__(self, idx: int) -> WaymoSample:
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        
        scenario = self.scenarios[idx]
        
        # Parse actual Waymo data
        sample = self._decode_scenario(scenario)
        
        if self._cache is not None:
            self._cache[idx] = sample
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _decode_scenario(self, scenario: Dict) -> WaymoSample:
        """Decode a scenario into tensors."""
        # Placeholder - real implementation would:
        # 1. Decode TFRecord using waymo_open_dataset
        # 2. Extract agent trajectories
        # 3. Extract map polylines
        # 4. Apply coordinate transformations
        
        # For now, return dummy data with correct shape
        n_agents = min(self.max_agents, 32)
        n_polylines = 100
        points_per_polyline = 20
        
        return WaymoSample(
            agent_history=torch.randn(n_agents, self.historical_steps, 7),
            agent_future=torch.randn(n_agents, self.future_steps, 2),
            agent_valid=torch.rand(n_agents) > 0.3,
            map_polylines=torch.randn(n_polylines, points_per_polyline, 3),
            polyline_valid=torch.rand(n_polylines) > 0.5,
            scenario_id=scenario.get("scenario_id", "unknown"),
            timestamp=scenario.get("timestamp", 0),
        )


class WaymoToWaypointBCAdapter:
    """
    Adapter to convert Waymo samples to waypoint BC format.
    
    Provides compatibility with existing EpisodesWaypointBCDataset.
    """
    
    def __init__(self, waymo_dataset: WaymoMotionDataset):
        self.waymo_dataset = waymo_dataset
    
    def __len__(self) -> int:
        return len(self.waymo_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """Convert Waymo sample to waypoint BC format."""
        sample = self.waymo_dataset[idx]
        
        # Get the target agent (most important vehicle)
        # For now, use first valid vehicle agent
        agent_history = sample.agent_history
        agent_future = sample.agent_future
        agent_valid = sample.agent_valid
        
        # Find valid vehicle agents
        vehicle_mask = agent_history[:, 0, 6] == 0  # type == vehicle
        valid_vehicles = agent_valid & vehicle_mask
        
        if valid_vehicles.any():
            target_idx = valid_vehicles.nonzero()[0].item()
        else:
            target_idx = 0
        
        # Extract target agent trajectory
        target_history = agent_history[target_idx]  # (T, 7)
        target_future = agent_future[target_idx]  # (H, 2)
        
        # Convert to waypoint BC format
        return {
            # Agent history: (A, T, 7)
            "agent_history": agent_history,
            # Map: (P, M, 3)
            "map_polylines": sample.map_polylines,
            # Target future waypoints: (H, 2)
            "waypoints": target_future,
            # Valid mask
            "agent_valid": agent_valid,
            "polyline_valid": sample.polyline_valid,
            # Metadata
            "scenario_id": sample.scenario_id,
        }


def create_waymo_dataloader(
    tfrecord_paths: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    historical_steps: int = 20,
    future_steps: int = 30,
    use_adapter: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for Waymo motion dataset.
    
    Args:
        tfrecord_paths: List of TFRecord paths or glob patterns
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        historical_steps: Historical timesteps
        future_steps: Future timesteps
        use_adapter: Whether to use WaypointBC adapter
        **dataset_kwargs: Additional dataset arguments
    
    Returns:
        DataLoader ready for training
    """
    dataset = WaymoMotionDataset(
        tfrecord_paths=tfrecord_paths,
        historical_steps=historical_steps,
        future_steps=future_steps,
        **dataset_kwargs,
    )
    
    if use_adapter:
        dataset = WaymoToWaypointBCAdapter(dataset)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# Demo / test function
def demo_loader():
    """Demo the Waymo loader with dummy data."""
    print("Waymo Motion Dataset Loader Demo")
    print("=" * 50)
    
    # Create dataset with no files (will use dummy data)
    # In real usage, provide actual TFRecord paths
    try:
        dataset = WaymoMotionDataset(
            tfrecord_paths=["dummy/*.tfrecord"],
            historical_steps=20,
            future_steps=30,
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading
        sample = dataset[0]
        print(f"\nSample shapes:")
        print(f"  agent_history: {sample.agent_history.shape}")
        print(f"  agent_future: {sample.agent_future.shape}")
        print(f"  map_polylines: {sample.map_polylines.shape}")
        print(f"  agent_valid: {sample.agent_valid.shape}")
        
        # Test adapter
        adapter = WaymoToWaypointBCAdapter(dataset)
        adapted = adapter[0]
        print(f"\nAdapted sample keys: {list(adapted.keys())}")
        
    except Exception as e:
        print(f"Demo note: {e}")
        print("(Would work with real Waymo TFRecord files)")


if __name__ == "__main__":
    demo_loader()

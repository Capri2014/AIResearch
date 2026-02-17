"""
Unified Dataset Interface for Autonomous Driving

Supports:
- Waymo Open Dataset
- NVIDIA Alpamayo-R1
- nuScenes

Usage:
    from training.data.unified_dataset import UnifiedDataset, DatasetRegistry
    
    # Register datasets
    DatasetRegistry.register("waymo", WaymoAdapter)
    DatasetRegistry.register("alpamayo", AlpamayoAdapter)
    DatasetRegistry.register("nuscenes", NuScenesAdapter)
    
    # Create unified dataset
    dataset = UnifiedDataset.from_config({
        "name": "waymo",
        "data_root": "/data/waymo",
        "split": "train"
    })
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


# ============================================================================
# Common Schema
# ============================================================================

@dataclass
class DrivingState:
    """Unified driving state representation."""
    ego_speed: float = 0.0           # m/s
    ego_heading: float = 0.0        # radians
    ego_acceleration: float = 0.0    # m/s²
    ego_angular_rate: float = 0.0    # rad/s
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    timestamp: float = 0.0
    road_type: str = "unknown"
    is_junction: bool = False
    is_intersection: bool = False
    traffic_light_state: int = 0
    
    def to_tensor(self) -> np.ndarray:
        return np.array([
            self.ego_speed, self.ego_heading, self.ego_acceleration,
            self.ego_angular_rate, self.position_x, self.position_y,
            self.position_z, self.timestamp,
            1.0 if self.is_junction else 0.0,
            1.0 if self.is_intersection else 0.0,
            self.traffic_light_state,
        ], dtype=np.float32)


@dataclass
class DetectedObject:
    """Unified detected object."""
    object_id: str
    object_type: str  # vehicle, pedestrian, cyclist, unknown
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    size: Tuple[float, float, float]
    heading: float = 0.0
    confidence: float = 1.0
    predicted_trajectory: Optional[List[Tuple]] = None
    intent: Optional[str] = None


@dataclass
class LaneContext:
    """Unified lane context."""
    lane_id: str = ""
    lane_type: str = "drivable"
    lane_marking: str = "solid"
    lateral_offset: float = 0.0
    longitudinal_offset: float = 0.0
    confidence: float = 1.0


@dataclass
class CoTTrace:
    """Unified Chain of Thought trace."""
    perception: str = ""
    prediction: str = ""
    planning: str = ""
    justification: str = ""
    confidence: float = 1.0
    source: str = "auto"  # auto, human, model
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        return {
            "perception": self.perception,
            "prediction": self.prediction,
            "planning": self.planning,
            "justification": self.justification,
            "confidence": self.confidence,
            "source": self.source,
            "version": self.version,
        }


@dataclass
class UnifiedSample:
    """Unified training sample."""
    sample_id: str
    dataset_name: str
    episode_id: str
    timestamp: float
    state: DrivingState
    images: Dict = field(default_factory=dict)
    lidar_points: Optional[np.ndarray] = None
    lidar_range: Tuple = (100.0, 100.0, 6.0)
    objects: List = field(default_factory=list)
    lanes: List = field(default_factory=list)
    expert_trajectory: np.ndarray = None
    trajectory_timestamps: np.ndarray = None
    cot_trace: Optional[CoTTrace] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.expert_trajectory is not None:
            if self.expert_trajectory.ndim != 2:
                raise ValueError("expert_trajectory must be 2D")
            if self.expert_trajectory.shape[1] != 3:
                raise ValueError("expert_trajectory must have 3 columns (x, y, heading)")
    
    def get_state_tensor(self) -> np.ndarray:
        return self.state.to_tensor()
    
    def get_waypoints(self) -> np.ndarray:
        return self.expert_trajectory


# ============================================================================
# Base Adapter Interface
# ============================================================================

class DatasetAdapter(ABC):
    """Base class for dataset adapters."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def supported_splits(self) -> List[str]:
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict) -> bool:
        pass
    
    @abstractmethod
    def get_available_episodes(self, split: str) -> List[str]:
        pass
    
    @abstractmethod
    def load_sample(self, sample_id: str) -> UnifiedSample:
        pass
    
    @abstractmethod
    def load_episode(self, episode_id: str) -> List[UnifiedSample]:
        pass
    
    @abstractmethod
    def generate_cot(self, sample: UnifiedSample) -> CoTTrace:
        pass


# ============================================================================
# Waymo Adapter
# ============================================================================

class WaymoAdapter(DatasetAdapter):
    """Adapter for Waymo Open Dataset."""
    
    name = "waymo"
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_root = Path(config["data_root"])
        self.cot_config = config.get("cot", {})
    
    def validate_config(self, config: Dict) -> bool:
        if "data_root" not in config:
            raise ValueError("Missing required config: data_root")
        return True
    
    @property
    def supported_splits(self) -> List[str]:
        return ["train", "val", "test"]
    
    def get_available_episodes(self, split: str) -> List[str]:
        split_dir = self.data_root / split
        if not split_dir.exists():
            return []
        return [p.name for p in split_dir.iterdir() if p.is_dir()]
    
    def load_sample(self, sample_id: str) -> UnifiedSample:
        episode_id, frame_id = sample_id.split("/")
        
        # Load state
        state = self._load_state(episode_id, frame_id)
        
        # Load images
        images = {"front": str(self.data_root / episode_id / "camera_front" / f"{frame_id}.jpg")}
        
        # Load objects
        objects = self._load_objects(episode_id, frame_id)
        
        # Load trajectory
        trajectory = self._load_trajectory(episode_id, frame_id)
        
        sample = UnifiedSample(
            sample_id=sample_id,
            dataset_name=self.name,
            episode_id=episode_id,
            timestamp=state.timestamp,
            state=state,
            images=images,
            objects=objects,
            expert_trajectory=trajectory,
        )
        
        if self.cot_config.get("enabled", False):
            sample.cot_trace = self.generate_cot(sample)
        
        return sample
    
    def load_episode(self, episode_id: str) -> List[UnifiedSample]:
        samples = []
        frame_ids = self._get_frame_ids(episode_id)
        for frame_id in frame_ids:
            samples.append(self.load_sample(f"{episode_id}/{frame_id}"))
        return samples
    
    def generate_cot(self, sample: UnifiedSample) -> CoTTrace:
        vehicles = [o for o in sample.objects if o.object_type == "vehicle"]
        
        if vehicles:
            lead = self._find_lead_vehicle(vehicles, sample.state)
            if lead:
                dist = np.linalg.norm([
                    lead.position[0] - sample.state.position_x,
                    lead.position[1] - sample.state.position_y,
                ])
                perception = f"Lead vehicle at {dist:.1f}m ahead"
            else:
                perception = f"{len(vehicles)} vehicles visible"
        else:
            perception = "Clear road ahead"
        
        return CoTTrace(
            perception=perception,
            prediction="Traffic stable",
            planning="Follow trajectory",
            source="auto",
        )
    
    def _load_state(self, episode_id: str, frame_id: str) -> DrivingState:
        pose_file = self.data_root / episode_id / f"{frame_id}.json"
        if pose_file.exists():
            pose = json.loads(pose_file.read_text())
            return DrivingState(
                ego_speed=pose.get("speed", 0.0),
                ego_heading=pose.get("heading", 0.0),
                position_x=pose.get("x", 0.0),
                position_y=pose.get("y", 0.0),
                timestamp=pose.get("timestamp", 0.0),
            )
        return DrivingState()
    
    def _load_objects(self, episode_id: str, frame_id: str) -> List[DetectedObject]:
        label_file = self.data_root / episode_id / f"{frame_id}_labels.json"
        if not label_file.exists():
            return []
        
        labels = json.loads(label_file.read_text())
        objects = []
        for label in labels.get("labels", []):
            objects.append(DetectedObject(
                object_id=label["id"],
                object_type=label["type"],
                position=(label["x"], label["y"], label.get("z", 0)),
                velocity=(label.get("vx", 0), label.get("vy", 0), 0),
                size=(label.get("length", 0), label.get("width", 0), label.get("height", 0)),
                heading=label.get("heading", 0),
            ))
        return objects
    
    def _load_trajectory(self, episode_id: str, frame_id: str) -> Optional[np.ndarray]:
        traj_file = self.data_root / episode_id / f"{frame_id}_trajectory.json"
        if not traj_file.exists():
            return None
        traj = json.loads(traj_file.read_text())
        return np.array(traj["waypoints"], dtype=np.float32)
    
    def _get_frame_ids(self, episode_id: str) -> List[str]:
        camera_dir = self.data_root / episode_id / "camera_front"
        if not camera_dir.exists():
            return []
        return [f.stem for f in camera_dir.glob("*.jpg")]
    
    def _find_lead_vehicle(self, vehicles: List, state: DrivingState) -> Optional:
        lead = None
        min_dist = float("inf")
        for v in vehicles:
            rel_x = v.position[0] - state.position_x
            rel_y = v.position[1] - state.position_y
            cos_h, sin_h = np.cos(state.ego_heading), np.sin(state.ego_heading)
            forward = rel_x * cos_h + rel_y * sin_h
            lateral = -rel_x * sin_h + rel_y * cos_h
            if forward > 0 and abs(lateral) < 2.0:
                dist = np.sqrt(forward**2 + lateral**2)
                if dist < min_dist:
                    min_dist = dist
                    lead = v
        return lead


# ============================================================================
# Alpamayo Adapter
# ============================================================================

class AlpamayoAdapter(DatasetAdapter):
    """Adapter for NVIDIA Alpamayo-R1."""
    
    name = "alpamayo"
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_root = Path(config["data_root"])
        self.cot_config = config.get("cot", {})
    
    def validate_config(self, config: Dict) -> bool:
        if "data_root" not in config:
            raise ValueError("Missing required config: data_root")
        return True
    
    @property
    def supported_splits(self) -> List[str]:
        return ["train", "val"]
    
    def get_available_episodes(self, split: str) -> List[str]:
        split_dir = self.data_root / split
        if not split_dir.exists():
            return []
        return [p.name for p in split_dir.iterdir() if p.is_dir()]
    
    def load_sample(self, sample_id: str) -> UnifiedSample:
        parts = sample_id.split("/")
        episode_id, frame_id = parts[0], "/".join(parts[1:])
        
        state = self._load_state(episode_id, frame_id)
        images = {"front": str(self.data_root / episode_id / "images" / f"{frame_id}.jpg")}
        trajectory = self._load_trajectory(episode_id, frame_id)
        
        sample = UnifiedSample(
            sample_id=sample_id,
            dataset_name=self.name,
            episode_id=episode_id,
            timestamp=state.timestamp,
            state=state,
            images=images,
            expert_trajectory=trajectory,
        )
        
        if self.cot_config.get("enabled", False):
            sample.cot_trace = self.generate_cot(sample)
        
        return sample
    
    def load_episode(self, episode_id: str) -> List[UnifiedSample]:
        samples = []
        for fid in self._get_frame_ids(episode_id):
            samples.append(self.load_sample(f"{episode_id}/{fid}"))
        return samples
    
    def generate_cot(self, sample: UnifiedSample) -> CoTTrace:
        state = sample.state
        perception = f"Ego at {state.ego_speed:.1f} m/s, heading {np.degrees(state.ego_heading):.1f}°"
        
        return CoTTrace(
            perception=perception,
            planning="Follow trajectory",
            source="model",
        )
    
    def _load_state(self, episode_id: str, frame_id: str) -> DrivingState:
        state_file = self.data_root / episode_id / "state" / f"{frame_id}.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            return DrivingState(
                ego_speed=data.get("speed", 0),
                ego_heading=data.get("heading", 0),
                position_x=data.get("x", 0),
                position_y=data.get("y", 0),
                timestamp=data.get("timestamp", 0),
            )
        return DrivingState()
    
    def _load_trajectory(self, episode_id: str, frame_id: str) -> Optional[np.ndarray]:
        traj_file = self.data_root / episode_id / "trajectory" / f"{frame_id}.json"
        if not traj_file.exists():
            return None
        traj = json.loads(traj_file.read_text())
        return np.array(traj["waypoints"], dtype=np.float32)
    
    def _get_frame_ids(self, episode_id: str) -> List[str]:
        img_dir = self.data_root / episode_id / "images"
        if not img_dir.exists():
            return []
        return [f.stem for f in img_dir.glob("*.jpg")]


# ============================================================================
# nuScenes Adapter
# ============================================================================

class NuScenesAdapter(DatasetAdapter):
    """Adapter for nuScenes dataset."""
    
    name = "nuscenes"
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_root = Path(config["data_root"])
        self.version = config.get("version", "v1.0-trainval")
        self.cot_config = config.get("cot", {})
        
        try:
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(version=self.version, dataroot=str(self.data_root), verbose=False)
        except ImportError:
            raise ImportError("nuScenes adapter requires: pip install nuscenes-devkit")
    
    def validate_config(self, config: Dict) -> bool:
        if "data_root" not in config:
            raise ValueError("Missing required config: data_root")
        return True
    
    @property
    def supported_splits(self) -> List[str]:
        return ["train", "val"]
    
    def get_available_episodes(self, split: str) -> List[str]:
        return [s["token"] for s in self.nusc.scene if split in s["name"].lower()]
    
    def load_sample(self, sample_id: str) -> UnifiedSample:
        scene_token, frame_idx = sample_id.split("/")
        frame_idx = int(frame_idx)
        
        # Find sample token
        sample_token = None
        for scene in self.nusc.scene:
            if scene["token"] == scene_token:
                current = scene["first_sample_token"]
                for i in range(frame_idx):
                    sample = self.nusc.get("sample", current)
                    current = sample["next"]
                sample_token = current
                break
        
        if not sample_token:
            raise ValueError(f"Unknown sample: {sample_id}")
        
        sample = self.nusc.get("sample", sample_token)
        state = self._load_state(sample)
        images = self._load_images(sample)
        objects = self._load_objects(sample)
        trajectory = self._load_trajectory(sample)
        
        unified = UnifiedSample(
            sample_id=sample_id,
            dataset_name=self.name,
            episode_id=scene_token,
            timestamp=sample["timestamp"] / 1e6,
            state=state,
            images=images,
            objects=objects,
            expert_trajectory=trajectory,
        )
        
        if self.cot_config.get("enabled", False):
            unified.cot_trace = self.generate_cot(unified)
        
        return unified
    
    def load_episode(self, episode_id: str) -> List[UnifiedSample]:
        samples = []
        scene = self.nusc.get("scene", episode_id)
        current = scene["first_sample_token"]
        frame_idx = 0
        
        while current:
            sample = self.nusc.get("sample", current)
            samples.append(self.load_sample(f"{episode_id}/{frame_idx}"))
            current = sample["next"]
            frame_idx += 1
        
        return samples
    
    def generate_cot(self, sample: UnifiedSample) -> CoTTrace:
        n_vehicles = len([o for o in sample.objects if o.object_type == "vehicle"])
        n_pedestrians = len([o for o in sample.objects if o.object_type == "pedestrian"])
        
        return CoTTrace(
            perception=f"{n_vehicles} vehicles, {n_pedestrians} pedestrians",
            prediction="Stable traffic",
            planning="Follow trajectory",
            source="auto",
        )
    
    def _load_state(self, sample: Dict) -> DrivingState:
        ego_pose = self.nusc.get("ego_pose", sample["data"]["ego_pose"])
        return DrivingState(
            ego_speed=ego_pose.get("speed", 0),
            ego_heading=ego_pose["rotation"][1],
            position_x=ego_pose["translation"][0],
            position_y=ego_pose["translation"][1],
            position_z=ego_pose["translation"][2],
            timestamp=sample["timestamp"] / 1e6,
        )
    
    def _load_images(self, sample: Dict) -> Dict[str, str]:
        images = {}
        cam_map = {
            "CAM_FRONT": "front",
            "CAM_FRONT_LEFT": "front_left",
            "CAM_FRONT_RIGHT": "front_right",
            "CAM_SIDE_LEFT": "side_left",
            "CAM_SIDE_RIGHT": "side_right",
        }
        for token, name in cam_map.items():
            if token in sample["data"]:
                sd = self.nusc.get("sample_data", sample["data"][token])
                images[name] = str(self.data_root / sd["filename"])
        return images
    
    def _load_objects(self, sample: Dict) -> List[DetectedObject]:
        objects = []
        for ann_token in sample.get("anns", []):
            ann = self.nusc.get("annotation", ann_token)
            cat = ann["category_name"].split(".")[-1].lower()
            if "vehicle" in cat:
                obj_type = "vehicle"
            elif "pedestrian" in cat:
                obj_type = "pedestrian"
            elif "bicycle" in cat or "motorcycle" in cat:
                obj_type = "cyclist"
            else:
                obj_type = "unknown"
            
            objects.append(DetectedObject(
                object_id=ann["token"],
                object_type=obj_type,
                position=tuple(ann["translation"][:3]),
                velocity=(0, 0, 0),
                size=tuple(ann["size"]),
                heading=ann["rotation"][1],
            ))
        return objects
    
    def _load_trajectory(self, sample: Dict) -> Optional[np.ndarray]:
        return None  # nuScenes doesn't have explicit waypoints


# ============================================================================
# Dataset Registry
# ============================================================================

class DatasetRegistry:
    """Registry for dataset adapters."""
    
    _adapters: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: type) -> None:
        cls._adapters[name.lower()] = adapter_class
    
    @classmethod
    def get(cls, name: str) -> type:
        name = name.lower()
        if name not in cls._adapters:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._adapters.keys())}")
        return cls._adapters[name]
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls._adapters.keys())
    
    @classmethod
    def register_all(cls) -> None:
        cls.register("waymo", WaymoAdapter)
        cls.register("alpamayo", AlpamayoAdapter)
        cls.register("nuscenes", NuScenesAdapter)


# ============================================================================
# Unified Dataset
# ============================================================================

class UnifiedDataset:
    """Unified dataset wrapping any registered adapter."""
    
    def __init__(self, adapter: DatasetAdapter, split: str = "train"):
        self.adapter = adapter
        self.split = split
        self.sample_ids = adapter.get_available_episodes(split)
    
    @classmethod
    def from_config(cls, config: Dict) -> "UnifiedDataset":
        DatasetRegistry.register_all()
        adapter_class = DatasetRegistry.get(config["name"])
        adapter = adapter_class(config)
        return cls(adapter, config.get("split", "train"))
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> UnifiedSample:
        return self.adapter.load_sample(self.sample_ids[idx])
    
    def __iter__(self):
        for sid in self.sample_ids:
            yield self.adapter.load_sample(sid)
    
    def get_episode(self, episode_id: str) -> List[UnifiedSample]:
        return self.adapter.load_episode(episode_id)
    
    def split_dataset(self, train_ratio: float = 0.8) -> Tuple["UnifiedDataset", "UnifiedDataset"]:
        n = len(self.sample_ids)
        split_idx = int(n * train_ratio)
        train_ids = self.sample_ids[:split_idx]
        val_ids = self.sample_ids[split_idx:]
        
        train_ds = UnifiedDataset(self.adapter, self.split)
        train_ds.sample_ids = train_ids
        
        val_ds = UnifiedDataset(self.adapter, self.split)
        val_ds.sample_ids = val_ids
        
        return train_ds, val_ds

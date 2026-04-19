#!/usr/bin/env python3
"""
Waypoint Trajectory Sampler - Generates diverse waypoint trajectories for BC training and evaluation.

This sampler addresses a key gap in the waypoint BC pipeline by providing:
- Diverse trajectory generation for training data augmentation
- Configurable sampling strategies (lane-following, lane-change, turning)
- Speed profile sampling for different driving behaviors
- Evaluation scenario generation for closed-loop testing

Usage:
    python training/bc/waypoint_trajectory_sampler.py --output-dir out/waypoint_samples --num-samples 1000
    python training/bc/waypoint_trajectory_sampler.py --generate-eval-scenarios --scenarios straight,turning,lane_change
    python training/bc/waypoint_trajectory_sampler.py --augment-existing data/waymo/episodes --output out/augmented
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SamplerConfig:
    """Configuration for waypoint trajectory sampling."""
    num_waypoints: int = 8
    horizon_seconds: float = 3.0
    sampling_rate_hz: float = 2.0
    # Trajectory parameters
    min_speed_mps: float = 0.0
    max_speed_mps: float = 15.0  # ~54 km/h
    min_curvature: float = -0.3
    max_curvature: float = 0.3
    # Lane parameters
    lane_width_m: float = 3.5
    num_lanes: int = 3
    # Sampling strategy weights
    lane_following_weight: float = 0.5
    lane_change_weight: float = 0.25
    turning_weight: float = 0.25
    # Speed profile
    cruise_weight: float = 0.4
    accelerating_weight: float = 0.3
    decelerating_weight: float = 0.3
    # Noise for augmentation
    position_noise_std: float = 0.1  # meters
    heading_noise_std: float = 0.02  # radians


@dataclass 
class WaypointSample:
    """Single waypoint in a trajectory."""
    x: float
    y: float
    timestep: int
    speed_mps: Optional[float] = None
    heading_rad: Optional[float] = None


@dataclass
class TrajectorySample:
    """Complete trajectory sample with waypoints and metadata."""
    trajectory_id: str
    waypoints: List[WaypointSample]
    strategy: str  # lane_following, lane_change, turning
    speed_profile: str  # cruise, accelerating, decelerating
    start_speed_mps: float
    end_speed_mps: float
    num_lane_changes: int = 0
    is_turning: bool = False
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "strategy": self.strategy,
            "speed_profile": self.speed_profile,
            "start_speed_mps": self.start_speed_mps,
            "end_speed_mps": self.end_speed_mps,
            "num_lane_changes": self.num_lane_changes,
            "is_turning": self.is_turning,
            "waypoints": [
                {
                    "x": wp.x,
                    "y": wp.y,
                    "timestep": wp.timestep,
                    "speed_mps": wp.speed_mps,
                    "heading_rad": wp.heading_rad,
                }
                for wp in self.waypoints
            ],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrajectorySample":
        waypoints = [
            WaypointSample(
                x=wp["x"],
                y=wp["y"],
                timestep=wp["timestep"],
                speed_mps=wp.get("speed_mps"),
                heading_rad=wp.get("heading_rad"),
            )
            for wp in data["waypoints"]
        ]
        return cls(
            trajectory_id=data["trajectory_id"],
            waypoints=waypoints,
            strategy=data["strategy"],
            speed_profile=data["speed_profile"],
            start_speed_mps=data["start_speed_mps"],
            end_speed_mps=data["end_speed_mps"],
            num_lane_changes=data.get("num_lane_changes", 0),
            is_turning=data.get("is_turning", False),
            metadata=data.get("metadata", {}),
        )


class WaypointTrajectorySampler:
    """Generates diverse waypoint trajectories for BC training."""
    
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.num_timesteps = int(config.horizon_seconds * config.sampling_rate_hz)
        self.dt = 1.0 / config.sampling_rate_hz
        
    def sample_strategy(self) -> str:
        """Sample a trajectory generation strategy."""
        r = random.random()
        if r < self.config.lane_following_weight:
            return "lane_following"
        elif r < self.config.lane_following_weight + self.config.lane_change_weight:
            return "lane_change"
        else:
            return "turning"
    
    def sample_speed_profile(self) -> str:
        """Sample a speed profile type."""
        r = random.random()
        if r < self.config.cruise_weight:
            return "cruise"
        elif r < self.config.cruise_weight + self.config.accelerating_weight:
            return "accelerating"
        else:
            return "decelerating"
    
    def sample_speed(self, profile: str, t: float) -> float:
        """Sample speed based on profile and timestep."""
        # Sample base speed
        base_speed = random.uniform(self.config.min_speed_mps, self.config.max_speed_mps)
        
        if profile == "cruise":
            return base_speed
        elif profile == "accelerating":
            # Gradually increase speed
            factor = min(1.0, t * 0.5)
            return base_speed * (1.0 + factor * 0.3)
        else:  # decelerating
            # Gradually decrease speed
            factor = min(1.0, t * 0.5)
            return base_speed * (1.0 - factor * 0.3)
    
    def sample_curvature(self, strategy: str) -> float:
        """Sample curvature based on strategy."""
        if strategy == "lane_following":
            # Small curvature for lane following
            return random.uniform(-0.05, 0.05)
        elif strategy == "lane_change":
            # Moderate curvature for lane changes
            return random.choice([-1, 1]) * random.uniform(0.1, 0.2)
        else:  # turning
            # Higher curvature for turns
            return random.choice([-1, 1]) * random.uniform(0.15, 0.3)
    
    def generate_trajectory(
        self, 
        trajectory_id: str,
        strategy: Optional[str] = None,
        speed_profile: Optional[str] = None,
        start_x: float = 0.0,
        start_y: float = 0.0,
        start_heading: float = 0.0,
    ) -> TrajectorySample:
        """Generate a single trajectory sample."""
        if strategy is None:
            strategy = self.sample_strategy()
        if speed_profile is None:
            speed_profile = self.sample_speed_profile()
        
        # Generate curvature
        curvature = self.sample_curvature(strategy)
        
        # Initial state
        x, y = start_x, start_y
        heading = start_heading
        speed = self.sample_speed(speed_profile, 0.0)
        
        waypoints = []
        num_lane_changes = 0
        is_turning = strategy == "turning"
        
        for t in range(self.num_timesteps):
            # Compute speed for this timestep
            current_speed = self.sample_speed(speed_profile, t * self.dt)
            
            # Add noise for augmentation
            if self.config.position_noise_std > 0:
                x += random.gauss(0, self.config.position_noise_std)
                y += random.gauss(0, self.config.position_noise_std)
            if self.config.heading_noise_std > 0:
                heading += random.gauss(0, self.config.heading_noise_std)
            
            # Create waypoint
            waypoint = WaypointSample(
                x=x,
                y=y,
                timestep=t,
                speed_mps=current_speed,
                heading_rad=heading,
            )
            waypoints.append(waypoint)
            
            # Update state using bicycle model kinematics
            # x += speed * math.cos(heading) * self.dt
            # y += speed * math.sin(heading) * self.dt
            # heading += curvature * speed * self.dt
            
            # Simplified update (arc parameterization)
            if abs(curvature) > 1e-6:
                # Curved trajectory
                radius = 1.0 / curvature
                theta = curvature * speed * self.dt
                x += radius * (math.sin(heading + theta) - math.sin(heading))
                y += radius * (math.cos(heading) - math.cos(heading + theta))
                heading += theta
            else:
                # Straight trajectory
                x += speed * math.cos(heading) * self.dt
                y += speed * math.sin(heading) * self.dt
        
        # Get final speed
        end_speed = waypoints[-1].speed_mps or speed
        
        return TrajectorySample(
            trajectory_id=trajectory_id,
            waypoints=waypoints,
            strategy=strategy,
            speed_profile=speed_profile,
            start_speed_mps=speed,
            end_speed_mps=end_speed,
            num_lane_changes=num_lane_changes,
            is_turning=is_turning,
            metadata={
                "curvature": curvature,
                "config": {
                    "num_waypoints": self.config.num_waypoints,
                    "horizon_seconds": self.config.horizon_seconds,
                    "sampling_rate_hz": self.config.sampling_rate_hz,
                }
            },
        )
    
    def generate_dataset(
        self, 
        num_samples: int,
        trajectory_prefix: str = "traj",
    ) -> List[TrajectorySample]:
        """Generate a dataset of trajectory samples."""
        trajectories = []
        
        for i in range(num_samples):
            traj = self.generate_trajectory(
                trajectory_id=f"{trajectory_prefix}_{i:05d}",
            )
            trajectories.append(traj)
        
        return trajectories
    
    def generate_eval_scenarios(
        self, 
        scenario_types: List[str],
    ) -> dict:
        """Generate evaluation scenarios."""
        scenarios = {}
        
        for scenario_type in scenario_types:
            if scenario_type == "straight":
                # Straight lane following
                traj = self.generate_trajectory(
                    trajectory_id="eval_straight_001",
                    strategy="lane_following",
                    speed_profile="cruise",
                    start_heading=0.0,
                )
                scenarios["straight"] = traj.to_dict()
                
            elif scenario_type == "turning":
                # Turning scenario
                for turn_type in ["left", "right"]:
                    traj = self.generate_trajectory(
                        trajectory_id=f"eval_turn_{turn_type}_001",
                        strategy="turning",
                        speed_profile="decelerating",
                        start_heading=0.0 if turn_type == "left" else math.pi,
                    )
                    scenarios[f"turn_{turn_type}"] = traj.to_dict()
                    
            elif scenario_type == "lane_change":
                # Lane change scenarios
                for direction in ["left", "right"]:
                    traj = self.generate_trajectory(
                        trajectory_id=f"eval_lane_change_{direction}_001",
                        strategy="lane_change",
                        speed_profile="cruise",
                        start_heading=0.0,
                    )
                    scenarios[f"lane_change_{direction}"] = traj.to_dict()
                    
            elif scenario_type == "stop":
                # Stopping scenario
                traj = self.generate_trajectory(
                    trajectory_id="eval_stop_001",
                    strategy="lane_following",
                    speed_profile="decelerating",
                )
                scenarios["stop"] = traj.to_dict()
        
        return scenarios
    
    def augment_episodes(
        self, 
        episodes_dir: str,
        aug_factor: int = 2,
    ) -> List[TrajectorySample]:
        """Augment existing episode data with noise transformations."""
        # This would load existing episodes and apply augmentations
        # For now, generate synthetic augmented samples
        augmented = []
        
        for i in range(aug_factor):
            # Generate with different noise levels
            config = SamplerConfig(
                position_noise_std=random.uniform(0.05, 0.2),
                heading_noise_std=random.uniform(0.01, 0.05),
            )
            sampler = WaypointTrajectorySampler(config)
            trajs = sampler.generate_dataset(
                num_samples=10,
                trajectory_prefix=f"aug_{i}",
            )
            augmented.extend(trajs)
        
        return augmented
    
    def save_dataset(
        self, 
        trajectories: List[TrajectorySample],
        output_dir: str,
        format: str = "jsonl",
    ) -> None:
        """Save trajectory dataset to file(s)."""
        os.makedirs(output_dir, exist_ok=True)
        
        if format == "jsonl":
            # Save as JSONL (one JSON per line)
            output_path = os.path.join(output_dir, "trajectories.jsonl")
            with open(output_path, "w") as f:
                for traj in trajectories:
                    f.write(json.dumps(traj.to_dict()) + "\n")
                    
        elif format == "json":
            # Save as single JSON array
            output_path = os.path.join(output_dir, "trajectories.json")
            with open(output_path, "w") as f:
                json.dump(
                    [traj.to_dict() for traj in trajectories],
                    f,
                    indent=2,
                )
        
        # Save metadata
        metadata = {
            "num_samples": len(trajectories),
            "config": {
                "num_waypoints": self.config.num_waypoints,
                "horizon_seconds": self.config.horizon_seconds,
                "sampling_rate_hz": self.config.sampling_rate_hz,
            },
            "strategy_counts": self._count_strategies(trajectories),
            "speed_profile_counts": self._count_speed_profiles(trajectories),
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _count_strategies(self, trajectories: List[TrajectorySample]) -> dict:
        counts = {}
        for traj in trajectories:
            counts[traj.strategy] = counts.get(traj.strategy, 0) + 1
        return counts
    
    def _count_speed_profiles(self, trajectories: List[TrajectorySample]) -> dict:
        counts = {}
        for traj in trajectories:
            counts[traj.speed_profile] = counts.get(traj.speed_profile, 0) + 1
        return counts
    
    def compute_statistics(self, trajectories: List[TrajectorySample]) -> dict:
        """Compute statistics over trajectory dataset."""
        if not trajectories:
            return {}
        
        # Collect data
        speeds = []
        curvatures = []
        
        for traj in trajectories:
            for wp in traj.waypoints:
                if wp.speed_mps is not None:
                    speeds.append(wp.speed_mps)
            if "curvature" in traj.metadata:
                curvatures.append(traj.metadata["curvature"])
        
        stats = {
            "num_samples": len(trajectories),
            "speed": {
                "mean": np.mean(speeds) if speeds else 0.0,
                "std": np.std(speeds) if speeds else 0.0,
                "min": np.min(speeds) if speeds else 0.0,
                "max": np.max(speeds) if speeds else 0.0,
            },
            "strategy_counts": self._count_strategies(trajectories),
            "speed_profile_counts": self._count_speed_profiles(trajectories),
        }
        
        if curvatures:
            stats["curvature"] = {
                "mean": np.mean(curvatures),
                "std": np.std(curvatures),
                "min": np.min(curvatures),
                "max": np.max(curvatures),
            }
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Waypoint Trajectory Sampler - Generate diverse waypoint trajectories for BC training"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="out/waypoint_sampler",
        help="Output directory for generated trajectories",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of trajectory samples to generate",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=8,
        help="Number of waypoints per trajectory",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=3.0,
        help="Trajectory horizon in seconds",
    )
    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        default=2.0,
        help="Waypoint sampling rate in Hz",
    )
    parser.add_argument(
        "--generate-eval-scenarios",
        action="store_true",
        help="Generate evaluation scenarios",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="straight,turning,lane_change,stop",
        help="Comma-separated list of scenario types",
    )
    parser.add_argument(
        "--augment-existing",
        type=str,
        help="Augment existing episode directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics instead of generating",
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    config = SamplerConfig(
        num_waypoints=args.num_waypoints,
        horizon_seconds=args.horizon_seconds,
        sampling_rate_hz=args.sampling_rate_hz,
    )
    sampler = WaypointTrajectorySampler(config)
    
    if args.generate_eval_scenarios:
        # Generate evaluation scenarios
        scenario_types = args.scenarios.split(",")
        scenarios = sampler.generate_eval_scenarios(scenario_types)
        
        output_path = os.path.join(args.output_dir, "eval_scenarios.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(scenarios, f, indent=2)
        
        print(f"Generated {len(scenarios)} evaluation scenarios")
        print(f"Saved to: {output_path}")
        
    elif args.augment_existing:
        # Augment existing episodes
        augmented = sampler.augment_episodes(args.augmentExisting)
        sampler.save_dataset(augmented, args.output_dir, args.format)
        
        print(f"Augmented with {len(augmented)} trajectories")
        print(f"Saved to: {args.output_dir}")
        
    else:
        # Generate new trajectory dataset
        trajectories = sampler.generate_dataset(
            num_samples=args.num_samples,
        )
        sampler.save_dataset(trajectories, args.output_dir, args.format)
        
        if args.stats:
            stats = sampler.compute_statistics(trajectories)
            print("Dataset Statistics:")
            print(json.dumps(stats, indent=2))
        else:
            print(f"Generated {len(trajectories)} trajectories")
            print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
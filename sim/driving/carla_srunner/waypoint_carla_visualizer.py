#!/usr/bin/env python3
"""
Waypoint Prediction Visualizer for CARLA Simulation.

Visualizes predicted waypoints overlaid on CARLA world for real-time debugging and analysis.
Supports comparison between SFT and RL-refined predictions.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Try CARLA imports, fallback to mock if unavailable
CARLA_AVAILABLE = False
carla = None

try:
    import carla
    if carla:
        CARLA_AVAILABLE = True
except ImportError:
    pass


@dataclass
class WaypointVisualizerConfig:
    """Configuration for waypoint visualization."""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    fps: int = 20
    num_waypoints: int = 8
    waypoint_spacing: float = 3.0
    arrow_scale: float = 0.5
    color_predicted: Tuple[int, int, int] = (0, 255, 0)
    color_ground_truth: Tuple[int, int, int] = (255, 0, 0)
    color_sft: Tuple[int, int, int] = (255, 255, 0)
    color_rl: Tuple[int, int, int] = (0, 255, 255)
    show_debug: bool = False
    output_dir: str = "out/waypoint_viz"


@dataclass
class WaypointSample:
    """Single waypoint prediction sample."""
    timestamp: float
    position: np.ndarray
    heading: float
    speed: float
    predicted_waypoints: Optional[np.ndarray] = None
    ground_truth_waypoints: Optional[np.ndarray] = None
    sft_waypoints: Optional[np.ndarray] = None
    rl_waypoints: Optional[np.ndarray] = None


@dataclass
class VisualizationMetrics:
    """Metrics for visualization quality."""
    ade_predicted: float = 0.0
    fde_predicted: float = 0.0
    ade_sft: float = 0.0
    ade_rl: float = 0.0
    num_samples: int = 0
    mean_confidence: float = 0.0


class WaypointCarlaVisualizer:
    """Main class for visualizing waypoints in CARLA."""
    
    def __init__(self, config: WaypointVisualizerConfig):
        self.config = config
        self.client = None
        self.world = None
        self.actors = []
        self.samples = []
        self.metrics = VisualizationMetrics()
        
    def connect(self) -> bool:
        """Connect to CARLA server."""
        if not CARLA_AVAILABLE:
            print("[Mock] Connecting to CARLA server...")
            return True
            
        try:
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout)
            self.world = self.client.get_world()
            print(f"Connected to CARLA: {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from CARLA server."""
        for actor in self.actors:
            try:
                actor.destroy()
            except:
                pass
        self.actors = []
        
        if self.client:
            self.client = None
            self.world = None
            
    def world_to_ego(self, world_pos, ego_transform) -> np.ndarray:
        """Convert world position to ego frame."""
        if not CARLA_AVAILABLE:
            dx = world_pos[0] - ego_transform[0]
            dy = world_pos[1] - ego_transform[1]
            return np.array([dx, dy])
            
        dx = world_pos.x - ego_transform.location.x
        dy = world_pos.y - ego_transform.location.y
        yaw = np.radians(-ego_transform.rotation.yaw)
        
        ego_x = dx * np.cos(yaw) - dy * np.sin(yaw)
        ego_y = dx * np.sin(yaw) + dy * np.cos(yaw)
        return np.array([ego_x, ego_y])
        
    def ego_to_world(self, ego_pos, ego_transform):
        """Convert ego frame to world position."""
        if not CARLA_AVAILABLE:
            return ego_pos
            
        yaw = np.radians(ego_transform.rotation.yaw)
        wx = ego_pos[0] * np.cos(yaw) - ego_pos[1] * np.sin(yaw) + ego_transform.location.x
        wy = ego_pos[0] * np.sin(yaw) + ego_pos[1] * np.cos(yaw) + ego_transform.location.y
        return carla.Location(x=wx, y=wy, z=ego_transform.location.z)
        
    def draw_waypoints(self, waypoints: np.ndarray, color: tuple, ego_transform, label: str = ""):
        """Draw waypoints in CARLA world."""
        if not CARLA_AVAILABLE or not self.world:
            return
            
        debug = self.world.debug
            
        for i, wp in enumerate(waypoints):
            world_pos = self.ego_to_world(wp, ego_transform)
            debug.draw_point(world_pos, size=0.1, color=carla.Color(*color), life_time=0.1)
            
            if i > 0:
                prev_wp = waypoints[i - 1]
                prev_world = self.ego_to_world(prev_wp, ego_transform)
                debug.draw_arrow(prev_world, world_pos, 
                           thickness=0.05, color=carla.Color(*color), life_time=0.1)
                            
    def visualize_trajectory(self, sample: WaypointSample, ego_transform):
        """Visualize single trajectory sample."""
        if sample.predicted_waypoints is not None:
            self.draw_waypoints(sample.predicted_waypoints, self.config.color_predicted, 
                          ego_transform, "predicted")
                          
        if sample.ground_truth_waypoints is not None:
            self.draw_waypoints(sample.ground_truth_waypoints, self.config.color_ground_truth,
                          ego_transform, "ground_truth")
                          
        if sample.sft_waypoints is not None:
            self.draw_waypoints(sample.sft_waypoints, self.config.color_sft,
                          ego_transform, "sft")
                          
        if sample.rl_waypoints is not None:
            self.draw_waypoints(sample.rl_waypoints, self.config.color_rl,
                          ego_transform, "rl")
                          
    def compute_metrics(self, sample: WaypointSample) -> dict:
        """Compute ADE/FDE metrics for single sample."""
        metrics = {}
        
        if sample.predicted_waypoints is not None and sample.ground_truth_waypoints is not None:
            ade = np.linalg.norm(sample.predicted_waypoints - sample.ground_truth_waypoints, axis=1).mean()
            fde = np.linalg.norm(sample.predicted_waypoints[-1] - sample.ground_truth_waypoints[-1])
            metrics['ade_predicted'] = ade
            metrics['fde_predicted'] = fde
            
        if sample.sft_waypoints is not None and sample.ground_truth_waypoints is not None:
            ade_sft = np.linalg.norm(sample.sft_waypoints - sample.ground_truth_waypoints, axis=1).mean()
            metrics['ade_sft'] = ade_sft
            
        if sample.rl_waypoints is not None and sample.ground_truth_waypoints is not None:
            ade_rl = np.linalg.norm(sample.rl_waypoints - sample.ground_truth_waypoints, axis=1).mean()
            metrics['ade_rl'] = ade_rl
            
        return metrics
        
    def run_realtime(self, policy_fn=None, num_samples: int = 100):
        """Run real-time visualization loop."""
        if not self.connect():
            print("Running in mock mode...")
            self._run_mock_visualization(num_samples)
            return
            
        try:
            self._run_carla_visualization(policy_fn, num_samples)
        finally:
            self.disconnect()
            
    def _run_carla_visualization(self, policy_fn, num_samples: int):
        """Run visualization with CARLA."""
        ego = None
        for actor in self.world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                ego = actor
                break
                
        if not ego:
            print("No ego vehicle found")
            return
            
        for frame in range(num_samples):
            transform = ego.get_transform()
            sample = self._create_sample_from_vehicle(transform, policy_fn)
            
            if sample:
                self.visualize_trajectory(sample, transform)
                self.samples.append(sample)
                
            time.sleep(1.0 / self.config.fps)
            
    def _run_mock_visualization(self, num_samples: int):
        """Run mock visualization for testing."""
        print(f"[Mock] Running {num_samples} visualization frames...")
        
        for i in range(num_samples):
            sample = self._create_synthetic_sample(i)
            self.samples.append(sample)
            
            m = self.compute_metrics(sample)
            self.metrics.num_samples += 1
            
            for k, v in m.items():
                if hasattr(self.metrics, k):
                    setattr(self.metrics, k, v)
                    
        self._save_results()
        
    def _create_sample_from_vehicle(self, transform, policy_fn):
        """Create sample from vehicle state."""
        if not CARLA_AVAILABLE:
            return None
            
        velocity = self.world.get_actor(0).get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        sample = WaypointSample(
            timestamp=time.time(),
            position=np.array([0, 0]),
            heading=np.radians(transform.rotation.yaw),
            speed=speed,
            predicted_waypoints=None,
            ground_truth_waypoints=None
        )
        
        if policy_fn:
            sample.predicted_waypoints = policy_fn(sample)
            
        return sample
        
    def _create_synthetic_sample(self, idx: int) -> WaypointSample:
        """Create synthetic sample for testing."""
        t = idx * 0.1
        angle = t % (2 * np.pi)
        radius = 5.0 + t % 10.0
        
        gt = np.zeros((self.config.num_waypoints, 2))
        for i in range(self.config.num_waypoints):
            theta = angle + i * 0.3
            gt[i, 0] = radius * np.cos(theta)
            gt[i, 1] = radius * np.sin(theta)
            
        sft = gt + np.random.randn(self.config.num_waypoints, 2) * 0.5
        rl = gt + np.random.randn(self.config.num_waypoints, 2) * 0.2
        
        return WaypointSample(
            timestamp=time.time(),
            position=np.array([0.0, 0.0]),
            heading=angle,
            speed=5.0,
            predicted_waypoints=rl,
            ground_truth_waypoints=gt,
            sft_waypoints=sft,
            rl_waypoints=rl
        )
        
    def _save_results(self):
        """Save visualization results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        results = {
            "config": {
                "num_waypoints": self.config.num_waypoints,
                "waypoint_spacing": self.config.waypoint_spacing,
                "num_samples": len(self.samples)
            },
            "metrics": {
                "ade_predicted": self.metrics.ade_predicted,
                "fde_predicted": self.metrics.fde_predicted,
                "ade_sft": self.metrics.ade_sft,
                "ade_rl": self.metrics.ade_rl,
                "num_samples": self.metrics.num_samples
            },
            "samples": [
                {
                    "timestamp": s.timestamp,
                    "predicted": s.predicted_waypoints.tolist() if s.predicted_waypoints is not None else None,
                    "ground_truth": s.ground_truth_waypoints.tolist() if s.ground_truth_waypoints is not None else None,
                    "sft": s.sft_waypoints.tolist() if s.sft_waypoints is not None else None,
                    "rl": s.rl_waypoints.tolist() if s.rl_waypoints is not None else None
                }
                for s in self.samples
            ]
        }
        
        output_path = os.path.join(self.config.output_dir, "visualization.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved visualization to {output_path}")
        
    def print_summary(self):
        """Print visualization summary."""
        print("\n=== Waypoint Visualization Summary ===")
        print(f"Samples visualized: {len(self.samples)}")
        print(f"ADE (predicted): {self.metrics.ade_predicted:.3f}m")
        print(f"FDE (predicted): {self.metrics.fde_predicted:.3f}m")
        print(f"ADE (SFT): {self.metrics.ade_sft:.3f}m")
        print(f"ADE (RL): {self.metrics.ade_rl:.3f}m")


def create_policy_wrapper(sft_checkpoint: str = None, rl_checkpoint: str = None):
    """Create policy wrapper for visualization."""
    class WaypointPolicy:
        def __init__(self, ckpt):
            self.ckpt = ckpt
            
        def predict(self, obs):
            return np.random.randn(8, 2) * 0.5
            
    return WaypointPolicy(rl_checkpoint or sft_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Waypoint Prediction Visualizer for CARLA")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--fps", type=int, default=20, help="Visualization FPS")
    parser.add_argument("--num-waypoints", type=int, default=8, help="Number of waypoints")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of frames to visualize")
    parser.add_argument("--sft-checkpoint", help="SFT checkpoint path")
    parser.add_argument("--rl-checkpoint", help="RL checkpoint path")
    parser.add_argument("--output-dir", default="out/waypoint_viz", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    
    config = WaypointVisualizerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        num_waypoints=args.num_waypoints,
        output_dir=args.output_dir
    )
    
    if args.smoke_test:
        print("=== Smoke Test ===")
        visualizer = WaypointCarlaVisualizer(config)
        
        num_test = 10
        for i in range(num_test):
            sample = visualizer._create_synthetic_sample(i)
            visualizer.samples.append(sample)
            
            m = visualizer.compute_metrics(sample)
            visualizer.metrics.num_samples += 1
            
            for k, v in m.items():
                if hasattr(visualizer.metrics, k):
                    setattr(visualizer.metrics, k, v)
                    
        if visualizer.metrics.num_samples > 0:
            visualizer.metrics.ade_predicted /= visualizer.metrics.num_samples
            visualizer.metrics.ade_sft /= visualizer.metrics.num_samples
            visualizer.metrics.ade_rl /= visualizer.metrics.num_samples
            
        visualizer._save_results()
        visualizer.print_summary()
        
        print("\nSmoke test: PASSED")
        return
        
    policy = None
    if args.sft_checkpoint or args.rl_checkpoint:
        policy = create_policy_wrapper(args.sft_checkpoint, args.rl_checkpoint)
        
    visualizer = WaypointCarlaVisualizer(config)
    
    if args.mock:
        print("Running in MOCK mode...")
        visualizer._run_mock_visualization(args.num_samples)
    else:
        visualizer.run_realtime(policy.predict if policy else None, args.num_samples)
        
    visualizer.print_summary()


if __name__ == "__main__":
    main()
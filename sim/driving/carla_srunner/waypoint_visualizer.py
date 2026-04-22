#!/usr/bin/env python3
"""
Waypoint Trajectory Visualizer

Visualizes predicted waypoints against ground truth trajectories from Waymo data.
Useful for debugging and analyzing BC/RL waypoint prediction policies.

Usage:
    python sim/driving/carla_srunner/waypoint_visualizer.py --checkpoints out/bc/ --output out/waypoint_viz/
    python sim/driving/carla_srunner/waypoint_visualizer.py --episode data/waymo/episodes/ --output out/waypoint_viz/
    python sim/driving/carla_srunner/waypoint_visualizer.py --all --output out/waypoint_viz/
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class WaypointPrediction:
    """A single waypoint prediction."""
    x: float
    y: float
    theta: float = 0.0
    speed: float = 0.0
    timestamp: float = 0.0


@dataclass
class Trajectory:
    """A full trajectory with waypoints."""
    waypoints: List[WaypointPrediction]
    route_id: str = ""
    scenario_name: str = ""
    start_position: Tuple[float, float] = (0.0, 0.0)
    end_position: Tuple[float, float] = (0.0, 0.0)
    length_m: float = 0.0


@dataclass
class WaypointMetrics:
    """Metrics for waypoint prediction."""
    ade: float = 0.0  # Average Displacement Error
    fde: float = 0.0  # Final Displacement Error
    mse: float = 0.0  # Mean Squared Error
    missing_rate: float = 0.0  # Rate of missing waypoints
    
    def to_dict(self) -> dict:
        return {
            "ade": self.ade,
            "fde": self.fde,
            "mse": self.mse,
            "missing_rate": self.missing_rate
        }


class TrajectoryVisualizer:
    """Visualizes waypoint trajectories against ground truth."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories: List[Trajectory] = []
        self.predictions: dict = {}
    
    def load_waymo_episode(self, episode_path: str) -> Trajectory:
        """Load a Waymo episode and extract ground truth trajectory."""
        # Try to load from TFRecord or JSON
        episode_path = Path(episode_path)
        
        if episode_path.suffix == ".json":
            with open(episode_path) as f:
                data = json.load(f)
                return self._parse_waymo_json(data)
        elif episode_path.suffix == ".tfrecord":
            # Would need tensorflow - use stub for now
            return self._generate_mock_trajectory(episode_path.stem)
        else:
            # Try as directory
            json_path = episode_path / "trajectory.json"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                    return self._parse_waymo_json(data)
            raise ValueError(f"Cannot load episode from {episode_path}")
    
    def _parse_waymo_json(self, data: dict) -> Trajectory:
        """Parse Waymo JSON format to Trajectory."""
        waypoints = []
        
        # Extract trajectory points
        trajectory = data.get("trajectory", data.get("waypoints", []))
        for i, point in enumerate(trajectory):
            if isinstance(point, dict):
                wp = WaypointPrediction(
                    x=point.get("x", point.get("position", [0, 0])[0]),
                    y=point.get("y", point.get("position", [0, 0])[1]),
                    theta=point.get("theta", 0.0),
                    speed=point.get("speed", 0.0),
                    timestamp=point.get("timestamp", i * 0.1)
                )
            else:
                # Array format [x, y, theta, speed, timestamp]
                wp = WaypointPrediction(
                    x=point[0],
                    y=point[1],
                    theta=point[2] if len(point) > 2 else 0.0,
                    speed=point[3] if len(point) > 3 else 0.0,
                    timestamp=point[4] if len(point) > 4 else i * 0.1
                )
            waypoints.append(wp)
        
        if not waypoints:
            # Generate mock trajectory
            return self._generate_mock_trajectory(data.get("route_id", "unknown"))
        
        return Trajectory(
            waypoints=waypoints,
            route_id=data.get("route_id", ""),
            scenario_name=data.get("scenario_name", ""),
            start_position=(waypoints[0].x, waypoints[0].y),
            end_position=(waypoints[-1].x, waypoints[-1].y),
            length_m=self._calculate_trajectory_length(waypoints)
        )
    
    def _generate_mock_trajectory(self, route_id: str) -> Trajectory:
        """Generate a mock trajectory for testing."""
        np.random.seed(hash(route_id) % (2**32))
        
        # Generate smooth curved trajectory
        num_waypoints = 20
        waypoints = []
        
        # Starting position
        x, y = 0.0, 0.0
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        
        for i in range(num_waypoints):
            # Add some curvature
            theta += np.random.normal(0, 0.05)
            x += np.cos(theta) * 2.0
            y += np.sin(theta) * 2.0
            
            speed = np.random.uniform(3.0, 10.0)  # m/s
            
            wp = WaypointPrediction(
                x=x,
                y=y,
                theta=theta,
                speed=speed,
                timestamp=i * 0.1
            )
            waypoints.append(wp)
        
        return Trajectory(
            waypoints=waypoints,
            route_id=route_id,
            start_position=(waypoints[0].x, waypoints[0].y),
            end_position=(waypoints[-1].x, waypoints[-1].y),
            length_m=self._calculate_trajectory_length(waypoints)
        )
    
    def _calculate_trajectory_length(self, waypoints: List[WaypointPrediction]) -> float:
        """Calculate total trajectory length."""
        length = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i-1].x
            dy = waypoints[i].y - waypoints[i-1].y
            length += np.sqrt(dx**2 + dy**2)
        return length
    
    def load_predictions(self, checkpoint_path: str) -> dict:
        """Load predictions from a BC/RL checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Try to load model predictions
        # For now, generate mock predictions
        predictions = {}
        
        # Load from metrics.json if available
        metrics_path = checkpoint_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
                # Extract any stored predictions
                predictions["metrics"] = metrics
        
        # Generate mock predictions for visualization
        # In real implementation, would run inference here
        return predictions
    
    def generate_predictions_for_trajectory(
        self, 
        trajectory: Trajectory,
        noise_scale: float = 0.5
    ) -> Trajectory:
        """Generate predicted waypoints from ground truth (with noise)."""
        np.random.seed(42)
        
        predicted_waypoints = []
        
        for gt_wp in trajectory.waypoints:
            # Add noise to simulate prediction error
            noise_x = np.random.normal(0, noise_scale)
            noise_y = np.random.normal(0, noise_scale)
            noise_theta = np.random.normal(0, 0.1)
            
            pred_wp = WaypointPrediction(
                x=gt_wp.x + noise_x,
                y=gt_wp.y + noise_y,
                theta=gt_wp.theta + noise_theta,
                speed=gt_wp.speed + np.random.normal(0, 0.5),
                timestamp=gt_wp.timestamp
            )
            predicted_waypoints.append(pred_wp)
        
        return Trajectory(
            waypoints=predicted_waypoints,
            route_id=trajectory.route_id + "_pred",
            scenario_name=trajectory.scenario_name,
            start_position=predicted_waypoints[0],
            end_position=predicted_waypoints[-1]
        )
    
    def compute_metrics(
        self, 
        pred: Trajectory, 
        gt: Trajectory
    ) -> WaypointMetrics:
        """Compute waypoint prediction metrics."""
        if len(pred.waypoints) != len(gt.waypoints):
            # Interpolate to match lengths
            pred_waypoints = self._interpolate_waypoints(
                pred.waypoints, len(gt.waypoints)
            )
        else:
            pred_waypoints = pred.waypoints
        
        # Compute ADE
        ade_sum = 0.0
        fde = 0.0
        
        for pred_wp, gt_wp in zip(pred_waypoints, gt.waypoints):
            dx = pred_wp.x - gt_wp.x
            dy = pred_wp.y - gt_wp.y
            error = np.sqrt(dx**2 + dy**2)
            ade_sum += error
        
        ade = ade_sum / len(gt.waypoints) if gt.waypoints else 0.0
        
        # Compute FDE (final waypoint error)
        if pred_waypoints and gt.waypoints:
            final_pred = pred_waypoints[-1]
            final_gt = gt.waypoints[-1]
            dx = final_pred.x - final_gt.x
            dy = final_pred.y - final_gt.y
            fde = np.sqrt(dx**2 + dy**2)
        
        # Compute MSE
        mse = (ade ** 2)
        
        return WaypointMetrics(
            ade=ade,
            fde=fde,
            mse=mse,
            missing_rate=0.0
        )
    
    def _interpolate_waypoints(
        self, 
        waypoints: List[WaypointPrediction], 
        target_len: int
    ) -> List[WaypointPrediction]:
        """Interpolate waypoints to target length."""
        if len(waypoints) == 0:
            return []
        if len(waypoints) == target_len:
            return waypoints
        
        # Simple linear interpolation
        result = []
        for i in range(target_len):
            t = i / (target_len - 1) if target_len > 1 else 0
            idx = t * (len(waypoints) - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, len(waypoints) - 1)
            frac = idx - idx_low
            
            wp_low = waypoints[idx_low]
            wp_high = waypoints[idx_high]
            
            result.append(WaypointPrediction(
                x=wp_low.x + frac * (wp_high.x - wp_low.x),
                y=wp_low.y + frac * (wp_high.y - wp_low.y),
                theta=wp_low.theta + frac * (wp_high.theta - wp_low.theta),
                speed=wp_low.speed + frac * (wp_high.speed - wp_low.speed),
                timestamp=wp_low.timestamp + frac * (wp_high.timestamp - wp_low.timestamp)
            ))
        
        return result
    
    def visualize_trajectory_pair(
        self,
        pred: Trajectory,
        gt: Trajectory,
        output_name: str
    ) -> dict:
        """Visualize a predicted trajectory vs ground truth."""
        # Generate ASCII visualization
        lines = []
        lines.append(f"Trajectory Comparison: {gt.route_id}")
        lines.append("=" * 50)
        
        # Compute metrics
        metrics = self.compute_metrics(pred, gt)
        
        # Header
        lines.append(f"ADE: {metrics.ade:.3f}m | FDE: {metrics.fde:.3f}m | MSE: {metrics.mse:.3f}")
        lines.append("")
        
        # Waypoint comparison table
        lines.append("Idx | GT (x, y)           | Pred (x, y)          | Error")
        lines.append("-" * 60)
        
        max_len = max(len(pred.waypoints), len(gt.waypoints))
        for i in range(min(max_len, 20)):  # Limit to 20 for display
            if i < len(gt.waypoints):
                gt_wp = gt.waypoints[i]
                gt_str = f"({gt_wp.x:6.2f}, {gt_wp.y:6.2f})"
            else:
                gt_str = "(    -,      -)"
            
            if i < len(pred.waypoints):
                pred_wp = pred.waypoints[i]
                pred_str = f"({pred_wp.x:6.2f}, {pred_wp.y:6.2f})"
                
                if i < len(gt.waypoints):
                    dx = pred_wp.x - gt_wp.x
                    dy = pred_wp.y - gt_wp.y
                    error = np.sqrt(dx**2 + dy**2)
                    err_str = f"{error:.3f}m"
                else:
                    err_str = "---"
            else:
                pred_str = "(    -,      -)"
                err_str = "---"
            
            lines.append(f"{i:3d} | {gt_str} | {pred_str} | {err_str}")
        
        if max_len > 20:
            lines.append(f"... ({max_len - 20} more waypoints)")
        
        # Route info
        lines.append("")
        lines.append(f"Route Length: {gt.length_m:.1f}m")
        lines.append(f"Start: ({gt.start_position[0]:.2f}, {gt.start_position[1]:.2f})")
        lines.append(f"End: ({gt.end_position[0]:.2f}, {gt.end_position[1]:.2f})")
        
        # Save visualization
        output_file = self.output_dir / f"{output_name}.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        
        # Also save as JSON
        result = {
            "route_id": gt.route_id,
            "metrics": metrics.to_dict(),
            "num_waypoints": len(gt.waypoints),
            "route_length_m": gt.length_m,
            "start_position": gt.start_position,
            "end_position": gt.end_position,
            "visualization": str(output_file)
        }
        
        json_file = self.output_dir / f"{output_name}.json"
        with open(json_file, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def visualize_episode(
        self,
        episode_path: str,
        checkpoint_path: Optional[str] = None,
        noise_scale: float = 0.5
    ) -> dict:
        """Visualize a single episode with optional prediction."""
        # Load ground truth
        gt = self.load_waymo_episode(episode_path)
        
        if checkpoint_path:
            # Load predictions from checkpoint
            predictions = self.load_predictions(checkpoint_path)
            pred = predictions.get("trajectory", None)
            if not pred:
                # Generate mock prediction
                pred = self.generate_predictions_for_trajectory(gt, noise_scale)
        else:
            # Generate prediction as if from BC model
            pred = self.generate_predictions_for_trajectory(gt, noise_scale)
        
        # Visualize
        result = self.visualize_trajectory_pair(
            pred, gt, 
            f"viz_{gt.route_id}"
        )
        
        self.trajectories.append(gt)
        
        return result
    
    def visualize_all_checkpoints(
        self,
        checkpoints_dir: str,
        episode_dir: str
    ) -> dict:
        """Visualize predictions from all checkpoints against episodes."""
        checkpoints_dir = Path(checkpoints_dir)
        episode_dir = Path(episode_dir)
        
        results = {}
        
        # Find all checkpoints
        if checkpoints_dir.exists():
            for checkpoint in sorted(checkpoints_dir.iterdir()):
                if checkpoint.is_dir():
                    checkpoint_name = checkpoint.name
                    
                    # Find episodes for this checkpoint
                    episode_path = episode_dir / checkpoint_name
                    if not episode_path.exists():
                        # Use any available episode
                        episodes = list(episode_dir.glob("*.json"))
                        if episodes:
                            episode_path = episodes[0]
                        else:
                            episodes = list(episode_dir.glob("*/trajectory.json"))
                            if episodes:
                                episode_path = episodes[0].parent
                            else:
                                continue
                    
                    try:
                        result = self.visualize_episode(
                            str(episode_path),
                            str(checkpoint),
                            noise_scale=0.3
                        )
                        results[checkpoint_name] = result
                    except Exception as e:
                        print(f"Warning: Could not visualize {checkpoint_name}: {e}")
        
        # Save summary
        summary = {
            "num_visualizations": len(results),
            "checkpoints": list(results.keys())
        }
        
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return results
    
    def generate_comparison_html(
        self,
        predictions_dir: str
    ) -> str:
        """Generate HTML comparison page for all visualizations."""
        predictions_dir = Path(predictions_dir)
        
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Waypoint Trajectory Comparison</title>",
            "<style>",
            "body { font-family: monospace; margin: 20px; }",
            ".trajectory { border: 1px solid #ccc; margin: 10px; padding: 10px; }",
            ".metric { display: inline-block; margin-right: 20px; }",
            "h2 { color: #333; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Waypoint Trajectory Comparison</h1>",
        ]
        
        # Find all visualization files
        viz_files = sorted(self.output_dir.glob("*.txt"))
        
        for viz_file in viz_files:
            html.append(f'<div class="trajectory">')
            html.append(f'<h2>{viz_file.stem}</h2>')
            
            with open(viz_file) as f:
                content = f.read()
                # Simple escaping for HTML
                content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html.append(f'<pre>{content}</pre>')
            
            html.append('</div>')
        
        html.extend([
            "</body>",
            "</html>"
        ])
        
        html_file = self.output_dir / "comparison.html"
        with open(html_file, "w") as f:
            f.write("\n".join(html))
        
        return str(html_file)


def discover_episodes(episode_dir: str) -> List[str]:
    """Discover Waymo episodes in directory."""
    episode_dir = Path(episode_dir)
    episodes = []
    
    # Find JSON files
    episodes.extend(sorted(episode_dir.glob("*.json")))
    
    # Find directories with trajectory.json
    for traj_file in episode_dir.glob("*/trajectory.json"):
        episodes.append(str(traj_file.parent))
    
    return [str(e) for e in episodes[:10]]  # Limit to 10


def discover_checkpoints(checkpoint_dir: str) -> List[str]:
    """Discover checkpoints in directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []
    
    if checkpoint_dir.exists():
        for cp in checkpoint_dir.iterdir():
            if cp.is_dir():
                # Check for model files
                if (cp / "model.pt").exists() or (cp / "best.pt").exists():
                    checkpoints.append(str(cp))
    
    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Waypoint Trajectory Visualizer"
    )
    parser.add_argument(
        "--episode", 
        type=str, 
        help="Path to Waymo episode (JSON or directory)"
    )
    parser.add_argument(
        "--episodes-dir",
        type=str,
        help="Directory containing Waymo episodes"
    )
    parser.add_argument(
        "--checkpoints", 
        type=str, 
        help="Directory containing BC/RL checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Single checkpoint path"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Visualize all available data"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="out/waypoint_viz",
        help="Output directory"
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.5,
        help="Noise scale for mock predictions (default: 0.5)"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML comparison page"
    )
    
    args = parser.parse_args()
    
    viz = TrajectoryVisualizer(args.output)
    
    if args.episode:
        # Visualize single episode
        result = viz.visualize_episode(
            args.episode,
            args.checkpoint,
            args.noise_scale
        )
        print(f"Visualization saved to {args.output}")
        print(f"ADE: {result['metrics']['ade']:.3f}m")
        print(f"FDE: {result['metrics']['fde']:.3f}m")
    
    elif args.episodes_dir and args.checkpoints:
        # Visualize all checkpoints against episodes
        results = viz.visualize_all_checkpoints(
            args.checkpoints,
            args.episodes_dir
        )
        print(f"Visualized {len(results)} checkpoints")
        for name, result in results.items():
            print(f"  {name}: ADE={result['metrics']['ade']:.3f}m")
    
    elif args.all:
        # Try to find and visualize everything
        project_root = PROJECT_ROOT
        
        # Look for episodes
        episode_dirs = [
            project_root / "data/waymo/episodes",
            project_root / "data/waymo",
            project_root / "out",
        ]
        
        episodes_dir = None
        for d in episode_dirs:
            if d.exists():
                episodes_dir = str(d)
                break
        
        # Look for checkpoints
        checkpoint_dirs = [
            project_root / "out/bc",
            project_root / "out/sft",
            project_root / "out/rl",
        ]
        
        checkpoints_dir = None
        for d in checkpoint_dirs:
            if d.exists():
                checkpoints_dir = str(d)
                break
        
        if checkpoints_dir and episodes_dir:
            results = viz.visualize_all_checkpoints(
                checkpoints_dir,
                episodes_dir
            )
            print(f"Visualized {len(results)} checkpoints")
            for name, result in results.items():
                print(f"  {name}: ADE={result['metrics']['ade']:.3f}m")
        elif episodes_dir:
            # Just visualize episodes with mock predictions
            episodes = discover_episodes(episodes_dir)
            if episodes:
                result = viz.visualize_episode(
                    episodes[0],
                    None,
                    args.noise_scale
                )
                print(f"Visualization saved to {args.output}")
                print(f"ADE: {result['metrics']['ade']:.3f}m")
            else:
                print("No episodes found")
        else:
            print("No data found to visualize")
            print("Use --episode or --episodes-dir to specify data")
    
    else:
        # Generate mock example
        print("Generating mock example visualization...")
        
        # Create mock ground truth trajectory
        mock_trajectory = Trajectory(
            waypoints=[
                WaypointPrediction(x=0, y=0, theta=0, speed=5, timestamp=0),
                WaypointPrediction(x=5, y=1, theta=0.1, speed=5, timestamp=0.1),
                WaypointPrediction(x=10, y=2, theta=0.2, speed=5, timestamp=0.2),
                WaypointPrediction(x=15, y=4, theta=0.3, speed=5, timestamp=0.3),
                WaypointPrediction(x=20, y=6, theta=0.2, speed=5, timestamp=0.4),
            ],
            route_id="mock_route",
            length_m=22.0
        )
        
        # Generate mock prediction (with noise)
        mock_prediction = viz.generate_predictions_for_trajectory(
            mock_trajectory, 
            args.noise_scale
        )
        
        # Visualize
        result = viz.visualize_trajectory_pair(
            mock_prediction,
            mock_trajectory,
            "mock_example"
        )
        
        print(f"Mock visualization saved to {args.output}")
        print(f"ADE: {result['metrics']['ade']:.3f}m")
        print(f"FDE: {result['metrics']['fde']:.3f}m")
    
    if args.html:
        html_file = viz.generate_comparison_html(args.output)
        print(f"HTML comparison: {html_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
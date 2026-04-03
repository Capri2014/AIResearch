#!/usr/bin/env python3
"""
Synthetic Episode Generator for Driving Pipeline Testing

Generates synthetic episode JSON files following the episode.json schema:
- Multi-camera observations (front, left, right, rear)
- Expert waypoints in ego frame
- Ego state (speed, yaw_rate)
- Configurable episode length and route complexity

Usage:
    python generate_synthetic_episodes.py --output-dir data/waymo/episodes --num-episodes 10 --frames 100
"""

import argparse
import json
import os
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Optional


def generate_trajectory(
    num_frames: int,
    dt: float = 0.1,
    start_pos: tuple = (0.0, 0.0),
    start_heading: float = 0.0,
    curvature: float = 0.0,
    speed: float = 10.0
) -> List[Dict[str, Any]]:
    """Generate a smooth vehicle trajectory.
    
    Args:
        num_frames: Number of frames
        dt: Time step in seconds
        start_pos: Starting (x, y) position
        start_heading: Starting heading in radians
        curvature: Path curvature (0 = straight, positive = left turn)
        speed: Average speed in m/s
        
    Returns:
        List of frame states with position, heading, speed
    """
    frames = []
    x, y = start_pos
    heading = start_heading
    
    for i in range(num_frames):
        # Add slight curvature variation
        curv = curvature + 0.001 * math.sin(i * 0.1)
        
        # Update heading
        heading += curv * speed * dt
        
        # Update position
        x += speed * math.cos(heading) * dt
        y += speed * math.sin(heading) * dt
        
        # Add small noise
        x += random.gauss(0, 0.05)
        y += random.gauss(0, 0.05)
        
        # Speed variation
        speed_var = speed + random.gauss(0, 0.5)
        
        frames.append({
            "x": x,
            "y": y,
            "heading": heading,
            "speed": max(0, speed_var)
        })
    
    return frames


def compute_waypoints(
    trajectory: List[Dict[str, Any]],
    horizon_steps: int = 8,
    dt: float = 0.1,
    spacing: float = 5.0
) -> List[List[float]]:
    """Compute future waypoints in ego frame.
    
    Args:
        trajectory: List of trajectory states
        horizon_steps: Number of future waypoints
        dt: Time between waypoints
        spacing: Distance between waypoints in meters
        
    Returns:
        List of waypoints [[x, y], ...] in ego frame
    """
    if len(trajectory) < 2:
        return [[0, 0]] * horizon_steps
    
    # Current state
    current = trajectory[-1]
    current_x = current["x"]
    current_y = current["y"]
    current_heading = current["heading"]
    
    waypoints = []
    
    for i in range(1, horizon_steps + 1):
        # Time in future
        t = i * (spacing / 10.0)  # Assume 10 m/s avg speed
        
        # Find position at that future time (or extrapolate)
        idx = min(len(trajectory) - 1, int(t / dt))
        future = trajectory[idx]
        
        # Transform to ego frame
        dx = future["x"] - current_x
        dy = future["y"] - current_y
        
        # Rotation to ego frame
        wx = dx * math.cos(-current_heading) - dy * math.sin(-current_heading)
        wy = dx * math.sin(-current_heading) + dy * math.cos(-current_heading)
        
        waypoints.append([round(wx, 2), round(wy, 2)])
    
    return waypoints


def generate_episode(
    episode_id: str,
    num_frames: int,
    cameras: List[str],
    waypoint_spec: Dict[str, Any],
    route_difficulty: str = "medium"
) -> Dict[str, Any]:
    """Generate a synthetic episode.
    
    Args:
        episode_id: Unique episode identifier
        num_frames: Number of frames in episode
        cameras: List of camera names
        waypoint_spec: Waypoint specification
        route_difficulty: easy, medium, or hard
        
    Returns:
        Episode dict following episode.json schema
    """
    # Route parameters based on difficulty
    if route_difficulty == "easy":
        curvature = 0.0
        speed = 8.0
    elif route_difficulty == "hard":
        curvature = 0.03
        speed = 15.0
    else:  # medium
        curvature = 0.01
        speed = 12.0
    
    # Generate trajectory
    trajectory = generate_trajectory(
        num_frames=num_frames,
        dt=0.1,
        start_pos=(0.0, 0.0),
        start_heading=0.0,
        curvature=curvature,
        speed=speed
    )
    
    # Build frames
    frames = []
    horizon_steps = waypoint_spec.get("horizon_steps", 8)
    
    for t, state in enumerate(trajectory):
        frame = {
            "t": round(t * 0.1, 2),
            "observations": {
                "cameras": {},
                "state": {
                    "speed": round(state["speed"], 2),
                    "yaw_rate": round(curvature * state["speed"], 3),
                    "heading": round(state["heading"], 3)
                }
            }
        }
        
        # Add camera observations (placeholder paths)
        for cam in cameras:
            frame["observations"]["cameras"][cam] = {
                "image_path": f"episode_{episode_id}/frame_{t:04d}/{cam}.png",
                "intrinsics": [400, 0, 400, 300, 0, 400],
                "extrinsics": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
            }
        
        # Add expert waypoints (from future trajectory)
        if t < len(trajectory) - horizon_steps:
            future_trajectory = trajectory[t:]
            waypoints = compute_waypoints(
                future_trajectory,
                horizon_steps=horizon_steps,
                spacing=waypoint_spec.get("spacing", 5.0)
            )
            frame["expert"] = {
                "waypoints": waypoints
            }
        else:
            # No future waypoints available
            frame["expert"] = {
                "waypoints": [[0, 0]] * horizon_steps
            }
        
        frames.append(frame)
    
    # Build episode
    episode = {
        "episode_id": episode_id,
        "domain": "driving",
        "source": {
            "dataset": "synthetic",
            "split": "test",
            "route_difficulty": route_difficulty
        },
        "cameras": cameras,
        "waypoint_spec": waypoint_spec,
        "frames": frames
    }
    
    return episode


def validate_episode(episode: Dict[str, Any]) -> bool:
    """Validate episode against schema."""
    required = ["episode_id", "domain", "frames"]
    for key in required:
        if key not in episode:
            print(f"Missing required field: {key}")
            return False
    
    if not episode["frames"]:
        print("No frames in episode")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic episodes")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/waymo/episodes",
        help="Output directory for episode JSON files"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to generate"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames per episode"
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["front", "left", "right", "rear"],
        help="Camera names"
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=8,
        help="Number of future waypoints"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "mixed"],
        default="mixed",
        help="Route difficulty"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each episode against schema"
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Waypoint spec
    waypoint_spec = {
        "horizon_steps": args.horizon_steps,
        "dt": 0.1,
        "frame": "ego",
        "units": "m",
        "spacing": 5.0
    }
    
    # Generate episodes
    difficulties = ["easy", "medium", "hard"]
    num_generated = 0
    
    for i in range(args.num_episodes):
        # Determine difficulty
        if args.difficulty == "mixed":
            difficulty = random.choice(difficulties)
        else:
            difficulty = args.difficulty
        
        episode_id = f"syn_{args.seed}_{i:04d}_{difficulty}"
        
        episode = generate_episode(
            episode_id=episode_id,
            num_frames=args.frames,
            cameras=args.cameras,
            waypoint_spec=waypoint_spec,
            route_difficulty=difficulty
        )
        
        # Validate
        if args.validate and not validate_episode(episode):
            print(f"Episode {episode_id} failed validation, skipping")
            continue
        
        # Save
        output_path = output_dir / f"{episode_id}.json"
        with open(output_path, "w") as f:
            json.dump(episode, f, indent=2)
        
        num_generated += 1
        
        # Print stats
        num_cameras = len(episode["cameras"])
        num_frames = len(episode["frames"])
        waypoint_horizon = episode["waypoint_spec"]["horizon_steps"]
        
        print(f"Generated: {episode_id}")
        print(f"  Frames: {num_frames}, Cameras: {num_cameras}, Waypoint horizon: {waypoint_horizon}")
    
    print(f"\nGenerated {num_generated} episodes in {output_dir}")
    
    # Summary stats
    total_frames = args.num_episodes * args.frames
    print(f"Total frames: {total_frames}")
    print(f"Total size estimate: ~{total_frames * 0.5:.1f} KB")


if __name__ == "__main__":
    main()
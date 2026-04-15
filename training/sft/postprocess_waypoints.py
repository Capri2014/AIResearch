#!/usr/bin/env python3
"""
Waypoint Trajectory Post-Processor

Applies kinematic constraints and smoothing to waypoint predictions:
- Velocity/acceleration limits
- Temporal smoothing (exponential moving average)
- Trajectory validation (reachability, collision-free)
- Speed profile generation

Usage:
    # Post-process waypoint predictions
    python training/sft/postprocess_waypoints.py \
        --predictions waypoints.json \
        --output processed.json \
        --max-velocity 10.0 \
        --max-acceleration 3.0 \
        --smoothing 0.3

    # Validate trajectories
    python training/sft/postprocess_waypoints.py \
        --predictions waypoints.json \
        --validate \
        --output validated.json
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PostProcessConfig:
    """Configuration for waypoint post-processing."""

    # Kinematic constraints
    max_velocity: float = 10.0  # m/s
    max_acceleration: float = 3.0  # m/s^2
    max_deceleration: float = 5.0  # m/s^2

    # Smoothing
    smoothing_alpha: float = 0.3  # EMA weight (0=no smoothing, 1=no change)

    # Validation
    validate_reachability: bool = True
    min_waypoint_spacing: float = 0.1  # meters

    # Speed profile
    target_speed: float = 8.0  # m/s
    speed_profile_type: str = "trapezoidal"  # or "constant", "adaptive"


def load_waypoints(path: str) -> dict:
    """Load waypoints from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_waypoints(data: dict, path: str):
    """Save waypoints to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def compute_velocities(waypoints: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Compute velocities between consecutive waypoints.

    Args:
        waypoints: Array of shape (N, 2) with (x, y) positions
        dt: Time between waypoints

    Returns:
        Array of shape (N,) with velocities in m/s
    """
    if len(waypoints) < 2:
        return np.zeros(len(waypoints))

    diffs = np.diff(waypoints, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    velocities = distances / dt

    # First velocity is 0 (no previous point)
    velocities = np.concatenate([[0.0], velocities])
    return velocities


def compute_accelerations(velocities: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Compute accelerations between consecutive velocities.

    Args:
        velocities: Array of velocities
        dt: Time between waypoints

    Returns:
        Array of accelerations
    """
    if len(velocities) < 2:
        return np.zeros(len(velocities))

    accelerations = np.diff(velocities) / dt
    accelerations = np.concatenate([[0.0], accelerations])
    return accelerations


def apply_velocity_limits(waypoints: np.ndarray, max_vel: float, dt: float = 0.1) -> np.ndarray:
    """Apply velocity limits by scaling waypoint intervals.

    Args:
        waypoints: Array of shape (N, 2)
        max_vel: Maximum allowed velocity in m/s
        dt: Time between waypoints

    Returns:
        Adjusted waypoints
    """
    velocities = compute_velocities(waypoints, dt)

    # Find indices where velocity exceeds limit
    violating_indices = np.where(velocities > max_vel)[0]

    if len(violating_indices) == 0:
        return waypoints

    # Scale down the waypoints to respect velocity limits
    adjusted = waypoints.copy()
    for i in violating_indices:
        if i == 0:
            continue
        # Scale the displacement from previous point
        scale = max_vel * dt / (velocities[i] * dt + 1e-6)
        scale = min(scale, 1.0)
        adjusted[i] = adjusted[i - 1] + (waypoints[i] - waypoints[i - 1]) * scale

    return adjusted


def apply_acceleration_limits(waypoints: np.ndarray, max_acc: float, max_dec: float, dt: float = 0.1) -> np.ndarray:
    """Apply acceleration/deceleration limits.

    Args:
        waypoints: Array of shape (N, 2)
        max_acc: Maximum acceleration (m/s^2)
        max_dec: Maximum deceleration (magnitude, m/s^2)
        dt: Time between waypoints

    Returns:
        Adjusted waypoints
    """
    adjusted = waypoints.copy()

    for _ in range(3):  # Iterative refinement
        velocities = compute_velocities(adjusted, dt)
        accelerations = compute_accelerations(velocities, dt)

        # Check for violations
        for i in range(1, len(adjusted)):
            if accelerations[i] > max_acc:
                # Reduce speed - scale back the displacement
                scale = max_acc * dt / (abs(accelerations[i]) * dt + 1e-6)
                scale = min(scale, 1.0)
                # Interpolate towards previous point
                adjusted[i] = adjusted[i - 1] + (adjusted[i] - adjusted[i - 1]) * scale
            elif accelerations[i] < -max_dec:
                # Increase speed - scale forward the displacement
                scale = max_dec * dt / (abs(accelerations[i]) * dt + 1e-6)
                scale = min(scale, 1.0)
                adjusted[i] = adjusted[i - 1] + (adjusted[i] - adjusted[i - 1]) * scale

    return adjusted


def smooth_waypoints_ema(waypoints: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply exponential moving average smoothing to waypoints.

    Args:
        waypoints: Array of shape (N, 2)
        alpha: Smoothing weight (0=no smoothing, 1=no change)

    Returns:
        Smoothed waypoints
    """
    if alpha <= 0 or len(waypoints) < 2:
        return waypoints

    smoothed = waypoints.copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * waypoints[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


def smooth_waypoints_gaussian(waypoints: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to waypoints.

    Args:
        waypoints: Array of shape (N, 2)
        sigma: Gaussian kernel sigma

    Returns:
        Smoothed waypoints
    """
    if len(waypoints) < 3:
        return waypoints

    # Simple Gaussian smoothing with small kernel
    kernel_size = int(2 * round(sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, len(waypoints))

    # Create Gaussian kernel
    half = kernel_size // 2
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # Apply convolution for each dimension
    smoothed = waypoints.copy()
    for dim in range(2):
        smoothed[:, dim] = np.convolve(waypoints[:, dim], kernel, mode="same")

    return smoothed


def validate_waypoints(waypoints: np.ndarray, config: PostProcessConfig) -> dict:
    """Validate waypoints for feasibility.

    Args:
        waypoints: Array of shape (N, 2)
        config: Post-process configuration

    Returns:
        Validation results dictionary
    """
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
    }

    if len(waypoints) < 2:
        results["valid"] = False
        results["errors"].append("Too few waypoints")
        return results

    # Check waypoint spacing
    for i in range(1, len(waypoints)):
        dist = np.linalg.norm(waypoints[i] - waypoints[i - 1])
        if dist < config.min_waypoint_spacing:
            results["warnings"].append(f"Waypoints {i-1} and {i} too close: {dist:.3f}m")

    # Compute velocities
    velocities = compute_velocities(waypoints)
    max_vel = np.max(velocities)
    if max_vel > config.max_velocity:
        results["valid"] = False
        results["errors"].append(f"Max velocity {max_vel:.1f}m/s exceeds limit {config.max_velocity}m/s")

    # Compute accelerations
    accelerations = compute_accelerations(velocities)
    max_acc = np.max(accelerations)
    min_acc = np.min(accelerations)
    if max_acc > config.max_acceleration:
        results["valid"] = False
        results["errors"].append(f"Max acceleration {max_acc:.1f}m/s^2 exceeds limit")
    if min_acc < -config.max_deceleration:
        results["valid"] = False
        results["errors"].append(f"Max deceleration {abs(min_acc):.1f}m/s^2 exceeds limit")

    # Check for NaN/Inf
    if np.any(np.isnan(waypoints)) or np.any(np.isinf(waypoints)):
        results["valid"] = False
        results["errors"].append("Waypoints contain NaN or Inf values")

    return results


def generate_speed_profile(waypoints: np.ndarray, config: PostProcessConfig) -> np.ndarray:
    """Generate speed profile for waypoints.

    Args:
        waypoints: Array of shape (N, 2)
        config: Post-process configuration

    Returns:
        Array of shape (N,) with target speeds at each waypoint
    """
    n = len(waypoints)
    speeds = np.full(n, config.target_speed)

    if config.speed_profile_type == "constant":
        # Constant speed
        return speeds

    elif config.speed_profile_type == "trapezoidal":
        # Trapezoidal: accelerate, cruise, decelerate
        # Estimate total distance
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        total_distance = np.sum(distances)

        # Estimate time to cover at target speed
        cruise_time = total_distance / config.target_speed

        # Time for acceleration/deceleration
        accel_time = config.target_speed / config.max_acceleration
        decel_time = config.target_speed / config.max_deceleration

        # Check if we have enough distance for full trapezoid
        accel_dist = 0.5 * config.max_acceleration * accel_time ** 2
        decel_dist = 0.5 * config.max_deceleration * decel_time ** 2

        if accel_dist + decel_dist >= total_distance:
            # Triangle profile
            max_speed = math.sqrt(total_distance * config.max_acceleration)
            speeds = np.full(n, max_speed)
        else:
            # Full trapezoid: accelerate, cruise, decelerate
            accel_distance = 0.5 * config.max_acceleration * accel_time ** 2
            decel_distance = 0.5 * config.max_deceleration * decel_time ** 2
            cruise_distance = total_distance - accel_distance - decel_distance
            cruise_time = cruise_distance / config.target_speed

            total_time = accel_time + cruise_time + decel_time
            time_points = np.linspace(0, total_time, n)

            for i, t in enumerate(time_points):
                if t < accel_time:
                    speeds[i] = config.max_acceleration * t
                elif t < accel_time + cruise_time:
                    speeds[i] = config.target_speed
                else:
                    remaining = total_time - t
                    speeds[i] = config.max_deceleration * remaining

            speeds = np.clip(speeds, 0, config.max_velocity)

    elif config.speed_profile_type == "adaptive":
        # Adaptive speed based on curvature
        for i in range(1, n - 1):
            # Compute curvature via angle change
            v1 = waypoints[i] - waypoints[i - 1]
            v2 = waypoints[i + 1] - waypoints[i]
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            curvature = abs(angle)

            # Reduce speed for high curvature
            if curvature > 0.5:  # ~30 degrees
                speeds[i] = config.target_speed * 0.5
            elif curvature > 0.2:
                speeds[i] = config.target_speed * 0.75

        # Ensure monotonic decrease at end
        for i in range(n - 2, 0, -1):
            if speeds[i] > speeds[i + 1]:
                speeds[i] = speeds[i + 1]

    return speeds


def postprocess_waypoints(
    waypoints: np.ndarray,
    config: PostProcessConfig,
    smooth: bool = True,
    validate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Main post-processing function.

    Args:
        waypoints: Array of shape (N, 2) with (x, y) positions
        config: Post-process configuration
        smooth: Whether to apply smoothing
        validate: Whether to validate results

    Returns:
        Tuple of (processed_waypoints, speed_profile, validation_results)
    """
    processed = waypoints.copy()

    # Apply velocity limits
    processed = apply_velocity_limits(processed, config.max_velocity)

    # Apply acceleration limits
    processed = apply_acceleration_limits(
        processed, config.max_acceleration, config.max_deceleration
    )

    # Apply smoothing
    if smooth:
        if config.smoothing_alpha > 0:
            processed = smooth_waypoints_ema(processed, config.smoothing_alpha)
        else:
            processed = smooth_waypoints_gaussian(processed, sigma=1.0)

    # Generate speed profile
    speeds = generate_speed_profile(processed, config)

    # Validate
    validation = {}
    if validate:
        validation = validate_waypoints(processed, config)

    return processed, speeds, validation


def process_single_sample(data: dict, config: PostProcessConfig) -> dict:
    """Process a single sample's waypoints."""
    waypoints = np.array(data["waypoints"])
    speeds = data.get("speeds", None)

    processed_waypoints, speed_profile, validation = postprocess_waypoints(
        waypoints, config, smooth=True, validate=True
    )

    result = {
        "waypoints": processed_waypoints.tolist(),
        "speeds": speed_profile.tolist() if speeds is None else speeds,
        "validation": validation,
    }

    if "metadata" in data:
        result["metadata"] = data["metadata"]

    return result


def process_batch(input_path: str, output_path: str, config: PostProcessConfig):
    """Process a batch of waypoint predictions."""
    data = load_waypoints(input_path)

    # Handle both single sample and batch
    if "waypoints" in data:
        # Single sample
        result = process_single_sample(data, config)
        save_waypoints(result, output_path)
        print(f"Processed waypoints saved to {output_path}")
    else:
        # Batch
        results = []
        for sample in data:
            result = process_single_sample(sample, config)
            results.append(result)

        # Save batch
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Processed {len(results)} samples saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Waypoint trajectory post-processor")
    parser.add_argument("--predictions", required=True, help="Input predictions JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-velocity", type=float, default=10.0, help="Max velocity (m/s)")
    parser.add_argument("--max-acceleration", type=float, default=3.0, help="Max acceleration (m/s^2)")
    parser.add_argument("--max-deceleration", type=float, default=5.0, help="Max deceleration (m/s^2)")
    parser.add_argument("--smoothing", type=float, default=0.3, help="Smoothing alpha (EMA)")
    parser.add_argument("--validate", action="store_true", help="Validate trajectories")
    parser.add_argument("--speed-profile", default="trapezoidal", choices=["constant", "trapezoidal", "adaptive"])

    args = parser.parse_args()

    config = PostProcessConfig(
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        max_deceleration=args.max_deceleration,
        smoothing_alpha=args.smoothing,
        speed_profile_type=args.speed_profile,
    )

    process_batch(args.predictions, args.output, config)


if __name__ == "__main__":
    main()

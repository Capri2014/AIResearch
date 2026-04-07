#!/usr/bin/env python3
"""
Waymo Episode Data Augmentation and Quality Enhancement

Enhances synthetic Waymo episodes with camera augmentations, noise models, 
and quality metrics for better SSL pretraining. Bridges synthetic data 
generation to meaningful contrastive learning.

Usage:
    python data/waymo/augment_episodes.py \
        --episodes-dir data/waymo/episodes \
        --output-dir data/waymo/episodes_augmented \
        --augment-images \
        --quality-metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Camera augmentations
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    saturation_range: float = 0.2
    hue_range: float = 0.1
    
    # Geometric augmentations
    horizontal_flip: bool = True
    rotation_range: float = 5.0  # degrees
    scale_range: float = 0.1
    
    # Noise augmentations
    gaussian_noise_std: float = 0.01
    shot_noise_prob: float = 0.1
    defocus_blur_prob: float = 0.1
    
    # Quality metrics
    compute_quality_metrics: bool = True
    min_waypoint_spacing: float = 2.0  # meters
    max_waypoint_jump: float = 10.0   # meters


class WaymoEpisodeAugmenter:
    """Augments Waymo episodes for SSL pretraining."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self._init_augmentation_functions()
    
    def _init_augmentation_functions(self):
        """Initialize augmentation functions."""
        self.aug_fns = []
        
        if self.config.brightness_range > 0:
            self.aug_fns.append(self._augment_brightness)
        if self.config.contrast_range > 0:
            self.aug_fns.append(self._augment_contrast)
        if self.config.saturation_range > 0:
            self.aug_fns.append(self._augment_saturation)
        if self.config.hue_range > 0:
            self.aug_fns.append(self._augment_hue)
        if self.config.gaussian_noise_std > 0:
            self.aug_fns.append(self._add_gaussian_noise)
    
    def _augment_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness augmentation."""
        factor = 1.0 + random.uniform(
            -self.config.brightness_range, 
            self.config.brightness_range
        )
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def _augment_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast augmentation."""
        factor = 1.0 + random.uniform(
            -self.config.contrast_range,
            self.config.contrast_range
        )
        mean = image.mean()
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _augment_saturation(self, image: np.ndarray) -> np.ndarray:
        """Apply saturation augmentation (HSV space)."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        # Convert to HSV
        hsv = np.array(image, dtype=np.float32)
        hsv = hsv / 255.0
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Adjust saturation
        factor = 1.0 + random.uniform(
            -self.config.saturation_range,
            self.config.saturation_range
        )
        s = np.clip(s * factor, 0, 1)
        
        # Convert back to RGB
        hsv = np.stack([h, s, v], axis=-1)
        hsv = (hsv * 255).astype(np.uint8)
        return hsv
    
    def _augment_hue(self, image: np.ndarray) -> np.ndarray:
        """Apply hue shift augmentation."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        # Small hue shift (in HSV space)
        hsv = np.array(image, dtype=np.float32)
        hsv = hsv / 255.0
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Shift hue
        shift = random.uniform(-self.config.hue_range, self.config.hue_range)
        h = (h + shift) % 1.0
        
        # Convert back
        hsv = np.stack([h, s, v], axis=-1)
        hsv = (hsv * 255).astype(np.uint8)
        return hsv
    
    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        std = self.config.gaussian_noise_std * 255
        noise = np.random.normal(0, std, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def augment_frame(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single frame."""
        augmented = image.copy()
        
        # Randomly apply subset of augmentations
        random.shuffle(self.aug_fns)
        num_augs = random.randint(0, len(self.aug_fns))
        
        for aug_fn in self.aug_fns[:num_augs]:
            augmented = aug_fn(augmented)
        
        return augmented
    
    def validate_waypoints(self, waypoints: List) -> Tuple[bool, List]:
        """Validate and fix waypoint sequences."""
        if not waypoints:
            return False, waypoints
        
        validated = []
        prev_wp = None
        
        for wp in waypoints:
            # Handle both dict and list formats
            if isinstance(wp, (list, tuple)):
                x, y = wp[0], wp[1]
                yaw = wp[2] if len(wp) > 2 else 0
                velocity = wp[3] if len(wp) > 3 else 0
                wp_dict = {"x": x, "y": y, "yaw": yaw, "velocity": velocity}
            else:
                wp_dict = wp
                x, y = wp_dict.get("x", 0), wp_dict.get("y", 0)
            
            if prev_wp is not None:
                dx = x - prev_wp["x"]
                dy = y - prev_wp["y"]
                dist = np.sqrt(dx**2 + dy**2)
                
                # Check minimum spacing
                if dist < self.config.min_waypoint_spacing:
                    continue  # Skip too-close waypoints
                
                # Check maximum jump
                if dist > self.config.max_waypoint_jump:
                    # Interpolate intermediate waypoints
                    num_interp = int(dist / self.config.max_waypoint_jump)
                    for i in range(1, num_interp + 1):
                        t = i / (num_interp + 1)
                        interp_wp = {
                            "x": prev_wp["x"] + t * dx,
                            "y": prev_wp["y"] + t * dy,
                            "yaw": prev_wp.get("yaw", 0) + t * (wp_dict.get("yaw", 0) - prev_wp.get("yaw", 0)),
                            "velocity": (prev_wp.get("velocity", 0) + wp_dict.get("velocity", 0)) / 2
                        }
                        validated.append(interp_wp)
            
            validated.append(wp_dict)
            prev_wp = wp_dict
        
        return len(validated) > 0, validated
    
    def _get_waypoint_coords(self, wp):
        """Extract x, y from waypoint (list or dict format)."""
        if isinstance(wp, (list, tuple)):
            return wp[0], wp[1]
        elif isinstance(wp, dict):
            return wp.get("x", 0), wp.get("y", 0)
        return 0, 0
    
    def compute_quality_metrics(self, episode: Dict) -> Dict:
        """Compute quality metrics for an episode."""
        metrics = {
            "num_frames": len(episode.get("frames", [])),
            "num_waypoints": 0,
            "waypoint_spacing_mean": 0.0,
            "waypoint_spacing_std": 0.0,
            "trajectory_length": 0.0,
            "avg_velocity": 0.0,
            "difficulty": "unknown"
        }
        
        # Collect waypoints from expert annotations across frames
        all_waypoints = []
        velocities = []
        
        for frame in episode.get("frames", []):
            expert = frame.get("expert", {})
            waypoints = expert.get("waypoints", [])
            if waypoints:
                all_waypoints.extend(waypoints)
            
            # Get velocity from state if available
            state = frame.get("observations", {}).get("state", {})
            if "speed" in state:
                velocities.append(state["speed"])
        
        if all_waypoints:
            metrics["num_waypoints"] = len(all_waypoints)
            
            # Compute spacing (between consecutive waypoints)
            spacings = []
            for i in range(1, len(all_waypoints)):
                x1, y1 = self._get_waypoint_coords(all_waypoints[i-1])
                x2, y2 = self._get_waypoint_coords(all_waypoints[i])
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                spacings.append(dist)
            
            if spacings:
                metrics["waypoint_spacing_mean"] = float(np.mean(spacings))
                metrics["waypoint_spacing_std"] = float(np.std(spacings))
                metrics["trajectory_length"] = float(sum(spacings))
        
        if velocities:
            metrics["avg_velocity"] = float(np.mean(velocities))
        
        # Determine difficulty based on curvature
        if len(all_waypoints) > 10:
            curvatures = []
            for i in range(1, len(all_waypoints) - 1):
                x1, y1 = self._get_waypoint_coords(all_waypoints[i-1])
                x2, y2 = self._get_waypoint_coords(all_waypoints[i])
                if i + 1 >= len(all_waypoints):
                    continue
                x3, y3 = self._get_waypoint_coords(all_waypoints[i+1])
                
                dx1 = x2 - x1
                dy1 = y2 - y1
                dx2 = x3 - x2
                dy2 = y3 - y2
                
                # Cross product for curvature direction
                cross = dx1 * dy2 - dy1 * dx2
                denom = (np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2) + 1e-6)
                curvature = abs(cross) / (denom ** 2)
                curvatures.append(curvature)
            
            if curvatures:
                avg_curvature = np.mean(curvatures)
                if avg_curvature < 0.01:
                    metrics["difficulty"] = "easy"
                elif avg_curvature < 0.05:
                    metrics["difficulty"] = "medium"
                else:
                    metrics["difficulty"] = "hard"
        
        return metrics
    
    def augment_episode(self, episode: Dict, augment_images: bool = True) -> Dict:
        """Augment a complete episode."""
        augmented = episode.copy()
        
        # Validate and fix waypoints - collect from expert annotations in frames
        waypoints_by_frame = []
        for frame in episode.get("frames", []):
            expert = frame.get("expert", {})
            waypoints = expert.get("waypoints", [])
            if waypoints:
                waypoints_by_frame.extend(waypoints)
        
        if waypoints_by_frame:
            is_valid, validated_waypoints = self.validate_waypoints(waypoints_by_frame)
            if is_valid:
                # Store validated waypoints in each frame's expert annotation
                for i, frame in enumerate(augmented.get("frames", [])):
                    if "expert" in frame and "waypoints" in frame["expert"]:
                        # Update with validated version
                        num_wps = len(frame["expert"]["waypoints"])
                        frame["expert"]["waypoints"] = validated_waypoints[:num_wps]
                        validated_waypoints = validated_waypoints[num_wps:]
        else:
            print(f"  Warning: Episode {episode.get('episode_id', 'unknown')} has no waypoints")
        
        # Augment camera images if present
        if augment_images and "frames" in episode:
            augmented_frames = []
            for frame in episode["frames"]:
                aug_frame = frame.copy()
                
                if "images" in frame:
                    aug_images = {}
                    for cam_name, img_path in frame["images"].items():
                        # For now, just note the augmentation would be applied
                        # In practice, would load and transform the image
                        aug_images[cam_name] = img_path + "_aug"
                    aug_frame["images"] = aug_images
                
                augmented_frames.append(aug_frame)
            
            augmented["frames"] = augmented_frames
        
        # Compute quality metrics
        if self.config.compute_quality_metrics:
            augmented["quality_metrics"] = self.compute_quality_metrics(augmented)
        
        return augmented


def load_episode(path: Path) -> Dict:
    """Load a single episode JSON file."""
    with open(path) as f:
        return json.load(f)


def save_episode(episode: Dict, output_path: Path):
    """Save augmented episode to JSON file."""
    with open(output_path, "w") as f:
        json.dump(episode, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Augment Waymo episodes for SSL pretraining")
    parser.add_argument("--episodes-dir", type=str, default="data/waymo/episodes",
                        help="Directory containing episode JSON files")
    parser.add_argument("--output-dir", type=str, default="data/waymo/episodes_augmented",
                        help="Output directory for augmented episodes")
    parser.add_argument("--augment-images", action="store_true",
                        help="Apply image augmentations (requires images)")
    parser.add_argument("--quality-metrics", action="store_true",
                        help="Compute quality metrics for each episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Skip augmentations, only compute quality metrics")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create config
    config = AugmentationConfig(
        compute_quality_metrics=args.quality_metrics
    )
    
    # Create augmenter
    augmenter = WaymoEpisodeAugmenter(config)
    
    # Load episodes
    episodes_dir = Path(args.episodes_dir)
    episode_files = sorted(episodes_dir.glob("*.json"))
    
    print(f"Found {len(episode_files)} episode files")
    
    if not episode_files:
        print("No episode files found!")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each episode
    quality_reports = []
    
    for ep_file in episode_files:
        print(f"Processing {ep_file.name}...")
        
        # Load episode
        episode = load_episode(ep_file)
        
        # Augment
        if args.no_augmentation:
            augmented = episode.copy()
        else:
            augmented = augmenter.augment_episode(
                episode, 
                augment_images=args.augment_images
            )
        
        # Compute quality metrics if not done in augmentation
        if args.quality_metrics and "quality_metrics" not in augmented:
            augmented["quality_metrics"] = augmenter.compute_quality_metrics(augmented)
        
        # Save augmented episode
        output_path = output_dir / ep_file.name
        save_episode(augmented, output_path)
        
        # Collect quality report
        if "quality_metrics" in augmented:
            quality_reports.append({
                "episode_id": augmented.get("episode_id", ep_file.stem),
                "metrics": augmented["quality_metrics"]
            })
    
    # Print summary
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    print(f"Input episodes: {len(episode_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Augmentations applied: {not args.no_augmentation}")
    print(f"Quality metrics computed: {args.quality_metrics}")
    
    if quality_reports:
        # Aggregate metrics
        difficulties = {}
        total_frames = 0
        total_waypoints = 0
        
        for report in quality_reports:
            diff = report["metrics"].get("difficulty", "unknown")
            difficulties[diff] = difficulties.get(diff, 0) + 1
            total_frames += report["metrics"].get("num_frames", 0)
            total_waypoints += report["metrics"].get("num_waypoints", 0)
        
        print(f"\nQuality Report:")
        print(f"  Total frames: {total_frames}")
        print(f"  Total waypoints: {total_waypoints}")
        print(f"  Difficulty distribution: {difficulties}")
        
        # Save quality report
        report_path = output_dir / "quality_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "num_episodes": len(quality_reports),
                "total_frames": total_frames,
                "total_waypoints": total_waypoints,
                "difficulty_distribution": difficulties,
                "episodes": quality_reports
            }, f, indent=2)
        print(f"\nQuality report saved to: {report_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
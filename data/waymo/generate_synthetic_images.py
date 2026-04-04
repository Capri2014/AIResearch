#!/usr/bin/env python3
"""
Generate synthetic images for Waymo-style episodes.

Creates placeholder images that match the episode JSON metadata structure.
This enables the contrastive SSL pipeline to work with synthetic data.

Usage:
    python generate_synthetic_images.py --episodes-dir data/waymo/episodes \
        --output-dir data/waymo/images --num-processes 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional
import numpy as np
from PIL import Image


def generate_camera_image(
    width: int = 800,
    height: int = 600,
    seed: int = 42,
    camera_type: str = "front",
    frame_idx: int = 0,
) -> np.ndarray:
    """Generate a synthetic camera image with procedural patterns.
    
    Creates varied images that differ by camera type, frame index, and seed
    to simulate temporal and viewpoint diversity.
    """
    np.random.seed(seed + frame_idx * 100 + hash(camera_type) % 1000)
    
    # Base color varies by camera
    base_colors = {
        "front": (60, 80, 100),    # Blue-ish road view
        "left": (80, 70, 60),       # Brown-ish left view
        "right": (70, 80, 70),      # Green-ish right view
        "rear": (90, 60, 60),       # Red-ish rear view
    }
    base = np.array(base_colors.get(camera_type, (70, 70, 70)), dtype=np.uint8)
    
    # Create base image with gradient
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky gradient (top portion)
    sky_height = int(height * 0.4)
    for y in range(sky_height):
        factor = y / sky_height
        img[y, :] = [
            int(base[0] + 40 * factor),
            int(base[1] + 30 * factor),
            int(base[2] + 50 * factor),
        ]
    
    # Ground gradient (bottom portion)
    for y in range(sky_height, height):
        factor = (y - sky_height) / (height - sky_height)
        img[y, :] = [
            int(base[0] - 20 * (1 - factor)),
            int(base[1] - 10 * (1 - factor)),
            int(base[2] - 30 * (1 - factor)),
        ]
    
    # Add road-like features (horizontal lines)
    road_y = int(height * 0.6)
    for offset in range(-3, 4):
        y = road_y + offset * 20
        if 0 <= y < height:
            # Road marking
            img[y, :] = [200, 200, 200]
    
    # Add some noise for texture
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add frame-specific variation (simulating temporal progression)
    brightness = 0.9 + 0.2 * np.sin(frame_idx * 0.1)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    return img


def process_episode(episode_path: Path, output_dir: Path) -> Dict:
    """Generate images for a single episode."""
    episode_id = episode_path.stem
    
    # Load episode JSON
    with open(episode_path, "r") as f:
        episode_data = json.load(f)
    
    # Create episode image directory
    cameras = episode_data.get("cameras", ["front", "left", "right", "rear"])
    num_frames = len(episode_data.get("frames", []))
    
    results = {
        "episode_id": episode_id,
        "cameras": cameras,
        "num_frames": num_frames,
        "images_created": 0,
        "images_failed": 0,
    }
    
    # Generate images for each frame and camera
    for frame_idx, frame in enumerate(episode_data.get("frames", [])):
        frame_dir = output_dir / f"episode_{episode_id}" / f"frame_{frame_idx:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        for cam in cameras:
            img_path = frame_dir / f"{cam}.png"
            
            try:
                # Generate image
                img = generate_camera_image(
                    width=800,
                    height=600,
                    seed=hash(episode_id) % 10000,
                    camera_type=cam,
                    frame_idx=frame_idx,
                )
                
                # Save
                Image.fromarray(img).save(img_path)
                results["images_created"] += 1
                
            except Exception as e:
                results["images_failed"] += 1
                print(f"  Failed to generate {img_path}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images for Waymo-style episodes"
    )
    parser.add_argument(
        "--episodes-dir",
        type=str,
        default="data/waymo/episodes",
        help="Directory containing episode JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/waymo/images",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of episodes to process",
    )
    
    args = parser.parse_args()
    
    episodes_dir = Path(args.episodes_dir)
    output_dir = Path(args.output_dir)
    
    # Find episode JSON files
    episode_files = sorted(episodes_dir.glob("syn_*.json"))
    
    if args.limit:
        episode_files = episode_files[:args.limit]
    
    print(f"Found {len(episode_files)} episode files")
    print(f"Output directory: {output_dir}")
    
    # Process episodes in parallel
    total_created = 0
    total_failed = 0
    
    if args.num_processes > 1:
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            futures = {
                executor.submit(process_episode, ep, output_dir): ep
                for ep in episode_files
            }
            
            for future in as_completed(futures):
                ep = futures[future]
                try:
                    result = future.result()
                    total_created += result["images_created"]
                    total_failed += result["images_failed"]
                    print(f"Processed {result['episode_id']}: "
                          f"{result['images_created']} images, "
                          f"{result['images_failed']} failed")
                except Exception as e:
                    print(f"Failed to process {ep}: {e}")
    else:
        # Sequential processing
        for ep in episode_files:
            result = process_episode(ep, output_dir)
            total_created += result["images_created"]
            total_failed += result["images_failed"]
            print(f"Processed {result['episode_id']}: "
                  f"{result['images_created']} images, "
                  f"{result['images_failed']} failed")
    
    print(f"\n=== Summary ===")
    print(f"Total images created: {total_created}")
    print(f"Total images failed: {total_failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
"""
Data Augmentation for Waymo Episodes.

Applies geometric and photometric transforms to Waymo episode data
for robust SSL pretraining:
- Random crop, flip, rotation
- Color jitter, gaussian blur
- Temporal augmentation (speed variation)
- Weather/time-of-day simulation

Usage:
    python3 training/pretrain/augment_episodes.py \
        --input data/waymo/episodes \
        --output data/waymo/augmented \
        --methods crop,flip,color_jitter \
        --num-workers 4
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisodeAugmentor:
    """Applies augmentations to Waymo episode data."""
    
    def __init__(self, methods: list[str], seed: int = 42):
        self.methods = methods
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)
        
    def apply(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Apply augmentations to a single episode."""
        augmented = episode.copy()
        
        for method in self.methods:
            if method == "crop":
                augmented = self._random_crop(augmented)
            elif method == "flip":
                augmented = self._random_flip(augmented)
            elif method == "rotate":
                augmented = self._random_rotate(augmented)
            elif method == "color_jitter":
                augmented = self._color_jitter(augmented)
            elif method == "gaussian_blur":
                augmented = self._gaussian_blur(augmented)
            elif method == "speed_variation":
                augmented = self._speed_variation(augmented)
            elif method == "weather_sim":
                augmented = self._weather_simulation(augmented)
                
        return augmented
    
    def _random_crop(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Apply random crop to images using torch grid_sample."""
        if "images" not in episode:
            return episode
            
        images = episode["images"]
        
        # Process each image
        augmented = []
        for img in images:
            h, w = img.shape[:2]
            
            # Random crop ratio
            crop_ratio = self.rng.uniform(0.7, 1.0)
            new_h = int(h * crop_ratio)
            new_w = int(w * crop_ratio)
            
            # Random crop position
            top = self.rng.integers(0, max(1, h - new_h + 1))
            left = self.rng.integers(0, max(1, w - new_w + 1))
            
            # Crop
            cropped = img[top:top+new_h, left:left+new_w]
            
            # Resize back to original using torch
            img_tensor = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            resized = F.interpolate(img_tensor, size=(h, w), mode='bilinear', align_corners=False)
            resized = (resized.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            augmented.append(resized)
            
        episode = episode.copy()
        episode["images"] = augmented
        episode["augmentation"] = episode.get("augmentation", []) + ["crop"]
        return episode
    
    def _random_flip(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Apply random horizontal flip."""
        if "images" not in episode:
            return episode
            
        if self.rng.random() > 0.5:
            images = [np.fliplr(img) for img in episode["images"]]
            episode = episode.copy()
            episode["images"] = images
            episode["augmentation"] = episode.get("augmentation", []) + ["flip"]
            
            # Flip waypoints too
            if "waypoints" in episode:
                waypoints = episode["waypoints"].copy()
                waypoints[:, 1] = -waypoints[:, 1]  # Flip Y coordinate
                episode["waypoints"] = waypoints
                
        return episode
    
    def _random_rotate(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Apply random rotation using torch."""
        if "images" not in episode:
            return episode
            
        angle = self.rng.uniform(-15, 15)
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        
        # Rotation matrix
        theta = torch.tensor([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        images = []
        for img in episode["images"]:
            h, w = img.shape[:2]
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Create grid
            grid = F.affine_grid(theta, (1, 3, h, w), align_corners=False)
            
            # Apply rotation
            rotated = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            rotated = (rotated.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(rotated)
            
        episode = episode.copy()
        episode["images"] = images
        episode["augmentation"] = episode.get("augmentation", []) + [f"rotate_{angle:.1f}"]
        return episode
    
    def _color_jitter(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Apply color jitter."""
        if "images" not in episode:
            return episode
            
        brightness = self.rng.uniform(0.7, 1.3)
        contrast = self.rng.uniform(0.7, 1.3)
        saturation = self.rng.uniform(0.7, 1.3)
        hue = self.rng.uniform(-0.1, 0.1)
        
        images = []
        for img in episode["images"]:
            # Convert to float
            img_float = img.astype(np.float32) / 255.0
            
            # Brightness
            img_float = img_float * brightness
            
            # Contrast
            mean = img_float.mean()
            img_float = (img_float - mean) * contrast + mean
            
            # Saturation (affect only chromatic channels)
            if len(img_float.shape) == 3 and img_float.shape[2] >= 3:
                gray = img_float[:, :, :3].mean(axis=2, keepdims=True)
                img_float[:, :, :3] = gray + (img_float[:, :, :3] - gray) * saturation
                
                # Hue shift
                if hue != 0:
                    # Simple hue shift by cycling channels
                    shift = int(hue * 3)
                    if shift != 0:
                        img_float[:, :, :3] = np.roll(img_float[:, :, :3], shift, axis=2)
            
            # Clip and convert back
            img_float = np.clip(img_float, 0, 1)
            images.append((img_float * 255).astype(np.uint8))
            
        episode = episode.copy()
        episode["images"] = images
        episode["augmentation"] = episode.get("augmentation", []) + ["color_jitter"]
        return episode
    
    def _gaussian_blur(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Apply Gaussian blur using torch."""
        if "images" not in episode:
            return episode
            
        sigma = self.rng.uniform(0.5, 2.0)
        kernel_size = int(sigma * 6)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        images = []
        for img in episode["images"]:
            # Convert to tensor and apply blur
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Create Gaussian kernel
            channel = img_tensor.shape[1]
            kernel = torch.zeros(channel, channel, kernel_size, kernel_size)
            
            mid = kernel_size // 2
            for c in range(channel):
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        kernel[c, c, i, j] = torch.exp(torch.tensor(-((i - mid)**2 + (j - mid)**2) / (2 * sigma**2)))
                        
            kernel = kernel / kernel.sum(dim=(2, 3), keepdim=True)
            
            # Apply convolution
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dim
            blurred = torch.nn.functional.conv2d(
                img_tensor.view(-1, 1, img_tensor.shape[-2], img_tensor.shape[-1]),
                kernel[0:1], padding=mid
            )
            blurred = blurred.view(channel, -1, -1)
            
            blurred = (blurred.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(blurred)
            
        episode = episode.copy()
        episode["images"] = images
        episode["augmentation"] = episode.get("augmentation", []) + [f"blur_{sigma:.1f}"]
        return episode
    
    def _speed_variation(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Apply temporal speed variation."""
        if "waypoints" not in episode:
            return episode
            
        # Vary the temporal spacing of waypoints
        speed_factor = self.rng.uniform(0.8, 1.2)
        
        # Adjust timestamps
        if "timestamps" in episode:
            timestamps = np.array(episode["timestamps"])
            timestamps = timestamps * speed_factor
            episode = episode.copy()
            episode["timestamps"] = timestamps.tolist()
            
        # Adjust velocities
        if "velocities" in episode:
            velocities = np.array(episode["velocities"])
            velocities = velocities / speed_factor
            episode = episode.copy()
            episode["velocities"] = velocities.tolist()
            
        episode["augmentation"] = episode.get("augmentation", []) + [f"speed_{speed_factor:.2f}"]
        return episode
    
    def _weather_simulation(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Simulate different weather conditions."""
        if "images" not in episode:
            return episode
            
        weather_types = ["clear", "rain", "fog", "night"]
        weather = self.rng.choice(weather_types)
        
        images = []
        for img in episode["images"]:
            img_float = img.astype(np.float32) / 255.0
            
            if weather == "rain":
                # Add noise and slight blur
                noise = self.rng.normal(0, 0.05, img_float.shape).astype(np.float32)
                blurred = self._simple_blur(img_float + noise)
                img_float = blurred
                
            elif weather == "fog":
                # Add white overlay
                fog_amount = self.rng.uniform(0.2, 0.4)
                img_float = img_float * (1 - fog_amount) + fog_amount
                
            elif weather == "night":
                # Darken and add blue tint
                darken = self.rng.uniform(0.3, 0.6)
                img_float = img_float * darken
                if img_float.shape[2] >= 3:
                    img_float[:, :, 0] *= 0.8  # Reduce red
                    img_float[:, :, 2] *= 1.2  # Boost blue
                    
            # Clip and convert
            img_float = np.clip(img_float, 0, 1)
            images.append((img_float * 255).astype(np.uint8))
            
        episode = episode.copy()
        episode["images"] = images
        episode["augmentation"] = episode.get("augmentation", []) + [f"weather_{weather}"]
        episode["weather"] = weather
        return episode
    
    def _simple_blur(self, img: np.ndarray) -> np.ndarray:
        """Simple box blur."""
        kernel_size = 3
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge')
        blurred = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                blurred[i, j] = padded[i:i+kernel_size, j:j+kernel_size].mean(axis=(0, 1))
        return blurred


def augment_directory(
    input_dir: Path,
    output_dir: Path,
    methods: list[str],
    num_workers: int = 4,
    seed: int = 42
) -> dict[str, Any]:
    """Augment all episodes in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all episode files
    episode_files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.npz"))
    logger.info(f"Found {len(episode_files)} episodes in {input_dir}")
    
    augmentor = EpisodeAugmentor(methods, seed=seed)
    
    stats = {"augmented": 0, "failed": 0, "methods": methods}
    
    for ep_file in episode_files:
        try:
            if ep_file.suffix == ".json":
                with open(ep_file) as f:
                    episode = json.load(f)
            else:
                episode = dict(np.load(ep_file))
                
            # Apply augmentations
            augmented = augmentor.apply(episode)
            
            # Save
            out_file = output_dir / ep_file.name
            if ep_file.suffix == ".json":
                with open(out_file, "w") as f:
                    json.dump(augmented, f)
            else:
                np.savez(out_file, **augmented)
                
            stats["augmented"] += 1
            
        except Exception as e:
            logger.error(f"Failed to augment {ep_file}: {e}")
            stats["failed"] += 1
            
    logger.info(f"Augmentation complete: {stats['augmented']} success, {stats['failed']} failed")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Augment Waymo episodes")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--methods", type=str, default="crop,flip,color_jitter",
                       help="Comma-separated list of augmentation methods")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    methods = args.methods.split(",")
    
    stats = augment_directory(
        args.input, args.output, methods,
        args.num_workers, args.seed
    )
    
    print(f"\n=== Augmentation Summary ===")
    print(f"Methods: {stats['methods']}")
    print(f"Augmented: {stats['augmented']}")
    print(f"Failed: {stats['failed']}")
    

if __name__ == "__main__":
    main()
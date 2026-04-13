#!/usr/bin/env python3
"""
Dataset Splitter for Waymo Episodes

Provides train/val/test splitting with stratification by:
- Route similarity (geographic clustering)
- Scenario type (lane following, turning, intersection)
- Difficulty level (speed, curvature)

Usage:
    # Simple random split (80/10/10)
    python training/episodes/dataset_splitter.py --episodes-glob "data/waymo/episodes/*.json" --output-dir out/splits

    # Stratified split by route
    python training/episodes/dataset_splitter.py --episodes-glob "data/waymo/episodes/*.json" --stratify route --output-dir out/splits

    # Custom ratios
    python training/episodes/dataset_splitter.py --episodes-glob "data/waymo/episodes/*.json" --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --output-dir out/splits

    # List existing splits
    python training/episodes/dataset_splitter.py --list-splits out/splits
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np


# Stratification keys
STRATIFY_ROUTE = "route"
STRATIFY_SCENARIO = "scenario"
STRATIFY_DIFFICULTY = "difficulty"
STRATIFY_NONE = "none"


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""

    # Input
    episodes_glob: str = "data/waymo/episodes/*.json"

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Stratification
    stratify: str = STRATIFY_NONE  # route | scenario | difficulty | none
    min_samples_per_stratum: int = 2  # Minimum samples per stratum for val/test

    # Output
    output_dir: str = "out/splits"
    experiment_name: Optional[str] = None

    # Options
    seed: int = 42
    dry_run: bool = False


@dataclass
class EpisodeInfo:
    """Basic info extracted from an episode for stratification."""

    episode_id: str
    route_id: Optional[str] = None
    scenario_type: Optional[str] = None
    max_speed: float = 0.0
    avg_curvature: float = 0.0
    num_waypoints: int = 0
    file_path: str = ""


@dataclass
class SplitResult:
    """Result from a dataset split."""

    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)

    train_count: int = 0
    val_count: int = 0
    test_count: int = 0

    stratify_key: str = ""
    seed: int = 42

    # Metadata
    config: dict = field(default_factory=dict)
    stratification_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)


def extract_episode_info(episode_path: str) -> EpisodeInfo:
    """
    Extract basic info from an episode JSON file for stratification.

    Args:
        episode_path: Path to episode JSON file

    Returns:
        EpisodeInfo with extracted metadata
    """
    episode_id = Path(episode_path).stem

    try:
        with open(episode_path, 'r') as f:
            episode = json.load(f)

        # Extract route_id if available
        route_id = episode.get("route_id") or episode.get("route", {}).get("id")

        # Extract scenario type
        scenario_type = episode.get("scenario_type") or episode.get("type", "unknown")

        # Compute max speed from waypoints if available
        waypoints = episode.get("waypoints", [])
        max_speed = 0.0
        avg_curvature = 0.0
        num_waypoints = len(waypoints)

        if waypoints:
            speeds = []
            for wp in waypoints:
                if isinstance(wp, dict):
                    speed = wp.get("speed", 0.0)
                elif isinstance(wp, (list, tuple)) and len(wp) >= 3:
                    speed = wp[2] if wp[2] is not None else 0.0
                else:
                    speed = 0.0
                speeds.append(speed)
            max_speed = max(speeds) if speeds else 0.0

            # Estimate curvature from waypoint direction changes
            if len(waypoints) >= 3:
                directions = []
                for i in range(1, len(waypoints)):
                    if isinstance(waypoints[i], dict) and isinstance(waypoints[i-1], dict):
                        dx = waypoints[i].get("x", 0) - waypoints[i-1].get("x", 0)
                        dy = waypoints[i].get("y", 0) - waypoints[i-1].get("y", 0)
                    elif isinstance(waypoints[i], (list, tuple)) and isinstance(waypoints[i-1], (list, tuple)):
                        dx = waypoints[i][0] - waypoints[i-1][0]
                        dy = waypoints[i][1] - waypoints[i-1][1]
                    else:
                        continue
                    if dx != 0 or dy != 0:
                        directions.append(np.arctan2(dy, dx))

                # Compute angular changes as proxy for curvature
                if len(directions) >= 2:
                    angle_changes = np.abs(np.diff(directions))
                    avg_curvature = float(np.mean(angle_changes))

        return EpisodeInfo(
            episode_id=episode_id,
            route_id=route_id,
            scenario_type=scenario_type,
            max_speed=max_speed,
            avg_curvature=avg_curvature,
            num_waypoints=num_waypoints,
            file_path=episode_path,
        )

    except Exception as e:
        # Return minimal info for failed episodes
        return EpisodeInfo(
            episode_id=episode_id,
            file_path=episode_path,
        )


def compute_stratification_key(info: EpisodeInfo, stratify_by: str) -> str:
    """
    Compute stratification key for an episode based on the stratification strategy.

    Args:
        info: Episode info
        stratify_by: Strategy (route | scenario | difficulty)

    Returns:
        Stratification key string
    """
    if stratify_by == STRATIFY_ROUTE:
        return info.route_id or info.episode_id.split("_")[0] or "unknown"
    elif stratify_by == STRATIFY_SCENARIO:
        return info.scenario_type or "unknown"
    elif stratify_by == STRATIFY_DIFFICULTY:
        # Bin difficulty by speed and curvature
        speed_bin = "low" if info.max_speed < 5.0 else "medium" if info.max_speed < 10.0 else "high"
        curve_bin = "straight" if info.avg_curvature < 0.05 else "curvy"
        return f"{speed_bin}_{curve_bin}"
    else:
        return "none"


def stratified_split(
    files: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify_keys: List[str],
    seed: int,
    min_per_stratum: int = 2,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Perform stratified train/val/test split.

    Args:
        files: List of episode file paths
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify_keys: Stratification key for each file
        seed: Random seed
        min_per_stratum: Minimum samples per stratum

    Returns:
        (train_files, val_files, test_files)
    """
    np.random.seed(seed)

    # Group files by stratum
    strata: Dict[str, List[int]] = {}
    for i, key in enumerate(stratify_keys):
        if key not in strata:
            strata[key] = []
        strata[key].append(i)

    train_files = []
    val_files = []
    test_files = []

    for stratum, indices in strata.items():
        np.random.shuffle(indices)
        n = len(indices)

        # Compute split counts for this stratum
        n_test = max(min_per_stratum, int(n * test_ratio))
        n_val = max(min_per_stratum, int(n * val_ratio))
        n_train = n - n_test - n_val

        # Ensure minimum counts
        if n_train < min_per_stratum:
            n_train = n - n_test
            n_val = 0

        train_files.extend([files[i] for i in indices[:n_train]])
        val_files.extend([files[i] for i in indices[n_train:n_train + n_val]])
        test_files.extend([files[i] for i in indices[n_train + n_val:]])

    # Shuffle each split
    np.random.shuffle(train_files)
    np.random.shuffle(val_files)
    np.random.shuffle(test_files)

    return train_files, val_files, test_files


def random_split(
    files: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Perform random train/val/test split.

    Args:
        files: List of episode file paths
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        (train_files, val_files, test_files)
    """
    np.random.seed(seed)
    indices = np.arange(len(files))
    np.random.shuffle(indices)

    n = len(files)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    test_files = [files[i] for i in test_idx]

    return train_files, val_files, test_files


def load_episode_files(glob_pattern: str) -> List[str]:
    """
    Load episode files matching a glob pattern.

    Args:
        glob_pattern: Glob pattern for episode files

    Returns:
        List of matching file paths
    """
    import glob as glob_module

    files = glob_module.glob(glob_pattern)
    files = sorted(files)

    if not files:
        print(f"Warning: No files matching pattern: {glob_pattern}", file=sys.stderr)

    return files


def save_split_result(result: SplitResult, output_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Save split result to output directory.

    Args:
        result: Split result
        output_dir: Output directory
        experiment_name: Optional experiment name

    Returns:
        Path to saved split index
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate split ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_id = experiment_name or f"split_{timestamp}"

    # Save split index
    split_file = output_dir / f"{split_id}.json"

    split_data = {
        "split_id": split_id,
        "train_files": result.train_files,
        "val_files": result.val_files,
        "test_files": result.test_files,
        "train_count": result.train_count,
        "val_count": result.val_count,
        "test_count": result.test_count,
        "stratify_key": result.stratify_key,
        "seed": result.seed,
        "config": result.config,
        "stratification_distribution": result.stratification_distribution,
    }

    with open(split_file, 'w') as f:
        json.dump(split_data, f, indent=2)

    # Also save as train.txt, val.txt, test.txt for easy loading
    for split_name, files in [
        ("train.txt", result.train_files),
        ("val.txt", result.val_files),
        ("test.txt", result.test_files),
    ]:
        split_list_file = output_dir / split_name
        with open(split_list_file, 'w') as f:
            f.write("\n".join(files))

    print(f"Saved split to: {split_file}")
    return str(split_file)


def load_split_result(split_file: str) -> SplitResult:
    """
    Load a split result from file.

    Args:
        split_file: Path to split JSON file

    Returns:
        SplitResult
    """
    with open(split_file, 'r') as f:
        data = json.load(f)

    return SplitResult(
        train_files=data.get("train_files", []),
        val_files=data.get("val_files", []),
        test_files=data.get("test_files", []),
        train_count=data.get("train_count", 0),
        val_count=data.get("val_count", 0),
        test_count=data.get("test_count", 0),
        stratify_key=data.get("stratify_key", ""),
        seed=data.get("seed", 42),
        config=data.get("config", {}),
        stratification_distribution=data.get("stratification_distribution", {}),
    )


def list_splits(split_dir: str) -> List[Dict]:
    """
    List all splits in a directory.

    Args:
        split_dir: Directory containing split files

    Returns:
        List of split info dictionaries
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []

    splits = []
    for split_file in split_dir.glob("*.json"):
        try:
            with open(split_file, 'r') as f:
                data = json.load(f)
            splits.append({
                "file": str(split_file),
                "split_id": data.get("split_id", split_file.stem),
                "train_count": data.get("train_count", 0),
                "val_count": data.get("val_count", 0),
                "test_count": data.get("test_count", 0),
                "stratify_key": data.get("stratify_key", ""),
            })
        except Exception:
            continue

    return sorted(splits, key=lambda x: x["file"])


def run_split(config: SplitConfig) -> SplitResult:
    """
    Run the dataset splitting pipeline.

    Args:
        config: Split configuration

    Returns:
        SplitResult
    """
    print(f"Loading episodes from: {config.episodes_glob}")

    # Load episode files
    files = load_episode_files(config.episodes_glob)
    print(f"Found {len(files)} episode files")

    if len(files) == 0:
        return SplitResult(config={}, seed=config.seed)

    # Extract episode info for stratification
    print(f"Extracting episode info (stratify={config.stratify})...")
    episode_infos = [extract_episode_info(f) for f in files]

    # Compute stratification keys
    if config.stratify != STRATIFY_NONE:
        stratify_keys = [compute_stratification_key(info, config.stratify) for info in episode_infos]

        # Compute distribution
        stratify_dist: Dict[str, Dict[str, int]] = {}
        for key, info in zip(stratify_keys, episode_infos):
            stratum = compute_stratification_key(info, config.stratify)
            key_str = f"{info.scenario_type or 'unknown'}"
            if stratum not in stratify_dist:
                stratify_dist[stratum] = {}
            stratify_dist[stratum][key_str] = stratify_dist[stratum].get(key_str, 0) + 1
    else:
        stratify_keys = ["none"] * len(files)
        stratify_dist = {}

    # Perform split
    print(f"Performing split (train={config.train_ratio}, val={config.val_ratio}, test={config.test_ratio})...")

    if config.stratify != STRATIFY_NONE:
        train_files, val_files, test_files = stratified_split(
            files,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
            stratify_keys,
            config.seed,
            config.min_samples_per_stratum,
        )
    else:
        train_files, val_files, test_files = random_split(
            files,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
            config.seed,
        )

    result = SplitResult(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        train_count=len(train_files),
        val_count=len(val_files),
        test_count=len(test_files),
        stratify_key=config.stratify,
        seed=config.seed,
        config={
            "episodes_glob": config.episodes_glob,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "stratify": config.stratify,
        },
        stratification_distribution=stratify_dist,
    )

    # Save result
    if not config.dry_run:
        save_split_result(result, config.output_dir, config.experiment_name)

    # Print summary
    print(f"\nSplit Summary:")
    print(f"  Train: {result.train_count} files ({result.train_count / len(files) * 100:.1f}%)")
    print(f"  Val:   {result.val_count} files ({result.val_count / len(files) * 100:.1f}%)")
    print(f"  Test:  {result.test_count} files ({result.test_count / len(files) * 100:.1f}%)")
    print(f"  Stratify: {config.stratify}")

    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Dataset Splitter for Waymo Episodes")

    # Input
    parser.add_argument("--episodes-glob", type=str, default="data/waymo/episodes/*.json",
                      help="Glob pattern for episode files")

    # Split ratios
    parser.add_argument("--train-ratio", type=float, default=0.8,
                      help="Training set ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                      help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                      help="Test set ratio (default: 0.1)")

    # Stratification
    parser.add_argument("--stratify", type=str, default=STRATIFY_NONE,
                      choices=[STRATIFY_NONE, STRATIFY_ROUTE, STRATIFY_SCENARIO, STRATIFY_DIFFICULTY],
                      help="Stratification strategy")
    parser.add_argument("--min-samples-per-stratum", type=int, default=2,
                      help="Minimum samples per stratum (default: 2)")

    # Output
    parser.add_argument("--output-dir", type=str, default="out/splits",
                      help="Output directory for split files")
    parser.add_argument("--experiment-name", type=str, default=None,
                      help="Experiment name for split ID")

    # Options
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                      help="Dry run (don't save)")

    # List mode
    parser.add_argument("--list-splits", type=str, metavar="DIR",
                      help="List existing splits in directory")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Ratios must sum to 1.0, got {total_ratio}")

    # List mode
    if args.list_splits:
        splits = list_splits(args.list_splits)
        if not splits:
            print(f"No splits found in: {args.list_splits}")
            return 1

        print(f"Found {len(splits)} splits:")
        for split in splits:
            print(f"  {split['split_id']}: train={split['train_count']}, val={split['val_count']}, "
                  f"test={split['test_count']}, stratify={split['stratify_key']}")
        return 0

    # Run split
    config = SplitConfig(
        episodes_glob=args.episodes_glob,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=args.stratify,
        min_samples_per_stratum=args.min_samples_per_stratum,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    result = run_split(config)
    return 0 if result.train_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
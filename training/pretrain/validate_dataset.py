#!/usr/bin/env python3
"""
Dataset Validation Utility for Pipeline Data Quality Assurance.

Validates Waymo episode datasets before SSL/Waypoint BC training.
Checks for: missing frames, corrupt data, out-of-range values, temporal gaps.
Computes: frame counts, temporal coverage, velocity distributions.

Usage:
    python training/pretrain/validate_dataset.py validate --episodes-glob "data/waymo/episodes/*.json"
    python training/pretrain/validate_dataset.py check --episode-path data/waymo/episodes/episode_12345.json
    python training/pretrain/validate_dataset.py stats --episodes-dir data/waymo/episodes/ --output stats.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ValidationIssue:
    """Single validation issue."""
    severity: str  # error, warning, info
    category: str  # missing, corrupt, range, temporal
    message: str
    episode_id: Optional[str] = None
    frame_idx: Optional[int] = None


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_id: str
    num_frames: int
    duration_s: float
    total_distance_m: float
    avg_speed_mps: float
    max_speed_mps: float
    num_agents: int
    has_valid_route: bool
    has_images: bool
    missing_frames: int = 0


@dataclass
class DatasetValidationResult:
    """Overall validation result for dataset."""
    total_episodes: int = 0
    valid_episodes: int = 0
    issues: list = field(default_factory=list)
    episode_stats: list = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


def load_episode(path: Path) -> dict:
    """Load episode JSON."""
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")
    except Exception as e:
        raise IOError(f"Cannot read {path}: {e}")


def validate_episode_structure(episode: dict, episode_id: str) -> list[ValidationIssue]:
    """Validate episode has required fields."""
    issues = []
    
    # Accept multiple schema variants
    has_frames = "frames" in episode
    has_episode_id = "episode_id" in episode
    
    if not has_frames:
        issues.append(ValidationIssue(
            severity="error",
            category="missing",
            message="Missing required field: frames",
            episode_id=episode_id
        ))
    
    if not has_episode_id:
        issues.append(ValidationIssue(
            severity="warning",
            category="missing",
            message="Missing episode_id field",
            episode_id=episode_id
        ))
    
    # Check for domain (Waymo vs synthetic)
    if "domain" not in episode and "source" not in episode:
        issues.append(ValidationIssue(
            severity="info",
            category="missing",
            message="Missing domain/source field",
            episode_id=episode_id
        ))
    
    return issues


def validate_frames(frames: list, episode_id: str) -> list[ValidationIssue]:
    """Validate frame data."""
    issues = []
    
    if not frames:
        issues.append(ValidationIssue(
            severity="error",
            category="missing",
            message="No frames in episode",
            episode_id=episode_id
        ))
        return issues
    
    # Check frame structure - supports multiple schemas:
    # Schema 1: {"t", "observations", "expert"}
    # Schema 2: {"timestamp", "agents", "image"}
    # Schema 3: {"timestamp", "agent_data", "image"}
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            issues.append(ValidationIssue(
                severity="error",
                category="corrupt",
                message=f"Frame {idx} is not a dict",
                episode_id=episode_id,
                frame_idx=idx
            ))
            continue
        
        # Check for valid observation data
        has_obs = "observations" in frame or "observation" in frame
        has_t = "t" in frame
        has_timestamp = "timestamp" in frame
        
        if not (has_obs or has_t or has_timestamp):
            issues.append(ValidationIssue(
                severity="warning",
                category="missing",
                message=f"Frame {idx} missing observation/timestamp",
                episode_id=episode_id,
                frame_idx=idx
            ))
        
        # Check for expert data / ego trajectory
        has_expert = "expert" in frame or "ego_state" in frame
        if not has_expert:
            issues.append(ValidationIssue(
                severity="warning",
                category="missing",
                message=f"Frame {idx} missing expert/ego_state",
                episode_id=episode_id,
                frame_idx=idx
            ))
        
        # Check image format if present
        if "image" in frame:
            img = frame["image"]
            if img is None:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="corrupt",
                    message=f"Frame {idx} has null image",
                    episode_id=episode_id,
                    frame_idx=idx
                ))
            elif isinstance(img, str) and len(img) < 100:
                # Very short base64/data URI, likely placeholder
                issues.append(ValidationIssue(
                    severity="warning",
                    category="corrupt",
                    message=f"Frame {idx} has short/placeholder image",
                    episode_id=episode_id,
                    frame_idx=idx
                ))
    
    # Check for temporal gaps - supports {"t": float} in ms
    timestamps = []
    for f in frames:
        if "t" in f:
            timestamps.append(f["t"])
        elif "timestamp" in f:
            timestamps.append(f["timestamp"])
    
    if len(timestamps) > 1:
        # Determine time unit (ms vs seconds)
        max_t = max(timestamps)
        if max_t > 1000:
            # Likely in milliseconds
            t_unit = 1000.0
        else:
            t_unit = 1.0
        
        gaps = []
        for i in range(len(timestamps) - 1):
            dt = (timestamps[i+1] - timestamps[i]) / t_unit
            if dt > 0:
                gaps.append(dt)
        
        if gaps:
            max_gap = max(gaps)
            if max_gap > 0.5:  # > 500ms gap
                issues.append(ValidationIssue(
                    severity="warning",
                    category="temporal",
                    message=f"Large temporal gap: {max_gap:.3f}s",
                    episode_id=episode_id
                ))
    
    return issues


def validate_trajectory(trajectory: list, episode_id: str) -> list[ValidationIssue]:
    """Validate ego trajectory."""
    issues = []
    
    if not trajectory:
        issues.append(ValidationIssue(
            severity="warning",
            category="missing",
            message="No ego trajectory",
            episode_id=episode_id
        ))
        return issues
    
    # Check waypoint format
    for idx, wp in enumerate(trajectory):
        if not isinstance(wp, (list, tuple)) or len(wp) < 2:
            issues.append(ValidationIssue(
                severity="error",
                category="corrupt",
                message=f"Waypoint {idx} has invalid format",
                episode_id=episode_id,
                frame_idx=idx
            ))
            continue
        
        x, y = wp[0], wp[1]
        
        # Check for reasonable coordinates (Waymo area ~ Boston)
        if not (-1000 < x < 1000 and -1000 < y < 1000):
            issues.append(ValidationIssue(
                severity="warning",
                category="range",
                message=f"Waypoint {idx} out of range: ({x:.1f}, {y:.1f})",
                episode_id=episode_id,
                frame_idx=idx
            ))
    
    # Check velocity reasonableness
    velocities = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i-1][0]
        dy = trajectory[i][1] - trajectory[i-1][1]
        dist = (dx**2 + dy**2)**0.5
        velocities.append(dist)
    
    if velocities:
        max_vel = max(velocities)
        if max_vel > 50:  # > 50m between frames is suspicious
            issues.append(ValidationIssue(
                severity="warning",
                category="range",
                message=f"Unusually large step: {max_vel:.1f}m",
                episode_id=episode_id
            ))
    
    return issues


def compute_episode_stats(episode: dict, episode_id: str) -> EpisodeStats:
    """Compute statistics for episode."""
    frames = episode.get("frames", [])
    trajectory = episode.get("trajectory", episode.get("ego_trajectory", []))
    metadata = episode.get("metadata", {})
    
    num_frames = len(frames)
    
    # Duration
    timestamps = [f.get("timestamp") for f in frames if f.get("timestamp")]
    duration_s = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0.0
    
    # Distance
    total_distance = 0.0
    for i in range(1, len(trajectory)):
        if len(trajectory[i]) >= 2 and len(trajectory[i-1]) >= 2:
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_distance += (dx**2 + dy**2)**0.5
    
    # Speed stats
    velocities = []
    for i in range(1, len(trajectory)):
        if len(trajectory[i]) >= 2 and len(trajectory[i-1]) >= 2:
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            dist = (dx**2 + dy**2)**0.5
            velocities.append(dist)
    
    avg_speed = sum(velocities) / len(velocities) if velocities else 0.0
    max_speed = max(velocities) if velocities else 0.0
    
    # Agents
    num_agents = 0
    for frame in frames:
        agents = frame.get("agents", frame.get("agent_data", []))
        if isinstance(agents, list):
            num_agents = max(num_agents, len(agents))
    
    # Valid checks
    has_valid_route = len(trajectory) > 0
    has_images = any(f.get("image") for f in frames)
    
    return EpisodeStats(
        episode_id=episode_id,
        num_frames=num_frames,
        duration_s=duration_s,
        total_distance_m=total_distance,
        avg_speed_mps=avg_speed,
        max_speed_mps=max_speed,
        num_agents=num_agents,
        has_valid_route=has_valid_route,
        has_images=has_images
    )


def validate_single_episode(episode_path: Path) -> tuple[EpisodeStats, list[ValidationIssue]]:
    """Validate a single episode file."""
    episode_id = episode_path.stem
    
    try:
        episode = load_episode(episode_path)
    except (ValueError, IOError) as e:
        return None, [ValidationIssue(
            severity="error",
            category="corrupt",
            message=str(e),
            episode_id=episode_id
        )]
    
    issues = []
    
    # Structure validation
    issues.extend(validate_episode_structure(episode, episode_id))
    
    # Frame validation
    frames = episode.get("frames", [])
    issues.extend(validate_frames(frames, episode_id))
    
    # Trajectory validation - from episode or per-frame expert data
    trajectory = episode.get("trajectory", episode.get("ego_trajectory", []))
    if not trajectory:
        # Check for per-frame expert waypoints
        for f in frames:
            exp = f.get("expert", {})
            wp = exp.get("waypoints", [])
            if wp:
                trajectory = wp
                break
    issues.extend(validate_trajectory(trajectory, episode_id))
    
    # Compute stats
    stats = compute_episode_stats(episode, episode_id)
    
    return stats, issues


def validate_episodes_glob(glob_pattern: str) -> DatasetValidationResult:
    """Validate all episodes matching glob pattern."""
    result = DatasetValidationResult()
    
    from glob import glob
    episode_paths = sorted(glob(glob_pattern))
    
    if not episode_paths:
        result.issues.append(ValidationIssue(
            severity="error",
            category="missing",
            message=f"No episodes found matching: {glob_pattern}",
            episode_id="*"
        ))
        return result
    
    result.total_episodes = len(episode_paths)
    
    valid_count = 0
    for path_str in episode_paths:
        path = Path(path_str)
        stats, issues = validate_single_episode(path)
        
        if stats:
            result.episode_stats.append(stats)
            valid_count += 1
        
        result.issues.extend(issues)
    
    result.valid_episodes = valid_count
    
    return result


def validate_episodes_dir(dir_path: Path) -> DatasetValidationResult:
    """Validate all episodes in directory."""
    result = DatasetValidationResult()
    
    episode_paths = sorted(dir_path.glob("*.json"))
    episode_paths.extend(sorted(dir_path.glob("*.jsonl")))
    
    if not episode_paths:
        result.issues.append(ValidationIssue(
            severity="error",
            category="missing",
            message=f"No episodes found in: {dir_path}",
            episode_id="*"
        ))
        return result
    
    result.total_episodes = len(episode_paths)
    
    valid_count = 0
    for path in episode_paths:
        stats, issues = validate_single_episode(path)
        
        if stats:
            result.episode_stats.append(stats)
            valid_count += 1
        
        result.issues.extend(issues)
    
    result.valid_episodes = valid_count
    
    return result


def print_validation_report(result: DatasetValidationResult) -> None:
    """Print human-readable validation report."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\nEpisodes: {result.valid_episodes}/{result.total_episodes} valid")
    print(f"Errors: {result.error_count}")
    print(f"Warnings: {result.warning_count}")
    
    if result.episode_stats:
        total_frames = sum(s.num_frames for s in result.episode_stats)
        total_distance = sum(s.total_distance_m for s in result.episode_stats)
        avg_speed = sum(s.avg_speed_mps for s in result.episode_stats) / len(result.episode_stats)
        
        print(f"\nAggregate Statistics:")
        print(f"  Total frames: {total_frames}")
        print(f"  Total distance: {total_distance:.1f}m")
        print(f"  Avg speed: {avg_speed:.2f}m/s")
        print(f"  Avg frames/episode: {total_frames/len(result.episode_stats):.1f}")
    
    # Show issues
    errors = [i for i in result.issues if i.severity == "error"]
    warnings = [i for i in result.issues if i.severity == "warning"]
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for issue in errors[:10]:
            print(f"  [{issue.category}] {issue.message}")
            if issue.episode_id:
                print(f"    Episode: {issue.episode_id}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for issue in warnings[:10]:
            print(f"  [{issue.category}] {issue.message}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    
    # Pass/fail
    print("\n" + "-" * 60)
    if result.error_count == 0:
        print("✅ PASSED - Dataset is valid for training")
    else:
        print("❌ FAILED - Fix errors before training")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for driving-first pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # validate subcommand
    val_parser = subparsers.add_parser("validate", help="Validate episodes by glob pattern")
    val_parser.add_argument("--episodes-glob", required=True, help="Glob pattern for episodes")
    val_parser.add_argument("--output", help="Output JSON report path")
    
    # check subcommand
    check_parser = subparsers.add_parser("check", help="Validate single episode")
    check_parser.add_argument("--episode-path", required=True, help="Path to episode JSON")
    
    # stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Compute dataset statistics")
    stats_parser.add_argument("--episodes-dir", required=True, help="Directory containing episodes")
    stats_parser.add_argument("--output", help="Output JSON path")
    
    args = parser.parse_args()
    
    result = None
    
    if args.command == "validate":
        result = validate_episodes_glob(args.episodes_glob)
    elif args.command == "check":
        stats, issues = validate_single_episode(Path(args.episode_path))
        result = DatasetValidationResult(
            total_episodes=1,
            valid_episodes=1 if stats else 0,
            episode_stats=[stats] if stats else [],
            issues=issues
        )
    elif args.command == "stats":
        result = validate_episodes_dir(Path(args.episodes_dir))
    
    if result:
        print_validation_report(result)
        
        if args.command == "validate" and args.output:
            output = {
                "total_episodes": result.total_episodes,
                "valid_episodes": result.valid_episodes,
                "error_count": result.error_count,
                "warning_count": result.warning_count,
                "issues": [
                    {
                        "severity": i.severity,
                        "category": i.category,
                        "message": i.message,
                        "episode_id": i.episode_id
                    }
                    for i in result.issues
                ],
                "stats": [
                    {
                        "episode_id": s.episode_id,
                        "num_frames": s.num_frames,
                        "duration_s": s.duration_s,
                        "total_distance_m": s.total_distance_m,
                        "avg_speed_mps": s.avg_speed_mps
                    }
                    for s in result.episode_stats
                ]
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Report saved to: {args.output}")
        
        # Exit code
        sys.exit(0 if result.error_count == 0 else 1)
    else:
        print("No validation performed")
        sys.exit(1)


if __name__ == "__main__":
    main()
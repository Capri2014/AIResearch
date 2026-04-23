#!/usr/bin/env python3
"""
Pipeline Data Manager

Orchestrates the complete data pipeline for driving-first pretraining:
Waymo episodes → SSL frame index → SSL pretrain → waypoint extraction → BC dataloader.

Provides unified data lineage tracking, cache management, and stage validation
across all data-dependent pipeline stages.

Usage:
    python training/pipeline_data_manager.py build --episodes data/waymo/episodes/*.json
    python training/pipeline_data_manager.py stats
    python training/pipeline_data_manager.py validate
    python training/pipeline_data_manager.py dataloader --stage bc --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Enums and dataclasses
# ──────────────────────────────────────────────────────────────────────────────

class DataStage(Enum):
    RAW_EPISODES = "raw_episodes"
    EPISODE_INDEX = "episode_index"
    SSL_PRETRAIN = "ssl_pretrain"
    WAYPOINT_EXTRACT = "waypoint_extract"
    BC_DATALOADER = "bc_dataloader"


@dataclass
class EpisodeSummary:
    """Summary of a single Waymo episode."""
    episode_id: str
    frame_count: int
    duration_s: float
    cameras: List[str]
    path: str
    file_size_mb: float


@dataclass
class DataStageInfo:
    """Info for a pipeline data stage."""
    stage: DataStage
    output_dir: Path
    exists: bool
    item_count: int = 0
    last_modified: Optional[str] = None
    file_size_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDataConfig:
    """Configuration for pipeline data management."""
    # Paths
    episodes_glob: str = "data/waymo/episodes/*.json"
    episodes_dir: str = "data/waymo/episodes"
    episode_index_path: str = "data/waymo/episode_index.json"
    waypoint_cache_dir: str = "data/waymo/waypoint_cache"

    # SSL index building
    frame_sample_rate: int = 1
    cameras: List[str] = field(
        default_factory=lambda: ["front", "front_left", "front_right", "side_left", "side_right"]
    )

    # Waypoint extraction
    num_waypoints: int = 8
    horizon_seconds: float = 3.0
    waypoint_sampling_hz: float = 2.0

    # BC dataloader
    bc_batch_size: int = 32
    bc_num_workers: int = 4
    bc_shuffle: bool = True

    # Output
    out_dir: Path = field(default_factory=lambda: Path("out/data_manager"))
    log_every: int = 10

    # Misc
    seed: int = 42
    verbose: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Episode scanner
# ──────────────────────────────────────────────────────────────────────────────

def _get_default_cameras() -> List[str]:
    return ["front", "front_left", "front_right", "side_left", "side_right"]


def scan_episodes(episodes_glob: str) -> List[EpisodeSummary]:
    """Scan directory for Waymo episode files and build summaries."""
    episodes: List[EpisodeSummary] = []
    glob_path = Path(episodes_glob)
    if not glob_path.parent.exists():
        return episodes

    pattern = glob_path.name
    for ep_file in sorted(glob_path.parent.glob(pattern)):
        try:
            size_mb = ep_file.stat().st_size / 1e6
            with open(ep_file) as f:
                data = json.load(f)

            frames = data.get("frames", [])
            cameras = _get_default_cameras()
            duration = len(frames) * 0.1 if frames else 0.0

            episodes.append(EpisodeSummary(
                episode_id=data.get("episode_id", ep_file.stem),
                frame_count=len(frames),
                duration_s=duration,
                cameras=cameras,
                path=str(ep_file),
                file_size_mb=size_mb,
            ))
        except (json.JSONDecodeError, KeyError, OSError):
            # Synthetic/placeholder episode
            episodes.append(EpisodeSummary(
                episode_id=ep_file.stem,
                frame_count=0,
                duration_s=0.0,
                cameras=_get_default_cameras(),
                path=str(ep_file),
                file_size_mb=ep_file.stat().st_size / 1e6,
            ))

    return episodes


def print_episode_table(episodes: List[EpisodeSummary]) -> None:
    """Print a formatted table of episode summaries."""
    if not episodes:
        print("  No episodes found.")
        return

    header = f"  {'Episode ID':<30} {'Frames':>8} {'Duration':>10} {'Size (MB)':>10} {'Cameras':>12}"
    print(header)
    print("  " + "-" * 75)

    total_frames = 0
    total_duration = 0.0
    total_size = 0.0

    for ep in episodes:
        print(
            f"  {ep.episode_id:<30} {ep.frame_count:>8} "
            f"{ep.duration_s:>9.1f}s {ep.file_size_mb:>9.1f} MB "
            f"{len(ep.cameras):>6} cam"
        )
        total_frames += ep.frame_count
        total_duration += ep.duration_s
        total_size += ep.file_size_mb

    print("  " + "-" * 75)
    print(
        f"  {'TOTAL':<30} {total_frames:>8} "
        f"{total_duration:>9.1f}s {total_size:>9.1f} MB "
        f"{len(episodes):>6} eps"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Episode index builder
# ──────────────────────────────────────────────────────────────────────────────

def build_episode_index(
    config: PipelineDataConfig,
    episodes: List[EpisodeSummary],
) -> Dict[str, Any]:
    """Build compact frame index from episodes for fast dataloader init."""
    index: Dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "config": {
            "frame_sample_rate": config.frame_sample_rate,
            "cameras": config.cameras,
            "num_cameras": len(config.cameras),
        },
        "episodes": [],
        "total_frames": 0,
        "total_episodes": len(episodes),
    }

    for ep in episodes:
        if ep.frame_count == 0:
            continue

        ep_entry = {
            "episode_id": ep.episode_id,
            "path": ep.path,
            "frame_count": ep.frame_count,
            "cameras": config.cameras,
            "frame_indices": [],
        }

        # Sample frames at frame_sample_rate
        for i in range(0, ep.frame_count, config.frame_sample_rate):
            ep_entry["frame_indices"].append(i)

        index["episodes"].append(ep_entry)
        index["total_frames"] += len(ep_entry["frame_indices"])

    return index


def save_episode_index(index: Dict[str, Any], output_path: str) -> Path:
    """Save episode index to file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(index, f, indent=2)
    return output


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint cache manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WaypointCacheInfo:
    """Info about waypoint cache state."""
    cache_dir: Path
    exists: bool
    episode_count: int = 0
    total_samples: int = 0
    file_size_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def ensure_waypoint_cache(
    config: PipelineDataConfig,
    episodes: List[EpisodeSummary],
) -> WaypointCacheInfo:
    """Ensure waypoint cache exists, creating it if needed."""
    cache_dir = Path(config.waypoint_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check existing cache
    existing = []
    total_size = 0.0
    total_samples = 0

    if cache_dir.exists():
        for cf in sorted(cache_dir.glob("*.jsonl")):
            total_size += cf.stat().st_size / 1e6
            try:
                with open(cf) as f:
                    lines = sum(1 for _ in f)
                    total_samples += lines
                    existing.append(cf.stem)
            except OSError:
                pass

    # Load or create cache metadata
    meta_path = cache_dir / "metadata.json"
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return WaypointCacheInfo(
        cache_dir=cache_dir,
        exists=cache_dir.exists(),
        episode_count=len(existing),
        total_samples=total_samples,
        file_size_mb=total_size,
        metadata=metadata,
    )


# ──────────────────────────────────────────────────────────────────────────────
# BC dataloader builder
# ──────────────────────────────────────────────────────────────────────────────

def build_bc_dataloader_info(config: PipelineDataConfig) -> Dict[str, Any]:
    """Build info about BC dataloader configuration."""
    cache_dir = Path(config.waypoint_cache_dir)

    # Check what waypoint data is available
    available_eps = []
    if cache_dir.exists():
        for f in sorted(cache_dir.glob("*.jsonl")):
            try:
                with open(f) as fp:
                    lines = sum(1 for _ in fp)
                available_eps.append({"episode_id": f.stem, "samples": lines})
            except OSError:
                pass

    total_samples = sum(e["samples"] for e in available_eps)
    num_batches = total_samples // config.bc_batch_size if total_samples > 0 else 0

    return {
        "batch_size": config.bc_batch_size,
        "num_workers": config.bc_num_workers,
        "shuffle": config.bc_shuffle,
        "available_episodes": len(available_eps),
        "total_samples": total_samples,
        "estimated_batches_per_epoch": num_batches,
        "episodes": available_eps[:5],  # First 5 for preview
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stage validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_data_pipeline(config: PipelineDataConfig) -> Dict[str, Any]:
    """Validate entire data pipeline and report gaps."""
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "stages": {},
        "pipeline_ready": True,
        "gaps": [],
        "recommendations": [],
    }

    # Stage 1: Raw episodes
    episodes = scan_episodes(config.episodes_glob)
    ep_dir = Path(config.episodes_dir)
    results["stages"]["raw_episodes"] = {
        "status": "PASS" if episodes else "WARN",
        "count": len(episodes),
        "total_frames": sum(e.frame_count for e in episodes),
        "path": str(ep_dir),
        "exists": ep_dir.exists(),
    }
    if not episodes:
        results["pipeline_ready"] = False
        results["gaps"].append("No Waymo episode files found")
        results["recommendations"].append(
            f"Place episode JSON files in {config.episodes_dir}/ or set --episodes-glob"
        )

    # Stage 2: Episode index
    index_path = Path(config.episode_index_path)
    if index_path.exists():
        try:
            with open(index_path) as f:
                index_data = json.load(f)
            results["stages"]["episode_index"] = {
                "status": "PASS",
                "path": str(index_path),
                "total_frames": index_data.get("total_frames", 0),
                "total_episodes": index_data.get("total_episodes", 0),
            }
        except (json.JSONDecodeError, OSError):
            results["stages"]["episode_index"] = {
                "status": "FAIL",
                "path": str(index_path),
                "error": "Corrupt index file",
            }
            results["pipeline_ready"] = False
    else:
        results["stages"]["episode_index"] = {
            "status": "MISSING",
            "path": str(index_path),
        }
        results["gaps"].append("Episode index not built")
        results["recommendations"].append(
            f"Run: python training/pipeline_data_manager.py build --episodes '{config.episodes_glob}'"
        )

    # Stage 3: SSL pretrain data readiness
    if index_path.exists() and episodes:
        results["stages"]["ssl_pretrain"] = {
            "status": "PASS",
            "ready": True,
            "frame_count": results["stages"].get("episode_index", {}).get("total_frames", 0),
        }
    else:
        results["stages"]["ssl_pretrain"] = {
            "status": "BLOCKED",
            "ready": False,
        }
        results["pipeline_ready"] = False

    # Stage 4: Waypoint cache
    wpi = ensure_waypoint_cache(config, episodes)
    results["stages"]["waypoint_cache"] = {
        "status": "PASS" if wpi.total_samples > 0 else "EMPTY",
        "cache_dir": str(wpi.cache_dir),
        "exists": wpi.exists,
        "episodes_cached": wpi.episode_count,
        "total_samples": wpi.total_samples,
        "size_mb": round(wpi.file_size_mb, 2),
    }
    if wpi.total_samples == 0 and episodes:
        results["gaps"].append("Waypoint cache is empty")
        results["recommendations"].append(
            "Run waypoint extraction before BC training: "
            "python training/pretrain/extract_waypoints.py --episodes-dir data/waymo/episodes"
        )

    # Stage 5: BC dataloader readiness
    bc_info = build_bc_dataloader_info(config)
    results["stages"]["bc_dataloader"] = {
        "status": "PASS" if bc_info["total_samples"] > 0 else "EMPTY",
        "batch_size": bc_info["batch_size"],
        "available_episodes": bc_info["available_episodes"],
        "total_samples": bc_info["total_samples"],
        "estimated_batches": bc_info["estimated_batches_per_epoch"],
    }
    if bc_info["total_samples"] == 0 and episodes:
        results["gaps"].append("BC dataloader has no samples")
        results["recommendations"].append(
            "BC dataloader requires waypoint extraction to complete first"
        )

    return results


def print_validation_results(validation: Dict[str, Any]) -> None:
    """Print formatted validation results."""
    print("\n📊 Pipeline Data Validation")
    print("=" * 60)

    for stage_name, info in validation["stages"].items():
        status = info.get("status", "UNKNOWN")
        icon = "✅" if status == "PASS" else ("⚠️" if status in ("WARN", "MISSING", "EMPTY") else "❌")
        print(f"\n{icon} {stage_name.replace('_', ' ').title()}")

        if stage_name == "raw_episodes":
            print(f"   Episodes: {info.get('count', 0)} | Frames: {info.get('total_frames', 0)} | Path: {info.get('path', 'N/A')}")
        elif stage_name == "episode_index":
            if status == "PASS":
                print(f"   Frames: {info.get('total_frames', 0)} | Episodes: {info.get('total_episodes', 0)}")
            print(f"   Path: {info.get('path', 'N/A')}")
        elif stage_name == "waypoint_cache":
            print(f"   Cached: {info.get('episodes_cached', 0)} eps | Samples: {info.get('total_samples', 0)} | Size: {info.get('size_mb', 0):.1f} MB")
        elif stage_name == "bc_dataloader":
            print(f"   Episodes: {info.get('available_episodes', 0)} | Samples: {info.get('total_samples', 0)} | Batches/epoch: {info.get('estimated_batches', 0)}")
        else:
            for k, v in info.items():
                if k != "status":
                    print(f"   {k}: {v}")

    gaps = validation.get("gaps", [])
    if gaps:
        print(f"\n⚠️  Gaps ({len(gaps)}):")
        for gap in gaps:
            print(f"   • {gap}")

    recs = validation.get("recommendations", [])
    if recs:
        print(f"\n💡 Recommendations:")
        for rec in recs:
            print(f"   → {rec}")

    ready = validation.get("pipeline_ready", False)
    print(f"\n{'✅ Pipeline data ready' if ready else '❌ Pipeline data NOT ready'}")


# ──────────────────────────────────────────────────────────────────────────────
# PipelineDataManager main class
# ──────────────────────────────────────────────────────────────────────────────

class PipelineDataManager:
    """
    Orchestrates the complete data pipeline for driving-first pretraining.

    Manages data flow across stages:
    1. Raw Waymo episodes → scan and summarize
    2. Episode index building → compact frame index for fast dataloader init
    3. SSL pretrain data readiness
    4. Waypoint extraction cache
    5. BC dataloader preparation

    Provides unified view of pipeline data state with lineage tracking.
    """

    def __init__(self, config: Optional[PipelineDataConfig] = None):
        self.config = config or PipelineDataConfig()
        self._episode_cache: List[EpisodeSummary] = []
        self._validation_cache: Optional[Dict[str, Any]] = None
        self._bc_loader_info: Optional[Dict[str, Any]] = None

    # ─── Properties ───────────────────────────────────────────────────────���───

    @property
    def episodes(self) -> List[EpisodeSummary]:
        """Lazy-load and cache episode scan results."""
        if not self._episode_cache:
            self._episode_cache = scan_episodes(self.config.episodes_glob)
        return self._episode_cache

    @property
    def stage_info(self) -> Dict[DataStage, DataStageInfo]:
        """Get info for all data stages."""
        info: Dict[DataStage, DataStageInfo] = {}

        # Raw episodes
        ep_dir = Path(self.config.episodes_dir)
        info[DataStage.RAW_EPISODES] = DataStageInfo(
            stage=DataStage.RAW_EPISODES,
            output_dir=ep_dir,
            exists=ep_dir.exists(),
            item_count=len(self.episodes),
        )

        # Episode index
        idx_path = Path(self.config.episode_index_path)
        idx_info = DataStageInfo(
            stage=DataStage.EPISODE_INDEX,
            output_dir=idx_path,
            exists=idx_path.exists(),
            last_modified=(
                datetime.fromtimestamp(idx_path.stat().st_mtime).isoformat()
                if idx_path.exists() else None
            ),
        )
        if idx_path.exists():
            try:
                with open(idx_path) as f:
                    idx_data = json.load(f)
                idx_info.item_count = idx_data.get("total_frames", 0)
                idx_info.file_size_mb = idx_path.stat().st_size / 1e6
                idx_info.metadata = {
                    "total_episodes": idx_data.get("total_episodes", 0),
                    "num_cameras": len(idx_data.get("config", {}).get("cameras", [])),
                }
            except (json.JSONDecodeError, OSError):
                pass
        info[DataStage.EPISODE_INDEX] = idx_info

        # Waypoint cache
        wpi = ensure_waypoint_cache(self.config, self.episodes)
        info[DataStage.WAYPOINT_EXTRACT] = DataStageInfo(
            stage=DataStage.WAYPOINT_EXTRACT,
            output_dir=wpi.cache_dir,
            exists=wpi.exists,
            item_count=wpi.episode_count,
            file_size_mb=wpi.file_size_mb,
            metadata={"total_samples": wpi.total_samples},
        )

        # BC dataloader
        bc_info = build_bc_dataloader_info(self.config)
        info[DataStage.BC_DATALOADER] = DataStageInfo(
            stage=DataStage.BC_DATALOADER,
            output_dir=Path(self.config.waypoint_cache_dir),
            exists=wpi.total_samples > 0,
            item_count=bc_info["total_samples"],
            metadata=bc_info,
        )

        return info

    # ─── Methods ──────────────────────────────────────────────────────────────

    def scan(self) -> List[EpisodeSummary]:
        """Scan and return episode summaries."""
        return self.episodes

    def build_index(self, save: bool = True) -> Dict[str, Any]:
        """Build episode frame index from episodes."""
        index = build_episode_index(self.config, self.episodes)
        if save:
            out_path = save_episode_index(index, self.config.episode_index_path)
            print(f"   Index saved to: {out_path}")
        return index

    def validate(self, use_cache: bool = True) -> Dict[str, Any]:
        """Validate pipeline data, optionally using cached results."""
        if use_cache and self._validation_cache:
            return self._validation_cache
        validation = validate_data_pipeline(self.config)
        self._validation_cache = validation
        return validation

    def get_bc_dataloader_info(self) -> Dict[str, Any]:
        """Get BC dataloader configuration info."""
        if self._bc_loader_info is None:
            self._bc_loader_info = build_bc_dataloader_info(self.config)
        return self._bc_loader_info

    def print_status(self) -> None:
        """Print human-readable pipeline data status."""
        print("\n📦 Pipeline Data Status")
        print("=" * 60)

        for stage, info in self.stage_info.items():
            icon = "✅" if info.exists else ("⚠️" if info.item_count > 0 else "❌")
            print(f"\n{icon} {stage.value}")

            if stage == DataStage.RAW_EPISODES:
                print(f"   Episodes found: {info.item_count}")
            elif stage == DataStage.EPISODE_INDEX:
                if info.exists:
                    print(f"   Frames indexed: {info.item_count}")
                    print(f"   Episodes covered: {info.metadata.get('total_episodes', '?')}")
                    print(f"   Size: {info.file_size_mb:.2f} MB")
                else:
                    print("   Not built yet")
            elif stage == DataStage.WAYPOINT_EXTRACT:
                print(f"   Cached episodes: {info.item_count}")
                print(f"   Total samples: {info.metadata.get('total_samples', 0)}")
                print(f"   Size: {info.file_size_mb:.2f} MB")
            elif stage == DataStage.BC_DATALOADER:
                meta = info.metadata
                print(f"   Batch size: {meta.get('batch_size', self.config.bc_batch_size)}")
                print(f"   Available samples: {meta.get('total_samples', 0)}")
                print(f"   Batches/epoch: {meta.get('estimated_batches_per_epoch', 0)}")

        # Print episode table (first 10)
        episodes = self.episodes
        if episodes:
            print(f"\n📁 Episode Files (showing {min(10, len(episodes))} of {len(episodes)}):")
            print_episode_table(episodes[:10])


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", default="data/waymo/episodes/*.json", help="Glob for episode files")
    parser.add_argument("--episodes-dir", default="data/waymo/episodes", help="Base episodes directory")
    parser.add_argument("--index-output", default="data/waymo/episode_index.json", help="Output path for index")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--waypoint-cache", default="data/waymo/waypoint_cache", help="Waypoint cache directory")
    parser.add_argument("--bc-batch-size", type=int, default=32, help="BC dataloader batch size")
    parser.add_argument("--bc-workers", type=int, default=4, dest="bc_num_workers", help="BC dataloader num_workers")
    parser.add_argument("--verbose", action=store_true)


def store_true(arg):
    setattr(arg, 'verbose', True)
    return arg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline Data Manager - orchestrate driving-first data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Scan episodes and build frame index")
    add_build_args(p_build)
    p_build.add_argument("--no-save", action="store_true", help="Dry run, don't save index")

    # scan
    p_scan = sub.add_parser("scan", help="Scan and list episode files")
    p_scan.add_argument("--episodes", default="data/waymo/episodes/*.json")

    # index-stats
    p_idx = sub.add_parser("index-stats", help="Show episode index statistics")
    p_idx.add_argument("--index", default="data/waymo/episode_index.json")

    # validate
    p_val = sub.add_parser("validate", help="Validate pipeline data readiness")
    p_val.add_argument("--episodes", default="data/waymo/episodes/*.json")
    p_val.add_argument("--episodes-dir", default="data/waymo/episodes")
    p_val.add_argument("--index", default="data/waymo/episode_index.json")
    p_val.add_argument("--waypoint-cache", default="data/waymo/waypoint_cache")
    p_val.add_argument("--bc-batch-size", type=int, default=32)

    # status
    p_stat = sub.add_parser("status", help="Show full pipeline data status")
    p_stat.add_argument("--episodes", default="data/waymo/episodes/*.json")
    p_stat.add_argument("--episodes-dir", default="data/waymo/episodes")
    p_stat.add_argument("--index", default="data/waymo/episode_index.json")
    p_stat.add_argument("--waypoint-cache", default="data/waymo/waypoint_cache")
    p_stat.add_argument("--bc-batch-size", type=int, default=32)

    # dataloader-info
    p_dl = sub.add_parser("dataloader-info", help="Show BC dataloader info")
    p_dl.add_argument("--waypoint-cache", default="data/waymo/waypoint_cache")
    p_dl.add_argument("--bc-batch-size", type=int, default=32)

    args = parser.parse_args()
    cmd = args.command

    if cmd == "build":
        config = PipelineDataConfig(
            episodes_glob=args.episodes,
            episodes_dir=args.episodes_dir,
            episode_index_path=args.index_output,
        )
        mgr = PipelineDataManager(config)
        episodes = mgr.scan()
        print(f"\n📦 Found {len(episodes)} episodes")
        print_episode_table(episodes[:20])
        print(f"\n🔧 Building frame index...")
        index = mgr.build_index(save=not args.no_save)
        print(f"   Total frames indexed: {index.get('total_frames', 0)}")
        print(f"   Total episodes: {index.get('total_episodes', 0)}")

    elif cmd == "scan":
        episodes = scan_episodes(args.episodes)
        print(f"\n📦 Found {len(episodes)} episodes")
        print_episode_table(episodes)

    elif cmd == "index-stats":
        idx_path = Path(args.index)
        if not idx_path.exists():
            print(f"❌ Index not found: {idx_path}")
            return 1
        with open(idx_path) as f:
            index = json.load(f)
        print(f"\n📊 Episode Index Stats")
        print("=" * 40)
        print(f"  Version:        {index.get('version', 'N/A')}")
        print(f"  Created:       {index.get('created_at', 'N/A')}")
        print(f"  Total frames:  {index.get('total_frames', 0):,}")
        print(f"  Total episodes:{index.get('total_episodes', 0)}")
        cfg = index.get("config", {})
        print(f"  Cameras:       {len(cfg.get('cameras', []))}")
        print(f"  Frame rate:    {cfg.get('frame_sample_rate', 1)}")

    elif cmd == "validate":
        config = PipelineDataConfig(
            episodes_glob=args.episodes,
            episodes_dir=args.episodes_dir,
            episode_index_path=args.index,
            waypoint_cache_dir=args.waypoint_cache,
            bc_batch_size=args.bc_batch_size,
        )
        mgr = PipelineDataManager(config)
        validation = mgr.validate()
        print_validation_results(validation)

    elif cmd == "status":
        config = PipelineDataConfig(
            episodes_glob=args.episodes,
            episodes_dir=args.episodes_dir,
            episode_index_path=args.index,
            waypoint_cache_dir=args.waypoint_cache,
            bc_batch_size=args.bc_batch_size,
        )
        mgr = PipelineDataManager(config)
        mgr.print_status()

    elif cmd == "dataloader-info":
        config = PipelineDataConfig(
            waypoint_cache_dir=args.waypoint_cache,
            bc_batch_size=args.bc_batch_size,
        )
        info = build_bc_dataloader_info(config)
        print(f"\n📦 BC Dataloader Info")
        print("=" * 40)
        print(f"  Batch size:          {info['batch_size']}")
        print(f"  Num workers:         {info['num_workers']}")
        print(f"  Shuffle:             {info['shuffle']}")
        print(f"  Available episodes:  {info['available_episodes']}")
        print(f"  Total samples:       {info['total_samples']:,}")
        print(f"  Batches/epoch:       {info['estimated_batches_per_epoch']:,}")
        if info["episodes"]:
            print(f"\n  Sample episodes:")
            for ep in info["episodes"]:
                print(f"    {ep['episode_id']}: {ep['samples']} samples")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
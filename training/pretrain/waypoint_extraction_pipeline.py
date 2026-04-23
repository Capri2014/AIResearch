"""Waypoint Extraction Pipeline - Connect Episode Index to Waypoint Cache.

This pipeline extracts waypoints from indexed episodes and populates the waypoint cache
with proper metadata for BC training. Integrates with PipelineDataManager's episode index.

Usage:
    python training/pretrain/waypoint_extraction_pipeline.py extract \
        --index data/waymo/episode_index.json \
        --output-dir data/waymo/waypoint_cache
    
    python training/pretrain/waypoint_extraction_pipeline.py validate \
        --cache-dir data/waymo/waypoint_cache
    
    python training/pretrain/waypoint_extraction_pipeline.py status \
        --cache-dir data/waymo/waypoint_cache
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

import torch
import numpy as np
from torch import Tensor


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class WaypointExtractionConfig:
    """Configuration for waypoint extraction."""
    # Input
    index_path: str = "data/waymo/episode_index.json"
    episodes_dir: Optional[str] = None  # Override episode dir from index
    
    # Output
    output_dir: str = "data/waymo/waypoint_cache"
    
    # Extraction
    batch_size: int = 32
    num_workers: int = 4
    max_waypoints: int = 8  # Max waypoints per frame
    horizon: float = 3.0  # Seconds into future
    sampling_rate: float = 0.5  # Sample every N seconds
    
    # Quality control
    min_waypoint_confidence: float = 0.5
    interpolate_missing: bool = True
    
    # Output
    save_every: int = 100  # Save checkpoint every N episodes
    verbose: bool = True


@dataclass
class EpisodeWaypoints:
    """Waypoints for a single episode."""
    episode_id: str
    frame_count: int
    waypoints: list[Tensor]  # [N, max_waypoints, 2] per frame
    timestamps: list[float]
    confidences: list[Tensor]
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of waypoint extraction for an episode."""
    episode_id: str
    success: bool
    frames_processed: int
    waypoints_extracted: int
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class CacheMetadata:
    """Metadata for the waypoint cache."""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    total_episodes: int = 0
    total_frames: int = 0
    total_waypoints: int = 0
    cache_version: str = "1.0"
    index_path: str = ""
    extraction_config: dict = field(default_factory=dict)


# =============================================================================
# Waypoint Extractor
# =============================================================================

class WaypointExtractorProtocol(Protocol):
    """Protocol for waypoint extraction backends."""
    
    def extract_frame_waypoints(self, frame: dict) -> Optional[tuple[Tensor, Tensor]]:
        """Extract waypoints from a single frame.
        
        Returns:
            Tuple of (waypoints [max_waypoints, 2], confidences [max_waypoints])
            or None if extraction failed.
        """
        ...


class MockWaypointExtractor:
    """Mock waypoint extractor for testing/demonstration."""
    
    def __init__(self, config: WaypointExtractionConfig):
        self.config = config
        self.rng = np.random.RandomState(42)
    
    def extract_frame_waypoints(
        self, 
        frame: dict
    ) -> Optional[tuple[Tensor, Tensor]]:
        """Extract mock waypoints from frame."""
        
        # Generate mock waypoints based on frame info
        episode_id = frame.get("episode_id", "unknown")
        frame_idx = frame.get("frame_index", 0)
        
        # Deterministic mock based on episode/frame
        rng = np.random.RandomState(hash(f"{episode_id}_{frame_idx}") % 2**31)
        
        num_waypoints = rng.randint(3, self.config.max_waypoints + 1)
        
        # Generate waypoints in smooth trajectory
        if num_waypoints > 0:
            # Start position (mock: center of frame)
            start_x = 0.5 + rng.randn() * 0.05
            start_y = 0.5 + rng.randn() * 0.05
            
            # Generate waypoints with small noise
            waypoints = []
            confidences = []
            
            for i in range(num_waypoints):
                # Slight progression toward goal
                t = (i + 1) / num_waypoints
                wp_x = start_x + t * 0.2 + rng.randn() * 0.02
                wp_y = start_y + t * 0.1 + rng.randn() * 0.02
                
                waypoints.append([wp_x, wp_y])
                confidences.append(0.7 + rng.rand() * 0.25)  # 0.7-0.95
        
        else:
            waypoints = [[0.5, 0.5]] * self.config.max_waypoints
            confidences = [0.0] * self.config.max_waypoints
        
        # Pad to max_waypoints
        while len(waypoints) < self.config.max_waypoints:
            waypoints.append([0.5, 0.5])
            confidences.append(0.0)
        
        return (
            torch.tensor(waypoints, dtype=torch.float32),
            torch.tensor(confidences, dtype=torch.float32)
        )


class WaypointExtractionPipeline:
    """Pipeline for extracting waypoints from indexed episodes."""
    
    def __init__(self, config: WaypointExtractionConfig):
        self.config = config
        self.extractor = MockWaypointExtractor(config)
        self.output_dir = Path(config.output_dir)
        self._lock = threading.Lock()
        
        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Core Extraction
    # -------------------------------------------------------------------------
    
    def load_episode_index(self) -> list[dict]:
        """Load episode index from JSON."""
        index_path = Path(self.config.index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Episode index not found: {index_path}")
        
        with open(index_path) as f:
            data = json.load(f)
        
        # Handle different index formats
        if "episodes" in data:
            return data["episodes"]
        elif "frames" in data:
            # Group frames by episode
            frames = data["frames"]
            episode_map = {}
            for frame in frames:
                ep_id = frame.get("episode_id")
                if ep_id not in episode_map:
                    episode_map[ep_id] = []
                episode_map[ep_id].append(frame)
            return [{"episode_id": ep_id, "frames": frames} for ep_id, frames in episode_map.items()]
        else:
            return []
    
    def extract_episode(
        self, 
        episode: dict,
        verbose: bool = True
    ) -> ExtractionResult:
        """Extract waypoints for a single episode."""
        
        episode_id = episode.get("episode_id", episode.get("id", "unknown"))
        frame_count = episode.get("frame_count", 0)
        
        if frame_count == 0:
            # Check for frames array
            frames = episode.get("frames", [])
            frame_count = len(frames) if frames else 0
        
        if frame_count == 0:
            return ExtractionResult(
                episode_id=episode_id,
                success=False,
                frames_processed=0,
                waypoints_extracted=0,
                errors=["No frames in episode"]
            )
        
        start_time = datetime.now()
        all_waypoints = []
        all_confidences = []
        timestamps = []
        errors = []
        
        # Extract waypoints for each frame index
        for frame_idx in range(frame_count):
            frame = {"episode_id": episode_id, "frame_index": frame_idx}
            try:
                result = self.extractor.extract_frame_waypoints(frame)
                if result is not None:
                    waypoints, confidences = result
                    
                    all_waypoints.append(waypoints)
                    all_confidences.append(confidences)
                    timestamps.append(float(frame_idx * 0.1))  # 10 FPS
                    
                else:
                    errors.append(f"Frame {frame_idx} extraction failed")
                    
            except Exception as e:
                errors.append(f"Frame {frame_idx} error: {e}")
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Save episode waypoints
        if all_waypoints:
            episode_output = {
                "episode_id": episode_id,
                "frame_count": len(all_waypoints),
                "waypoints_dim": list(all_waypoints[0].shape),
                "timestamps": timestamps,
                "waypoints": [
                    wp.cpu().numpy().tolist() for wp in all_waypoints
                ],
                "confidences": [
                    conf.cpu().numpy().tolist() for conf in all_confidences
                ]
            }
            
            output_file = self.output_dir / f"{episode_id}.json"
            with open(output_file, "w") as f:
                json.dump(episode_output, f, indent=2)
        
        return ExtractionResult(
            episode_id=episode_id,
            success=len(all_waypoints) > 0,
            frames_processed=frame_count,
            waypoints_extracted=len(all_waypoints),
            errors=errors,
            duration_ms=duration_ms
        )
    
    def extract_all(
        self,
        indices: Optional[list[dict]] = None,
        verbose: Optional[bool] = None
    ) -> list[ExtractionResult]:
        """Extract waypoints for all indexed episodes."""
        
        if verbose is None:
            verbose = self.config.verbose
        
        if indices is None:
            indices = self.load_episode_index()
        
        results = []
        
        for i, episode in enumerate(indices):
            ep_id = episode.get("episode_id", f"episode_{i}")
            
            if verbose:
                print(f"[{i+1}/{len(indices)}] Extracting: {ep_id}")
            
            result = self.extract_episode(episode, verbose=verbose)
            results.append(result)
            
            if verbose:
                status = "✓" if result.success else "✗"
                print(f"  {status} {result.waypoints_extracted}/{result.frames_processed} frames "
                      f"({result.duration_ms:.0f}ms)")
        
        # Update cache metadata
        self._update_metadata(results)
        
        return results
    
    def _update_metadata(self, results: list[ExtractionResult]):
        """Update cache metadata after extraction."""
        
        total_frames = sum(r.frames_processed for r in results)
        total_waypoints = sum(r.waypoints_extracted for r in results)
        
        metadata = CacheMetadata(
            last_updated=datetime.now().isoformat(),
            total_episodes=len([r for r in results if r.success]),
            total_frames=total_frames,
            total_waypoints=total_waypoints,
            index_path=self.config.index_path,
            extraction_config={
                "max_waypoints": self.config.max_waypoints,
                "horizon": self.config.horizon,
                "sampling_rate": self.config.sampling_rate
            }
        )
        
        metadata_path = self.output_dir / "metadata.json"
        
        # Load existing if present
        existing = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                existing = json.load(f)
        
        # Merge
        metadata.created_at = existing.get("created_at", metadata.created_at)
        metadata.total_episodes = metadata.total_episodes + existing.get("total_episodes", 0)
        
        with open(metadata_path, "w") as f:
            json.dump({
                "created_at": metadata.created_at,
                "last_updated": metadata.last_updated,
                "total_episodes": metadata.total_episodes,
                "total_frames": total_frames,
                "total_waypoints": total_waypoints,
                "cache_version": metadata.cache_version,
                "index_path": metadata.index_path,
                "extraction_config": metadata.extraction_config
            }, f, indent=2)
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    def validate_cache(self) -> dict:
        """Validate waypoint cache readiness."""
        
        metadata_path = self.output_dir / "metadata.json"
        
        if not metadata_path.exists():
            return {
                "valid": False,
                "status": "MISSING",
                "message": "Waypoint cache not initialized",
                "recommendations": ["Run extraction first: waypoint_extraction_pipeline.py extract"]
            }
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load episode index for comparison
        index = self.load_episode_index()
        index_episodes = len(index)
        
        cache_episodes = metadata.get("total_episodes", 0)
        
        # Check completeness
        if cache_episodes < index_episodes:
            return {
                "valid": False,
                "status": "INCOMPLETE",
                "message": f"Cache has {cache_episodes}/{index_episodes} episodes",
                "recommendations": [
                    f"Re-extract to fill gaps: {index_episodes - cache_episodes} episodes missing"
                ],
                "metadata": metadata
            }
        
        # Check minimum frames
        min_frames = 500  # Minimum for BC training
        total_frames = metadata.get("total_frames", 0)
        
        if total_frames < min_frames:
            return {
                "valid": False,
                "status": "INSUFFICIENT",
                "message": f"Only {total_frames} frames, need {min_frames} for BC training",
                "recommendations": ["Add more episodes or augment data"],
                "metadata": metadata
            }
        
        return {
            "valid": True,
            "status": "READY",
            "message": f"Waypoint cache ready: {cache_episodes} episodes, {total_frames} frames",
            "metadata": metadata
        }
    
    def print_cache_status(self):
        """Print formatted cache status."""
        
        validation = self.validate_cache()
        
        print("=" * 60)
        print("Waypoint Cache Status")
        print("=" * 60)
        print(f"  Status:    {validation['status']}")
        print(f"  Valid:    {validation['valid']}")
        print(f"  Message:  {validation['message']}")
        
        if "metadata" in validation:
            meta = validation["metadata"]
            print(f"  Episodes: {meta.get('total_episodes', 0)}")
            print(f"  Frames:   {meta.get('total_frames', 0)}")
            print(f"  Waypoints:{meta.get('total_waypoints', 0)}")
            
            if meta.get("last_updated"):
                print(f"  Updated:  {meta['last_updated'][:19]}")
        
        print("-" * 60)
        
        if validation.get("recommendations"):
            print("Recommendations:")
            for rec in validation["recommendations"]:
                print(f"  • {rec}")
        
        print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Waypoint Extraction Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract waypoints from episodes")
    extract_parser.add_argument(
        "--index", 
        default="data/waymo/episode_index.json",
        help="Episode index JSON path"
    )
    extract_parser.add_argument(
        "--output-dir",
        default="data/waymo/waypoint_cache",
        help="Output directory for waypoint cache"
    )
    extract_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for extraction"
    )
    extract_parser.add_argument(
        "--max-waypoints",
        type=int,
        default=8,
        help="Maximum waypoints per frame"
    )
    extract_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate waypoint cache")
    validate_parser.add_argument(
        "--cache-dir",
        default="data/waymo/waypoint_cache",
        help="Waypoint cache directory"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show cache status")
    status_parser.add_argument(
        "--cache-dir",
        default="data/waymo/waypoint_cache",
        help="Waypoint cache directory"
    )
    
    args = parser.parse_args()
    
    if args.command == "extract":
        config = WaypointExtractionConfig(
            index_path=args.index,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_waypoints=args.max_waypoints,
            verbose=args.verbose
        )
        
        pipeline = WaypointExtractionPipeline(config)
        
        print(f"Extracting waypoints from: {args.index}")
        print(f"Output directory: {args.output_dir}")
        print("-" * 40)
        
        results = pipeline.extract_all(verbose=args.verbose)
        
        # Summary
        success = sum(1 for r in results if r.success)
        total_frames = sum(r.frames_processed for r in results)
        
        print("-" * 40)
        print(f"Extracted: {success}/{len(results)} episodes, {total_frames} frames")
        
        # Print status
        pipeline.print_cache_status()
    
    elif args.command == "validate":
        config = WaypointExtractionConfig(
            output_dir=args.cache_dir
        )
        
        pipeline = WaypointExtractionPipeline(config)
        validation = pipeline.validate_cache()
        
        print(f"Status: {validation['status']}")
        print(f"Valid:  {validation['valid']}")
        print(f"Message: {validation['message']}")
        
        if validation.get("recommendations"):
            print("\nRecommendations:")
            for rec in validation["recommendations"]:
                print(f"  • {rec}")
    
    elif args.command == "status":
        config = WaypointExtractionConfig(
            output_dir=args.cache_dir
        )
        
        pipeline = WaypointExtractionPipeline(config)
        pipeline.print_cache_status()


if __name__ == "__main__":
    main()
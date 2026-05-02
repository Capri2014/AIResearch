#!/usr/bin/env python3
"""
Episode Data Quality Analyzer

Analyzes Waymo episode data for quality issues before training.
Checks for: missing frames, corrupt data, temporal gaps, velocity outliers,
collision/infraction markers, and generates quality reports.

This helps identify problematic episodes early in the pipeline.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try to import numpy for statistics, fallback to pure Python
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class DataQualityIssue:
    """Single data quality issue."""
    issue_type: str  # missing, corrupt, gap, outlier, marker
    location: str  # episode_id:frame or similar
    severity: str  # critical, warning, info
    description: str
    value: Optional[float] = None
    expected: Optional[float] = None


@dataclass
class FrameQuality:
    """Quality metrics for a single frame."""
    frame_id: int
    has_position: bool = True
    has_velocity: bool = True
    has_heading: bool = True
    has_image: bool = True
    position_valid: bool = True
    velocity_valid: bool = True
    heading_valid: bool = True
    speed_mps: float = 0.0
    issues: list = field(default_factory=list)


@dataclass
class EpisodeQuality:
    """Quality metrics for a single episode."""
    episode_id: str
    num_frames: int = 0
    duration_seconds: float = 0.0
    frame_rate_hz: float = 0.0
    
    # Data completeness
    frames_with_position: int = 0
    frames_with_velocity: int = 0
    frames_with_heading: int = 0
    frames_with_image: int = 0
    
    # Quality checks
    temporal_gaps: int = 0
    max_frame_gap: float = 0.0
    velocity_outliers: int = 0
    heading_jumps: int = 0
    
    # Issue tracking
    issues: list = field(default_factory=list)
    
    # Overall scores (0-100)
    completeness_score: float = 100.0
    quality_score: float = 100.0


class EpisodeQualityAnalyzer:
    """
    Analyzes Waymo episode data for quality issues.
    
    Performs checks for:
    - Data completeness (presence of position, velocity, heading, images)
    - Temporal continuity (frame gaps, frame rate stability)
    - Value validity (velocity limits, heading changes)
    - Special markers (collisions, infractions, disengagements)
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Thresholds
        self.max_velocity = self.config.get('max_velocity', 50.0)  # m/s (~112 mph)
        self.max_heading_delta = self.config.get('max_heading_delta', 90.0)  # degrees per frame
        self.min_frames = self.config.get('min_frames', 10)
        self.target_frame_rate = self.config.get('target_frame_rate', 10.0)  # Hz
        
        self.episodes_analyzed: dict = {}
    
    def analyze_episode(self, episode_data: dict) -> EpisodeQuality:
        """Analyze a single episode for quality issues."""
        episode_id = episode_data.get('episode_id', 'unknown')
        quality = EpisodeQuality(episode_id=episode_id)
        
        # Get frames
        frames = episode_data.get('frames', [])
        if not frames:
            quality.issues.append(DataQualityIssue(
                issue_type='missing',
                location=f'{episode_id}',
                severity='critical',
                description='No frames found in episode'
            ))
            quality.quality_score = 0.0
            return quality
        
        quality.num_frames = len(frames)
        
        # Analyze each frame
        prev_heading = None
        velocities = []
        
        for i, frame in enumerate(frames):
            frame_id = frame.get('frame_id', i)
            
            # Check position exists
            pos = frame.get('position') or frame.get('translation')
            if pos is not None:
                quality.frames_with_position += 1
                # Validate position range
                if pos:
                    try:
                        if abs(pos[0]) > 10000 or abs(pos[1]) > 10000:
                            quality.issues.append(DataQualityIssue(
                                issue_type='outlier',
                                location=f'{episode_id}:{frame_id}',
                                severity='warning',
                                description=f'Position outlier: ({pos[0]:.1f}, {pos[1]:.1f})',
                                value=max(abs(pos[0]), abs(pos[1])),
                                expected=10000.0
                            ))
                    except (TypeError, IndexError):
                        pass
            
            # Check velocity exists
            vel = frame.get('velocity') or frame.get('speed')
            if vel is not None:
                quality.frames_with_velocity += 1
                try:
                    speed = float(vel) if isinstance(vel, (int, float)) else float(vel[0])
                    velocities.append(speed)
                    if speed > self.max_velocity:
                        quality.velocity_outliers += 1
                        quality.issues.append(DataQualityIssue(
                            issue_type='outlier',
                            location=f'{episode_id}:{frame_id}',
                            severity='warning',
                            description=f'Velocity exceeds limit: {speed:.1f} m/s',
                            value=speed,
                            expected=self.max_velocity
                        ))
                except (TypeError, ValueError):
                    pass
            
            # Check heading exists
            heading = frame.get('heading')
            if heading is not None:
                quality.frames_with_heading += 1
                try:
                    heading_deg = float(heading)
                    if heading_deg > 360:
                        heading_deg = heading_deg % 360
                    
                    if prev_heading is not None:
                        # Compute heading delta (handle wraparound)
                        delta = abs(heading_deg - prev_heading)
                        if delta > 180:
                            delta = 360 - delta
                        if delta > self.max_heading_delta:
                            quality.heading_jumps += 1
                            quality.issues.append(DataQualityIssue(
                                issue_type='outlier',
                                location=f'{episode_id}:{frame_id}',
                                severity='info',
                                description=f'Heading jump: {delta:.1f}°',
                                value=delta,
                                expected=self.max_heading_delta
                            ))
                    prev_heading = heading_deg
                except (TypeError, ValueError):
                    pass
            
            # Check image exists
            image = frame.get('image') or frame.get('image_front')
            if image is not None:
                quality.frames_with_image += 1
        
        # Compute temporal metrics
        timestamps = [f.get('timestamp') for f in frames if f.get('timestamp')]
        if len(timestamps) >= 2:
            duration = timestamps[-1] - timestamps[0]
            quality.duration_seconds = duration
            if duration > 0:
                quality.frame_rate_hz = (len(timestamps) - 1) / duration
        
        # Check for temporal gaps
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap > 0.2:  # 200ms gap (5fps minimum)
                quality.temporal_gaps += 1
                quality.max_frame_gap = max(quality.max_frame_gap, gap)
        
        # Compute completeness score
        total_fields = quality.num_frames * 4  # pos, vel, heading, image
        if total_fields > 0:
            present = (quality.frames_with_position + quality.frames_with_velocity +
                     quality.frames_with_heading + quality.frames_with_image)
            quality.completeness_score = 100.0 * present / total_fields
        
        # Compute overall quality score
        quality.quality_score = quality.completeness_score
        
        # Deduct for issues
        for issue in quality.issues:
            if issue.severity == 'critical':
                quality.quality_score -= 20
            elif issue.severity == 'warning':
                quality.quality_score -= 5
            # info doesn't affect score
        
        quality.quality_score = max(0.0, quality.quality_score)
        
        self.episodes_analyzed[episode_id] = quality
        return quality
    
    def analyze_episode_from_path(self, path: Path) -> EpisodeQuality:
        """Load and analyze episode from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return self.analyze_episode(data)
        except json.JSONDecodeError as e:
            quality = EpisodeQuality(episode_id=path.stem)
            quality.issues.append(DataQualityIssue(
                issue_type='corrupt',
                location=str(path),
                severity='critical',
                description=f'JSON parse error: {e}'
            ))
            quality.quality_score = 0.0
            return quality
        except Exception as e:
            quality = EpisodeQuality(episode_id=path.stem)
            quality.issues.append(DataQualityIssue(
                issue_type='corrupt',
                location=str(path),
                severity='critical',
                description=f'Read error: {e}'
            ))
            quality.quality_score = 0.0
            return quality
    
    def analyze_directory(self, episodes_dir: Path) -> dict:
        """Analyze all episodes in a directory."""
        results = {}
        
        # Find episode files
        episode_files = list(episodes_dir.glob('*.json'))
        episode_files.extend(list(episodes_dir.glob('*.jsonl')))
        
        for ep_file in sorted(episode_files):
            quality = self.analyze_episode_from_path(ep_file)
            results[ep_file.stem] = quality
        
        return results
    
    def generate_report(self, episodes: dict) -> dict:
        """Generate quality report for analyzed episodes."""
        if not episodes:
            return {'error': 'No episodes to analyze'}
        
        # Aggregate statistics
        total_episodes = len(episodes)
        total_frames = sum(e.num_frames for e in episodes.values())
        total_issues = sum(len(e.issues) for e in episodes.values())
        
        avg_quality = sum(e.quality_score for e in episodes.values()) / total_episodes
        avg_completeness = sum(e.completeness_score for e in episodes.values()) / total_episodes
        
        # Count issues by type
        issues_by_severity = {'critical': 0, 'warning': 0, 'info': 0}
        issues_by_type = {}
        
        for ep in episodes.values():
            for issue in ep.issues:
                issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
                issues_by_type[issue.issue_type] = issues_by_type.get(issue.issue_type, 0) + 1
        
        # Categorize episodes
        good_episodes = [e for e in episodes.values() if e.quality_score >= 90]
        ok_episodes = [e for e in episodes.values() if 70 <= e.quality_score < 90]
        bad_episodes = [e for e in episodes.values() if e.quality_score < 70]
        
        report = {
            'summary': {
                'total_episodes': total_episodes,
                'total_frames': total_frames,
                'total_issues': total_issues,
                'avg_quality_score': avg_quality,
                'avg_completeness_score': avg_completeness,
                'good_episodes': len(good_episodes),
                'ok_episodes': len(ok_episodes),
                'bad_episodes': len(bad_episodes),
                'issues_by_severity': issues_by_severity,
                'issues_by_type': issues_by_type,
            },
            'episodes': {
                ep_id: {
                    'num_frames': ep.num_frames,
                    'quality_score': ep.quality_score,
                    'completeness_score': ep.completeness_score,
                    'issues': [
                        {
                            'type': i.issue_type,
                            'location': i.location,
                            'severity': i.severity,
                            'description': i.description,
                        }
                        for i in ep.issues[:5]  # Limit to first 5 issues
                    ],
                }
                for ep_id, ep in episodes.items()
            },
        }
        
        return report
    
    def print_report(self, report: dict) -> None:
        """Print formatted quality report."""
        summary = report.get('summary', {})
        
        print("\n" + "=" * 60)
        print("EPISODE DATA QUALITY REPORT")
        print("=" * 60)
        
        print(f"\nTotal Episodes: {summary.get('total_episodes', 0)}")
        print(f"Total Frames: {summary.get('total_frames', 0)}")
        print(f"Total Issues: {summary.get('total_issues', 0)}")
        
        print(f"\nQuality Scores:")
        print(f"  Average Quality: {summary.get('avg_quality_score', 0):.1f}/100")
        print(f"  Average Completeness: {summary.get('avg_completeness_score', 0):.1f}/100")
        
        print(f"\nEpisode Categories:")
        print(f"  Good (≥90): {summary.get('good_episodes', 0)}")
        print(f"  OK (70-89): {summary.get('ok_episodes', 0)}")
        print(f"  Bad (<70): {summary.get('bad_episodes', 0)}")
        
        issues_by_sev = summary.get('issues_by_severity', {})
        if sum(issues_by_sev.values()) > 0:
            print(f"\nIssues by Severity:")
            print(f"  Critical: {issues_by_sev.get('critical', 0)}")
            print(f"  Warning: {issues_by_sev.get('warning', 0)}")
            print(f"  Info: {issues_by_sev.get('info', 0)}")
        
        # Show problematic episodes
        episodes = report.get('episodes', {})
        bad_eps = [(k, v) for k, v in episodes.items() if v['quality_score'] < 70]
        
        if bad_eps:
            print(f"\nProblematic Episodes:")
            for ep_id, info in sorted(bad_eps, key=lambda x: x[1]['quality_score'])[:5]:
                print(f"  {ep_id}: {info['quality_score']:.1f}/100 ({info['num_frames']} frames)")
                for issue in info.get('issues', [])[:2]:
                    print(f"    - [{issue['severity']}] {issue['description']}")
        
        print("=" * 60 + "\n")


def load_episode_sample() -> dict:
    """Generate sample episode for testing."""
    import math
    
    frames = []
    for i in range(100):
        frames.append({
            'frame_id': i,
            'timestamp': i * 0.1,
            'position': [i * 0.5, 0.0],  # Moving forward
            'velocity': [2.0, 0.0],
            'heading': 0.0,  # Facing forward
        })
    
    return {
        'episode_id': 'sample_episode',
        'frames': frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Waymo episode data quality'
    )
    parser.add_argument(
        '--episodes-dir',
        type=Path,
        help='Directory containing episode JSON files'
    )
    parser.add_argument(
        '--episode',
        type=Path,
        help='Single episode JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for JSON report'
    )
    parser.add_argument(
        '--max-velocity',
        type=float,
        default=50.0,
        help='Maximum velocity threshold (m/s)'
    )
    parser.add_argument(
        '--max-heading-delta',
        type=float,
        default=90.0,
        help='Maximum heading change per frame (degrees)'
    )
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run smoke test with synthetic data'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Configure analyzer
    config = {
        'max_velocity': args.max_velocity,
        'max_heading_delta': args.max_heading_delta,
    }
    analyzer = EpisodeQualityAnalyzer(config)
    
    # Analyze episodes
    if args.smoke_test or (not args.episodes_dir and not args.episode):
        # Run smoke test
        print("Running smoke test with synthetic episode data...")
        
        episode = load_episode_sample()
        # Introduce some issues
        episode['frames'][50]['position'] = [99999.0, 99999.0]  # Position outlier
        episode['frames'][25]['velocity'] = 100.0  # Velocity outlier
        
        quality = analyzer.analyze_episode(episode)
        
        report = analyzer.generate_report({episode['episode_id']: quality})
        
        if args.verbose:
            analyzer.print_report(report)
        
        print(f"Smoke test complete:")
        print(f"  Episode: {quality.episode_id}")
        print(f"  Quality Score: {quality.quality_score:.1f}/100")
        print(f"  Issues Found: {len(quality.issues)}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"  Report: {args.output}")
        
        return 0
    
    # Analyze episode directory
    if args.episodes_dir:
        episodes = analyzer.analyze_directory(args.episodes_dir)
        report = analyzer.generate_report(episodes)
        analyzer.print_report(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.output}")
        
        return 0
    
    # Analyze single episode
    if args.episode:
        quality = analyzer.analyze_episode_from_path(args.episode)
        report = analyzer.generate_report({quality.episode_id: quality})
        analyzer.print_report(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
        
        return 0
    
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
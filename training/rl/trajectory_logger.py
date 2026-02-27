"""
Episode Trajectory Logger for RL Training.

Records detailed trajectory information for each episode including:
- State, action, reward at each timestep
- SFT vs RL waypoints comparison
- Goal reach status and episode metrics
- Collision/timeout information

Usage:
    from trajectory_logger import TrajectoryLogger
    
    logger = TrajectoryLogger(output_dir='out/trajectories')
    
    # During training
    logger.start_episode(episode_num)
    logger.log_step(state, action, reward, sft_waypoints, rl_waypoints)
    logger.end_episode(episode_reward, goal_reached, info)
    
    # After training
    logger.save()
    logger.get_summary()
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np


class TrajectoryLogger:
    """
    Logger for recording detailed episode trajectories.
    
    Useful for:
    - Debugging training issues
    - Analyzing SFT vs RL behavior differences
    - Computing custom metrics offline
    - Generating visualization datasets
    """
    
    def __init__(
        self,
        output_dir: str = 'out/trajectories',
        max_episodes: int = 1000,
        save_frequency: int = 50,
        record_states: bool = True,
        record_waypoints: bool = True,
        record_rewards: bool = True,
    ):
        """
        Initialize trajectory logger.
        
        Args:
            output_dir: Directory to save trajectory files
            max_episodes: Maximum episodes to keep in memory before flushing
            save_frequency: Save to disk every N episodes
            record_states: Whether to record full state trajectories
            record_waypoints: Whether to record SFT and RL waypoints
            record_rewards: Whether to record reward components
        """
        self.output_dir = output_dir
        self.max_episodes = max_episodes
        self.save_frequency = save_frequency
        self.record_states = record_states
        self.record_waypoints = record_waypoints
        self.record_rewards = record_rewards
        
        os.makedirs(output_dir, exist_ok=True)
        
        # In-memory storage
        self.episodes: List[Dict[str, Any]] = []
        self.current_episode: Optional[Dict[str, Any]] = None
        self.episode_count = 0
        
        # Summary statistics
        self.stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'total_reward': 0.0,
            'avg_episode_length': 0.0,
        }
    
    def start_episode(self, episode_num: int):
        """Start recording a new episode."""
        self.current_episode = {
            'episode_num': episode_num,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'sft_waypoints': [],
            'rl_waypoints': [],
            'final_state': None,
            'episode_reward': 0.0,
            'goal_reached': False,
            'terminated': False,
            'truncated': False,
        }
    
    def log_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        sft_waypoints: Optional[np.ndarray] = None,
        rl_waypoints: Optional[np.ndarray] = None,
        reward_components: Optional[Dict[str, float]] = None,
    ):
        """
        Log a single timestep within the current episode.
        
        Args:
            state: Current state vector
            action: Action taken
            reward: Total reward received
            sft_waypoints: SFT model waypoints (if available)
            rl_waypoints: RL model waypoints (if available)
            reward_components: Dictionary of reward components
        """
        if self.current_episode is None:
            raise RuntimeError("Must call start_episode() before log_step()")
        
        step_data = {'timestep': len(self.current_episode['steps'])}
        
        if self.record_states:
            step_data['state'] = state.tolist() if isinstance(state, np.ndarray) else list(state)
        
        step_data['action'] = action.tolist() if isinstance(action, np.ndarray) else list(action)
        
        if self.record_rewards:
            step_data['reward'] = float(reward)
            if reward_components:
                step_data['reward_components'] = reward_components
        
        if self.record_waypoints:
            if sft_waypoints is not None:
                step_data['sft_waypoints'] = sft_waypoints.tolist() if isinstance(sft_waypoints, np.ndarray) else sft_waypoints
            if rl_waypoints is not None:
                step_data['rl_waypoints'] = rl_waypoints.tolist() if isinstance(rl_waypoints, np.ndarray) else rl_waypoints
        
        self.current_episode['steps'].append(step_data)
        
        # Also keep lists for easy access
        if self.record_states:
            self.current_episode['states'].append(state)
        self.current_episode['actions'].append(action)
        if self.record_rewards:
            self.current_episode['rewards'].append(reward)
        if self.record_waypoints:
            if sft_waypoints is not None:
                self.current_episode['sft_waypoints'].append(sft_waypoints)
            if rl_waypoints is not None:
                self.current_episode['rl_waypoints'].append(rl_waypoints)
    
    def end_episode(
        self,
        episode_reward: float,
        goal_reached: bool = False,
        info: Optional[Dict[str, Any]] = None,
        terminated: bool = False,
        truncated: bool = False,
    ):
        """
        Finish recording the current episode.
        
        Args:
            episode_reward: Total episode reward
            goal_reached: Whether the agent reached the goal
            info: Additional episode info (collision, timeout, etc.)
            terminated: Whether episode terminated naturally
            truncated: Whether episode was truncated (timeout)
        """
        if self.current_episode is None:
            raise RuntimeError("Must call start_episode() before end_episode()")
        
        self.current_episode['end_time'] = datetime.now().isoformat()
        self.current_episode['episode_reward'] = float(episode_reward)
        self.current_episode['goal_reached'] = bool(goal_reached)
        self.current_episode['terminated'] = bool(terminated)
        self.current_episode['truncated'] = bool(truncated)
        self.current_episode['episode_length'] = len(self.current_episode['steps'])
        
        if info:
            self.current_episode['info'] = info
        
        # Compute ADE if we have waypoints
        if self.current_episode['sft_waypoints'] and self.current_episode['rl_waypoints']:
            sft_wp = np.array(self.current_episode['sft_waypoints'])
            rl_wp = np.array(self.current_episode['rl_waypoints'])
            
            # Average displacement error
            if len(sft_wp) > 0 and len(rl_wp) > 0:
                min_len = min(len(sft_wp), len(rl_wp))
                ade = np.mean(np.abs(sft_wp[:min_len] - rl_wp[:min_len]))
                self.current_episode['ade_sft_rl'] = float(ade)
        
        # Add to episodes list
        self.episodes.append(self.current_episode)
        self.episode_count += 1
        
        # Update stats
        self.stats['total_episodes'] += 1
        if goal_reached:
            self.stats['successful_episodes'] += 1
        self.stats['total_reward'] += episode_reward
        
        # Clear current episode
        self.current_episode = None
        
        # Periodic save
        if self.episode_count % self.save_frequency == 0:
            self._save_to_disk()
    
    def _save_to_disk(self):
        """Save episodes to disk (incremental)."""
        if not self.episodes:
            return
        
        # Save latest episodes
        filepath = os.path.join(
            self.output_dir,
            f'trajectories_{self.episode_count // self.save_frequency * self.save_frequency:06d}.json'
        )
        
        with open(filepath, 'w') as f:
            json.dump(self.episodes, f, indent=2)
        
        print(f"Saved {len(self.episodes)} trajectories to {filepath}")
        
        # Clear memory (keep last batch)
        self.episodes = []
    
    def save(self):
        """Save all remaining episodes to disk."""
        if self.episodes:
            filepath = os.path.join(
                self.output_dir,
                f'trajectories_final_{self.episode_count:06d}.json'
            )
            with open(filepath, 'w') as f:
                json.dump(self.episodes, f, indent=2)
            print(f"Saved final {len(self.episodes)} trajectories to {filepath}")
            self.episodes = []
        
        # Save summary stats
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        print(f"Saved summary to {summary_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self.stats['total_episodes'] > 0:
            self.stats['avg_reward'] = self.stats['total_reward'] / self.stats['total_episodes']
            self.stats['success_rate'] = self.stats['successful_episodes'] / self.stats['total_episodes']
        
        self.stats['episode_count'] = self.episode_count
        self.stats['output_dir'] = self.output_dir
        
        return self.stats.copy()
    
    def get_episode(self, episode_num: int) -> Optional[Dict[str, Any]]:
        """Get a specific episode by number (from disk if not in memory)."""
        # For now, only return if in memory
        for ep in self.episodes:
            if ep['episode_num'] == episode_num:
                return ep
        return None
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.episodes:
            self._save_to_disk()


class TrajectoryAnalyzer:
    """
    Analyze recorded trajectories for insights.
    
    Methods for:
    - Computing trajectory-level metrics
    - Comparing SFT vs RL behavior
    - Finding failure modes
    - Generating training insights
    """
    
    def __init__(self, trajectory_dir: str):
        """
        Initialize analyzer with trajectory directory.
        
        Args:
            trajectory_dir: Directory containing trajectory files
        """
        self.trajectory_dir = trajectory_dir
        self.episodes: List[Dict[str, Any]] = []
        self._load_all()
    
    def _load_all(self):
        """Load all trajectory files."""
        for filename in os.listdir(self.trajectory_dir):
            if filename.endswith('.json') and filename.startswith('trajectories'):
                filepath = os.path.join(self.trajectory_dir, filename)
                with open(filepath, 'r') as f:
                    episodes = json.load(f)
                    self.episodes.extend(episodes)
    
    def get_success_rate(self) -> float:
        """Compute overall success rate."""
        if not self.episodes:
            return 0.0
        return sum(1 for ep in self.episodes if ep.get('goal_reached', False)) / len(self.episodes)
    
    def get_avg_reward(self) -> float:
        """Compute average episode reward."""
        if not self.episodes:
            return 0.0
        rewards = [ep.get('episode_reward', 0) for ep in self.episodes]
        return sum(rewards) / len(rewards)
    
    def get_avg_episode_length(self) -> float:
        """Compute average episode length."""
        if not self.episodes:
            return 0.0
        lengths = [ep.get('episode_length', 0) for ep in self.episodes]
        return sum(lengths) / len(lengths)
    
    def get_failure_modes(self) -> Dict[str, int]:
        """
        Categorize failure modes.
        
        Returns:
            Dict mapping failure type to count
        """
        failures = {
            'timeout': 0,
            'collision': 0,
            'off_road': 0,
            'unknown': 0,
        }
        
        for ep in self.episodes:
            if ep.get('goal_reached', False):
                continue
            
            info = ep.get('info', {})
            if info.get('timeout'):
                failures['timeout'] += 1
            elif info.get('collision'):
                failures['collision'] += 1
            elif info.get('off_road'):
                failures['off_road'] += 1
            else:
                failures['unknown'] += 1
        
        return failures
    
    def compare_sft_rl(self) -> Dict[str, float]:
        """
        Compare SFT and RL waypoint predictions.
        
        Returns:
            Dict with comparison metrics
        """
        ade_values = []
        
        for ep in self.episodes:
            if 'ade_sft_rl' in ep:
                ade_values.append(ep['ade_sft_rl'])
        
        if not ade_values:
            return {}
        
        return {
            'mean_ade': float(np.mean(ade_values)),
            'std_ade': float(np.std(ade_values)),
            'min_ade': float(np.min(ade_values)),
            'max_ade': float(np.max(ade_values)),
        }
    
    def generate_report(self) -> str:
        """Generate a markdown report of trajectory analysis."""
        lines = [
            "# Trajectory Analysis Report",
            "",
            f"**Total Episodes:** {len(self.episodes)}",
            f"**Success Rate:** {self.get_success_rate():.2%}",
            f"**Avg Reward:** {self.get_avg_reward():.2f}",
            f"**Avg Episode Length:** {self.get_avg_episode_length():.1f} steps",
            "",
        ]
        
        # Failure modes
        failures = self.get_failure_modes()
        if sum(failures.values()) > 0:
            lines.append("## Failure Modes")
            lines.append("")
            for mode, count in failures.items():
                lines.append(f"- {mode}: {count}")
            lines.append("")
        
        # SFT vs RL comparison
        sft_rl = self.compare_sft_rl()
        if sft_rl:
            lines.append("## SFT vs RL Comparison")
            lines.append("")
            lines.append(f"- Mean ADE: {sft_rl.get('mean_ade', 0):.4f}")
            lines.append(f"- Std ADE: {sft_rl.get('std_ade', 0):.4f}")
            lines.append("")
        
        return "\n".join(lines)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training trajectories')
    parser.add_argument('--dir', type=str, default='out/trajectories',
                        help='Trajectory directory')
    parser.add_argument('--report', action='store_true',
                        help='Generate analysis report')
    
    args = parser.parse_args()
    
    if args.report:
        analyzer = TrajectoryAnalyzer(args.dir)
        print(analyzer.generate_report())
    else:
        print(f"Trajectory directory: {args.dir}")
        print(f"Found {len(os.listdir(args.dir))} files")

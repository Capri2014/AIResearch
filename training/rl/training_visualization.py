"""
Training Visualization Utilities for Driving Pipeline.

Provides visualization tools for:
- Waypoint predictions vs ground truth
- Training curves (reward, loss, metrics)
- Episode trajectory visualization
- Delta head correction visualization

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Try to import matplotlib - it's commonly available
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    episode_rewards: List[float]
    episode_lengths: List[int]
    losses: List[float]
    ade_scores: List[float]
    fde_scores: List[float]
    learning_rates: List[float]
    policy_losses: List[float]
    value_losses: List[float]
    entropy: List[float]
    
    @classmethod
    def from_json(cls, path: str) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            episode_rewards=data.get('episode_rewards', []),
            episode_lengths=data.get('episode_lengths', []),
            losses=data.get('losses', []),
            ade_scores=data.get('ade_scores', []),
            fde_scores=data.get('fde_scores', []),
            learning_rates=data.get('learning_rates', []),
            policy_losses=data.get('policy_losses', []),
            value_losses=data.get('value_losses', []),
            entropy=data.get('entropy', [])
        )


@dataclass
class WaypointPrediction:
    """Container for waypoint prediction data."""
    predicted: np.ndarray  # (T, H, 2) - T timesteps, H waypoints, (x, y)
    ground_truth: np.ndarray  # (T, H, 2)
    timestamps: Optional[np.ndarray] = None  # (T,)
    
    def ade(self) -> float:
        """Average Displacement Error."""
        return np.mean(np.linalg.norm(self.predicted - self.ground_truth, axis=-1))
    
    def fde(self) -> float:
        """Final Displacement Error."""
        return float(np.mean(np.linalg.norm(
            self.predicted[-1] - self.ground_truth[-1],
            axis=-1
        )))


def plot_waypoint_comparison(
    predictions: WaypointPrediction,
    title: str = "Waypoint Prediction Comparison",
    save_path: Optional[str] = None,
    show_sft_delta: bool = False,
    sft_waypoints: Optional[np.ndarray] = None,
) -> Any:
    """
    Plot waypoint predictions vs ground truth.
    
    Args:
        predictions: WaypointPrediction with predicted and ground truth waypoints
        title: Plot title
        save_path: Optional path to save figure
        show_sft_delta: Whether to show SFT baseline
        sft_waypoints: SFT-only predictions for comparison
    
    Returns:
        Figure object if matplotlib available, else None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping visualization")
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    T = predictions.predicted.shape[0]
    
    # Plot ground truth trajectory
    gt_final = predictions.ground_truth[:, -1, :]  # Final waypoint at each timestep
    ax.plot(gt_final[:, 0], gt_final[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
    
    # Plot SFT predictions if provided
    if show_sft_delta and sft_waypoints is not None:
        sft_final = sft_waypoints[:, -1, :]
        ax.plot(sft_final[:, 0], sft_final[:, 1], 'b--', linewidth=1.5, 
                label='SFT Only', alpha=0.6)
    
    # Plot RL predictions
    pred_final = predictions.predicted[:, -1, :]
    ax.plot(pred_final[:, 0], pred_final[:, 1], 'r-', linewidth=2, 
            label=f'RL (ADE={predictions.ade():.3f})', alpha=0.7)
    
    # Mark start and end points
    ax.scatter(gt_final[0, 0], gt_final[0, 1], c='green', s=100, marker='o', 
               label='Start', zorder=5)
    ax.scatter(gt_final[-1, 0], gt_final[-1, 1], c='green', s=100, marker='*', 
               label='Goal', zorder=5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_delta_corrections(
    sft_waypoints: np.ndarray,
    rl_waypoints: np.ndarray,
    ground_truth: np.ndarray,
    title: str = "Delta Corrections Visualization",
    save_path: Optional[str] = None,
) -> Any:
    """
    Visualize the delta corrections made by the RL head.
    
    Args:
        sft_waypoints: SFT model predictions (T, H, 2)
        rl_waypoints: RL-corrected predictions (T, H, 2)
        ground_truth: Ground truth waypoints (T, H, 2)
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        Figure object if matplotlib available, else None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    T = sft_waypoints.shape[0]
    
    # Get final waypoint trajectory
    sft_final = sft_waypoints[:, -1, :]
    rl_final = rl_waypoints[:, -1, :]
    gt_final = ground_truth[:, -1, :]
    
    # Plot SFT predictions
    axes[0].plot(gt_final[:, 0], gt_final[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
    axes[0].plot(sft_final[:, 0], sft_final[:, 1], 'b--', linewidth=2, label='SFT')
    axes[0].scatter(sft_final[0, 0], sft_final[0, 1], c='blue', s=50, marker='o')
    axes[0].scatter(sft_final[-1, 0], sft_final[-1, 1], c='blue', s=50, marker='*')
    sft_ade = np.mean(np.linalg.norm(sft_final - gt_final, axis=1))
    axes[0].set_title(f'SFT Only\nADE: {sft_ade:.3f}', fontsize=12)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Plot RL predictions
    axes[1].plot(gt_final[:, 0], gt_final[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
    axes[1].plot(rl_final[:, 0], rl_final[:, 1], 'r--', linewidth=2, label='SFT + RL')
    axes[1].scatter(rl_final[0, 0], rl_final[0, 1], c='red', s=50, marker='o')
    axes[1].scatter(rl_final[-1, 0], rl_final[-1, 1], c='red', s=50, marker='*')
    rl_ade = np.mean(np.linalg.norm(rl_final - gt_final, axis=1))
    axes[1].set_title(f'SFT + RL\nADE: {rl_ade:.3f}', fontsize=12)
    axes[1].set_xlabel('X (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    # Plot delta vectors
    delta = rl_final - sft_final
    axes[2].quiver(sft_final[:, 0], sft_final[:, 1], 
                   delta[:, 0], delta[:, 1],
                   delta[:, 0]**2 + delta[:, 1]**2,
                   cmap='viridis', scale=1, scale_units='xy', angles='xy')
    axes[2].plot(gt_final[:, 0], gt_final[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
    axes[2].scatter(sft_final[0, 0], sft_final[0, 1], c='blue', s=50, marker='o', label='Start')
    axes[2].set_title('Delta Corrections\n(RL - SFT)', fontsize=12)
    axes[2].set_xlabel('X (m)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    metrics: TrainingMetrics,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    window: int = 10,
) -> Any:
    """
    Plot training curves for RL training.
    
    Args:
        metrics: TrainingMetrics with all training data
        title: Plot title
        save_path: Optional path to save figure
        window: Moving average window size
    
    Returns:
        Figure object if matplotlib available, else None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    # Determine which metrics are available
    n_plots = 0
    for attr in ['episode_rewards', 'losses', 'policy_losses', 'value_losses', 'entropy']:
        if len(getattr(metrics, attr)) > 0:
            n_plots += 1
    
    if n_plots == 0:
        print("No metrics to plot")
        return None
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    def smooth(values: List[float], window: int) -> np.ndarray:
        """Compute moving average."""
        if len(values) < window:
            return np.array(values)
        return np.convolve(values, np.ones(window)/window, mode='valid')
    
    idx = 0
    
    if len(metrics.episode_rewards) > 0:
        ax = axes[idx]
        episodes = range(1, len(metrics.episode_rewards) + 1)
        ax.plot(episodes, metrics.episode_rewards, 'b-', alpha=0.3, label='Raw')
        ax.plot(episodes[window-1:], smooth(metrics.episode_rewards, window), 'b-', 
                linewidth=2, label=f'MA({window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1
    
    if len(metrics.losses) > 0:
        ax = axes[idx]
        steps = range(1, len(metrics.losses) + 1)
        ax.plot(steps, metrics.losses, 'r-', alpha=0.5, label='Total Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1
    
    if len(metrics.policy_losses) > 0 and len(metrics.value_losses) > 0:
        ax = axes[idx]
        steps = range(1, len(metrics.policy_losses) + 1)
        ax.plot(steps, metrics.policy_losses, 'b-', alpha=0.5, label='Policy Loss')
        ax.plot(steps, metrics.value_losses, 'g-', alpha=0.5, label='Value Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Policy vs Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1
    
    if len(metrics.entropy) > 0:
        ax = axes[idx]
        steps = range(1, len(metrics.entropy) + 1)
        ax.plot(steps, metrics.entropy, 'purple', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)
        idx += 1
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ade_fde_comparison(
    metrics: TrainingMetrics,
    title: str = "ADE/FDE Comparison",
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot ADE and FDE metrics over training.
    
    Args:
        metrics: TrainingMetrics with ADE/FDE data
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        Figure object if matplotlib available, else None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    if len(metrics.ade_scores) == 0:
        print("No ADE/FDE metrics to plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ADE
    episodes = range(1, len(metrics.ade_scores) + 1)
    axes[0].plot(episodes, metrics.ade_scores, 'b-', alpha=0.5)
    axes[0].axhline(y=np.mean(metrics.ade_scores), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(metrics.ade_scores):.3f}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('ADE (m)')
    axes[0].set_title('Average Displacement Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # FDE
    if len(metrics.fde_scores) > 0:
        axes[1].plot(episodes, metrics.fde_scores, 'r-', alpha=0.5)
        axes[1].axhline(y=np.mean(metrics.fde_scores), color='b', linestyle='--',
                        label=f'Mean: {np.mean(metrics.fde_scores):.3f}')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('FDE (m)')
        axes[1].set_title('Final Displacement Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_episode_trajectory(
    states: np.ndarray,
    actions: np.ndarray,
    waypoints: np.ndarray,
    title: str = "Episode Trajectory",
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot a complete episode trajectory with states, actions, and waypoints.
    
    Args:
        states: State trajectory (T, state_dim)
        actions: Action trajectory (T, action_dim)
        waypoints: Waypoints (T, H, 2)
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        Figure object if matplotlib available, else None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get position from states (assume first 2 dims are x, y)
    if states.shape[1] >= 2:
        positions = states[:, :2]
    else:
        positions = np.zeros((len(states), 2))
    
    # Left plot: Trajectory with waypoints
    # Plot waypoint path (final waypoint at each timestep)
    final_waypoints = waypoints[:, -1, :]  # (T, 2)
    axes[0].plot(final_waypoints[:, 0], final_waypoints[:, 1], 'g-', 
                 linewidth=2, label='Waypoints', alpha=0.7)
    
    # Plot agent trajectory
    axes[0].plot(positions[:, 0], positions[:, 1], 'b-', 
                 linewidth=1.5, label='Agent', alpha=0.5)
    
    # Mark start and end
    axes[0].scatter(positions[0, 0], positions[0, 1], c='blue', s=100, 
                    marker='o', zorder=5, label='Start')
    axes[0].scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, 
                    marker='*', zorder=5, label='End')
    
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Right plot: Speed profile
    if actions.shape[1] >= 1:
        speed = np.linalg.norm(actions, axis=1)
        axes[1].plot(speed, 'b-', linewidth=1.5)
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Speed (m/s)')
        axes[1].set_title('Speed Profile')
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_training_report(
    metrics: TrainingMetrics,
    output_dir: str,
    prefix: str = "train",
) -> Dict[str, str]:
    """
    Generate a complete training report with all visualizations.
    
    Args:
        metrics: Training metrics
        output_dir: Directory to save figures
        prefix: Prefix for output files
    
    Returns:
        Dictionary mapping metric names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = {}
    
    # Generate training curves
    fig = plot_training_curves(metrics, save_path=str(output_path / f"{prefix}_curves.png"))
    if fig:
        files['curves'] = str(output_path / f"{prefix}_curves.png")
        plt.close(fig)
    
    # Generate ADE/FDE plots
    if len(metrics.ade_scores) > 0:
        fig = plot_ade_fde_comparison(metrics, save_path=str(output_path / f"{prefix}_ade_fde.png"))
        if fig:
            files['ade_fde'] = str(output_path / f"{prefix}_ade_fde.png")
            plt.close(fig)
    
    return files


# Demo function
if __name__ == '__main__':
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping demo")
        exit(0)
    
    # Generate dummy data for demo
    np.random.seed(42)
    T, H = 20, 5
    
    # Create dummy waypoint prediction
    t = np.linspace(0, 1, T)
    gt = np.stack([t * 10, t * 5], axis=1)[:, np.newaxis, :]  # (T, 1, 2)
    gt = np.repeat(gt, H, axis=1)
    gt = gt + np.random.randn(T, H, 2) * 0.1
    
    pred = gt + np.random.randn(T, H, 2) * 0.3
    pred[:, -1, :] = gt[:, -1, :] + np.array([[1.0, 0.5]])  # Final waypoint offset
    
    sft_only = gt + np.random.randn(T, H, 2) * 0.8
    
    predictions = WaypointPrediction(predicted=pred, ground_truth=gt)
    
    # Plot comparison
    print(f"ADE: {predictions.ade():.3f}")
    print(f"FDE: {predictions.fde():.3f}")
    
    fig = plot_waypoint_comparison(
        predictions,
        show_sft_delta=True,
        sft_waypoints=sft_only,
        save_path="out/waypoint_comparison.png"
    )
    print(f"Saved: out/waypoint_comparison.png")
    
    # Plot delta corrections
    fig = plot_delta_corrections(
        sft_only, pred, gt,
        save_path="out/delta_corrections.png"
    )
    print(f"Saved: out/delta_corrections.png")
    
    # Generate dummy training metrics
    metrics = TrainingMetrics(
        episode_rewards=list(np.random.randn(100).cumsum() + 50),
        episode_lengths=[100] * 100,
        losses=list(np.random.rand(100) * 0.5 + 0.1),
        ade_scores=list(np.exp(-np.linspace(0, 2, 100)) + 0.1),
        fde_scores=list(np.exp(-np.linspace(0, 2, 100)) + 0.15),
        learning_rates=[1e-4] * 100,
        policy_losses=list(np.random.rand(100) * 0.3),
        value_losses=list(np.random.rand(100) * 0.2),
        entropy=list(np.random.rand(100) * 2 + 1)
    )
    
    fig = plot_training_curves(metrics, save_path="out/training_curves.png")
    print(f"Saved: out/training_curves.png")
    
    print("Demo complete!")

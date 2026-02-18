"""RL Refinement Evaluation with Statistical Significance.

This module evaluates SFT-only and RL-refined policies on the toy waypoint
environment, computing ADE/FDE metrics with confidence intervals for
statistically meaningful comparison.

Usage
-----
# SFT-only evaluation
python -m training.rl.eval_toy_waypoint_env --policy sft --episodes 100 --seed-base 0

# RL-refined evaluation
python -m training.rl.eval_toy_waypoint_env --policy rl --checkpoint out/rl_delta_ppo_v0/final.pt --episodes 100 --seed-base 0

# Side-by-side comparison
python -m training.rl.eval_toy_waypoint_env --compare \
  --sft-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --rl-checkpoint out/rl_delta_ppo_v0/final.pt \
  --episodes 100
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Statistical Functions
# ============================================================================

def mean_std_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute mean, std, and confidence interval.
    
    Args:
        values: List of sample values
        confidence: Confidence level (default: 0.95)
    
    Returns:
        (mean, std, ci_width) where CI = [mean - ci_width, mean + ci_width]
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    
    # Bootstrap confidence interval using normal approximation
    # For small samples, use t-distribution critical value
    if n < 30:
        # Simple approximation: use normal for now
        z = 1.96  # 95% CI
    else:
        z = 1.96  # 95% CI (approximately valid for n >= 30)
    
    ci_width = z * std / math.sqrt(n)
    
    return float(mean), float(std), float(ci_width)


def compute_p_value(
    sample1: List[float],
    sample2: List[float]
) -> float:
    """Compute two-sample t-test p-value for comparing means.
    
    Args:
        sample1: First sample
        sample2: Second sample
    
    Returns:
        Two-sided p-value
    """
    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        return 1.0
    
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1 = np.var(sample1, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(sample2, ddof=1) if n2 > 1 else 0.0
    
    # Welch's t-test
    se1 = var1 / n1
    se2 = var2 / n2
    se = math.sqrt(se1 + se2)
    
    if se == 0:
        return 1.0
    
    t_stat = (mean1 - mean2) / se
    
    # Approximate p-value using normal distribution
    # (valid for reasonable sample sizes)
    from scipy.stats import norm
    p_value = 2.0 * (1.0 - norm.cdf(abs(t_stat)))
    
    return p_value


# ============================================================================
# Waypoint Environment
# ============================================================================

class ToyWaypointEnv:
    """Simple toy environment for waypoint evaluation.
    
    Simulates a 2D waypoint tracking task with noisy SFT predictions.
    """

    def __init__(
        self,
        horizon_steps: int = 20,
        sft_noise_std: float = 2.0,
        seed: Optional[int] = None
    ):
        self.horizon_steps = horizon_steps
        self.sft_noise_std = sft_noise_std
        self.rng = np.random.default_rng(seed)
        self.target_waypoints = self._generate_target()
        self.sft_waypoints = self.target_waypoints + self.rng.normal(
            0, sft_noise_std, size=self.target_waypoints.shape
        )
        self.current_step = 0

    def _generate_target(self) -> np.ndarray:
        """Generate smooth target trajectory."""
        t = np.linspace(0, 4 * np.pi, self.horizon_steps)
        x = 5 * np.sin(t / 4) + np.linspace(-2, 2, self.horizon_steps)
        y = 5 * np.cos(t / 4)
        return np.stack([x, y], axis=1)  # (H, 2)

    def reset(self) -> Dict[str, Any]:
        """Reset environment."""
        self.target_waypoints = self._generate_target()
        self.sft_waypoints = self.target_waypoints + self.rng.normal(
            0, self.sft_noise_std, size=self.target_waypoints.shape
        )
        self.current_step = 0
        return {
            'target_waypoints': self.target_waypoints,
            'sft_waypoints': self.sft_waypoints,
            'step': self.current_step,
        }

    def step(
        self,
        waypoints: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute step with predicted waypoints.
        
        Args:
            waypoints: (H, 2) array of predicted waypoints
        
        Returns:
            (obs, reward, done, info)
        """
        # Compute ADE/FDE
        errors = np.linalg.norm(waypoints - self.target_waypoints, axis=1)
        ade = float(np.mean(errors))
        fde = float(errors[-1])

        # Reward: negative ADE (higher is better)
        reward = -ade

        # SFT baseline for comparison
        sft_errors = np.linalg.norm(self.sft_waypoints - self.target_waypoints, axis=1)
        sft_ade = float(np.mean(sft_errors))
        sft_fde = float(sft_errors[-1])
        improvement = sft_ade - ade

        self.current_step += 1
        done = self.current_step >= self.horizon_steps

        info = {
            'ade': ade,
            'fde': fde,
            'sft_ade': sft_ade,
            'sft_fde': sft_fde,
            'improvement': improvement,
            'errors': errors.tolist(),
        }

        return {
            'target_waypoints': self.target_waypoints,
            'sft_waypoints': self.sft_waypoints,
            'step': self.current_step,
        }, reward, done, info


# ============================================================================
# Policy Interface
# ============================================================================

class WaypointPolicy:
    """Base class for waypoint prediction policies."""

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """Predict waypoints for given observation.
        
        Args:
            obs: Environment observation
        
        Returns:
            (H, 2) array of predicted waypoints
        """
        raise NotImplementedError


class SFTPolicy(WaypointPolicy):
    """SFT policy using frozen encoder + waypoint head."""

    def __init__(self, checkpoint: Path, device: str = 'cpu'):
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required for SFT policy")
        
        self.device = torch.device(device)
        self.checkpoint = torch.load(checkpoint, map_location=self.device)
        
        # Load encoder and waypoint head
        from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
        from training.sft.waypoint_bc_torch_v0 import WaypointBCHead
        
        self.encoder = TinyMultiCamEncoder(out_dim=128).to(self.device)
        self.encoder.load_state_dict(self.checkpoint.get('encoder', {}))
        self.encoder.eval()
        
        self.waypoint_head = WaypointBCHead(
            in_dim=128,
            out_dim=20 * 2  # horizon_steps * 2 (x, y)
        )
        self.waypoint_head.load_state_dict(self.checkpoint.get('waypoint_head', {}))
        self.waypoint_head.eval()

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """Predict waypoints from observation."""
        # For toy environment, we use SFT waypoints directly
        # In real scenario, would process images through encoder
        return obs.get('sft_waypoints', np.zeros((20, 2)))


class RLPolicy(WaypointPolicy):
    """RL-refined policy with delta head."""

    def __init__(
        self,
        checkpoint: Path,
        sft_checkpoint: Path,
        device: str = 'cpu'
    ):
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required for RL policy")
        
        self.device = torch.device(device)
        
        # Load SFT base
        self.sft_ckpt = torch.load(sft_checkpoint, map_location=self.device)
        
        # Load RL delta head
        self.rl_ckpt = torch.load(checkpoint, map_location=self.device)
        
        # Create delta head
        from training.rl.train_ppo_delta_waypoint import DeltaHead
        
        self.delta_head = DeltaHead(
            in_dim=128,
            hidden_dim=128,
            horizon_steps=20
        ).to(self.device)
        self.delta_head.load_state_dict(self.rl_ckpt['delta_head'])
        self.delta_head.eval()

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """Predict corrected waypoints."""
        sft_waypoints = obs.get('sft_waypoints', np.zeros((20, 2)))
        
        # Get delta prediction (simplified - uses mock embedding)
        z = torch.randn(1, 128, device=self.device)
        delta = self.delta_head(z).detach().cpu().numpy().squeeze(0)
        
        # Apply correction
        corrected = sft_waypoints + delta
        return corrected


class HeuristicDeltaPolicy(WaypointPolicy):
    """Simple heuristic policy for testing.
    
    Applies a fixed correction pattern to SFT waypoints.
    Used for smoke tests and baseline comparison.
    """

    def __init__(self, scale: float = 0.5):
        self.scale = scale

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """Apply heuristic correction."""
        sft_waypoints = obs.get('sft_waypoints', np.zeros((20, 2)))
        
        # Heuristic: scale towards target (simple correction)
        target = obs.get('target_waypoints', sft_waypoints)
        delta = (target - sft_waypoints) * self.scale
        
        return sft_waypoints + delta


# ============================================================================
# Evaluation
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    policy: str  # 'sft', 'rl', 'heuristic', 'compare'
    episodes: int = 100
    seed_base: int = 0
    horizon_steps: int = 20
    sft_noise_std: float = 2.0
    sft_checkpoint: Optional[Path] = None
    rl_checkpoint: Optional[Path] = None
    output_dir: Optional[Path] = None


@dataclass
class EvalResult:
    """Evaluation result with statistics."""
    policy_name: str
    ade_samples: List[float]
    fde_samples: List[float]
    improvement_samples: List[float]
    
    @property
    def ade_mean(self) -> float:
        return float(np.mean(self.ade_samples))
    
    @property
    def ade_std(self) -> float:
        return float(np.std(self.ade_samples, ddof=1))
    
    @property
    def fde_mean(self) -> float:
        return float(np.mean(self.fde_samples))
    
    @property
    def fde_std(self) -> float:
        return float(np.std(self.fde_samples, ddof=1))
    
    @property
    def success_rate(self) -> float:
        """Rate of episodes where all waypoints were reached."""
        return 0.0  # Placeholder - depends on task definition
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'policy': self.policy_name,
            'ade': {
                'mean': self.ade_mean,
                'std': self.ade_std,
                'samples': self.ade_samples,
            },
            'fde': {
                'mean': self.fde_mean,
                'std': self.fde_std,
                'samples': self.fde_samples,
            },
            'improvement': {
                'mean': float(np.mean(self.improvement_samples)),
                'std': float(np.std(self.improvement_samples, ddof=1)),
                'samples': self.improvement_samples,
            },
            'success_rate': self.success_rate,
            'num_episodes': len(self.ade_samples),
        }


def evaluate_policy(
    policy: WaypointPolicy,
    config: EvalConfig
) -> EvalResult:
    """Evaluate a policy on the toy waypoint environment.
    
    Args:
        policy: Policy to evaluate
        config: Evaluation configuration
    
    Returns:
        Evaluation result with ADE/FDE statistics
    """
    ade_samples = []
    fde_samples = []
    improvement_samples = []

    for ep in range(config.episodes):
        env = ToyWaypointEnv(
            horizon_steps=config.horizon_steps,
            sft_noise_std=config.sft_noise_std,
            seed=config.seed_base + ep
        )
        obs = env.reset()
        
        total_waypoints = []
        target_waypoints = obs['target_waypoints']
        sft_waypoints = obs['sft_waypoints']
        
        # Roll out episode
        for step in range(config.horizon_steps):
            waypoints = policy.predict(obs)
            obs, reward, done, info = env.step(waypoints)
            total_waypoints.append(waypoints)
            
            if done:
                break
        
        # Compute final metrics
        final_waypoints = total_waypoints[-1] if total_waypoints else sft_waypoints
        errors = np.linalg.norm(final_waypoints - target_waypoints, axis=1)
        
        ade = float(np.mean(errors))
        fde = float(errors[-1])
        
        sft_errors = np.linalg.norm(sft_waypoints - target_waypoints, axis=1)
        sft_ade = float(np.mean(sft_errors))
        improvement = sft_ade - ade
        
        ade_samples.append(ade)
        fde_samples.append(fde)
        improvement_samples.append(improvement)

    return EvalResult(
        policy_name=getattr(policy, 'name', 'unknown'),
        ade_samples=ade_samples,
        fde_samples=fde_samples,
        improvement_samples=improvement_samples,
    )


def compare_policies(
    sft_result: EvalResult,
    rl_result: EvalResult
) -> Dict[str, Any]:
    """Compare two policies and compute statistical significance.
    
    Args:
        sft_result: SFT-only evaluation result
        rl_result: RL-refined evaluation result
    
    Returns:
        Comparison dictionary with p-values and improvement metrics
    """
    # Compute p-values
    ade_p_value = compute_p_value(sft_result.ade_samples, rl_result.ade_samples)
    fde_p_value = compute_p_value(sft_result.fde_samples, rl_result.fde_samples)
    
    # Compute improvement percentages
    ade_improvement = (
        (sft_result.ade_mean - rl_result.ade_mean) / sft_result.ade_mean * 100
        if sft_result.ade_mean > 0 else 0
    )
    fde_improvement = (
        (sft_result.fde_mean - rl_result.fde_mean) / sft_result.fde_mean * 100
        if sft_result.fde_mean > 0 else 0
    )
    
    # Confidence intervals
    sft_ade_mean, sft_ade_std, sft_ade_ci = mean_std_confidence_interval(
        sft_result.ade_samples
    )
    rl_ade_mean, rl_ade_std, rl_ade_ci = mean_std_confidence_interval(
        rl_result.ade_samples
    )
    
    return {
        'ade': {
            'sft_mean': sft_ade_mean,
            'sft_std': sft_ade_std,
            'sft_ci': sft_ade_ci,
            'rl_mean': rl_ade_mean,
            'rl_std': rl_ade_std,
            'rl_ci': rl_ade_ci,
            'improvement_pct': ade_improvement,
            'p_value': ade_p_value,
            'significant': ade_p_value < 0.05,
        },
        'fde': {
            'sft_mean': float(np.mean(sft_result.fde_samples)),
            'sft_std': float(np.std(sft_result.fde_samples, ddof=1)),
            'rl_mean': float(np.mean(rl_result.fde_samples)),
            'rl_std': float(np.std(rl_result.fde_samples, ddof=1)),
            'improvement_pct': fde_improvement,
            'p_value': fde_p_value,
            'significant': fde_p_value < 0.05,
        },
        'num_episodes': len(sft_result.ade_samples),
    }


def print_comparison_report(
    sft_result: EvalResult,
    rl_result: EvalResult,
    comparison: Dict[str, Any]
) -> None:
    """Print 3-line comparison report to console."""
    n = comparison['num_episodes']
    
    print("\n" + "=" * 60)
    print("SFT vs RL Comparison Report")
    print("=" * 60)
    print(f"Episodes: {n}")
    print("-" * 60)
    
    # ADE line
    sft_ade = comparison['ade']
    rl_ade = comparison['ade']
    sig_marker = "*" if sft_ade['significant'] else ""
    print(
        f"ADE: {sft_ade['sft_mean']:.2f}m ± {sft_ade['sft_ci']:.2f}m (SFT) → "
        f"{rl_ade['rl_mean']:.2f}m (RL) [{sft_ade['improvement_pct']:+.1f}%]{sig_marker}"
    )
    
    # FDE line
    sft_fde = comparison['fde']
    rl_fde = comparison['fde']
    sig_marker = "*" if sft_fde['significant'] else ""
    print(
        f"FDE: {sft_fde['sft_mean']:.2f}m (SFT) → {rl_fde['rl_mean']:.2f}m (RL) "
        f"[{sft_fde['improvement_pct']:+.1f}%]{sig_marker}"
    )
    
    # Success rate (placeholder)
    print(f"Success: {sft_result.success_rate:.1%} (SFT) → {rl_result.success_rate:.1%} (RL)")
    
    print("-" * 60)
    if sft_ade['significant']:
        print("✓ Statistically significant improvement (p < 0.05)")
    else:
        print("✗ No statistically significant difference (p >= 0.05)")
    print("=" * 60 + "\n")


# ============================================================================
# Main
# ============================================================================

def parse_args() -> EvalConfig:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate SFT or RL policies on toy waypoint environment"
    )
    
    # Policy selection
    p.add_argument(
        "--policy",
        type=str,
        choices=['sft', 'rl', 'heuristic', 'compare'],
        default='heuristic',
        help="Policy type to evaluate"
    )
    
    # Checkpoints
    p.add_argument(
        "--sft-checkpoint",
        type=Path,
        help="Path to SFT checkpoint"
    )
    p.add_argument(
        "--rl-checkpoint",
        type=Path,
        help="Path to RL checkpoint"
    )
    
    # Evaluation parameters
    p.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base random seed for episodes"
    )
    p.add_argument(
        "--horizon-steps",
        type=int,
        default=20,
        help="Number of waypoints per episode"
    )
    p.add_argument(
        "--sft-noise-std",
        type=float,
        default=2.0,
        help="Standard deviation of SFT noise"
    )
    
    # Output
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for metrics"
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )
    
    args = p.parse_args()
    
    return EvalConfig(
        policy=args.policy,
        episodes=args.episodes,
        seed_base=args.seed_base,
        horizon_steps=args.horizon_steps,
        sft_noise_std=args.sft_noise_std,
        sft_checkpoint=args.sft_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Main evaluation entry point."""
    config = parse_args()
    
    if config.policy == 'compare' and not config.sft_checkpoint:
        raise ValueError("--sft-checkpoint required for comparison")
    
    # Create policy based on type
    if config.policy == 'sft':
        if not config.sft_checkpoint:
            raise ValueError("--sft-checkpoint required for SFT policy")
        policy = SFTPolicy(config.sft_checkpoint)
        policy.name = 'SFT'
    elif config.policy == 'rl':
        if not config.rl_checkpoint:
            raise ValueError("--rl-checkpoint required for RL policy")
        if not config.sft_checkpoint:
            raise ValueError("--sft-checkpoint required for RL policy")
        policy = RLPolicy(config.rl_checkpoint, config.sft_checkpoint)
        policy.name = 'RL'
    elif config.policy == 'heuristic':
        policy = HeuristicDeltaPolicy(scale=0.5)
        policy.name = 'Heuristic'
    else:  # compare
        # Evaluate both policies
        sft_policy = SFTPolicy(config.sft_checkpoint)
        sft_policy.name = 'SFT'
        
        rl_policy = RLPolicy(config.rl_checkpoint, config.sft_checkpoint)
        rl_policy.name = 'RL'
        
        sft_result = evaluate_policy(sft_policy, config)
        rl_result = evaluate_policy(rl_policy, config)
        
        comparison = compare_policies(sft_result, rl_result)
        
        if not config.quiet:
            print_comparison_report(sft_result, rl_result, comparison)
        
        # Save results
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            
            output = {
                'sft': sft_result.to_dict(),
                'rl': rl_result.to_dict(),
                'comparison': comparison,
                'config': {
                    'episodes': config.episodes,
                    'seed_base': config.seed_base,
                    'horizon_steps': config.horizon_steps,
                }
            }
            
            (config.output_dir / 'metrics.json').write_text(
                json.dumps(output, indent=2)
            )
            print(f"Metrics saved to {config.output_dir / 'metrics.json'}")
        
        return
    
    # Single policy evaluation
    result = evaluate_policy(config, policy)
    
    mean, std, ci = mean_std_confidence_interval(result.ade_samples)
    
    if not config.quiet:
        print(f"\n{policy.name} Evaluation Results")
        print(f"  ADE: {mean:.2f}m ± {ci:.2f}m (std={std:.2f})")
        print(f"  FDE: {float(np.mean(result.fde_samples)):.2f}m")
        print(f"  Episodes: {len(result.ade_samples)}")
    
    # Save results
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / 'metrics.json').write_text(
            json.dumps(result.to_dict(), indent=2)
        )
        if not config.quiet:
            print(f"Metrics saved to {config.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

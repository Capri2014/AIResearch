"""
Unified Policy Evaluation Framework for Autonomous Driving

Compares SFT baseline, PPO-trained, and GRPO-trained policies on:
- Toy Waypoint Environment
- ADE/FDE metrics
- Success rate
- Per-episode detailed metrics

Usage:
    python -m training.rl.unified_eval --output out/eval/unified_2026-02-16
"""

import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv
from training.rl.waypoint_policy_torch import SFTWaypointPolicy, PPOWaypointPolicy
from training.rl.grpo_waypoint import GRPOWaypointModel, GRPOConfig


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluation episode."""
    episode_id: str
    success: bool
    ade: float  # Average Displacement Error
    fde: float  # Final Displacement Error
    return_value: float
    steps: int
    trajectory: List[List[float]]
    target_trajectory: List[List[float]]
    policy_type: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PolicyComparison:
    """Comparison results across policies."""
    policy_type: str
    n_episodes: int
    success_rate: float
    mean_ade: float
    mean_fde: float
    mean_return: float
    std_ade: float
    std_fde: float
    episodes: List[EpisodeMetrics]
    
    def to_dict(self) -> Dict:
        return {
            'policy_type': self.policy_type,
            'n_episodes': self.n_episodes,
            'success_rate': self.success_rate,
            'mean_ade': self.mean_ade,
            'mean_fde': self.mean_fde,
            'mean_return': self.mean_return,
            'std_ade': self.std_ade,
            'std_fde': self.std_fde,
            'episodes': [e.to_dict() for e in self.episodes],
        }


class UnifiedEvaluator:
    """Unified evaluator for comparing SFT, PPO, and GRPO policies."""
    
    def __init__(self, output_dir: str = "out/eval/unified"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.results: Dict[str, PolicyComparison] = {}
    
    def evaluate_sft(
        self,
        n_episodes: int = 50,
        seed_base: int = 42,
        max_steps: int = 50
    ) -> PolicyComparison:
        """Evaluate SFT baseline policy."""
        print(f"\n{'='*60}")
        print(f"Evaluating SFT Baseline ({n_episodes} episodes)")
        print(f"{'='*60}")
        
        env = ToyWaypointEnv(seed=seed_base)
        policy = SFTWaypointPolicy()
        
        episodes = []
        for ep_id in range(n_episodes):
            env = ToyWaypointEnv(seed=seed_base + ep_id)
            obs, info = env.reset()
            
            trajectory = []
            target_trajectory = []
            done = False
            steps = 0
            total_return = 0.0
            
            while not done and steps < max_steps:
                if isinstance(obs, tuple):
                    state = obs[0]
                else:
                    state = obs
                
                action = policy.predict(state)
                obs, reward, done, info = env.step(action)
                
                trajectory.append(action.tolist() if hasattr(action, 'tolist') else [float(action)])
                target_trajectory.append(info.get('target_waypoint', []))
                total_return += reward
                steps += 1
            
            # Compute metrics
            traj_array = np.array(trajectory)
            target_array = np.array(target_trajectory)
            
            ade = float(np.mean(np.linalg.norm(traj_array - target_array, axis=1)))
            fde = float(np.linalg.norm(traj_array[-1] - target_array[-1]))
            success = bool(info.get('success', False))
            
            episodes.append(EpisodeMetrics(
                episode_id=f"sft_{ep_id:04d}",
                success=success,
                ade=ade,
                fde=fde,
                return_value=total_return,
                steps=steps,
                trajectory=trajectory,
                target_trajectory=target_trajectory,
                policy_type="sft"
            ))
            
            if (ep_id + 1) % 10 == 0:
                print(f"  Completed {ep_id + 1}/{n_episodes} episodes")
        
        success_rate = sum(1 for e in episodes if e.success) / len(episodes)
        mean_ade = np.mean([e.ade for e in episodes])
        mean_fde = np.mean([e.fde for e in episodes])
        mean_return = np.mean([e.return_value for e in episodes])
        std_ade = np.std([e.ade for e in episodes])
        std_fde = np.std([e.fde for e in episodes])
        
        comparison = PolicyComparison(
            policy_type="sft",
            n_episodes=n_episodes,
            success_rate=success_rate,
            mean_ade=mean_ade,
            mean_fde=mean_fde,
            mean_return=mean_return,
            std_ade=std_ade,
            std_fde=std_fde,
            episodes=episodes
        )
        
        self.results['sft'] = comparison
        
        print(f"\nSFT Results:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Mean ADE: {mean_ade:.2f}m")
        print(f"  Mean FDE: {mean_fde:.2f}m")
        print(f"  Mean Return: {mean_return:.2f}")
        
        return comparison
    
    def evaluate_ppo(
        self,
        checkpoint_path: Optional[str] = None,
        n_episodes: int = 50,
        seed_base: int = 42,
        max_steps: int = 50
    ) -> PolicyComparison:
        """Evaluate PPO-trained policy."""
        print(f"\n{'='*60}")
        print(f"Evaluating PPO Policy ({n_episodes} episodes)")
        print(f"{'='*60}")
        
        env = ToyWaypointEnv(seed=seed_base)
        
        # Load policy if checkpoint provided
        if checkpoint_path and Path(checkpoint_path).exists():
            policy = PPOWaypointPolicy.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            policy = PPOWaypointPolicy()
            print("Using untrained PPO policy (for comparison)")
        
        episodes = []
        for ep_id in range(n_episodes):
            env = ToyWaypointEnv(seed=seed_base + ep_id)
            obs, info = env.reset()
            
            trajectory = []
            target_trajectory = []
            done = False
            steps = 0
            total_return = 0.0
            
            while not done and steps < max_steps:
                if isinstance(obs, tuple):
                    state = obs[0]
                else:
                    state = obs
                
                action = policy.predict(state)
                obs, reward, done, info = env.step(action)
                
                trajectory.append(action.tolist() if hasattr(action, 'tolist') else [float(action)])
                target_trajectory.append(info.get('target_waypoint', []))
                total_return += reward
                steps += 1
            
            traj_array = np.array(trajectory)
            target_array = np.array(target_trajectory)
            
            ade = float(np.mean(np.linalg.norm(traj_array - target_array, axis=1)))
            fde = float(np.linalg.norm(traj_array[-1] - target_array[-1]))
            success = bool(info.get('success', False))
            
            episodes.append(EpisodeMetrics(
                episode_id=f"ppo_{ep_id:04d}",
                success=success,
                ade=ade,
                fde=fde,
                return_value=total_return,
                steps=steps,
                trajectory=trajectory,
                target_trajectory=target_trajectory,
                policy_type="ppo"
            ))
            
            if (ep_id + 1) % 10 == 0:
                print(f"  Completed {ep_id + 1}/{n_episodes} episodes")
        
        success_rate = sum(1 for e in episodes if e.success) / len(episodes)
        mean_ade = np.mean([e.ade for e in episodes])
        mean_fde = np.mean([e.fde for e in episodes])
        mean_return = np.mean([e.return_value for e in episodes])
        std_ade = np.std([e.ade for e in episodes])
        std_fde = np.std([e.fde for e in episodes])
        
        comparison = PolicyComparison(
            policy_type="ppo",
            n_episodes=n_episodes,
            success_rate=success_rate,
            mean_ade=mean_ade,
            mean_fde=mean_fde,
            mean_return=mean_return,
            std_ade=std_ade,
            std_fde=std_fde,
            episodes=episodes
        )
        
        self.results['ppo'] = comparison
        
        print(f"\nPPO Results:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Mean ADE: {mean_ade:.2f}m")
        print(f"  Mean FDE: {mean_fde:.2f}m")
        print(f"  Mean Return: {mean_return:.2f}")
        
        return comparison
    
    def evaluate_grpo(
        self,
        checkpoint_path: Optional[str] = None,
        n_episodes: int = 50,
        seed_base: int = 42,
        max_steps: int = 50
    ) -> PolicyComparison:
        """Evaluate GRPO-trained policy."""
        print(f"\n{'='*60}")
        print(f"Evaluating GRPO Policy ({n_episodes} episodes)")
        print(f"{'='*60}")
        
        # Create model
        config = GRPOConfig()
        model = GRPOWaypointModel(config)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded GRPO checkpoint from {checkpoint_path}")
        else:
            print("Using untrained GRPO policy (for comparison)")
        
        model.eval()
        
        episodes = []
        for ep_id in range(n_episodes):
            env = ToyWaypointEnv(seed=seed_base + ep_id)
            obs, info = env.reset()
            
            trajectory = []
            target_trajectory = []
            done = False
            steps = 0
            total_return = 0.0
            
            while not done and steps < max_steps:
                if isinstance(obs, tuple):
                    state = obs[0]
                else:
                    state = obs
                
                # Convert state to tensor and get action from GRPO model
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    waypoints = model.sample_trajectory(state_tensor)
                    action = waypoints.squeeze(0).tolist()  # Take first waypoint as action
                
                obs, reward, done, info = env.step(action)
                
                trajectory.append(action if isinstance(action, list) else [float(action)])
                target_trajectory.append(info.get('target_waypoint', []))
                total_return += reward
                steps += 1
            
            traj_array = np.array(trajectory)
            target_array = np.array(target_trajectory)
            
            ade = float(np.mean(np.linalg.norm(traj_array - target_array, axis=1)))
            fde = float(np.linalg.norm(traj_array[-1] - target_array[-1]))
            success = bool(info.get('success', False))
            
            episodes.append(EpisodeMetrics(
                episode_id=f"grpo_{ep_id:04d}",
                success=success,
                ade=ade,
                fde=fde,
                return_value=total_return,
                steps=steps,
                trajectory=trajectory,
                target_trajectory=target_trajectory,
                policy_type="grpo"
            ))
            
            if (ep_id + 1) % 10 == 0:
                print(f"  Completed {ep_id + 1}/{n_episodes} episodes")
        
        success_rate = sum(1 for e in episodes if e.success) / len(episodes)
        mean_ade = np.mean([e.ade for e in episodes])
        mean_fde = np.mean([e.fde for e in episodes])
        mean_return = np.mean([e.return_value for e in episodes])
        std_ade = np.std([e.ade for e in episodes])
        std_fde = np.std([e.fde for e in episodes])
        
        comparison = PolicyComparison(
            policy_type="grpo",
            n_episodes=n_episodes,
            success_rate=success_rate,
            mean_ade=mean_ade,
            mean_fde=mean_fde,
            mean_return=mean_return,
            std_ade=std_ade,
            std_fde=std_fde,
            episodes=episodes
        )
        
        self.results['grpo'] = comparison
        
        print(f"\nGRPO Results:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Mean ADE: {mean_ade:.2f}m")
        print(f"  Mean FDE: {mean_fde:.2f}m")
        print(f"  Mean Return: {mean_return:.2f}")
        
        return comparison
    
    def compare_all(
        self,
        n_episodes: int = 50,
        seed_base: int = 42,
        max_steps: int = 50
    ) -> Dict[str, PolicyComparison]:
        """Evaluate all policies and return comparison."""
        # Evaluate all policies
        self.evaluate_sft(n_episodes, seed_base, max_steps)
        self.evaluate_ppo(n_episodes, seed_base, max_steps)
        self.evaluate_grpo(n_episodes, seed_base, max_steps)
        
        # Save results
        self.save_results()
        
        # Print comparison
        self.print_comparison()
        
        return self.results
    
    def print_comparison(self):
        """Print 3-line comparison report."""
        print(f"\n{'='*60}")
        print("POLICY COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        if 'sft' not in self.results:
            print("No results to compare")
            return
        
        sft = self.results['sft']
        ppo = self.results.get('ppo', None)
        grpo = self.results.get('grpo', None)
        
        # ADE comparison
        ade_ppo_str = ""
        if ppo:
            ade_pct = ((ppo.mean_ade - sft.mean_ade) / sft.mean_ade) * 100 if sft.mean_ade > 0 else 0
            ade_ppo_str = f" → {ppo.mean_ade:.2f}m ({'+' if ade_pct > 0 else ''}{ade_pct:.0f}%)"
        
        ade_grpo_str = ""
        if grpo:
            ade_pct = ((grpo.mean_ade - sft.mean_ade) / sft.mean_ade) * 100 if sft.mean_ade > 0 else 0
            ade_grpo_str = f" → {grpo.mean_ade:.2f}m ({'+' if ade_pct > 0 else ''}{ade_pct:.0f}%)"
        
        print(f"ADE: {sft.mean_ade:.2f}m (SFT){ade_ppo_str}{ade_grpo_str}")
        
        # FDE comparison
        fde_ppo_str = ""
        if ppo:
            fde_pct = ((ppo.mean_fde - sft.mean_fde) / sft.mean_fde) * 100 if sft.mean_fde > 0 else 0
            fde_ppo_str = f" → {ppo.mean_fde:.2f}m ({'+' if fde_pct > 0 else ''}{fde_pct:.0f}%)"
        
        fde_grpo_str = ""
        if grpo:
            fde_pct = ((grpo.mean_fde - sft.mean_fde) / sft.mean_fde) * 100 if sft.mean_fde > 0 else 0
            fde_grpo_str = f" → {grpo.mean_fde:.2f}m ({'+' if fde_pct > 0 else ''}{fde_pct:.0f}%)"
        
        print(f"FDE: {sft.mean_fde:.2f}m (SFT){fde_ppo_str}{fde_grpo_str}")
        
        # Success rate comparison
        sr_ppo_str = ""
        if ppo:
            sr_delta = (ppo.success_rate - sft.success_rate) * 100
            sr_ppo_str = f" → {ppo.success_rate:.1%} ({'+' if sr_delta > 0 else ''}{sr_delta:.0f}pp)"
        
        sr_grpo_str = ""
        if grpo:
            sr_delta = (grpo.success_rate - sft.success_rate) * 100
            sr_grpo_str = f" → {grpo.success_rate:.1%} ({'+' if sr_delta > 0 else ''}{sr_delta:.0f}pp)"
        
        print(f"Success: {sft.success_rate:.1%} (SFT){sr_ppo_str}{sr_grpo_str}")
    
    def save_results(self):
        """Save all results to output directory."""
        # Save individual policy results
        for policy_type, comparison in self.results.items():
            policy_dir = self.output_dir / policy_type
            policy_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics JSON
            with open(policy_dir / 'metrics.json', 'w') as f:
                json.dump(comparison.to_dict(), f, indent=2)
        
        # Save unified comparison
        unified_results = {
            'evaluation_timestamp': str(Path(self.output_dir).name),
            'policies': {
                k: v.to_dict() for k, v in self.results.items()
            }
        }
        
        with open(self.output_dir / 'comparison.json', 'w') as f:
            json.dump(unified_results, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate markdown summary report."""
        if 'sft' not in self.results:
            return
        
        sft = self.results['sft']
        ppo = self.results.get('ppo', None)
        grpo = self.results.get('grpo', None)
        
        report = f"""# Unified Policy Comparison Report

## Evaluation Configuration
- **Output Directory**: {self.output_dir}
- **Episodes per Policy**: {sft.n_episodes}

## Results Summary

### SFT Baseline
- **Success Rate**: {sft.success_rate:.1%}
- **Mean ADE**: {sft.mean_ade:.2f}m (±{sft.std_ade:.2f})
- **Mean FDE**: {sft.mean_fde:.2f}m (±{sft.std_fde:.2f})
- **Mean Return**: {sft.mean_return:.2f}

"""
        
        if ppo:
            ade_pct = ((ppo.mean_ade - sft.mean_ade) / sft.mean_ade) * 100 if sft.mean_ade > 0 else 0
            fde_pct = ((ppo.mean_fde - sft.mean_fde) / sft.mean_fde) * 100 if sft.mean_fde > 0 else 0
            sr_delta = (ppo.success_rate - sft.success_rate) * 100
            
            report += f"""### PPO
- **Success Rate**: {ppo.success_rate:.1%} ({'+' if sr_delta > 0 else ''}{sr_delta:.0f}pp vs SFT)
- **Mean ADE**: {ppo.mean_ade:.2f}m ({'+' if ade_pct > 0 else ''}{ade_pct:.0f}% vs SFT) (±{ppo.std_ade:.2f})
- **Mean FDE**: {ppo.mean_fde:.2f}m ({'+' if fde_pct > 0 else ''}{fde_pct:.0f}% vs SFT) (±{ppo.std_fde:.2f})
- **Mean Return**: {ppo.mean_return:.2f}

"""
        
        if grpo:
            ade_pct = ((grpo.mean_ade - sft.mean_ade) / sft.mean_ade) * 100 if sft.mean_ade > 0 else 0
            fde_pct = ((grpo.mean_fde - sft.mean_fde) / sft.mean_fde) * 100 if sft.mean_fde > 0 else 0
            sr_delta = (grpo.success_rate - sft.success_rate) * 100
            
            report += f"""### GRPO
- **Success Rate**: {grpo.success_rate:.1%} ({'+' if sr_delta > 0 else ''}{sr_delta:.0f}pp vs SFT)
- **Mean ADE**: {grpo.mean_ade:.2f}m ({'+' if ade_pct > 0 else ''}{ade_pct:.0f}% vs SFT) (±{grpo.std_ade:.2f})
- **Mean FDE**: {grpo.mean_fde:.2f}m ({'+' if fde_pct > 0 else ''}{fde_pct:.0f}% vs SFT) (±{grpo.std_fde:.2f})
- **Mean Return**: {grpo.mean_return:.2f}

"""
        
        report += """## 3-Line Summary

"""
        
        # ADE line
        ade_parts = [f"{sft.mean_ade:.2f}m (SFT)"]
        if ppo:
            ade_pct = ((ppo.mean_ade - sft.mean_ade) / sft.mean_ade) * 100 if sft.mean_ade > 0 else 0
            ade_parts.append(f"→ {ppo.mean_ade:.2f}m (PPO {ade_pct:+.0f}%)")
        if grpo:
            ade_pct = ((grpo.mean_ade - sft.mean_ade) / sft.mean_ade) * 100 if sft.mean_ade > 0 else 0
            ade_parts.append(f"→ {grpo.mean_ade:.2f}m (GRPO {ade_pct:+.0f}%)")
        report += f"ADE: " + " ".join(ade_parts) + "\n"
        
        # FDE line
        fde_parts = [f"{sft.mean_fde:.2f}m (SFT)"]
        if ppo:
            fde_pct = ((ppo.mean_fde - sft.mean_fde) / sft.mean_fde) * 100 if sft.mean_fde > 0 else 0
            fde_parts.append(f"→ {ppo.mean_fde:.2f}m (PPO {fde_pct:+.0f}%)")
        if grpo:
            fde_pct = ((grpo.mean_fde - sft.mean_fde) / sft.mean_fde) * 100 if sft.mean_fde > 0 else 0
            fde_parts.append(f"→ {grpo.mean_fde:.2f}m (GRPO {fde_pct:+.0f}%)")
        report += f"FDE: " + " ".join(fde_parts) + "\n"
        
        # Success rate line
        sr_parts = [f"{sft.success_rate:.1%} (SFT)"]
        if ppo:
            sr_delta = (ppo.success_rate - sft.success_rate) * 100
            sr_parts.append(f"→ {ppo.success_rate:.1%} (PPO {sr_delta:+.0f}pp)")
        if grpo:
            sr_delta = (grpo.success_rate - sft.success_rate) * 100
            sr_parts.append(f"→ {grpo.success_rate:.1%} (GRPO {sr_delta:+.0f}pp)")
        report += f"Success: " + " ".join(sr_parts) + "\n"
        
        with open(self.output_dir / 'report.md', 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to {self.output_dir / 'report.md'}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Unified Policy Evaluation Framework'
    )
    
    parser.add_argument('--output', type=str, default='out/eval/unified_2026-02-16')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--policies', type=str, default='all',
                        choices=['sft', 'ppo', 'grpo', 'all'])
    parser.add_argument('--ppo-checkpoint', type=str, default=None)
    parser.add_argument('--grpo-checkpoint', type=str, default=None)
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNIFIED POLICY EVALUATION FRAMEWORK")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Episodes per policy: {args.episodes}")
    
    evaluator = UnifiedEvaluator(output_dir=args.output)
    
    if args.policies in ['sft', 'all']:
        evaluator.evaluate_sft(args.episodes, args.seed_base, args.max_steps)
    
    if args.policies in ['ppo', 'all']:
        evaluator.evaluate_ppo(args.ppo_checkpoint, args.episodes, args.seed_base, args.max_steps)
    
    if args.policies in ['grpo', 'all']:
        evaluator.evaluate_grpo(args.grpo_checkpoint, args.episodes, args.seed_base, args.max_steps)
    
    # Print comparison
    evaluator.print_comparison()
    
    # Save results
    evaluator.save_results()
    
    print(f"\nEvaluation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()

"""
Training Diagnostics Module for RL After SFT Pipeline.

Analyzes training runs to identify failure modes, provide actionable insights,
and help debug training issues. Integrates with existing checkpoint management
and evaluation metrics.

Usage:
    python -m training.rl.training_diagnostics --run-id <run_id>
    python -m training.rl.training_diagnostics --latest --report
    python -m training.rl.training_diagnostics --run-id <run_id> --failure-modes
"""
import os
import sys
import json
import argparse
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class TrainingDiagnostics:
    """Container for training diagnostics results."""
    run_id: str
    checkpoint_path: str
    
    # Training metrics
    total_episodes: int = 0
    total_updates: int = 0
    final_reward: float = 0.0
    best_reward: float = 0.0
    reward_std: float = 0.0
    
    # Learning dynamics
    learning_rate: float = 0.0
    entropy: float = 0.0
    grad_norm: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    
    # Convergence metrics
    reward_trend: str = "unknown"  # increasing, decreasing, stable, oscillating
    entropy_trend: str = "unknown"
    value_alignment: float = 0.0  # How well value function predicts returns
    
    # Evaluation metrics (if available)
    eval_ade: Optional[float] = None
    eval_fde: Optional[float] = None
    eval_success_rate: Optional[float] = None
    
    # Failure mode analysis
    failure_modes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Raw data for detailed analysis
    reward_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)


class TrainingDiagnosticsAnalyzer:
    """
    Analyzes training runs to identify issues and provide recommendations.
    
    Features:
    - Failure mode detection (instability, divergence, plateaus)
    - Learning dynamics analysis (reward curves, entropy, gradient norms)
    - Convergence quality assessment
    - Actionable recommendations
    """
    
    def __init__(self, out_dir: str = "out"):
        self.out_dir = out_dir
        self.run_dirs = self._scan_runs()
    
    def _scan_runs(self) -> List[str]:
        """Scan for training run directories."""
        pattern = os.path.join(self.out_dir, "*/")
        runs = []
        for d in glob.glob(pattern):
            if os.path.isdir(d):
                # Check if it has training metrics
                train_metrics = os.path.join(d, "train_metrics.json")
                if os.path.exists(train_metrics):
                    runs.append(d.rstrip('/'))
        return sorted(runs, key=os.path.getmtime, reverse=True)
    
    def _load_train_metrics(self, run_dir: str) -> Optional[Dict]:
        """Load training metrics from run directory."""
        metrics_path = os.path.join(run_dir, "train_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    # Handle both dict and list formats
                    if isinstance(data, list):
                        # Convert list format to dict
                        return {"episode_rewards": data}
                    return data
            except (json.JSONDecodeError, IOError):
                return None
        return None
    
    def _load_eval_metrics(self, run_dir: str) -> Optional[Dict]:
        """Load evaluation metrics if available."""
        # Try different evaluation output patterns
        eval_patterns = [
            os.path.join(run_dir, "eval", "metrics.json"),
            os.path.join(run_dir, "metrics.json"),
            os.path.join(run_dir, "*", "metrics.json"),
        ]
        for pattern in eval_patterns:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], 'r') as f:
                    return json.load(f)
        return None
    
    def _detect_reward_trend(self, rewards: List[float], window: int = 10) -> str:
        """Detect reward curve trend."""
        if len(rewards) < window:
            return "insufficient_data"
        
        # Compute rolling mean
        rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        if len(rolling) < 2:
            return "insufficient_data"
        
        # Check trend
        first_half = np.mean(rolling[:len(rolling)//2])
        second_half = np.mean(rolling[len(rolling)//2:])
        
        diff = second_half - first_half
        std = np.std(rolling)
        
        if std > 0:
            normalized_diff = diff / std
        else:
            normalized_diff = 0
        
        # Detect oscillation (high variance in rolling mean)
        rolling_diff = np.diff(rolling)
        oscillation = np.std(rolling_diff) / (np.abs(np.mean(rolling_diff)) + 1e-6)
        
        if oscillation > 0.5:
            return "oscillating"
        elif normalized_diff > 0.3:
            return "increasing"
        elif normalized_diff < -0.3:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_entropy_trend(self, entropies: List[float]) -> str:
        """Detect entropy trend (exploration behavior)."""
        if len(entropies) < 5:
            return "insufficient_data"
        
        # Compute trend
        x = np.arange(len(entropies))
        coeffs = np.polyfit(x, entropies, 1)
        slope = coeffs[0]
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope < -0.05:
            return "decreasing"  # Less exploration over time (normal)
        elif slope > 0.05:
            return "increasing"  # More exploration (unusual)
        else:
            return "stable"
    
    def _detect_value_alignment(self, returns: List[float], values: List[float]) -> float:
        """Detect how well value function aligns with actual returns."""
        if len(returns) < 10 or len(values) < 10:
            return 0.0
        
        # Simple correlation
        try:
            corr = np.corrcoef(returns[:len(values)], values)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _analyze_failure_modes(
        self,
        rewards: List[float],
        entropies: List[float],
        grad_norms: List[float],
        value_losses: List[float],
    ) -> Tuple[List[str], List[str]]:
        """Analyze and detect failure modes."""
        failures = []
        warnings = []
        
        # Check 1: Training instability (high gradient norms)
        if grad_norms:
            max_grad = max(grad_norms)
            if max_grad > 100:
                failures.append(f"gradient_explosion: max_grad={max_grad:.1f}")
            elif max_grad > 50:
                warnings.append(f"high_gradients: max_grad={max_grad:.1f}")
        
        # Check 2: Reward collapse
        if len(rewards) >= 20:
            recent = rewards[-20:]
            if all(r <= 0 for r in recent):
                failures.append("reward_collapse: all recent rewards <= 0")
        
        # Check 3: Entropy collapse (too deterministic too early)
        if len(entropies) >= 20:
            early_entropy = np.mean(entropies[:10])
            late_entropy = np.mean(entropies[-10:])
            if late_entropy < 0.1 * early_entropy and early_entropy > 0.5:
                failures.append("entropy_collapse: policy became deterministic too early")
        
        # Check 4: Training plateau (no improvement)
        if len(rewards) >= 50:
            early = np.mean(rewards[:25])
            late = np.mean(rewards[-25:])
            if late <= early + 0.1 * abs(early):
                warnings.append(f"training_plateau: no significant improvement ({early:.2f} -> {late:.2f})")
        
        # Check 5: Value function issues
        if value_losses:
            max_vloss = max(value_losses)
            if max_vloss > 100:
                failures.append(f"value_function_divergence: max_vloss={max_vloss:.1f}")
        
        # Check 6: Oscillating rewards
        if len(rewards) >= 20:
            trend = self._detect_reward_trend(rewards)
            if trend == "oscillating":
                warnings.append("oscillating_rewards: training may be unstable")
        
        return failures, warnings
    
    def _generate_recommendations(
        self,
        failures: List[str],
        warnings: List[str],
        reward_trend: str,
        entropy_trend: str,
        eval_metrics: Optional[Dict],
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recs = []
        
        # Based on failures
        if any("gradient" in f for f in failures):
            recs.append("Reduce learning rate or increase gradient clipping (--max-grad-norm)")
            recs.append("Try smaller batch size or add gradient accumulation")
        
        if any("entropy_collapse" in f for f in failures):
            recs.append("Add entropy bonus (--entropy-coef 0.01-0.1)")
            recs.append("Increase exploration noise or decrease KL penalty")
        
        if any("reward_collapse" in f for f in failures):
            recs.append("Check reward function implementation")
            recs.append("Verify environment reset and goal placement")
            recs.append("Try dense rewards (--use-dense-rewards)")
        
        if any("plateau" in w for w in warnings):
            recs.append("Increase learning rate or use learning rate warmup")
            recs.append("Try curriculum learning for harder scenarios")
            recs.append("Add exploration noise (--action-noise 0.1)")
        
        # Based on trends
        if reward_trend == "decreasing":
            recs.append("Learning is diverging - reduce learning rate")
            recs.append("Check reward scaling and normalization")
        
        if entropy_trend == "decreasing" and "entropy_collapse" not in failures:
            recs.append("Normal exploration decay - consider entropy schedule")
        
        # Based on eval metrics
        if eval_metrics:
            summary = eval_metrics.get("summary", {})
            ade = summary.get("ade_mean", float('inf'))
            fde = summary.get("fde_mean", float('inf'))
            success = summary.get("success_rate", 0)
            
            if ade > 10:
                recs.append(f"High ADE ({ade:.2f}m) - RL needs more training or better rewards")
            if success < 20:
                recs.append(f"Low success rate ({success:.1f}%) - check goal conditions and rewards")
        
        if not recs:
            recs.append("Training appears healthy - consider longer training or hyperparameter tuning")
        
        return recs
    
    def analyze_run(self, run_dir: str) -> TrainingDiagnostics:
        """Analyze a single training run."""
        # Extract run_id from path
        run_id = os.path.basename(run_dir)
        
        # Load metrics
        train_metrics = self._load_train_metrics(run_dir)
        eval_metrics = self._load_eval_metrics(run_dir)
        
        # Initialize diagnostics
        diag = TrainingDiagnostics(
            run_id=run_id,
            checkpoint_path=run_dir,
        )
        
        if train_metrics is None:
            diag.warnings.append("No training metrics found")
            return diag
        
        # Extract training metrics - handle different key names
        episode_rewards = None
        if train_metrics and isinstance(train_metrics, dict):
            episode_rewards = (
                train_metrics.get("episode_returns") or 
                train_metrics.get("episode_rewards") or
                train_metrics.get("rewards") or
                train_metrics.get("avg_rewards")
            )
        
        if episode_rewards and isinstance(episode_rewards, list):
            try:
                diag.reward_history = [float(x) for x in episode_rewards if isinstance(x, (int, float))]
                diag.total_episodes = len(diag.reward_history)
                if diag.total_episodes > 0:
                    diag.final_reward = diag.reward_history[-1]
                    diag.best_reward = max(diag.reward_history)
                    diag.reward_std = float(np.std(diag.reward_history))
            except (ValueError, TypeError):
                pass
        
        if "update_count" in train_metrics:
            diag.total_updates = train_metrics["update_count"]
        
        if "learning_rates" in train_metrics and train_metrics["learning_rates"]:
            diag.learning_rate = train_metrics["learning_rates"][-1]
        
        if "entropies" in train_metrics and train_metrics["entropies"]:
            diag.entropy_history = train_metrics["entropies"]
            diag.entropy = train_metrics["entropies"][-1]
        
        if "grad_norms" in train_metrics and train_metrics["grad_norms"]:
            diag.grad_norm = np.mean(train_metrics["grad_norms"][-10:])
        
        if "value_losses" in train_metrics and train_metrics["value_losses"]:
            diag.value_loss = np.mean(train_metrics["value_losses"][-10:])
        
        if "policy_losses" in train_metrics and train_metrics["policy_losses"]:
            diag.policy_loss = np.mean(train_metrics["policy_losses"][-10:])
        
        # Detect trends
        diag.reward_trend = self._detect_reward_trend(diag.reward_history)
        diag.entropy_trend = self._detect_entropy_trend(diag.entropy_history)
        
        # Value alignment
        if "returns" in train_metrics and "values" in train_metrics:
            diag.value_alignment = self._detect_value_alignment(
                train_metrics["returns"], train_metrics["values"]
            )
        
        # Load eval metrics
        if eval_metrics:
            summary = eval_metrics.get("summary", {})
            diag.eval_ade = summary.get("ade_mean")
            diag.eval_fde = summary.get("fde_mean")
            diag.eval_success_rate = summary.get("success_rate")
        
        # Analyze failure modes
        grad_norms = train_metrics.get("grad_norms", [])
        value_losses = train_metrics.get("value_losses", [])
        failures, warnings = self._analyze_failure_modes(
            diag.reward_history,
            diag.entropy_history,
            grad_norms,
            value_losses,
        )
        diag.failure_modes = failures
        diag.warnings.extend(warnings)
        
        # Generate recommendations
        diag.recommendations = self._generate_recommendations(
            failures,
            warnings,
            diag.reward_trend,
            diag.entropy_trend,
            eval_metrics,
        )
        
        return diag
    
    def get_latest_run(self) -> Optional[str]:
        """Get the most recent training run."""
        return self.run_dirs[0] if self.run_dirs else None
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all available training runs with basic info."""
        runs = []
        for run_dir in self.run_dirs:
            run_id = os.path.basename(run_dir)
            
            try:
                train_metrics = self._load_train_metrics(run_dir)
            except Exception:
                train_metrics = None
                
            try:
                eval_metrics = self._load_eval_metrics(run_dir)
            except Exception:
                eval_metrics = None
            
            # Handle different key names - try multiple
            episode_rewards = None
            if train_metrics and isinstance(train_metrics, dict):
                episode_rewards = (
                    train_metrics.get("episode_returns") or 
                    train_metrics.get("episode_rewards") or
                    train_metrics.get("rewards") or
                    train_metrics.get("avg_rewards")
                )
            
            final_reward = None
            episodes_count = 0
            if episode_rewards and isinstance(episode_rewards, list):
                try:
                    final_reward = float(episode_rewards[-1]) if episode_rewards else None
                    episodes_count = len(episode_rewards)
                except (ValueError, TypeError, IndexError):
                    pass
            
            info = {
                "run_id": run_id,
                "path": run_dir,
                "episodes": episodes_count,
                "final_reward": final_reward,
                "has_eval": eval_metrics is not None,
            }
            
            if eval_metrics:
                summary = eval_metrics.get("summary", {})
                info["ade"] = summary.get("ade_mean")
                info["fde"] = summary.get("fde_mean")
                info["success"] = summary.get("success_rate")
            
            runs.append(info)
        
        return runs
    
    def generate_report(self, run_dir: str) -> str:
        """Generate a markdown report for a training run."""
        diag = self.analyze_run(run_dir)
        
        lines = [
            f"# Training Diagnostics: {diag.run_id}",
            "",
            f"**Checkpoint:** `{diag.checkpoint_path}`",
            "",
            "## Training Summary",
            "",
            f"- Episodes: {diag.total_episodes}",
            f"- Updates: {diag.total_updates}",
            f"- Final Reward: `{diag.final_reward:.2f}`",
            f"- Best Reward: `{diag.best_reward:.2f}`",
            f"- Reward Std: `{diag.reward_std:.2f}`",
            "",
            "## Learning Dynamics",
            "",
            f"- Learning Rate: `{diag.learning_rate:.6f}`",
            f"- Entropy: `{diag.entropy:.4f}` (trend: {diag.entropy_trend})",
            f"- Gradient Norm: `{diag.grad_norm:.2f}`",
            f"- Value Loss: `{diag.value_loss:.4f}`",
            f"- Policy Loss: `{diag.policy_loss:.4f}`",
            f"- Value Alignment: `{diag.value_alignment:.3f}`",
            f"- Reward Trend: **{diag.reward_trend}**",
            "",
        ]
        
        if diag.eval_ade is not None:
            lines.extend([
                "## Evaluation Metrics",
                "",
                f"- ADE: `{diag.eval_ade:.3f}m`",
                f"- FDE: `{diag.eval_fde:.3f}m`",
                f"- Success Rate: `{diag.eval_success_rate:.1f}%`",
                "",
            ])
        
        if diag.failure_modes:
            lines.extend([
                "## ⚠️ Failure Modes Detected",
                "",
            ])
            for fm in diag.failure_modes:
                lines.append(f"- **{fm}**")
            lines.append("")
        
        if diag.warnings:
            lines.extend([
                "## Warnings",
                "",
            ])
            for w in diag.warnings:
                lines.append(f"- {w}")
            lines.append("")
        
        lines.extend([
            "## Recommendations",
            "",
        ])
        for i, rec in enumerate(diag.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Training Diagnostics for RL After SFT")
    parser.add_argument("--run-id", type=str, help="Specific run ID to analyze")
    parser.add_argument("--latest", action="store_true", help="Analyze latest run")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--failure-modes", action="store_true", help="Show failure mode analysis")
    parser.add_argument("--output-dir", type=str, default="out", help="Output directory")
    parser.add_argument("--output", type=str, help="Save report to file")
    
    args = parser.parse_args()
    
    analyzer = TrainingDiagnosticsAnalyzer(args.output_dir)
    
    # List runs
    if args.list:
        print("Available training runs:")
        print("-" * 80)
        for run in analyzer.list_runs():
            status = "✓" if run["has_eval"] else "○"
            reward = f"{run['final_reward']:.2f}" if run["final_reward"] is not None else "N/A"
            ade = f"{run['ade']:.3f}m" if run.get("ade") else "N/A"
            success = f"{run['success']:.1f}%" if run.get("success") else "N/A"
            print(f"{status} {run['run_id'][:50]:<50} | {run['episodes']:>4} ep | {reward:>8} reward | ADE: {ade:>8} | Success: {success:>8}")
        return
    
    # Determine run to analyze
    run_dir = None
    if args.run_id:
        run_dir = os.path.join(args.output_dir, args.run_id)
    elif args.latest:
        run_dir = analyzer.get_latest_run()
    
    if run_dir is None:
        print("No run specified. Use --run-id or --latest")
        print("Available runs:")
        for run in analyzer.list_runs()[:5]:
            print(f"  - {run['run_id']}")
        return
    
    if not os.path.exists(run_dir):
        print(f"Run not found: {run_dir}")
        return
    
    # Analyze run
    if args.report:
        report = analyzer.generate_report(run_dir)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        else:
            print(report)
    elif args.failure_modes:
        diag = analyzer.analyze_run(run_dir)
        print(f"Failure modes for {diag.run_id}:")
        print("-" * 40)
        if diag.failure_modes:
            for fm in diag.failure_modes:
                print(f"  ⚠️ {fm}")
        else:
            print("  ✓ No critical failure modes detected")
        
        if diag.warnings:
            print("\nWarnings:")
            for w in diag.warnings:
                print(f"  ⚡ {w}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(diag.recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        # Default: show summary
        diag = analyzer.analyze_run(run_dir)
        print(f"Training Diagnostics: {diag.run_id}")
        print("=" * 60)
        print(f"Episodes: {diag.total_episodes} | Updates: {diag.total_updates}")
        print(f"Final Reward: {diag.final_reward:.2f} | Best: {diag.best_reward:.2f}")
        print(f"Reward Trend: {diag.reward_trend} | Entropy Trend: {diag.entropy_trend}")
        
        if diag.eval_ade is not None:
            print(f"\nEval ADE: {diag.eval_ade:.3f}m | FDE: {diag.eval_fde:.3f}m | Success: {diag.eval_success_rate:.1f}%")
        
        if diag.failure_modes:
            print(f"\n⚠️ Failure Modes: {len(diag.failure_modes)}")
            for fm in diag.failure_modes[:3]:
                print(f"  - {fm}")
        
        if args.output:
            report = analyzer.generate_report(run_dir)
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()

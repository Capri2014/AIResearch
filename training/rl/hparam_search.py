#!/usr/bin/env python3
"""
Hyperparameter Search for RL Delta Waypoint Training.

Supports grid search and random search over key hyperparameters:
- Learning rate
- LoRA rank
- Entropy coefficient
- Clip ratio
- Horizon
- Gamma

Usage:
    # Grid search over learning rates
    python hparam_search.py --method grid --param lr:1e-4,3e-4,1e-3 --episodes 200 --seeds 3
    
    # Random search with 20 trials
    python hparam_search.py --method random --trials 20 --episodes 100 --seeds 2
    
    # Targeted search based on previous results
    python hparam_search.py --method successive-halving --param lr:1e-5,1e-4,1e-3 --episodes 300
"""
import os
import sys
import json
import argparse
import numpy as np
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_seed_train import run_training_with_seed, TrainingConfig


@dataclass
class HParamConfig:
    """Configuration for a single hyperparameter trial."""
    # Core training params
    lr: float = 3e-4
    horizon: int = 20
    gamma: float = 0.99
    lam: float = 0.95
    
    # PPO/GRPO params
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_epochs: int = 10
    
    # LoRA params
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Environment
    scenario: str = "highway"
    
    # Search metadata
    trial_id: str = ""
    search_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding metadata."""
        d = asdict(self)
        d.pop('trial_id', None)
        d.pop('search_method', None)
        return d
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI argument list."""
        args = [
            "--lr", str(self.lr),
            "--horizon", str(self.horizon),
            "--gamma", str(self.gamma),
            "--lam", str(self.lam),
            "--clip-epsilon", str(self.clip_epsilon),
            "--entropy-coef", str(self.entropy_coef),
            "--value-coef", str(self.value_coef),
            "--max-epochs", str(self.max_epochs),
            "--scenario", self.scenario,
        ]
        if self.use_lora:
            args.extend([
                "--use-lora",
                "--lora-rank", str(self.lora_rank),
                "--lora-alpha", str(self.lora_alpha),
                "--lora-dropout", str(self.lora_dropout),
            ])
        return args


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_id: str
    config: HParamConfig
    metrics: Dict[str, float]
    runtime_seconds: float
    status: str  # "completed", "failed", "timeout"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "status": self.status,
            "error": self.error,
        }


class HParamSearcher:
    """Hyperparameter search manager."""
    
    # Default search spaces
    DEFAULT_SEARCH_SPACES = {
        "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "horizon": [10, 15, 20, 30],
        "gamma": [0.95, 0.99, 0.995],
        "lam": [0.9, 0.95, 0.98],
        "clip_epsilon": [0.1, 0.2, 0.3],
        "entropy_coef": [0.0, 0.001, 0.01, 0.05],
        "value_coef": [0.25, 0.5, 1.0],
        "lora_rank": [2, 4, 8, 16, 32],
        "lora_alpha": [8, 16, 32, 64],
    }
    
    def __init__(
        self,
        method: str = "grid",
        output_dir: str = "out/hparam_search",
        max_parallel: int = 4,
        timeout_minutes: int = 30,
    ):
        self.method = method
        self.output_dir = output_dir
        self.max_parallel = max_parallel
        self.timeout_minutes = timeout_minutes
        self.trials: List[TrialResult] = []
        
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_grid(self, param_specs: Dict[str, str]) -> List[HParamConfig]:
        """Generate trials from grid specification."""
        # Parse param specs like "lr:1e-4,3e-4,1e-3"
        param_grid = {}
        for spec in param_specs.values():
            key, values = spec.split(":")
            if "," in values:
                parsed_values = []
                for v in values.split(","):
                    try:
                        parsed_values.append(float(v))
                    except ValueError:
                        parsed_values.append(v)
                param_grid[key] = parsed_values
            else:
                try:
                    param_grid[key] = [float(values)]
                except ValueError:
                    param_grid[key] = [values]
        
        # Generate all combinations
        keys = list(param_grid.keys())
        combinations = list(product(*[param_grid[k] for k in keys]))
        
        configs = []
        for i, combo in enumerate(combinations):
            config = HParamConfig()
            config.trial_id = f"grid_{i:03d}"
            config.search_method = "grid"
            for key, value in zip(keys, combo):
                setattr(config, key, value)
            configs.append(config)
        
        return configs
    
    def generate_random(
        self,
        param_specs: Dict[str, str],
        n_trials: int = 20,
    ) -> List[HParamConfig]:
        """Generate random search trials."""
        configs = []
        for i in range(n_trials):
            config = HParamConfig()
            config.trial_id = f"random_{i:03d}"
            config.search_method = "random"
            
            # Parse and randomize each parameter
            for spec in param_specs.values():
                key, values = spec.split(":")
                if "," in values:
                    value_list = []
                    for v in values.split(","):
                        try:
                            value_list.append(float(v))
                        except ValueError:
                            value_list.append(v)
                    setattr(config, key, random.choice(value_list))
                else:
                    # For continuous params, use log-uniform sampling
                    try:
                        float_val = float(values)
                        # Generate log-uniform between 1e-5 and 1e-1
                        log_min = -5
                        log_max = -1
                        log_val = random.uniform(log_min, log_max)
                        setattr(config, key, 10 ** log_val)
                    except ValueError:
                        setattr(config, key, values)
            
            configs.append(config)
        
        return configs
    
    def generate_successive_halving(
        self,
        param_specs: Dict[str, str],
        n_initial: int = 16,
        reduction_factor: int = 2,
    ) -> List[HParamConfig]:
        """Generate successive halving trials."""
        # Start with random sampling
        configs = self.generate_random(param_specs, n_initial)
        
        # Mark for successive halving
        for config in configs:
            config.trial_id = config.trial_id.replace("random", "halving")
        
        return configs
    
    def run_trial(self, config: HParamConfig, train_script: str) -> TrialResult:
        """Run a single trial."""
        import time
        start_time = time.time()
        
        try:
            # Build command
            cmd = ["python", train_script] + config.to_cli_args()
            cmd.extend(["--episodes", "50"])  # Quick smoke test first
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_minutes * 60,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            
            runtime = time.time() - start_time
            
            if result.returncode == 0:
                # Parse metrics from output
                metrics = self._parse_metrics(result.stdout)
                return TrialResult(
                    trial_id=config.trial_id,
                    config=config,
                    metrics=metrics,
                    runtime_seconds=runtime,
                    status="completed",
                )
            else:
                return TrialResult(
                    trial_id=config.trial_id,
                    config=config,
                    metrics={},
                    runtime_seconds=runtime,
                    status="failed",
                    error=result.stderr[:500],
                )
                
        except subprocess.TimeoutExpired:
            return TrialResult(
                trial_id=config.trial_id,
                config=config,
                metrics={},
                runtime_seconds=self.timeout_minutes * 60,
                status="timeout",
            )
        except Exception as e:
            return TrialResult(
                trial_id=config.trial_id,
                config=config,
                metrics={},
                runtime_seconds=time.time() - start_time,
                status="failed",
                error=str(e)[:500],
            )
    
    def _parse_metrics(self, stdout: str) -> Dict[str, float]:
        """Parse metrics from training output."""
        metrics = {}
        
        # Look for common metric patterns
        for line in stdout.split("\n"):
            if "avg_reward" in line.lower() or "return" in line.lower():
                try:
                    # Extract number after colon or equals
                    parts = line.split(":")
                    if len(parts) > 1:
                        val = float(parts[-1].strip().split()[0])
                        metrics["avg_reward"] = val
                except:
                    pass
            
            if "success" in line.lower() or "goal" in line.lower():
                try:
                    parts = line.split(":")
                    if len(parts) > 1:
                        val = float(parts[-1].strip().split()[0].replace("%", ""))
                        metrics["success_rate"] = val
                except:
                    pass
        
        return metrics
    
    def run_search(
        self,
        configs: List[HParamConfig],
        train_script: str = "train_grpo_delta_waypoint.py",
    ) -> List[TrialResult]:
        """Run hyperparameter search."""
        print(f"Starting {self.method} search with {len(configs)} trials")
        print(f"Output directory: {self.output_dir}")
        
        results = []
        
        # Run trials (sequentially for stability)
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Running trial {config.trial_id}")
            print(f"  Config: {config.to_dict()}")
            
            result = self.run_trial(config, train_script)
            results.append(result)
            
            # Save intermediate results
            self._save_results(results)
            
            print(f"  Status: {result.status}")
            if result.metrics:
                print(f"  Metrics: {result.metrics}")
        
        self.trials = results
        return results
    
    def _save_results(self, results: List[TrialResult]):
        """Save intermediate results."""
        output_file = os.path.join(
            self.output_dir, 
            f"results_{self.timestamp}.json"
        )
        
        data = {
            "timestamp": self.timestamp,
            "method": self.method,
            "n_trials": len(results),
            "trials": [r.to_dict() for r in results],
        }
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_best_config(self, metric: str = "avg_reward") -> Tuple[HParamConfig, float]:
        """Get best configuration by metric."""
        completed = [r for r in self.trials if r.status == "completed"]
        
        if not completed:
            raise ValueError("No completed trials")
        
        # Find best by metric (higher is better)
        best = max(completed, key=lambda r: r.metrics.get(metric, -float("inf")))
        
        return best.config, best.metrics.get(metric, 0.0)
    
    def generate_report(self) -> str:
        """Generate markdown report of search results."""
        report_lines = [
            f"# Hyperparameter Search Report",
            f"",
            f"**Method:** {self.method}",
            f"**Timestamp:** {self.timestamp}",
            f"**Total Trials:** {len(self.trials)}",
            f"",
            f"## Summary",
            f"",
        ]
        
        completed = [r for r in self.trials if r.status == "completed"]
        failed = [r for r in self.trials if r.status == "failed"]
        
        report_lines.append(f"- Completed: {len(completed)}")
        report_lines.append(f"- Failed: {len(failed)}")
        report_lines.append("")
        
        if completed:
            best_config, best_metric = self.get_best_config()
            report_lines.append(f"## Best Configuration")
            report_lines.append(f"")
            report_lines.append(f"**Metric:** {best_metric:.4f}")
            report_lines.append(f"")
            for key, value in best_config.to_dict().items():
                report_lines.append(f"- {key}: {value}")
            report_lines.append("")
        
        report_lines.append(f"## All Trials")
        report_lines.append(f"")
        report_lines.append(f"| Trial ID | Status | " + 
                          " | ".join([k for k in ['reward', 'success'] if any(r.metrics.get(k) for r in completed)]) + 
                          " | Runtime (s) |")
        report_lines.append(f"|----------|--------|" + 
                          "|---" * len([k for k in ['reward', 'success'] if any(r.metrics.get(k) for r in completed)]) +
                          "|-------------|")
        
        for result in self.trials:
            metrics_str = " | ".join([
                f"{result.metrics.get(k, '-'):.2f}" 
                for k in ['reward', 'success'] 
                if any(r.metrics.get(k) for r in completed)
            ])
            report_lines.append(
                f"| {result.trial_id} | {result.status} | {metrics_str} | {result.runtime_seconds:.1f} |"
            )
        
        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search")
    parser.add_argument("--method", type=str, default="grid",
                        choices=["grid", "random", "successive-halving"],
                        help="Search method")
    parser.add_argument("--param", type=str, nargs="+", default=[],
                        help="Parameter specs like lr:1e-4,3e-4,1e-3")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials for random search")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes per trial")
    parser.add_argument("--seeds", type=int, default=2,
                        help="Number of seeds per trial")
    parser.add_argument("--output-dir", type=str, default="out/hparam_search",
                        help="Output directory")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Max parallel trials")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout per trial (minutes)")
    parser.add_argument("--train-script", type=str, 
                        default="train_grpo_delta_waypoint.py",
                        help="Training script to use")
    
    args = parser.parse_args()
    
    # Parse param specs
    param_specs = {}
    for p in args.param:
        if ":" in p:
            key, values = p.split(":", 1)
            param_specs[key] = f"{key}:{values}"
    
    # Create searcher
    searcher = HParamSearcher(
        method=args.method,
        output_dir=args.output_dir,
        max_parallel=args.max_parallel,
        timeout_minutes=args.timeout,
    )
    
    # Generate configs
    if args.method == "grid":
        configs = searcher.generate_grid(param_specs)
    elif args.method == "random":
        configs = searcher.generate_random(param_specs, args.trials)
    else:
        configs = searcher.generate_successive_halving(param_specs, args.trials)
    
    # Run search
    results = searcher.run_search(configs, args.train_script)
    
    # Generate and save report
    report = searcher.generate_report()
    report_file = os.path.join(args.output_dir, f"report_{searcher.timestamp}.md")
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\n{report}")
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()

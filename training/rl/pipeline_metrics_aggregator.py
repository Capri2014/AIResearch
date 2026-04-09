#!/usr/bin/env python3
"""
Pipeline Metrics Aggregator

Aggregates metrics across all pipeline stages (SSL, BC, RL, eval) and provides
unified performance tracking and reporting.

This module bridges the driving-first pipeline with centralized metrics collection:
Waymo episodes → PyTorch SSL pretrain → waypoint BC → CARLA evaluation

Usage:
    python pipeline_metrics_aggregator.py --stage ssl --metrics-dir checkpoints/ssl
    python pipeline_metrics_aggregator.py --stage bc --metrics-dir checkpoints/bc
    python pipeline_metrics_aggregator.py --report --output metrics_report.json
"""

import argparse
import json
import os
import glob
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class SSLMetrics:
    """Self-supervised pretraining metrics."""
    encoder_path: str
    epoch: int
    loss: float
    val_loss: float
    learning_rate: float
    batch_size: int
    training_time_hours: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BCMetrics:
    """Behavior cloning (waypoint prediction) metrics."""
    checkpoint_path: str
    epoch: int
    train_loss: float
    val_loss: float
    waypoint_mae: float       # meters
    waypoint_mse: float
    heading_mae: float        # radians
    speed_mae: float          # m/s
    batch_size: int
    learning_rate: float
    num_training_samples: int
    training_time_hours: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RLMetrics:
    """Reinforcement learning metrics."""
    checkpoint_path: str
    episode: int
    mean_reward: float
    std_reward: float
    success_rate: float
    collision_rate: float
    offroad_rate: float
    route_completion: float
    avg_episode_length: float
    num_env_steps: int
    learning_rate: float
    entropy_coef: float
    value_loss: float
    policy_loss: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvalMetrics:
    """CARLA evaluation metrics."""
    eval_dir: str
    num_episodes: int
    route_completion_avg: float
    route_completion_std: float
    collision_rate: float
    offroad_rate: float
    avg_deviation_m: float
    success_rate: float
    avg_episode_time: float
    timestamp: str
    town_stats: Dict[str, float] = field(default_factory=dict)
    scenario_stats: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        return result


@dataclass
class PipelineRun:
    """Complete pipeline run record."""
    run_id: str
    start_time: str
    end_time: Optional[str]
    status: str  # running, completed, failed
    ssl: Optional[SSLMetrics] = None
    bc: Optional[BCMetrics] = None
    rl: Optional[RLMetrics] = None
    eval: Optional[EvalMetrics] = None
    notes: str = ""
    
    def to_dict(self) -> Dict:
        result = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "notes": self.notes
        }
        if self.ssl:
            result["ssl"] = self.ssl.to_dict()
        if self.bc:
            result["bc"] = self.bc.to_dict()
        if self.rl:
            result["rl"] = self.rl.to_dict()
        if self.eval:
            result["eval"] = self.eval.to_dict()
        return result


class SSLMetricsCollector:
    """Collects and parses SSL pretraining metrics."""
    
    METRIC_PATTERNS = [
        "checkpoints/ssl_*/metrics.json",
        "checkpoints/pretrain_*/metrics.json",
        "checkpoints/encoder_*/metrics.json"
    ]
    
    @classmethod
    def discover(cls, checkpoint_dir: str = "checkpoints") -> List[SSLMetrics]:
        """Discover SSL metric files in checkpoint directory."""
        metrics_list = []
        
        for pattern in cls.METRIC_PATTERNS:
            for metric_file in glob.glob(os.path.join(checkpoint_dir, pattern.replace("checkpoints/", ""))):
                if os.path.exists(metric_file.replace("metrics.json", "metrics.json")):
                    try:
                        with open(metric_file) as f:
                            data = json.load(f)
                            metrics_list.append(SSLMetrics(
                                encoder_path=data.get("checkpoint_path", metric_file.replace("/metrics.json", ".pt")),
                                epoch=data.get("epoch", 0),
                                loss=data.get("loss", 0.0),
                                val_loss=data.get("val_loss", data.get("loss", 0.0)),
                                learning_rate=data.get("learning_rate", 1e-4),
                                batch_size=data.get("batch_size", 32),
                                training_time_hours=data.get("training_time_hours", 0.0),
                                timestamp=data.get("timestamp", datetime.now().isoformat())
                            ))
                    except (json.JSONDecodeError, KeyError):
                        pass
        
        return metrics_list
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> Optional[SSLMetrics]:
        """Load metrics from specific checkpoint."""
        metrics_file = checkpoint_path.replace(".pt", "_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                data = json.load(f)
                return SSLMetrics(
                    encoder_path=checkpoint_path,
                    epoch=data.get("epoch", 0),
                    loss=data.get("loss", 0.0),
                    val_loss=data.get("val_loss", data.get("loss", 0.0)),
                    learning_rate=data.get("learning_rate", 1e-4),
                    batch_size=data.get("batch_size", 32),
                    training_time_hours=data.get("training_time_hours", 0.0),
                    timestamp=data.get("timestamp", datetime.now().isoformat())
                )
        return None


class BCMetricsCollector:
    """Collects and parses BC (waypoint prediction) metrics."""
    
    METRIC_PATTERNS = [
        "checkpoints/bc_*/metrics.json",
        "checkpoints/waypoint_bc_*/metrics.json"
    ]
    
    @classmethod
    def discover(cls, checkpoint_dir: str = "checkpoints") -> List[BCMetrics]:
        """Discover BC metric files."""
        metrics_list = []
        
        for pattern in cls.METRIC_PATTERNS:
            for metric_file in glob.glob(os.path.join(checkpoint_dir, pattern.replace("checkpoints/", ""))):
                if os.path.exists(metric_file):
                    try:
                        with open(metric_file) as f:
                            data = json.load(f)
                            metrics_list.append(BCMetrics(
                                checkpoint_path=data.get("checkpoint_path", metric_file.replace("/metrics.json", ".pt")),
                                epoch=data.get("epoch", 0),
                                train_loss=data.get("train_loss", data.get("loss", 0.0)),
                                val_loss=data.get("val_loss", 0.0),
                                waypoint_mae=data.get("waypoint_mae", 0.0),
                                waypoint_mse=data.get("waypoint_mse", 0.0),
                                heading_mae=data.get("heading_mae", 0.0),
                                speed_mae=data.get("speed_mae", 0.0),
                                batch_size=data.get("batch_size", 64),
                                learning_rate=data.get("learning_rate", 1e-3),
                                num_training_samples=data.get("num_samples", data.get("num_training_samples", 0)),
                                training_time_hours=data.get("training_time_hours", 0.0),
                                timestamp=data.get("timestamp", datetime.now().isoformat())
                            ))
                    except (json.JSONDecodeError, KeyError):
                        pass
        
        return metrics_list
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> Optional[BCMetrics]:
        """Load metrics from specific checkpoint."""
        metrics_file = checkpoint_path.replace(".pt", "_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                data = json.load(f)
                return BCMetrics(
                    checkpoint_path=checkpoint_path,
                    epoch=data.get("epoch", 0),
                    train_loss=data.get("train_loss", data.get("loss", 0.0)),
                    val_loss=data.get("val_loss", 0.0),
                    waypoint_mae=data.get("waypoint_mae", 0.0),
                    waypoint_mse=data.get("waypoint_mse", 0.0),
                    heading_mae=data.get("heading_mae", 0.0),
                    speed_mae=data.get("speed_mae", 0.0),
                    batch_size=data.get("batch_size", 64),
                    learning_rate=data.get("learning_rate", 1e-3),
                    num_training_samples=data.get("num_samples", 0),
                    training_time_hours=data.get("training_time_hours", 0.0),
                    timestamp=data.get("timestamp", datetime.now().isoformat())
                )
        return None


class RLMetricsCollector:
    """Collects and parses RL training metrics."""
    
    METRIC_PATTERNS = [
        "checkpoints/rl_*/metrics.json",
        "checkpoints/ppo_*/metrics.json"
    ]
    
    @classmethod
    def discover(cls, checkpoint_dir: str = "checkpoints") -> List[RLMetrics]:
        """Discover RL metric files."""
        metrics_list = []
        
        for pattern in cls.METRIC_PATTERNS:
            for metric_file in glob.glob(os.path.join(checkpoint_dir, pattern.replace("checkpoints/", ""))):
                if os.path.exists(metric_file):
                    try:
                        with open(metric_file) as f:
                            data = json.load(f)
                            metrics_list.append(RLMetrics(
                                checkpoint_path=data.get("checkpoint_path", metric_file.replace("/metrics.json", ".pt")),
                                episode=data.get("episode", data.get("step", 0)),
                                mean_reward=data.get("mean_reward", 0.0),
                                std_reward=data.get("std_reward", 0.0),
                                success_rate=data.get("success_rate", 0.0),
                                collision_rate=data.get("collision_rate", 0.0),
                                offroad_rate=data.get("offroad_rate", 0.0),
                                route_completion=data.get("route_completion", 0.0),
                                avg_episode_length=data.get("avg_episode_length", 0.0),
                                num_env_steps=data.get("num_env_steps", 0),
                                learning_rate=data.get("learning_rate", 3e-4),
                                entropy_coef=data.get("entropy_coef", 0.01),
                                value_loss=data.get("value_loss", 0.0),
                                policy_loss=data.get("policy_loss", 0.0),
                                timestamp=data.get("timestamp", datetime.now().isoformat())
                            ))
                    except (json.JSONDecodeError, KeyError):
                        pass
        
        return metrics_list
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> Optional[RLMetrics]:
        """Load metrics from specific checkpoint."""
        metrics_file = checkpoint_path.replace(".pt", "_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                data = json.load(f)
                return RLMetrics(
                    checkpoint_path=checkpoint_path,
                    episode=data.get("episode", 0),
                    mean_reward=data.get("mean_reward", 0.0),
                    std_reward=data.get("std_reward", 0.0),
                    success_rate=data.get("success_rate", 0.0),
                    collision_rate=data.get("collision_rate", 0.0),
                    offroad_rate=data.get("offroad_rate", 0.0),
                    route_completion=data.get("route_completion", 0.0),
                    avg_episode_length=data.get("avg_episode_length", 0.0),
                    num_env_steps=data.get("num_env_steps", 0),
                    learning_rate=data.get("learning_rate", 3e-4),
                    entropy_coef=data.get("entropy_coef", 0.01),
                    value_loss=data.get("value_loss", 0.0),
                    policy_loss=data.get("policy_loss", 0.0),
                    timestamp=data.get("timestamp", datetime.now().isoformat())
                )
        return None


class EvalMetricsCollector:
    """Collects and parses CARLA evaluation metrics."""
    
    METRIC_PATTERNS = [
        "checkpoints/eval_*/metrics.json",
        "eval/*/metrics.json",
        "out/eval_*/metrics.json"
    ]
    
    @classmethod
    def discover(cls, base_dir: str = "checkpoints") -> List[EvalMetrics]:
        """Discover evaluation metric files."""
        metrics_list = []
        
        for pattern in cls.METRIC_PATTERNS:
            for metric_file in glob.glob(os.path.join(base_dir, pattern.replace("checkpoints/", "").replace("eval/", ""))):
                if os.path.exists(metric_file):
                    try:
                        with open(metric_file) as f:
                            data = json.load(f)
                            metrics_list.append(EvalMetrics(
                                eval_dir=os.path.dirname(metric_file),
                                num_episodes=data.get("num_episodes", 0),
                                route_completion_avg=data.get("route_completion_avg", 0.0),
                                route_completion_std=data.get("route_completion_std", 0.0),
                                collision_rate=data.get("collision_rate", 0.0),
                                offroad_rate=data.get("offroad_rate", 0.0),
                                avg_deviation_m=data.get("avg_deviation_m", 0.0),
                                success_rate=data.get("success_rate", 0.0),
                                avg_episode_time=data.get("avg_episode_time", 0.0),
                                town_stats=data.get("town_stats", {}),
                                scenario_stats=data.get("scenario_stats", {}),
                                timestamp=data.get("timestamp", datetime.now().isoformat())
                            ))
                    except (json.JSONDecodeError, KeyError):
                        pass
        
        return metrics_list


class PipelineMetricsAggregator:
    """Aggregates metrics across all pipeline stages."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", output_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.runs_file = os.path.join(output_dir, "pipeline_runs.json")
        self.runs: List[PipelineRun] = self._load_runs()
    
    def _load_runs(self) -> List[PipelineRun]:
        """Load saved pipeline runs."""
        if os.path.exists(self.runs_file):
            with open(self.runs_file) as f:
                data = json.load(f)
                runs = []
                for r in data.get("runs", []):
                    run = PipelineRun(
                        run_id=r["run_id"],
                        start_time=r["start_time"],
                        end_time=r.get("end_time"),
                        status=r["status"],
                        notes=r.get("notes", "")
                    )
                    if "ssl" in r:
                        run.ssl = SSLMetrics(**r["ssl"])
                    if "bc" in r:
                        run.bc = BCMetrics(**r["bc"])
                    if "rl" in r:
                        run.rl = RLMetrics(**r["rl"])
                    if "eval" in r:
                        run.eval = EvalMetrics(**r["eval"])
                    runs.append(run)
                return runs
        return []
    
    def _save_runs(self):
        """Save pipeline runs to file."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.runs_file, 'w') as f:
            json.dump({
                "runs": [r.to_dict() for r in self.runs],
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    
    def create_run(self, run_id: str = None) -> PipelineRun:
        """Create a new pipeline run."""
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = PipelineRun(
            run_id=run_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            status="running"
        )
        self.runs.append(run)
        self._save_runs()
        return run
    
    def update_run(self, run_id: str, ssl: SSLMetrics = None, bc: BCMetrics = None, 
                   rl: RLMetrics = None, eval: EvalMetrics = None, status: str = None):
        """Update a pipeline run with new metrics."""
        for run in self.runs:
            if run.run_id == run_id:
                if ssl:
                    run.ssl = ssl
                if bc:
                    run.bc = bc
                if rl:
                    run.rl = rl
                if eval:
                    run.eval = eval
                if status:
                    run.status = status
                    if status in ["completed", "failed"]:
                        run.end_time = datetime.now().isoformat()
                self._save_runs()
                return True
        return False
    
    def get_latest_metrics(self, stage: str) -> Optional[Any]:
        """Get the latest metrics for a stage."""
        if stage == "ssl":
            return SSLMetricsCollector.discover(self.checkpoint_dir)[-1] if SSLMetricsCollector.discover(self.checkpoint_dir) else None
        elif stage == "bc":
            return BCMetricsCollector.discover(self.checkpoint_dir)[-1] if BCMetricsCollector.discover(self.checkpoint_dir) else None
        elif stage == "rl":
            return RLMetricsCollector.discover(self.checkpoint_dir)[-1] if RLMetricsCollector.discover(self.checkpoint_dir) else None
        elif stage == "eval":
            return EvalMetricsCollector.discover(self.checkpoint_dir)[-1] if EvalMetricsCollector.discover(self.checkpoint_dir) else None
        return None
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive metrics report."""
        ssl_metrics = SSLMetricsCollector.discover(self.checkpoint_dir)
        bc_metrics = BCMetricsCollector.discover(self.checkpoint_dir)
        rl_metrics = RLMetricsCollector.discover(self.checkpoint_dir)
        eval_metrics = EvalMetricsCollector.discover(self.checkpoint_dir)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "checkpoint_dir": self.checkpoint_dir,
            "summary": {
                "total_ssl_runs": len(ssl_metrics),
                "total_bc_runs": len(bc_metrics),
                "total_rl_runs": len(rl_metrics),
                "total_eval_runs": len(eval_metrics),
                "total_pipeline_runs": len(self.runs)
            },
            "ssl": {
                "latest": ssl_metrics[-1].to_dict() if ssl_metrics else None,
                "all": [m.to_dict() for m in ssl_metrics]
            },
            "bc": {
                "latest": bc_metrics[-1].to_dict() if bc_metrics else None,
                "all": [m.to_dict() for m in bc_metrics]
            },
            "rl": {
                "latest": rl_metrics[-1].to_dict() if rl_metrics else None,
                "all": [m.to_dict() for m in rl_metrics]
            },
            "eval": {
                "latest": eval_metrics[-1].to_dict() if eval_metrics else None,
                "all": [m.to_dict() for m in eval_metrics]
            },
            "pipeline_runs": [r.to_dict() for r in self.runs]
        }
        
        return report
    
    def print_summary(self):
        """Print a human-readable metrics summary."""
        report = self.generate_report()
        
        print("=" * 60)
        print("PIPELINE METRICS SUMMARY")
        print("=" * 60)
        print(f"Generated: {report['generated_at']}")
        print(f"Checkpoint Dir: {report['checkpoint_dir']}")
        print()
        
        summary = report["summary"]
        print(f"Total Runs: SSL={summary['total_ssl_runs']} | BC={summary['total_bc_runs']} | RL={summary['total_rl_runs']} | Eval={summary['total_eval_runs']}")
        print()
        
        # SSL
        if report["ssl"]["latest"]:
            ssl = report["ssl"]["latest"]
            print(f"SSL Latest: epoch={ssl['epoch']}, loss={ssl['loss']:.4f}, val_loss={ssl['val_loss']:.4f}")
        
        # BC
        if report["bc"]["latest"]:
            bc = report["bc"]["latest"]
            print(f"BC Latest: epoch={bc['epoch']}, train_loss={bc['train_loss']:.4f}, val_loss={bc['val_loss']:.4f}")
            print(f"           waypoint_mae={bc['waypoint_mae']:.3f}m, heading_mae={bc['heading_mae']:.3f}rad")
        
        # RL
        if report["rl"]["latest"]:
            rl = report["rl"]["latest"]
            print(f"RL Latest: episode={rl['episode']}, reward={rl['mean_reward']:.2f}±{rl['std_reward']:.2f}")
            print(f"           success={rl['success_rate']:.1%}, collision={rl['collision_rate']:.1%}")
        
        # Eval
        if report["eval"]["latest"]:
            ev = report["eval"]["latest"]
            print(f"Eval Latest: episodes={ev['num_episodes']}, route_completion={ev['route_completion_avg']:.1%}")
            print(f"             success={ev['success_rate']:.1%}, collision_rate={ev['collision_rate']:.1%}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Metrics Aggregator")
    parser.add_argument("--stage", choices=["ssl", "bc", "rl", "eval", "all"], 
                        default="all", help="Pipeline stage to collect metrics for")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--output-dir", default="checkpoints",
                        help="Output directory for metrics")
    parser.add_argument("--metrics-file", 
                        help="Specific metrics file to load")
    parser.add_argument("--report", action="store_true",
                        help="Generate full metrics report")
    parser.add_argument("--output", default="metrics_report.json",
                        help="Output file for report")
    parser.add_argument("--run-id", 
                        help="Create or update a pipeline run")
    parser.add_argument("--status", choices=["running", "completed", "failed"],
                        help="Set pipeline run status")
    
    args = parser.parse_args()
    
    aggregator = PipelineMetricsAggregator(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )
    
    if args.run_id:
        if args.status:
            aggregator.update_run(args.run_id, status=args.status)
            print(f"Updated run {args.run_id} status to {args.status}")
        else:
            run = aggregator.create_run(args.run_id)
            print(f"Created new pipeline run: {run.run_id}")
    
    if args.report:
        report = aggregator.generate_report()
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
        aggregator.print_summary()
    else:
        # Print stage-specific metrics
        if args.stage in ["ssl", "all"]:
            ssl_metrics = SSLMetricsCollector.discover(args.checkpoint_dir)
            print(f"SSL Metrics: {len(ssl_metrics)} runs found")
            for m in ssl_metrics[-3:]:
                print(f"  - epoch={m.epoch}, loss={m.loss:.4f}")
        
        if args.stage in ["bc", "all"]:
            bc_metrics = BCMetricsCollector.discover(args.checkpoint_dir)
            print(f"BC Metrics: {len(bc_metrics)} runs found")
            for m in bc_metrics[-3:]:
                print(f"  - epoch={m.epoch}, val_loss={m.val_loss:.4f}, waypoint_mae={m.waypoint_mae:.3f}m")
        
        if args.stage in ["rl", "all"]:
            rl_metrics = RLMetricsCollector.discover(args.checkpoint_dir)
            print(f"RL Metrics: {len(rl_metrics)} runs found")
            for m in rl_metrics[-3:]:
                print(f"  - episode={m.episode}, reward={m.mean_reward:.2f}, success={m.success_rate:.1%}")
        
        if args.stage in ["eval", "all"]:
            eval_metrics = EvalMetricsCollector.discover(args.checkpoint_dir)
            print(f"Eval Metrics: {len(eval_metrics)} runs found")
            for m in eval_metrics[-3:]:
                print(f"  - episodes={m.num_episodes}, success={m.success_rate:.1%}, route_comp={m.route_completion_avg:.1%}")


if __name__ == "__main__":
    main()
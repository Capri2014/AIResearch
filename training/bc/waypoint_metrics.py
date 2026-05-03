"""
Waypoint Prediction Metrics for Behavior Cloning.

Computes comprehensive metrics for waypoint prediction quality including:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error) 
- Speed prediction accuracy
- Progress prediction accuracy
- Success rate at waypoint thresholds
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class WaypointMetricsConfig:
    """Configuration for waypoint metrics computation."""
    num_waypoints: int = 8
    horizon_seconds: float = 3.0
    sampling_rate_hz: float = 2.0
    ade_threshold_m: float = 2.0  # meters
    fde_threshold_m: float = 5.0  # meters
    speed_threshold_mps: float = 1.0  # m/s
    progress_threshold: float = 0.1  # 10%
    

@dataclass
class WaypointSampleMetrics:
    """Metrics for a single waypoint prediction sample."""
    sample_id: str
    ade: float  # Average Displacement Error (m)
    fde: float  # Final Displacement Error (m)
    max_de: float  # Maximum Displacement Error (m)
    speed_rmse: float  # Speed RMSE (m/s)
    speed_mae: float  # Speed MAE (m/s)
    progress_rmse: float
    progress_mae: float
    success_ade: bool  # Within ade_threshold_m
    success_fde: bool  # Within fde_threshold_m
    waypoint_errors: List[float] = field(default_factory=list)


@dataclass
class AggregatedWaypointMetrics:
    """Aggregated metrics over multiple samples."""
    num_samples: int
    ade_mean: float
    ade_std: float
    ade_median: float
    ade_min: float
    ade_max: float
    fde_mean: float
    fde_std: float
    fde_median: float
    fde_min: float
    fde_max: float
    max_de_mean: float
    speed_rmse_mean: float
    speed_mae_mean: float
    progress_rmse_mean: float
    progress_mae_mean: float
    success_rate_ade: float  # % samples with ade < threshold
    success_rate_fde: float  # % samples with fde < threshold
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_samples": self.num_samples,
            "ADE": {
                "mean": self.ade_mean,
                "std": self.ade_std,
                "median": self.ade_median,
                "min": self.ade_min,
                "max": self.ade_max,
            },
            "FDE": {
                "mean": self.fde_mean,
                "std": self.fde_std,
                "median": self.fde_median,
                "min": self.fde_min,
                "max": self.fde_max,
            },
            "max_DE_mean": self.max_de_mean,
            "speed_RMSE": self.speed_rmse_mean,
            "speed_MAE": self.speed_mae_mean,
            "progress_RMSE": self.progress_rmse_mean,
            "progress_MAE": self.progress_mae_mean,
            "success_rate_ADE": self.success_rate_ade,
            "success_rate_FDE": self.success_rate_fde,
        }


class WaypointMetricsComputer:
    """Compute metrics for waypoint predictions."""
    
    def __init__(self, config: Optional[WaypointMetricsConfig] = None):
        self.config = config or WaypointMetricsConfig()
        self.samples: List[WaypointSampleMetrics] = []
        
    def compute_single_sample(
        self,
        sample_id: str,
        pred_waypoints: np.ndarray,
        gt_waypoints: np.ndarray,
        pred_speeds: Optional[np.ndarray] = None,
        gt_speeds: Optional[np.ndarray] = None,
        pred_progress: Optional[np.ndarray] = None,
        gt_progress: Optional[np.ndarray] = None,
    ) -> WaypointSampleMetrics:
        """
        Compute metrics for a single prediction sample.
        
        Args:
            sample_id: Unique identifier for this sample
            pred_waypoints: Predicted waypoints (num_waypoints, 2) in meters
            gt_waypoints: Ground truth waypoints (num_waypoints, 2) in meters
            pred_speeds: Predicted speeds (num_waypoints,) in m/s
            gt_speeds: Ground truth speeds (num_waypoints,) in m/s
            pred_progress: Predicted progress (num_waypoints,) in [0, 1]
            gt_progress: Ground truth progress (num_waypoints,) in [0, 1]
            
        Returns:
            WaypointSampleMetrics for this sample
        """
        num_waypoints = min(len(pred_waypoints), len(gt_waypoints))
        
        # Compute waypoint errors
        waypoint_errors = []
        for i in range(num_waypoints):
            error = np.linalg.norm(pred_waypoints[i] - gt_waypoints[i])
            waypoint_errors.append(error)
            
        # ADE: mean of all waypoint errors
        ade = np.mean(waypoint_errors)
        
        # FDE: final waypoint error
        fde = waypoint_errors[-1] if waypoint_errors else 0.0
        
        # Max displacement error
        max_de = max(waypoint_errors) if waypoint_errors else 0.0
        
        # Speed metrics
        speed_rmse = 0.0
        speed_mae = 0.0
        if pred_speeds is not None and gt_speeds is not None:
            num_speeds = min(len(pred_speeds), len(gt_speeds))
            if num_speeds > 0:
                speed_diff = pred_speeds[:num_speeds] - gt_speeds[:num_speeds]
                speed_rmse = np.sqrt(np.mean(speed_diff ** 2))
                speed_mae = np.mean(np.abs(speed_diff))
                
        # Progress metrics
        progress_rmse = 0.0
        progress_mae = 0.0
        if pred_progress is not None and gt_progress is not None:
            num_progress = min(len(pred_progress), len(gt_progress))
            if num_progress > 0:
                progress_diff = pred_progress[:num_progress] - gt_progress[:num_progress]
                progress_rmse = np.sqrt(np.mean(progress_diff ** 2))
                progress_mae = np.mean(np.abs(progress_diff))
        
        # Success rates
        success_ade = ade < self.config.ade_threshold_m
        success_fde = fde < self.config.fde_threshold_m
        
        return WaypointSampleMetrics(
            sample_id=sample_id,
            ade=ade,
            fde=fde,
            max_de=max_de,
            speed_rmse=speed_rmse,
            speed_mae=speed_mae,
            progress_rmse=progress_rmse,
            progress_mae=progress_mae,
            success_ade=success_ade,
            success_fde=success_fde,
            waypoint_errors=waypoint_errors,
        )
        
    def add_sample(self, metrics: WaypointSampleMetrics) -> None:
        """Add a sample to the collection."""
        self.samples.append(metrics)
        
    def add_batch(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        pred_speeds: Optional[np.ndarray] = None,
        gt_speeds: Optional[np.ndarray] = None,
        pred_progress: Optional[np.ndarray] = None,
        gt_progress: Optional[np.ndarray] = None,
    ) -> List[WaypointSampleMetrics]:
        """
        Add a batch of predictions.
        
        Args:
            predictions: Predicted waypoints (batch_size, num_waypoints, 2)
            ground_truth: Ground truth waypoints (batch_size, num_waypoints, 2)
            pred_speeds: Predicted speeds (batch_size, num_waypoints)
            gt_speeds: Ground truth speeds (batch_size, num_waypoints)
            pred_progress: Predicted progress (batch_size, num_waypoints)
            gt_progress: Ground truth progress (batch_size, num_waypoints)
            
        Returns:
            List of WaypointSampleMetrics
        """
        batch_size = predictions.shape[0]
        results = []
        
        for i in range(batch_size):
            sample_id = f"sample_{i}"
            metrics = self.compute_single_sample(
                sample_id=sample_id,
                pred_waypoints=predictions[i],
                gt_waypoints=ground_truth[i],
                pred_speeds=pred_speeds[i] if pred_speeds is not None else None,
                gt_speeds=gt_speeds[i] if gt_speeds is not None else None,
                pred_progress=pred_progress[i] if pred_progress is not None else None,
                gt_progress=gt_progress[i] if gt_progress is not None else None,
            )
            self.samples.append(metrics)
            results.append(metrics)
            
        return results
        
    def aggregate(self) -> AggregatedWaypointMetrics:
        """Aggregate all collected samples."""
        if not self.samples:
            raise ValueError("No samples to aggregate")
            
        ades = [s.ade for s in self.samples]
        fdes = [s.fde for s in self.samples]
        max_des = [s.max_de for s in self.samples]
        speed_rmses = [s.speed_rmse for s in self.samples]
        speed_maes = [s.speed_mae for s in self.samples]
        progress_rmses = [s.progress_rmse for s in self.samples]
        progress_maes = [s.progress_mae for s in self.samples]
        success_ade = [s.success_ade for s in self.samples]
        success_fde = [s.success_fde for s in self.samples]
        
        return AggregatedWaypointMetrics(
            num_samples=len(self.samples),
            ade_mean=np.mean(ades),
            ade_std=np.std(ades),
            ade_median=np.median(ades),
            ade_min=np.min(ades),
            ade_max=np.max(ades),
            fde_mean=np.mean(fdes),
            fde_std=np.std(fdes),
            fde_median=np.median(fdes),
            fde_min=np.min(fdes),
            fde_max=np.max(fdes),
            max_de_mean=np.mean(max_des),
            speed_rmse_mean=np.mean(speed_rmses),
            speed_mae_mean=np.mean(speed_maes),
            progress_rmse_mean=np.mean(progress_rmses),
            progress_mae_mean=np.mean(progress_maes),
            success_rate_ade=np.mean(success_ade) * 100,
            success_rate_fde=np.mean(success_fde) * 100,
        )
        
    def save_metrics(self, output_path: str) -> None:
        """Save metrics to JSON file."""
        aggregated = self.aggregate()
        
        output = {
            "config": {
                "num_waypoints": self.config.num_waypoints,
                "horizon_seconds": self.config.horizon_seconds,
                "sampling_rate_hz": self.config.sampling_rate_hz,
                "ade_threshold_m": self.config.ade_threshold_m,
                "fde_threshold_m": self.config.fde_threshold_m,
                "speed_threshold_mps": self.config.speed_threshold_mps,
                "progress_threshold": self.config.progress_threshold,
            },
            "metrics": aggregated.to_dict(),
            "per_sample": [
                {
                    "sample_id": s.sample_id,
                    "ADE": s.ade,
                    "FDE": s.fde,
                    "max_DE": s.max_de,
                    "speed_RMSE": s.speed_rmse,
                    "speed_MAE": s.speed_mae,
                    "progress_RMSE": s.progress_rmse,
                    "progress_MAE": s.progress_mae,
                    "success_ADE": bool(s.success_ade),
                    "success_FDE": bool(s.success_fde),
                }
                for s in self.samples
            ],
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
            
    def print_summary(self) -> None:
        """Print a summary of the metrics."""
        if not self.samples:
            print("No samples to summarize")
            return
            
        agg = self.aggregate()
        
        print(f"\n{'='*50}")
        print("Waypoint Prediction Metrics Summary")
        print(f"{'='*50}")
        print(f"Number of samples: {agg.num_samples}")
        print(f"\nADE (Average Displacement Error):")
        print(f"  Mean:   {agg.ade_mean:.3f} m")
        print(f"  Std:    {agg.ade_std:.3f} m")
        print(f"  Median: {agg.ade_median:.3f} m")
        print(f"  Min:    {agg.ade_min:.3f} m")
        print(f"  Max:    {agg.ade_max:.3f} m")
        print(f"\nFDE (Final Displacement Error):")
        print(f"  Mean:   {agg.fde_mean:.3f} m")
        print(f"  Std:    {agg.fde_std:.3f} m")
        print(f"  Median: {agg.fde_median:.3f} m")
        print(f"  Min:    {agg.fde_min:.3f} m")
        print(f"  Max:    {agg.fde_max:.3f} m")
        print(f"\nSpeed Prediction:")
        print(f"  RMSE: {agg.speed_rmse_mean:.3f} m/s")
        print(f"  MAE:  {agg.speed_mae_mean:.3f} m/s")
        print(f"\nSuccess Rates:")
        print(f"  ADE < {self.config.ade_threshold_m}m: {agg.success_rate_ade:.1f}%")
        print(f"  FDE < {self.config.fde_threshold_m}m: {agg.success_rate_fde:.1f}%")
        print(f"{'='*50}\n")


def create_synthetic_data(
    num_samples: int = 10,
    num_waypoints: int = 8,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic waypoint data for testing."""
    np.random.seed(seed)
    
    # Ground truth: straight line trajectory
    t = np.linspace(0, 3.0, num_waypoints)
    gt_waypoints = np.column_stack([t * 2.0, np.zeros(num_waypoints)])  # (num_waypoints, 2)
    
    # Repeat for batch
    ground_truth = np.tile(gt_waypoints, (num_samples, 1, 1))  # (num_samples, num_waypoints, 2)
    
    # Add noise to predictions
    noise = np.random.randn(num_samples, num_waypoints, 2) * noise_std
    predictions = ground_truth + noise
    
    return predictions, ground_truth


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute waypoint prediction metrics")
    parser.add_argument("--predictions", type=str, help="Path to predictions JSON/NPY")
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth JSON/NPY")
    parser.add_argument("--output", type=str, default="out/waypoint_metrics/metrics.json",
                      help="Output path for metrics")
    parser.add_argument("--num-samples", type=int, default=10,
                      help="Number of synthetic samples")
    parser.add_argument("--num-waypoints", type=int, default=8,
                      help="Number of waypoints")
    parser.add_argument("--ade-threshold", type=float, default=2.0,
                      help="ADE success threshold in meters")
    parser.add_argument("--fde-threshold", type=float, default=5.0,
                      help="FDE success threshold in meters")
    parser.add_argument("--smoke-test", action="store_true",
                      help="Run smoke test with synthetic data")
    parser.add_argument("--verbose", action="store_true",
                      help="Print verbose output")
    
    args = parser.parse_args()
    
    if args.smoke_test or (not args.predictions and not args.ground_truth):
        # Synthetic smoke test
        print("Running smoke test with synthetic data...")
        
        predictions, ground_truth = create_synthetic_data(
            num_samples=args.num_samples,
            num_waypoints=args.num_waypoints,
            noise_std=0.5,
        )
        
        config = WaypointMetricsConfig(
            num_waypoints=args.num_waypoints,
            ade_threshold_m=args.ade_threshold,
            fde_threshold_m=args.fde_threshold,
        )
        
        computer = WaypointMetricsComputer(config)
        computer.add_batch(predictions, ground_truth)
        
        if args.verbose:
            computer.print_summary()
            
        aggregated = computer.aggregate()
        print(f" Smoke test results:")
        print(f"   ADE: {aggregated.ade_mean:.3f} ± {aggregated.ade_std:.3f} m")
        print(f"   FDE: {aggregated.fde_mean:.3f} ± {aggregated.fde_std:.3f} m")
        print(f"   Success (ADE < {args.ade_threshold}m): {aggregated.success_rate_ade:.1f}%")
        print(f"   Success (FDE < {args.fde_threshold}m): {aggregated.success_rate_fde:.1f}%")
        
        computer.save_metrics(args.output)
        print(f" Saved to: {args.output}")
        
    else:
        raise NotImplementedError("Loading from files not yet implemented")


if __name__ == "__main__":
    main()
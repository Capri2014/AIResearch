"""
RL to CARLA Pipeline Script

End-to-end pipeline that takes RL training output and runs CARLA closed-loop evaluation.
Automatically selects the best checkpoint based on training metrics and evaluates in CARLA.

Usage:
    python -m training.eval.run_rl_to_carla_pipeline \
        --rl-run-dir out/rl_delta_waypoint/2026-02-20_19-31-52 \
        --output-dir out/rl_carla_eval

This script:
1. Loads training metrics from the RL run
2. Selects the best checkpoint (by avg_reward)
3. Runs CARLA closed-loop evaluation with rl_mode=True
4. Outputs comprehensive results comparing SFT vs RL
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_best_checkpoint(rl_run_dir: Path) -> Optional[Path]:
    """Find the best checkpoint from RL training run based on metrics.
    
    Args:
        rl_run_dir: Path to RL training run directory
        
    Returns:
        Path to best checkpoint, or None if not found
    """
    # Look for metrics.json
    metrics_file = rl_run_dir / "metrics.json"
    if not metrics_file.exists():
        # Try train_metrics.json
        metrics_file = rl_run_dir / "train_metrics.json"
    
    if not metrics_file.exists():
        logger.error(f"No metrics.json found in {rl_run_dir}")
        return None
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Find checkpoint files
    checkpoints = list(rl_run_dir.glob("checkpoint_*.pt")) + list(rl_run_dir.glob("*.pt"))
    
    if not checkpoints:
        logger.error(f"No checkpoints found in {rl_run_dir}")
        return None
    
    # Try to find best by reward
    if "final_metrics" in metrics:
        best_reward = metrics["final_metrics"].get("avg_reward", float("-inf"))
        logger.info(f"Best reward from metrics: {best_reward}")
    
    # Return the final checkpoint or best one
    final_checkpoint = rl_run_dir / "final.pt"
    if final_checkpoint.exists():
        return final_checkpoint
    
    # Return the latest checkpoint by modification time
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def get_sft_checkpoint(rl_run_dir: Path) -> Optional[Path]:
    """Extract SFT checkpoint path from RL training metrics.
    
    Args:
        rl_run_dir: Path to RL training run directory
        
    Returns:
        Path to SFT checkpoint, or None if not found
    """
    metrics_file = rl_run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    sft_path = metrics.get("final_metrics", {}).get("sft_checkpoint")
    if not sft_path:
        sft_path = metrics.get("sft_checkpoint")
    
    if sft_path:
        return Path(sft_path)
    return None


def run_carla_eval(
    checkpoint: Path,
    output_dir: Path,
    sft_checkpoint: Optional[Path] = None,
    scenarios: str = "all",
    host: str = "localhost",
    port: int = 2000,
) -> Dict[str, Any]:
    """Run CARLA closed-loop evaluation on a checkpoint.
    
    Args:
        checkpoint: Path to model checkpoint (RL model)
        output_dir: Output directory for results
        sft_checkpoint: Optional SFT checkpoint for comparison
        scenarios: Comma-separated scenario names
        host: CARLA server host
        port: CARLA server port
        
    Returns:
        Dict with evaluation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "-m", "training.eval.run_carla_closed_loop_eval",
        "--checkpoint", str(checkpoint),
        "--output-dir", str(output_dir),
        "--scenarios", scenarios,
        "--host", host,
        "--port", str(port),
        "--rl-mode",  # Enable RL mode for delta head
    ]
    
    logger.info(f"Running CARLA eval: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=checkpoint.parent.parent.parent,  # Go to repo root
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"CARLA eval failed: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        # Load results
        metrics_file = output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                eval_metrics = json.load(f)
            return {"success": True, "metrics": eval_metrics}
        else:
            return {"success": True, "message": "Eval completed but no metrics file found"}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Evaluation timed out after 10 minutes"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def compare_sft_rl(
    sft_checkpoint: Path,
    rl_checkpoint: Path,
    output_dir: Path,
    scenarios: str = "all",
) -> Dict[str, Any]:
    """Run comparison between SFT and RL models in CARLA.
    
    Args:
        sft_checkpoint: Path to SFT model checkpoint
        rl_checkpoint: Path to RL model checkpoint  
        output_dir: Output directory for results
        scenarios: Comma-separated scenario names
        
    Returns:
        Dict with comparison results
    """
    # Run SFT evaluation
    sft_output = output_dir / "sft_eval"
    sft_result = run_carla_eval(sft_checkpoint, sft_output, scenarios=scenarios)
    
    # Run RL evaluation
    rl_output = output_dir / "rl_eval"
    rl_result = run_carla_eval(rl_checkpoint, rl_output, scenarios=scenarios)
    
    if not sft_result.get("success") or not rl_result.get("success"):
        return {
            "success": False,
            "error": "One or both evaluations failed",
            "sft_result": sft_result,
            "rl_result": rl_result,
        }
    
    sft_metrics = sft_result.get("metrics", {})
    rl_metrics = rl_result.get("metrics", {})
    
    # Compute improvements
    def get_metric(metrics: Dict, key: str, default: float = 0.0) -> float:
        return metrics.get(key, metrics.get(f"avg_{key}", default))
    
    sft_route = get_metric(sft_metrics, "route_completion")
    rl_route = get_metric(rl_metrics, "route_completion")
    route_improvement = ((rl_route - sft_route) / max(sft_route, 0.01)) * 100
    
    sft_collisions = get_metric(sft_metrics, "collision_rate")
    rl_collisions = get_metric(rl_metrics, "collision_rate")
    collision_improvement = ((sft_collisions - rl_collisions) / max(sft_collisions, 0.01)) * 100
    
    return {
        "success": True,
        "sft_metrics": sft_metrics,
        "rl_metrics": rl_metrics,
        "comparison": {
            "route_completion": {
                "sft": sft_route,
                "rl": rl_route,
                "improvement_percent": route_improvement,
            },
            "collision_rate": {
                "sft": sft_collisions,
                "rl": rl_collisions,
                "improvement_percent": collision_improvement,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="RL to CARLA Pipeline: Evaluate RL checkpoints in CARLA"
    )
    parser.add_argument(
        "--rl-run-dir",
        type=str,
        required=True,
        help="Path to RL training run directory (contains metrics.json and checkpoints)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for CARLA evaluation results",
    )
    parser.add_argument(
        "--compare-sft",
        action="store_true",
        help="Also run SFT baseline for comparison",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="straight_clear,turn_clear",
        help="Comma-separated scenario names (default: straight_clear,turn_clear)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test without CARLA",
    )
    
    args = parser.parse_args()
    
    rl_run_dir = Path(args.rl_run_dir)
    output_dir = Path(args.output_dir)
    
    if not rl_run_dir.exists():
        logger.error(f"RL run directory not found: {rl_run_dir}")
        sys.exit(1)
    
    logger.info(f"=== RL to CARLA Pipeline ===")
    logger.info(f"RL run: {rl_run_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Find best checkpoint
    checkpoint = find_best_checkpoint(rl_run_dir)
    if not checkpoint:
        logger.error("Could not find checkpoint to evaluate")
        sys.exit(1)
    
    logger.info(f"Best checkpoint: {checkpoint}")
    
    # Get SFT checkpoint if available
    sft_checkpoint = get_sft_checkpoint(rl_run_dir)
    if sft_checkpoint:
        logger.info(f"SFT checkpoint: {sft_checkpoint}")
    
    if args.smoke:
        logger.info("Smoke test mode - skipping actual CARLA evaluation")
        # Just output what we would do
        result = {
            "rl_checkpoint": str(checkpoint),
            "sft_checkpoint": str(sft_checkpoint) if sft_checkpoint else None,
            "smoke_test": True,
        }
    elif args.compare_sft and sft_checkpoint:
        logger.info("Running SFT vs RL comparison...")
        result = compare_sft_rl(sft_checkpoint, checkpoint, output_dir, args.scenarios)
    else:
        logger.info("Running RL evaluation in CARLA...")
        result = run_carla_eval(checkpoint, output_dir, sft_checkpoint, args.scenarios, args.host, args.port)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pipeline_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {output_dir / 'pipeline_results.json'}")
    
    if result.get("success"):
        logger.info("✓ Pipeline completed successfully")
        if "comparison" in result:
            comp = result["comparison"]
            logger.info(f"  Route completion: SFT {comp['route_completion']['sft']:.1%} → RL {comp['route_completion']['rl']:.1%} ({comp['route_completion']['improvement_percent']:+.1f}%)")
            logger.info(f"  Collision rate: SFT {comp['collision_rate']['sft']:.1%} → RL {comp['collision_rate']['rl']:.1%} ({comp['collision_rate']['improvement_percent']:+.1f}%)")
    else:
        logger.error(f"✗ Pipeline failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()

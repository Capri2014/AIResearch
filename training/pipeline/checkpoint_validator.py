#!/usr/bin/env python3
"""
Pipeline Checkpoint Validator

Validates discovered checkpoints are healthy and loadable.
Complements checkpoint_discovery.py with validation and health checks.

Usage:
    python checkpoint_validator.py --base-dir out/ --stage bc --validate
    python checkpoint_validator.py --base-dir out/ --all --smoke-test
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ValidationConfig:
    """Configuration for checkpoint validation."""
    base_dir: str = "out/"
    stage: Optional[str] = None  # ssl, bc, rl, eval
    run_id: Optional[str] = None
    load_weights: bool = True
    check_schema: bool = True
    verbose: bool = False


@dataclass
class CheckpointHealth:
    """Health status for a checkpoint."""
    path: str
    exists: bool = False
    readable: bool = False
    loadable: bool = False
    has_state_dict: bool = False
    has_optimizer: bool = False
    has_metadata: bool = False
    schema_valid: bool = False
    errors: list = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return self.exists and self.readable and self.loadable
    
    @property
    def health_score(self) -> float:
        """Compute health score 0-1."""
        score = 0.0
        if self.exists: score += 0.1
        if self.readable: score += 0.1
        if self.loadable: score += 0.2
        if self.has_state_dict: score += 0.3
        if self.has_optimizer: score += 0.1
        if self.has_metadata: score += 0.1
        if self.schema_valid: score += 0.1
        return score


@dataclass
class ValidationResult:
    """Result from validating a checkpoint."""
    checkpoint_path: str
    stage: str
    run_id: Optional[str] = None
    epoch: Optional[int] = None
    health: Optional[CheckpointHealth] = None
    metrics: dict = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        return self.health is not None and self.health.is_healthy


class PipelineCheckpointValidator:
    """Validates pipeline checkpoints for health and loadability."""
    
    STAGE_DIRS = {
        "ssl": ["pretrain", "ssl"],
        "bc": ["bc", "waypoint_bc"],
        "rl": ["rl", "rl_after_sft"],
        "eval": ["eval", "evaluation"],
    }
    
    VALID_SCHEMAS = ["metrics.json", "train_metrics.json"]
    
    # Priority order for checkpoint files
    CHECKPOINT_PRIORITY = ["final.pt", "best.pt", "best_reward.pt", "best_ade.pt", "checkpoint.pt"]
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results: list[ValidationResult] = []
    
    def find_checkpoints(self, stage: str) -> list[Path]:
        """Find checkpoints for a stage."""
        base = Path(self.config.base_dir)
        checkpoints = []
        
        # Search in stage-specific directories
        for dir_name in self.STAGE_DIRS.get(stage, [stage]):
            for search_dir in [base / dir_name, base / "checkpoints" / dir_name]:
                if not search_dir.exists():
                    continue
                for ckpt_file in search_dir.rglob("*.pt"):
                    if "final" in ckpt_file.name or "best" in ckpt_file.name or "epoch" in ckpt_file.name:
                        checkpoints.append(ckpt_file)
        
        return list(set(checkpoints))
    
    def validate_checkpoint(self, ckpt_path: Path, stage: str) -> ValidationResult:
        """Validate a single checkpoint."""
        result = ValidationResult(
            checkpoint_path=str(ckpt_path),
            stage=stage,
        )
        
        # Extract run_id and epoch from path
        parts = ckpt_path.parts
        for i, part in enumerate(parts):
            if part in ["pretrain", "ssl", "bc", "rl", "eval"]:
                result.run_id = parts[i + 1] if i + 1 < len(parts) else None
                break
        
        # Check for epoch in filename
        name = ckpt_path.name
        if "epoch_" in name:
            try:
                result.epoch = int(name.split("_")[1].replace(".pt", ""))
            except ValueError:
                pass
        
        # Health check
        health = CheckpointHealth(path=str(ckpt_path))
        
        # 1. Exists
        health.exists = ckpt_path.exists()
        if not health.exists:
            health.errors.append("File does not exist")
            result.health = health
            return result
        
        # 2. Readable (file size > 0)
        try:
            size = ckpt_path.stat().st_size
            health.readable = size > 0
            if not health.readable:
                health.errors.append(f"File is empty (size={size})")
        except OSError as e:
            health.errors.append(f"Cannot stat file: {e}")
            result.health = health
            return result
        
        # 3. Loadable (torch.load)
        if self.config.load_weights:
            try:
                # Try loading with weights_only for security
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                health.loadable = True
                health.has_state_dict = isinstance(state, dict)
                
                # Check for optimizer state
                if health.has_state_dict:
                    health.has_optimizer = "optimizer" in state or "optimizer_state" in state
                    health.has_metadata = "epoch" in state or "step" in state or "metrics" in state
                
                # Release memory
                del state
            except Exception as e:
                # Try without weights_only
                try:
                    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    health.loadable = True
                    health.has_state_dict = isinstance(state, dict)
                    del state
                except Exception as e2:
                    health.errors.append(f"Cannot load: {e2}")
        
        # 4. Check for schema files
        if self.config.check_schema:
            schema_dir = ckpt_path.parent
            for schema_file in self.VALID_SCHEMAS:
                if (schema_dir / schema_file).exists():
                    health.schema_valid = True
                    # Try to load metrics
                    try:
                        with open(schema_dir / schema_file) as f:
                            metrics = json.load(f)
                            result.metrics = metrics
                    except Exception:
                        pass
                    break
        
        result.health = health
        return result
    
    def validate_stage(self, stage: str) -> list[ValidationResult]:
        """Validate all checkpoints for a stage."""
        checkpoints = self.find_checkpoints(stage)
        results = []
        
        for ckpt in checkpoints:
            result = self.validate_checkpoint(ckpt, stage)
            results.append(result)
        
        # Sort by health score (descending)
        results.sort(
            key=lambda r: r.health.health_score if r.health else 0,
            reverse=True
        )
        
        return results
    
    def validate_all(self) -> list[ValidationResult]:
        """Validate all stages."""
        stages = [self.config.stage] if self.config.stage else list(self.STAGE_DIRS.keys())
        
        all_results = []
        for stage in stages:
            if self.config.verbose:
                print(f"Validating stage: {stage}")
            results = self.validate_stage(stage)
            all_results.extend(results)
        
        self.results = all_results
        return all_results
    
    def print_summary(self) -> str:
        """Generate summary text."""
        lines = []
        
        # Group by stage
        by_stage: dict[str, list[ValidationResult]] = {}
        for result in self.results:
            by_stage.setdefault(result.stage, []).append(result)
        
        for stage, results in by_stage.items():
            valid = sum(1 for r in results if r.is_valid())
            total = len(results)
            
            lines.append(f"\n## {stage.upper()} Stage")
            lines.append(f"- Checkpoints: {total}, Valid: {valid}")
            
            for result in results[:5]:  # Show top 5
                score = result.health.health_score if result.health else 0
                status = "✓" if result.is_valid() else "✗"
                lines.append(f"  {status} {result.checkpoint_path.split('/')[-1]}: {score:.1%}")
        
        return "\n".join(lines)
    
    def save_summary(self, output_path: str) -> None:
        """Save validation summary to JSON."""
        data = {
            "validation_config": {
                "base_dir": self.config.base_dir,
                "stage": self.config.stage,
            },
            "results": [
                {
                    "checkpoint_path": r.checkpoint_path,
                    "stage": r.stage,
                    "run_id": r.run_id,
                    "epoch": r.epoch,
                    "is_valid": r.is_valid(),
                    "health_score": r.health.health_score if r.health else 0,
                    "metrics": r.metrics,
                }
                for r in self.results
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def create_smoke_test(config: ValidationConfig) -> bool:
    """Run smoke test with synthetic checkpoint."""
    import tempfile
    
    # Create temp checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Save dummy checkpoint
        state = {
            "model_state_dict": {"weight": torch.randn(10, 10)},
            "optimizer_state_dict": {},
            "epoch": 1,
            "metrics": {"loss": 0.5},
        }
        torch.save(state, temp_path)
        
        # Validate
        validator = PipelineCheckpointValidator(config)
        result = validator.validate_checkpoint(temp_path, "bc")
        
        print(f"Smoke test: {result.checkpoint_path}")
        print(f"  exists: {result.health.exists}")
        print(f"  readable: {result.health.readable}")
        print(f"  loadable: {result.health.loadable}")
        print(f"  has_state_dict: {result.health.has_state_dict}")
        print(f"  has_metadata: {result.health.has_metadata}")
        
        return result.is_valid()
    
    finally:
        # Cleanup
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline checkpoints")
    parser.add_argument("--base-dir", default="out/", help="Base directory for checkpoints")
    parser.add_argument("--stage", help="Stage to validate (ssl, bc, rl, eval)")
    parser.add_argument("--run-id", help="Filter by run ID")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--list", action="store_true", help="List checkpoints")
    parser.add_argument("--all", action="store_true", help="Validate all stages")
    parser.add_argument("--output", help="Output JSON path for summary")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = ValidationConfig(
        base_dir=args.base_dir,
        stage=args.stage,
        run_id=args.run_id,
        verbose=args.verbose,
    )
    
    if args.smoke_test:
        success = create_smoke_test(config)
        sys.exit(0 if success else 1)
    
    validator = PipelineCheckpointValidator(config)
    
    if args.all or not args.stage:
        results = validator.validate_all()
    else:
        results = validator.validate_stage(args.stage)
    
    print(validator.print_summary())
    
    if args.output:
        validator.save_summary(args.output)
        print(f"\nSaved to: {args.output}")
    
    # Count valid
    valid = sum(1 for r in results if r.is_valid())
    print(f"\nTotal: {len(results)}, Valid: {valid}")
    
    sys.exit(0 if valid > 0 else 1)


if __name__ == "__main__":
    main()
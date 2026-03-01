#!/usr/bin/env python3
"""Validate evaluation metrics against the schema.

Validates metrics.json files against data/schema/metrics.json.

Usage
-----
# Validate a single metrics file
python -m training.rl.validate_metrics out/eval/20260228-213121_sft/metrics.json

# Validate multiple files
python -m training.rl.validate_metrics out/eval/*/metrics.json

# Check comparison output
python -m training.rl.validate_metrics --compare out/eval/20260228-213121_sft/metrics.json out/eval/20260228-213121_rl/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default schema path (relative to repo root)
DEFAULT_SCHEMA = Path(__file__).resolve().parents[2] / "data" / "schema" / "metrics.json"


def load_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the metrics JSON schema."""
    path = schema_path or DEFAULT_SCHEMA
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    with open(path) as f:
        return json.load(f)


def validate_metrics(
    metrics: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Validate metrics against schema.
    
    Args:
        metrics: Metrics dictionary to validate
        schema: JSON schema (loads default if not provided)
    
    Returns:
        (is_valid, errors_or_warnings)
    """
    if schema is None:
        schema = load_schema()
    
    errors = []
    
    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in metrics:
            errors.append(f"Missing required field: '{field}'")
    
    # Check domain value
    if "domain" in metrics:
        domain_schema = schema.get("properties", {}).get("domain", {})
        allowed_domains = domain_schema.get("enum", [])
        if metrics["domain"] not in allowed_domains:
            errors.append(f"Invalid domain: '{metrics['domain']}'. Must be one of {allowed_domains}")
    
    # Check scenarios array
    if "scenarios" in metrics:
        if not isinstance(metrics["scenarios"], list):
            errors.append("'scenarios' must be an array")
        else:
            scenario_schema = schema.get("properties", {}).get("scenarios", {}).get("items", {})
            for i, scenario in enumerate(metrics["scenarios"]):
                # Check required scenario fields
                scenario_required = scenario_schema.get("required", [])
                for field in scenario_required:
                    if field not in scenario:
                        errors.append(f"Scenario {i}: missing required field '{field}'")
    
    # Check summary fields (if present)
    if "summary" in metrics:
        summary_schema = schema.get("properties", {}).get("summary", {})
        # Just check types for known fields - additionalProperties is allowed
        summary = metrics["summary"]
        if "ade_mean" in summary and not isinstance(summary["ade_mean"], (int, float)):
            errors.append("summary.ade_mean must be a number")
        if "fde_mean" in summary and not isinstance(summary["fde_mean"], (int, float)):
            errors.append("summary.fde_mean must be a number")
        if "success_rate" in summary and not isinstance(summary["success_rate"], (int, float)):
            errors.append("summary.success_rate must be a number")
        if "num_episodes" in summary and not isinstance(summary["num_episodes"], int):
            errors.append("summary.num_episodes must be an integer")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def compare_metrics(
    sft_metrics: Dict[str, Any],
    rl_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare SFT vs RL metrics and compute improvements.
    
    Args:
        sft_metrics: Metrics from SFT policy evaluation
        rl_metrics: Metrics from RL policy evaluation
    
    Returns:
        Dict with comparison results
    """
    sft_summary = sft_metrics.get("summary", {})
    rl_summary = rl_metrics.get("summary", {})
    
    sft_ade = sft_summary.get("ade_mean", 0) or 0
    rl_ade = rl_summary.get("ade_mean", 0) or 0
    sft_fde = sft_summary.get("fde_mean", 0) or 0
    rl_fde = rl_summary.get("fde_mean", 0) or 0
    sft_success = sft_summary.get("success_rate", 0) or 0
    rl_success = rl_summary.get("success_rate", 0) or 0
    
    ade_improvement = sft_ade - rl_ade
    fde_improvement = sft_fde - rl_fde
    success_improvement = rl_success - sft_success
    
    return {
        "sft": sft_summary,
        "rl": rl_summary,
        "improvement": {
            "ade_delta": ade_improvement,
            "ade_percent": (ade_improvement / sft_ade * 100) if sft_ade > 0 else 0,
            "fde_delta": fde_improvement,
            "fde_percent": (fde_improvement / sft_fde * 100) if sft_fde > 0 else 0,
            "success_delta": success_improvement,
        },
        "3_line": {
            "ade": f"{sft_ade:.2f}m (SFT) → {rl_ade:.2f}m (RL) [{ade_improvement/sft_ade*100:+.0f}%]" if sft_ade > 0 else "N/A",
            "fde": f"{sft_fde:.2f}m (SFT) → {rl_fde:.2f}m (RL) [{fde_improvement/sft_fde*100:+.0f}%]" if sft_fde > 0 else "N/A",
            "success": f"{sft_success:.0%} (SFT) → {rl_success:.0%} (RL) [{success_improvement:+.0%}]",
        },
    }


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evaluation metrics against schema")
    parser.add_argument("metrics_files", nargs="+", type=Path, help="Metrics JSON files to validate")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="Path to metrics schema")
    parser.add_argument("--compare", action="store_true", help="Compare first two files as SFT vs RL")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output errors")
    args = parser.parse_args()
    
    schema = load_schema(args.schema)
    
    if args.compare and len(args.metrics_files) >= 2:
        # Compare two metrics files
        sft_metrics = load_metrics(args.metrics_files[0])
        rl_metrics = load_metrics(args.metrics_files[1])
        
        # Validate both
        sft_valid, sft_errors = validate_metrics(sft_metrics, schema)
        rl_valid, rl_errors = validate_metrics(rl_metrics, schema)
        
        if not args.quiet:
            print(f"Validating {args.metrics_files[0]}:")
            if sft_valid:
                print("  ✓ Valid")
            else:
                for err in sft_errors:
                    print(f"  ✗ {err}")
            
            print(f"\nValidating {args.metrics_files[1]}:")
            if rl_valid:
                print("  ✓ Valid")
            else:
                for err in rl_errors:
                    print(f"  ✗ {err}")
        
        # Compare
        comparison = compare_metrics(sft_metrics, rl_metrics)
        
        print("\n" + "=" * 60)
        print("METRICS COMPARISON (from saved files)")
        print("=" * 60)
        print(f"\nSFT Policy ({args.metrics_files[0].name}):")
        print(f"  ADE: {comparison['sft'].get('ade_mean', 'N/A'):.4f} ± {comparison['sft'].get('ade_std', 0):.4f}m")
        print(f"  FDE: {comparison['sft'].get('fde_mean', 'N/A'):.4f} ± {comparison['sft'].get('ade_std', 0):.4f}m")
        print(f"  Success Rate: {comparison['sft'].get('success_rate', 0):.1%}")
        
        print(f"\nRL Policy ({args.metrics_files[1].name}):")
        print(f"  ADE: {comparison['rl'].get('ade_mean', 'N/A'):.4f} ± {comparison['rl'].get('ade_std', 0):.4f}m")
        print(f"  FDE: {comparison['rl'].get('fde_mean', 'N/A'):.4f} ± {comparison['rl'].get('ade_std', 0):.4f}m")
        print(f"  Success Rate: {comparison['rl'].get('success_rate', 0):.1%}")
        
        print(f"\nImprovement:")
        print(f"  ADE: {comparison['improvement']['ade_delta']:+.4f}m ({comparison['improvement']['ade_percent']:+.1f}%)")
        print(f"  FDE: {comparison['improvement']['fde_delta']:+.4f}m ({comparison['improvement']['fde_percent']:+.1f}%)")
        print(f"  Success: {comparison['improvement']['success_delta']:+.1%}")
        
        print("\n" + "-" * 60)
        print("3-LINE SUMMARY:")
        print("-" * 60)
        print(comparison['3_line']['ade'])
        print(comparison['3_line']['fde'])
        print(comparison['3_line']['success'])
        print("=" * 60)
        
        # Exit code based on validation
        sys.exit(0 if (sft_valid and rl_valid) else 1)
    
    else:
        # Validate each file
        all_valid = True
        for metrics_file in args.metrics_files:
            try:
                metrics = load_metrics(metrics_file)
                is_valid, errors = validate_metrics(metrics, schema)
                
                if not args.quiet:
                    print(f"\n{metrics_file}:")
                    if is_valid:
                        print("  ✓ Valid")
                    else:
                        for err in errors:
                            print(f"  ✗ {err}")
                
                if not is_valid:
                    all_valid = False
                    
            except Exception as e:
                print(f"\n{metrics_file}:")
                print(f"  ✗ Error: {e}")
                all_valid = False
        
        sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Metrics Schema Validator.

Validates RL/evaluation metrics JSON output against the standard schema.

Usage:
    python -m training.eval.validate_metrics out/eval/<run_id>/metrics.json
    python -m training.eval.validate_metrics --schema data/schema/metrics.json out/eval/<run_id>/metrics.json
"""
import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_schema(path: str) -> Dict[str, Any]:
    """Load JSON schema."""
    with open(path, 'r') as f:
        return json.load(f)


def validate_required_fields(metrics: Dict[str, Any]) -> List[str]:
    """Check required fields exist."""
    required = ['run_id', 'domain', 'scenarios']
    errors = []
    for field in required:
        if field not in metrics:
            errors.append(f"Missing required field: {field}")
    return errors


def validate_domain(metrics: Dict[str, Any]) -> List[str]:
    """Validate domain field."""
    errors = []
    domain = metrics.get('domain')
    valid_domains = ['driving', 'robotics', 'rl']
    if domain and domain not in valid_domains:
        errors.append(f"Invalid domain '{domain}'. Must be one of: {valid_domains}")
    return errors


def validate_scenarios(metrics: Dict[str, Any]) -> List[str]:
    """Validate scenarios array."""
    errors = []
    scenarios = metrics.get('scenarios', [])
    
    if not isinstance(scenarios, list):
        errors.append("'scenarios' must be an array")
        return errors
    
    if len(scenarios) == 0:
        errors.append("'scenarios' array is empty")
        return errors
    
    for i, scenario in enumerate(scenarios):
        # Check required fields per scenario
        if 'scenario_id' not in scenario:
            errors.append(f"Scenario {i}: missing 'scenario_id'")
        if 'success' not in scenario:
            errors.append(f"Scenario {i}: missing 'success'")
        
        # Validate success is boolean
        if 'success' in scenario and not isinstance(scenario['success'], bool):
            errors.append(f"Scenario {i}: 'success' must be boolean, got {type(scenario['success'])}")
        
        # Validate numeric fields
        for field in ['ade', 'fde', 'return', 'steps', 'final_dist']:
            if field in scenario:
                val = scenario[field]
                if val is not None and not isinstance(val, (int, float)):
                    errors.append(f"Scenario {i}: '{field}' must be numeric, got {type(val)}")
                if isinstance(val, float) and (val != val or val == float('inf')):
                    # NaN or inf - might be ok but warn
                    pass
    
    return errors


def validate_summary(metrics: Dict[str, Any]) -> List[str]:
    """Validate summary section."""
    errors = []
    summary = metrics.get('summary')
    
    if summary is None:
        # Summary is optional but warn if missing
        return errors
    
    if not isinstance(summary, dict):
        errors.append("'summary' must be an object")
        return errors
    
    # Validate summary stats are numeric
    for field in ['ade_mean', 'ade_std', 'fde_mean', 'fde_std', 'success_rate', 'return_mean']:
        if field in summary:
            val = summary[field]
            if not isinstance(val, (int, float)):
                errors.append(f"Summary '{field}' must be numeric, got {type(val)}")
    
    return errors


def validate_comparison(metrics: Dict[str, Any]) -> List[str]:
    """Validate comparison section if present."""
    errors = []
    comparison = metrics.get('comparison')
    
    if comparison is None:
        return errors
    
    if not isinstance(comparison, dict):
        errors.append("'comparison' must be an object")
        return errors
    
    # Validate improvement percentages are numeric
    for field in ['ade_improvement_pct', 'fde_improvement_pct', 'success_rate_diff']:
        if field in comparison:
            val = comparison[field]
            if not isinstance(val, (int, float)):
                errors.append(f"Comparison '{field}' must be numeric, got {type(val)}")
    
    return errors


def validate_metrics(metrics: Dict[str, Any], schema_path: str = None) -> Tuple[bool, List[str]]:
    """
    Validate metrics against schema.
    
    Returns:
        (is_valid, list_of_errors)
    """
    all_errors = []
    
    # Check required fields
    all_errors.extend(validate_required_fields(metrics))
    
    # Validate domain
    all_errors.extend(validate_domain(metrics))
    
    # Validate scenarios
    all_errors.extend(validate_scenarios(metrics))
    
    # Validate summary
    all_errors.extend(validate_summary(metrics))
    
    # Validate comparison
    all_errors.extend(validate_comparison(metrics))
    
    # If schema provided, could do deeper validation with jsonschema
    # For now, we do structural validation
    
    is_valid = len(all_errors) == 0
    return is_valid, all_errors


def print_validation_report(is_valid: bool, errors: List[str], metrics_path: str):
    """Print human-readable validation report."""
    print(f"Validating: {metrics_path}")
    print("=" * 60)
    
    if is_valid:
        print("✓ PASSED: Metrics are valid")
    else:
        print("✗ FAILED: Found issues:")
        for error in errors:
            print(f"  - {error}")
    
    # Print summary info
    metrics = load_json(metrics_path)
    print("\n--- Summary ---")
    print(f"Run ID: {metrics.get('run_id', 'unknown')}")
    print(f"Domain: {metrics.get('domain', 'unknown')}")
    print(f"Scenarios: {len(metrics.get('scenarios', []))}")
    
    summary = metrics.get('summary')
    if summary:
        if 'sft' in summary and 'rl' in summary:
            print(f"SFT:  ADE={summary['sft'].get('ade_mean', 'N/A'):.3f}, Success={summary['sft'].get('success_rate', 'N/A'):.1%}")
            print(f"RL:   ADE={summary['rl'].get('ade_mean', 'N/A'):.3f}, Success={summary['rl'].get('success_rate', 'N/A'):.1%}")
        elif 'ade_mean' in summary:
            print(f"ADE: {summary.get('ade_mean', 'N/A'):.3f} ± {summary.get('ade_std', 0):.3f}")
            print(f"FDE: {summary.get('fde_mean', 'N/A'):.3f} ± {summary.get('fde_std', 0):.3f}")
            print(f"Success rate: {summary.get('success_rate', 'N/A'):.1%}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Validate metrics JSON against schema'
    )
    parser.add_argument('metrics_path', nargs='?', 
                        help='Path to metrics.json file')
    parser.add_argument('--schema', type=str, default='data/schema/metrics.json',
                        help='Path to metrics schema (default: data/schema/metrics.json)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.metrics_path:
        # List recent eval runs
        eval_dir = 'out/eval'
        if os.path.exists(eval_dir):
            runs = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])
            print("Recent eval runs:")
            for run in runs[-10:]:
                metrics_file = os.path.join(eval_dir, run, 'metrics.json')
                if os.path.exists(metrics_file):
                    print(f"  {run}")
            print(f"\nUsage: {args.schema.split('/')[-1]} <metrics_path>")
            return 0
    
    # Resolve paths
    metrics_path = args.metrics_path
    if not os.path.isabs(metrics_path):
        metrics_path = os.path.join(os.getcwd(), metrics_path)
    
    schema_path = args.schema
    if not os.path.isabs(schema_path):
        schema_path = os.path.join(os.getcwd(), schema_path)
    
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found: {metrics_path}")
        return 1
    
    if not os.path.exists(schema_path):
        print(f"Warning: Schema not found: {schema_path}, skipping schema validation")
        schema_path = None
    
    # Load and validate
    try:
        metrics = load_json(metrics_path)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return 1
    
    is_valid, errors = validate_metrics(metrics, schema_path)
    
    print_validation_report(is_valid, errors, metrics_path)
    
    return 0 if is_valid else 1


if __name__ == '__main__':
    exit(main())

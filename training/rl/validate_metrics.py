"""
Enhanced Metrics Schema Validator with Strict Validation.

Validates evaluation metrics output against the standard schema
with detailed error reporting and suggestions.
"""
import json
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set


def load_schema(schema_path: str = "data/schema/metrics.json") -> Dict:
    """Load the metrics schema."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def get_schema_field_type(schema: Dict, path: str) -> str:
    """Get the expected type for a schema field path."""
    parts = path.split('.')
    current = schema
    for part in parts:
        if 'properties' in current and part in current['properties']:
            current = current['properties'][part]
        elif 'items' in current:
            current = current['items']
    return current.get('type', 'unknown')


def validate_field_strict(
    value: Any, 
    schema: Dict, 
    path: str,
    required_fields: Set[str] = None,
    depth: int = 0
) -> List[str]:
    """
    Validate a single field against schema with strict checks.
    
    Args:
        value: The value to validate
        schema: The schema definition
        path: Current path in the object hierarchy
        required_fields: Set of required field names at current level
        depth: Current recursion depth (for limiting)
    
    Returns:
        List of error messages
    """
    errors = []
    
    # Limit recursion depth
    if depth > 20:
        return [f"{path}: Maximum nesting depth exceeded"]
    
    # Type validation
    if "type" in schema:
        expected_type = schema["type"]
        actual_type = type(value).__name__
        
        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "integer": int
        }
        
        expected_python = type_map.get(expected_type, object)
        
        if expected_type == "number":
            # Accept both int and float for number type
            if not isinstance(value, (int, float)):
                errors.append(f"{path}: expected {expected_type}, got {actual_type}")
        elif expected_type == "object":
            if not isinstance(value, dict):
                errors.append(f"{path}: expected {expected_type}, got {actual_type}")
        elif expected_type == "array":
            if not isinstance(value, list):
                errors.append(f"{path}: expected {expected_type}, got {actual_type}")
        else:
            if not isinstance(value, expected_python):
                errors.append(f"{path}: expected {expected_type}, got {actual_type}")
    
    # Enum validation
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value '{value}' not in allowed values {schema['enum']}")
    
    # Range validation
    if "minimum" in schema and isinstance(value, (int, float)):
        if value < schema["minimum"]:
            errors.append(f"{path}: value {value} below minimum {schema['minimum']}")
    
    if "maximum" in schema and isinstance(value, (int, float)):
        if value > schema["maximum"]:
            errors.append(f"{path}: value {value} above maximum {schema['maximum']}")
    
    # Validate object properties
    if "properties" in schema and isinstance(value, dict):
        # Check for unexpected fields
        schema_keys = set(schema["properties"].keys())
        value_keys = set(value.keys())
        
        # Skip additionalProperties check if not specified or if True
        additional = schema.get("additionalProperties", True)
        if not additional:
            extra = value_keys - schema_keys
            if extra:
                errors.append(f"{path}: unexpected fields: {', '.join(sorted(extra))}")
        
        # Validate each property
        for key, val in value.items():
            if key in schema["properties"]:
                child_errors = validate_field_strict(
                    val, 
                    schema["properties"][key], 
                    f"{path}.{key}",
                    depth=depth+1
                )
                errors.extend(child_errors)
    
    # Validate array items
    if "items" in schema and isinstance(value, list):
        for i, item in enumerate(value):
            child_errors = validate_field_strict(
                item, 
                schema["items"], 
                f"{path}[{i}]",
                depth=depth+1
            )
            errors.extend(child_errors)
    
    return errors


def validate_metrics_strict(
    metrics: Dict, 
    schema: Dict,
    require_all_fields: bool = False
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate metrics against schema with detailed reporting.
    
    Args:
        metrics: The metrics dictionary to validate
        schema: The schema to validate against
        require_all_fields: If True, warn about missing recommended fields
    
    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check required top-level fields
    required = set(schema.get("required", []))
    present = set(metrics.keys())
    
    missing_required = required - present
    for field in sorted(missing_required):
        errors.append(f"Missing required field: {field}")
    
    # Check for recommended fields that might be useful
    if require_all_fields and "properties" in schema:
        recommended = set(schema["properties"].keys()) - required
        missing_recommended = recommended - present
        for field in sorted(missing_recommended):
            warnings.append(f"Missing recommended field: {field}")
    
    # Validate each present field against schema
    if "properties" in schema:
        for field, value in metrics.items():
            if field in schema["properties"]:
                field_errors = validate_field_strict(
                    value,
                    schema["properties"][field],
                    field,
                    depth=1
                )
                errors.extend(field_errors)
    
    return len(errors) == 0, errors, warnings


def validate_metrics(metrics: Dict, schema: Dict) -> Tuple[bool, List[str]]:
    """
    Legacy validate_metrics - calls strict validation.
    
    Returns:
        (is_valid, errors)
    """
    is_valid, errors, _ = validate_metrics_strict(metrics, schema)
    return is_valid, errors


def print_report(
    is_valid: bool, 
    errors: List[str], 
    warnings: List[str],
    metrics_path: str,
    metrics: Dict = None
):
    """Print validation report with detailed information."""
    print("=" * 60)
    print("METRICS SCHEMA VALIDATION REPORT")
    print("=" * 60)
    print(f"File: {metrics_path}")
    
    if is_valid:
        print(f"Status: ✅ VALID")
    else:
        print(f"Status: ❌ INVALID")
    
    # Print summary info if available
    if metrics:
        run_id = metrics.get('run_id', 'unknown')
        domain = metrics.get('domain', 'unknown')
        num_scenarios = len(metrics.get('scenarios', []))
        print(f"\nSummary:")
        print(f"  Run ID: {run_id}")
        print(f"  Domain: {domain}")
        print(f"  Scenarios: {num_scenarios}")
        
        if 'summary' in metrics:
            summary = metrics['summary']
            print(f"  Summary Metrics:")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
    
    # Print errors
    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    # Print warnings
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("=" * 60)
    
    # Print suggestions
    if errors:
        print("\n💡 Suggestions:")
        if any("Missing required field" in e for e in errors):
            print("  - Add missing required fields to your metrics output")
        if any("expected number" in e or "expected string" in e for e in errors):
            print("  - Check field types match schema (number vs string vs boolean)")
        if any("unexpected field" in e for e in errors):
            print("  - Remove extra fields or update schema to allow them")
        if any("not in allowed values" in e for e in errors):
            print("  - Check enum values match allowed schema values")


def find_latest_checkpoint(base_dir: str = "out", pattern: str = "rl_checkpoint.pt") -> Tuple[str, Dict]:
    """
    Find the most recent checkpoint matching the pattern.
    
    Args:
        base_dir: Base directory to search
        pattern: Filename pattern to match
    
    Returns:
        (path, metadata) or ("", {}) if not found
    """
    import glob
    from datetime import datetime
    
    # Find all matching checkpoints
    search_pattern = os.path.join(base_dir, "**", pattern)
    matches = glob.glob(search_pattern, recursive=True)
    
    if not matches:
        return "", {}
    
    # Get most recent by modification time
    latest = max(matches, key=os.path.getmtime)
    mtime = os.path.getmtime(latest)
    timestamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    # Try to find associated metrics
    run_dir = os.path.dirname(latest)
    metrics_path = os.path.join(run_dir, "metrics.json")
    
    metadata = {
        "checkpoint": latest,
        "run_dir": run_dir,
        "modified": timestamp,
        "mtime": mtime
    }
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metadata["metrics"] = json.load(f)
    
    return latest, metadata


def find_latest_sft_checkpoint(base_dir: str = "out") -> Tuple[str, Dict]:
    """Find the most recent SFT checkpoint."""
    return find_latest_checkpoint(base_dir, "sft_checkpoint.pt")


def main():
    parser = argparse.ArgumentParser(
        description='Validate metrics against schema with strict validation'
    )
    parser.add_argument('metrics_file', type=str, nargs='?', 
                        help='Path to metrics.json (optional if using --auto)')
    parser.add_argument('--schema', type=str, default='data/schema/metrics.json',
                        help='Path to schema file')
    parser.add_argument('--strict', action='store_true',
                        help='Enable strict validation with warnings')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-find latest checkpoint and validate its metrics')
    parser.add_argument('--find-checkpoint', action='store_true',
                        help='Find latest checkpoint without validation')
    args = parser.parse_args()
    
    # Handle --find-checkpoint
    if args.find_checkpoint:
        print("Finding latest checkpoints...")
        
        rl_ckpt, rl_meta = find_latest_checkpoint()
        if rl_ckpt:
            print(f"\nLatest RL checkpoint:")
            print(f"  Path: {rl_ckpt}")
            print(f"  Modified: {rl_meta.get('modified', 'unknown')}")
            if 'metrics' in rl_meta:
                m = rl_meta['metrics']
                print(f"  RL metrics: avg_reward={m.get('rl_metrics', {}).get('final_avg_reward', 'N/A')}")
        
        sft_ckpt, sft_meta = find_latest_sft_checkpoint()
        if sft_ckpt:
            print(f"\nLatest SFT checkpoint:")
            print(f"  Path: {sft_ckpt}")
            print(f"  Modified: {sft_meta.get('modified', 'unknown')}")
            if 'metrics' in sft_meta:
                m = sft_meta['metrics']
                print(f"  SFT metrics: val_loss={m.get('sft_metrics', {}).get('final_val_loss', 'N/A')}")
        
        return 0
    
    # Handle --auto
    if args.auto:
        print("Auto-finding latest checkpoints...")
        rl_ckpt, rl_meta = find_latest_checkpoint()
        
        if not rl_ckpt:
            print("No RL checkpoint found")
            return 1
        
        run_dir = rl_meta["run_dir"]
        metrics_file = os.path.join(run_dir, "metrics.json")
        
        if not os.path.exists(metrics_file):
            print(f"Metrics file not found: {metrics_file}")
            return 1
        
        args.metrics_file = metrics_file
    
    # Validate schema file exists
    if not os.path.exists(args.schema):
        print(f"Error: Schema not found at {args.schema}")
        return 1
    
    schema = load_schema(args.schema)
    
    # Validate metrics file exists
    if not args.metrics_file:
        print("Error: No metrics file specified (use --auto or provide path)")
        return 1
    
    if not os.path.exists(args.metrics_file):
        print(f"Error: Metrics file not found at {args.metrics_file}")
        return 1
    
    # Load and validate
    with open(args.metrics_file, 'r') as f:
        metrics = json.load(f)
    
    is_valid, errors, warnings = validate_metrics_strict(
        metrics, 
        schema, 
        require_all_fields=args.strict
    )
    
    # Print report
    print_report(is_valid, errors, warnings, args.metrics_file, metrics)
    
    return 0 if is_valid else 1


if __name__ == '__main__':
    exit(main())

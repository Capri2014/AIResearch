"""
Metrics Schema Validator.

Validates evaluation metrics output against the standard schema.
"""
import json
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_schema(schema_path: str = "data/schema/metrics.json") -> Dict:
    """Load the metrics schema."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_field(value: Any, schema: Dict, path: str) -> List[str]:
    """Validate a single field against schema."""
    errors = []
    
    if "type" in schema:
        expected_type = schema["type"]
        if expected_type == "object" and not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"{path}: expected array, got {type(value).__name__}")
        elif expected_type == "string" and not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"{path}: expected number, got {type(value).__name__}")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"{path}: expected boolean, got {type(value).__name__}")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"{path}: expected integer, got {type(value).__name__}")
    
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value '{value}' not in allowed values {schema['enum']}")
    
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: value {value} below minimum {schema['minimum']}")
    
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{path}: value {value} above maximum {schema['maximum']}")
    
    if "properties" in schema and isinstance(value, dict):
        for key, val in value.items():
            if key in schema["properties"]:
                child_errors = validate_field(val, schema["properties"][key], f"{path}.{key}")
                errors.extend(child_errors)
    
    if "items" in schema and isinstance(value, list):
        for i, item in enumerate(value):
            child_errors = validate_field(item, schema["items"], f"{path}[{i}]")
            errors.extend(child_errors)
    
    return errors


def validate_metrics(metrics: Dict, schema: Dict) -> Tuple[bool, List[str]]:
    """
    Validate metrics against schema.
    
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in metrics:
            errors.append(f"Missing required field: {field}")
    
    # Validate fields that exist
    if "run_id" in metrics:
        errors.extend(validate_field(metrics["run_id"], schema["properties"]["run_id"], "run_id"))
    
    if "domain" in metrics:
        errors.extend(validate_field(metrics["domain"], schema["properties"]["domain"], "domain"))
    
    if "scenarios" in metrics:
        errors.extend(validate_field(metrics["scenarios"], schema["properties"]["scenarios"], "scenarios"))
    
    if "summary" in metrics:
        errors.extend(validate_field(metrics["summary"], schema["properties"]["summary"], "summary"))
    
    if "comparison" in metrics:
        errors.extend(validate_field(metrics["comparison"], schema["properties"]["comparison"], "comparison"))
    
    return len(errors) == 0, errors


def print_report(is_valid: bool, errors: List[str], metrics_path: str):
    """Print validation report."""
    print("=" * 60)
    print("METRICS SCHEMA VALIDATION REPORT")
    print("=" * 60)
    print(f"File: {metrics_path}")
    
    if is_valid:
        print(f"Status: ✅ VALID")
    else:
        print(f"Status: ❌ INVALID")
        print(f"Errors: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Validate metrics against schema')
    parser.add_argument('metrics_file', type=str, help='Path to metrics.json')
    parser.add_argument('--schema', type=str, default='data/schema/metrics.json',
                        help='Path to schema file')
    args = parser.parse_args()
    
    # Load schema
    if not os.path.exists(args.schema):
        print(f"Error: Schema not found at {args.schema}")
        return 1
    
    schema = load_schema(args.schema)
    
    # Load metrics
    if not os.path.exists(args.metrics_file):
        print(f"Error: Metrics file not found at {args.metrics_file}")
        return 1
    
    with open(args.metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Validate
    is_valid, errors = validate_metrics(metrics, schema)
    
    # Print report
    print_report(is_valid, errors, args.metrics_file)
    
    return 0 if is_valid else 1


if __name__ == '__main__':
    exit(main())

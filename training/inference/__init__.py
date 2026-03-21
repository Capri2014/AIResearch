"""
Inference module for BC → RL driving models.

Exports:
- PipelineInference: Main inference class
- InferenceConfig: Configuration dataclass
- InferenceResult: Result dataclass
- load_inference_pipeline: Convenience loader
"""

from training.inference.pipeline_inference import (
    PipelineInference,
    InferenceConfig,
    InferenceResult,
    load_inference_pipeline,
)

__all__ = [
    "PipelineInference",
    "InferenceConfig",
    "InferenceResult",
    "load_inference_pipeline",
]

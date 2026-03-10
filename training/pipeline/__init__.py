"""Training Pipeline Module.

Provides unified pipeline runners for the driving-first approach:
  Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval

Submodules:
- run_driving_pipeline: Unified pipeline runner (BC -> RL -> Eval)
"""

from .run_driving_pipeline import (
    DrivingPipeline,
    PipelineConfig,
    PipelineResult,
    STAGE_BC,
    STAGE_RL,
    STAGE_EVAL,
    STAGE_FULL,
)

__all__ = [
    "DrivingPipeline",
    "PipelineConfig",
    "PipelineResult",
    "STAGE_BC",
    "STAGE_RL",
    "STAGE_EVAL",
    "STAGE_FULL",
]

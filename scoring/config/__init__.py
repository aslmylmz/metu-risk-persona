"""Configurable task definition: a single source of truth for the BART.

``TaskConfig`` drives both how balloons burst (the task) and where the
EV-optimum sits (scoring), so the two cannot disagree.
"""

from __future__ import annotations

from scoring.config.curve import BalloonCurve, balloon_curve
from scoring.config.hazards import (
    ConstantHazard,
    ExponentialHazard,
    GompertzHazard,
    HazardSpec,
    LinearHazard,
    LogisticHazard,
    LognormalHazard,
    RayleighHazard,
    StepHazard,
    TabularHazard,
    UniformHazard,
    WeibullHazard,
)
from scoring.config.task_config import DEFAULT_TASK_CONFIG, ColorProfile, TaskConfig

__all__ = [
    "DEFAULT_TASK_CONFIG",
    "BalloonCurve",
    "ColorProfile",
    "ConstantHazard",
    "ExponentialHazard",
    "GompertzHazard",
    "HazardSpec",
    "LinearHazard",
    "LogisticHazard",
    "LognormalHazard",
    "RayleighHazard",
    "StepHazard",
    "TabularHazard",
    "TaskConfig",
    "UniformHazard",
    "WeibullHazard",
    "balloon_curve",
]

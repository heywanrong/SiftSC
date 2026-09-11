"""SiftSC: selective self-consistency for small language models."""

from .backends import Backend, GenerationCost, MeteredBackend, MLXBackend
from .calibration import FittedLogisticProfile, fit_logistic
from .gates import ConfidenceGate, LogisticGate, list_profiles, load_profile
from .prompts import math_prompt
from .router import SiftSC
from .types import GateDecision, Generation, SiftResult
from .voting import parse_answer, plurality_vote

__all__ = [
    "Backend",
    "ConfidenceGate",
    "FittedLogisticProfile",
    "GateDecision",
    "Generation",
    "GenerationCost",
    "LogisticGate",
    "MLXBackend",
    "MeteredBackend",
    "SiftResult",
    "SiftSC",
    "fit_logistic",
    "list_profiles",
    "load_profile",
    "math_prompt",
    "parse_answer",
    "plurality_vote",
]

__version__ = "0.2.0"

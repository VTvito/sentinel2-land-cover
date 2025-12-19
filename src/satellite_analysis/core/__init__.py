"""Core types and configuration for satellite analysis."""

from .config import AnalysisConfig
from .types import (
    DataLocation,
    ClassificationResult,
    AnalysisResult,
    ClassifierType,
)

__all__ = [
    "AnalysisConfig",
    "DataLocation",
    "ClassificationResult", 
    "AnalysisResult",
    "ClassifierType",
]

"""Core types and configuration for satellite analysis."""

from .config import AnalysisConfig
from .types import (
    DataLocation,
    ClassificationResult,
    AnalysisResult,
    ClassifierType,
)
from .ports import ClassifierPort, AreaSelectorPort, OutputManagerPort

__all__ = [
    "AnalysisConfig",
    "DataLocation",
    "ClassificationResult", 
    "AnalysisResult",
    "ClassifierType",
    "ClassifierPort",
    "AreaSelectorPort",
    "OutputManagerPort",
]

"""Satellite Image Analysis Toolkit.

A comprehensive toolkit for downloading, processing, and analyzing Sentinel-2
satellite imagery using machine learning techniques.

Quick Start:
    >>> from satellite_analysis import analyze
    >>> result = analyze("Florence")
    >>> print(f"Confidence: {result.avg_confidence:.1%}")

Batch Analysis:
    >>> from satellite_analysis import analyze_batch
    >>> results = analyze_batch(["Milan", "Rome", "Florence"])

Change Detection:
    >>> from satellite_analysis import compare
    >>> changes = compare("Milan", "2023-06", "2024-06")
    >>> print(f"Changed: {changes.changed_percentage:.1%}")

Export Results:
    >>> from satellite_analysis import export_geotiff, export_report, export_image
    >>> export_geotiff(result, "output.tif")
    >>> export_report(result, "report.html")
    >>> export_image(result, "summary.png")
"""

__version__ = "2.1.0"
__author__ = "Vito Delia"

# Main API
from satellite_analysis.api import (
    analyze,
    quick_preview,
    analyze_batch,
    export_geotiff,
    export_report,
    export_json,
    export_image,
)

# Change Detection
from satellite_analysis.change_detection import (
    compare,
    ChangeResult,
    export_change_report,
)

# Core types
from satellite_analysis.core.config import AnalysisConfig
from satellite_analysis.core.types import AnalysisResult, ClassificationResult

# Export utilities (for advanced usage)
from satellite_analysis.exports import (
    export_colored_geotiff,
    LAND_COVER_CLASSES,
)

# Legacy
from satellite_analysis.config import Config

__all__ = [
    # Main API
    "analyze",
    "quick_preview",
    "analyze_batch",
    # Change Detection
    "compare",
    "ChangeResult",
    "export_change_report",
    # Exports
    "export_geotiff",
    "export_colored_geotiff",
    "export_report",
    "export_json",
    "export_image",
    # Types
    "AnalysisConfig",
    "AnalysisResult",
    "ClassificationResult",
    # Constants
    "LAND_COVER_CLASSES",
    # Legacy
    "Config",
    "__version__",
]

"""Type definitions for satellite analysis pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import numpy as np


# Type aliases
ClassifierType = Literal["kmeans", "spectral", "consensus"]
BandFormat = Literal["tif", "jp2"]
DataSource = Literal["local", "downloaded"]


@dataclass
class DataLocation:
    """Location of satellite data."""
    
    path: Path
    source: DataSource
    format: BandFormat
    
    def __post_init__(self):
        self.path = Path(self.path)


@dataclass
class ClassificationResult:
    """Result of land cover classification."""
    
    labels: np.ndarray
    confidence: np.ndarray
    uncertainty_mask: np.ndarray
    statistics: Dict[str, Any]
    
    @property
    def shape(self) -> tuple:
        return self.labels.shape
    
    @property
    def n_classes(self) -> int:
        return len(np.unique(self.labels))
    
    @property
    def mean_confidence(self) -> float:
        return float(self.confidence.mean())


@dataclass
class AnalysisResult:
    """Complete result of analysis pipeline."""
    
    city: str
    classification: ClassificationResult
    output_dir: Path
    execution_time: float
    original_shape: tuple
    processed_shape: tuple
    config_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Cropping/geographic metadata
    bbox: Optional[list] = None
    city_center: Optional[tuple] = None
    radius_km: Optional[float] = None
    cropped_to_city: bool = True
    
    # Convenience accessors
    @property
    def labels(self) -> np.ndarray:
        return self.classification.labels
    
    @property
    def confidence(self) -> np.ndarray:
        return self.classification.confidence
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence across all pixels."""
        return self.classification.mean_confidence
    
    @property
    def total_pixels(self) -> int:
        return self.labels.size
    
    @property
    def was_downsampled(self) -> bool:
        return self.original_shape != self.processed_shape
    
    def class_distribution(self) -> Dict[int, Dict[str, Any]]:
        """Get distribution of classes."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = counts.sum()
        return {
            int(cls): {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
            for cls, count in zip(unique, counts)
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Analysis Result for {self.city}\n"
            f"  Shape: {self.processed_shape}\n"
            f"  Classes: {self.classification.n_classes}\n"
            f"  Confidence: {self.classification.mean_confidence:.1%}\n"
            f"  Downsampled: {self.was_downsampled}\n"
            f"  Output: {self.output_dir}\n"
            f"  Time: {self.execution_time:.1f}s"
        )

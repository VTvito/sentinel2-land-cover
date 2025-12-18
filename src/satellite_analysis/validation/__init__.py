"""
Validation module for land cover classification.

Provides metrics, visualization, and validation against
ESA Scene Classification Layer (SCL).
"""

from satellite_analysis.validation.metrics import (
    compute_accuracy,
    compute_kappa,
    compute_f1_scores,
    compute_confusion_matrix,
    compute_producer_user_accuracy,
    ValidationReport
)

from satellite_analysis.validation.confusion_matrix import (
    plot_confusion_matrix,
    plot_classification_comparison,
    plot_confidence_map,
    plot_consensus_analysis
)

from satellite_analysis.validation.scl_validator import (
    SCL_CLASSES,
    SCL_TO_CONSENSUS,
    map_scl_to_consensus,
    get_scl_statistics,
    filter_valid_pixels,
    SCLValidator
)

__all__ = [
    # Metrics
    'compute_accuracy',
    'compute_kappa',
    'compute_f1_scores',
    'compute_confusion_matrix',
    'compute_producer_user_accuracy',
    'ValidationReport',
    
    # Visualization
    'plot_confusion_matrix',
    'plot_classification_comparison',
    'plot_confidence_map',
    'plot_consensus_analysis',
    
    # SCL Validation
    'SCL_CLASSES',
    'SCL_TO_CONSENSUS',
    'map_scl_to_consensus',
    'get_scl_statistics',
    'filter_valid_pixels',
    'SCLValidator'
]

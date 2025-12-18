"""
ESA Scene Classification Layer (SCL) utilities.

Provides tools for working with ESA's Scene Classification Layer
from Sentinel-2 Level-2A products for validation purposes.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ESA SCL Classes (from Sentinel-2 Level-2A products)
SCL_CLASSES = {
    0: 'No Data',
    1: 'Saturated/Defective',
    2: 'Dark Area Pixels',
    3: 'Cloud Shadows',
    4: 'Vegetation',
    5: 'Not Vegetated',
    6: 'Water',
    7: 'Unclassified',
    8: 'Cloud Medium Probability',
    9: 'Cloud High Probability',
    10: 'Thin Cirrus',
    11: 'Snow/Ice'
}

# Simplified SCL classes for land cover validation
# Grouping related classes for comparison with our classification
SCL_SIMPLIFIED = {
    0: 'INVALID',        # No Data
    1: 'INVALID',        # Saturated
    2: 'SHADOWS',        # Dark Areas (shadows)
    3: 'SHADOWS',        # Cloud Shadows
    4: 'VEGETATION',     # Vegetation
    5: 'BARE_URBAN',     # Not Vegetated (bare soil, urban, etc.)
    6: 'WATER',          # Water
    7: 'MIXED',          # Unclassified
    8: 'CLOUD',          # Cloud
    9: 'CLOUD',          # Cloud High
    10: 'CLOUD',         # Cirrus
    11: 'SNOW'           # Snow/Ice
}

# Mapping from SCL to our consensus classes
# Our classes: 0=WATER, 1=VEGETATION, 2=BARE_SOIL, 3=URBAN, 4=BRIGHT_SURFACES, 5=SHADOWS_MIXED
SCL_TO_CONSENSUS = {
    0: None,    # No Data -> exclude from validation
    1: None,    # Saturated -> exclude
    2: 5,       # Dark Areas -> SHADOWS_MIXED
    3: 5,       # Cloud Shadows -> SHADOWS_MIXED
    4: 1,       # Vegetation -> VEGETATION
    5: 2,       # Not Vegetated -> BARE_SOIL (could also be URBAN)
    6: 0,       # Water -> WATER
    7: 5,       # Unclassified -> SHADOWS_MIXED
    8: None,    # Cloud -> exclude
    9: None,    # Cloud High -> exclude
    10: None,   # Cirrus -> exclude
    11: 4       # Snow/Ice -> BRIGHT_SURFACES
}


def map_scl_to_consensus(scl_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map ESA SCL labels to our consensus classification classes.
    
    Args:
        scl_labels: 2D array of SCL labels (0-11)
        
    Returns:
        Tuple of (mapped_labels, valid_mask):
            - mapped_labels: Array with mapped class values (0-5)
            - valid_mask: Boolean mask, True where SCL has valid land cover
    """
    # Initialize output
    mapped = np.zeros_like(scl_labels, dtype=np.uint8)
    valid_mask = np.ones(scl_labels.shape, dtype=bool)
    
    # Apply mapping
    for scl_class, consensus_class in SCL_TO_CONSENSUS.items():
        mask = scl_labels == scl_class
        
        if consensus_class is None:
            # Mark as invalid (clouds, no data, etc.)
            valid_mask[mask] = False
        else:
            mapped[mask] = consensus_class
    
    # Count statistics
    total_pixels = scl_labels.size
    valid_pixels = np.sum(valid_mask)
    invalid_pixels = total_pixels - valid_pixels
    
    logger.info(f"SCL mapping complete: {valid_pixels:,} valid pixels "
               f"({valid_pixels/total_pixels*100:.1f}%), "
               f"{invalid_pixels:,} excluded (clouds, no data)")
    
    return mapped, valid_mask


def get_scl_statistics(scl_labels: np.ndarray) -> Dict[str, Dict]:
    """
    Compute statistics for SCL labels.
    
    Args:
        scl_labels: 2D array of SCL labels
        
    Returns:
        Dict with statistics per SCL class
    """
    unique, counts = np.unique(scl_labels, return_counts=True)
    total = scl_labels.size
    
    stats = {}
    for label, count in zip(unique, counts):
        scl_name = SCL_CLASSES.get(label, f"Unknown ({label})")
        simplified = SCL_SIMPLIFIED.get(label, "UNKNOWN")
        
        stats[scl_name] = {
            'scl_code': int(label),
            'simplified': simplified,
            'count': int(count),
            'percentage': float(count / total * 100)
        }
    
    return stats


def filter_valid_pixels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scl_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter predictions to only include pixels with valid SCL labels.
    
    Excludes clouds, cloud shadows (optionally), no data, and saturated pixels.
    
    Args:
        y_true: Ground truth (or SCL-derived) labels
        y_pred: Predicted labels
        scl_labels: Original SCL labels for filtering
        
    Returns:
        Tuple of (filtered_true, filtered_pred) - flattened arrays
    """
    # Get valid mask
    _, valid_mask = map_scl_to_consensus(scl_labels)
    
    # Apply mask
    y_true_filtered = y_true[valid_mask]
    y_pred_filtered = y_pred[valid_mask]
    
    logger.info(f"Filtered to {len(y_true_filtered):,} valid pixels "
               f"(from {y_true.size:,} total)")
    
    return y_true_filtered, y_pred_filtered


class SCLValidator:
    """
    Validator using ESA Scene Classification Layer as reference.
    
    Handles mapping between SCL classes and our classification scheme,
    and computes validation metrics excluding invalid pixels.
    """
    
    # Class definitions for our consensus scheme
    CONSENSUS_CLASSES = {
        0: 'WATER',
        1: 'VEGETATION',
        2: 'BARE_SOIL',
        3: 'URBAN',
        4: 'BRIGHT_SURFACES',
        5: 'SHADOWS_MIXED'
    }
    
    def __init__(
        self,
        scl_labels: np.ndarray,
        exclude_clouds: bool = True,
        exclude_shadows: bool = False
    ):
        """
        Initialize SCL Validator.
        
        Args:
            scl_labels: 2D array of ESA SCL labels
            exclude_clouds: If True, exclude cloudy pixels from validation
            exclude_shadows: If True, exclude shadow pixels from validation
        """
        self.scl_labels = scl_labels
        self.exclude_clouds = exclude_clouds
        self.exclude_shadows = exclude_shadows
        
        # Map SCL to consensus classes
        self.reference_labels, self.valid_mask = map_scl_to_consensus(scl_labels)
        
        # Optionally exclude shadows
        if exclude_shadows:
            shadow_mask = np.isin(scl_labels, [2, 3])  # Dark areas, cloud shadows
            self.valid_mask = self.valid_mask & ~shadow_mask
        
        # Statistics
        self.scl_stats = get_scl_statistics(scl_labels)
        
        logger.info(f"SCLValidator initialized: {np.sum(self.valid_mask):,} valid pixels")
    
    def validate(self, predicted_labels: np.ndarray) -> Dict:
        """
        Validate predicted labels against SCL-derived reference.
        
        Args:
            predicted_labels: 2D array of predicted labels (0-5)
            
        Returns:
            Dict with validation metrics and statistics
        """
        from satellite_analysis.validation.metrics import (
            compute_accuracy, compute_kappa, compute_f1_scores,
            compute_confusion_matrix, ValidationReport
        )
        
        # Ensure same shape
        if predicted_labels.shape != self.scl_labels.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted_labels.shape}, "
                f"SCL {self.scl_labels.shape}"
            )
        
        # Filter to valid pixels
        y_true = self.reference_labels[self.valid_mask].flatten()
        y_pred = predicted_labels[self.valid_mask].flatten()
        
        logger.info(f"Validating {len(y_true):,} pixels...")
        
        # Compute metrics
        accuracy = compute_accuracy(y_true, y_pred)
        kappa = compute_kappa(y_true, y_pred)
        f1_weighted = compute_f1_scores(y_true, y_pred, self.CONSENSUS_CLASSES, average='weighted')
        f1_per_class = compute_f1_scores(y_true, y_pred, self.CONSENSUS_CLASSES, average=None)
        
        # Confusion matrix
        classes = np.arange(6)
        cm = compute_confusion_matrix(y_true, y_pred, classes)
        
        # Create full report
        report = ValidationReport(y_true, y_pred, self.CONSENSUS_CLASSES)
        
        return {
            'overall_accuracy': accuracy,
            'kappa': kappa,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'n_valid_pixels': len(y_true),
            'n_total_pixels': self.scl_labels.size,
            'valid_percentage': len(y_true) / self.scl_labels.size * 100,
            'scl_statistics': self.scl_stats,
            'report': report
        }

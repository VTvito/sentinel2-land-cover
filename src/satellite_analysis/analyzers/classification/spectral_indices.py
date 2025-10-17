"""
Spectral Indices Classifier for automatic land cover classification.

Based on scientifically validated indices (NDVI, MNDWI, NDBI, BSI) used by:
- ESA Scene Classification Layer (SCL)
- USGS Land Cover
- Google Earth Engine

Zero labeling required, infinitely scalable.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectralIndicesClassifier:
    """
    Automatic land cover classification using spectral indices.
    
    Classes:
        0: WATER
        1: FOREST (dense vegetation)
        2: GRASSLAND (moderate vegetation)
        3: URBAN (built-up areas)
        4: BARE_SOIL
        5: MIXED (uncertain)
    
    Attributes:
        thresholds: Dict with classification thresholds (customizable per region/season)
    """
    
    # Class mapping
    CLASSES = {
        0: 'WATER',
        1: 'FOREST',
        2: 'GRASSLAND',
        3: 'URBAN',
        4: 'BARE_SOIL',
        5: 'MIXED'
    }
    
    # Default thresholds (tunable per region)
    DEFAULT_THRESHOLDS = {
        'mndwi_water': 0.3,
        'ndwi_water': 0.3,
        'ndvi_forest': 0.6,
        'ndvi_grassland_min': 0.3,
        'ndvi_grassland_max': 0.6,
        'ndbi_urban': 0.0,
        'ndvi_urban_max': 0.3,
        'bsi_soil': 0.1,
        'ndvi_soil_max': 0.2
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize classifier with custom or default thresholds.
        
        Args:
            thresholds: Dict with custom thresholds (overrides defaults)
        """
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
        
        logger.info(f"SpectralIndicesClassifier initialized with thresholds: {self.thresholds}")
    
    def compute_indices(self, raster: np.ndarray, band_indices: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Compute spectral indices from multi-band raster.
        
        Args:
            raster: Array of shape (height, width, bands) or (bands, height, width)
            band_indices: Dict mapping band names to array indices
                Required: 'B02' (BLUE), 'B03' (GREEN), 'B04' (RED), 
                          'B08' (NIR), 'B11' (SWIR1), 'B12' (SWIR2)
        
        Returns:
            Dict with computed indices: {'NDVI': array, 'MNDWI': array, ...}
        """
        # Handle both (H,W,C) and (C,H,W) formats
        if raster.shape[0] < 20:  # Assume (C,H,W) if first dim is small
            raster = np.transpose(raster, (1, 2, 0))
        
        # Extract bands (convert to float to avoid integer division)
        blue = raster[..., band_indices['B02']].astype(np.float32)
        green = raster[..., band_indices['B03']].astype(np.float32)
        red = raster[..., band_indices['B04']].astype(np.float32)
        nir = raster[..., band_indices['B08']].astype(np.float32)
        swir1 = raster[..., band_indices['B11']].astype(np.float32)
        swir2 = raster[..., band_indices['B12']].astype(np.float32)
        
        # Compute indices with safe division (avoid /0)
        indices = {}
        
        # NDVI (Normalized Difference Vegetation Index)
        indices['NDVI'] = self._safe_divide(nir - red, nir + red)
        
        # MNDWI (Modified Normalized Difference Water Index)
        indices['MNDWI'] = self._safe_divide(green - swir1, green + swir1)
        
        # NDWI (Normalized Difference Water Index - alternative)
        indices['NDWI'] = self._safe_divide(green - nir, green + nir)
        
        # NDBI (Normalized Difference Built-up Index)
        indices['NDBI'] = self._safe_divide(swir1 - nir, swir1 + nir)
        
        # BSI (Bare Soil Index)
        numerator = (swir1 + red) - (nir + blue)
        denominator = (swir1 + red) + (nir + blue)
        indices['BSI'] = self._safe_divide(numerator, denominator)
        
        logger.info(f"Computed spectral indices. Shape: {indices['NDVI'].shape}")
        return indices
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """
        Safe division avoiding division by zero.
        
        Args:
            numerator: Numerator array
            denominator: Denominator array
            fill_value: Value to use when denominator is zero
        
        Returns:
            Division result with fill_value where denominator is zero
        """
        result = np.full_like(numerator, fill_value, dtype=np.float32)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        return result
    
    def classify(self, indices: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Classify pixels based on spectral indices using decision tree.
        
        Args:
            indices: Dict with computed indices from compute_indices()
        
        Returns:
            Array of shape (height, width) with class labels (0-5)
        """
        ndvi = indices['NDVI']
        mndwi = indices['MNDWI']
        ndwi = indices['NDWI']
        ndbi = indices['NDBI']
        bsi = indices['BSI']
        
        # Initialize with MIXED class (5)
        labels = np.full(ndvi.shape, 5, dtype=np.uint8)
        
        # Priority 1: WATER (most distinctive)
        water_mask = (mndwi > self.thresholds['mndwi_water']) | (ndwi > self.thresholds['ndwi_water'])
        labels[water_mask] = 0
        
        # Priority 2: FOREST (dense vegetation)
        forest_mask = (ndvi > self.thresholds['ndvi_forest']) & ~water_mask
        labels[forest_mask] = 1
        
        # Priority 3: GRASSLAND (moderate vegetation)
        grassland_mask = (
            (ndvi > self.thresholds['ndvi_grassland_min']) & 
            (ndvi <= self.thresholds['ndvi_grassland_max']) & 
            ~water_mask
        )
        labels[grassland_mask] = 2
        
        # Priority 4: URBAN (built-up areas)
        urban_mask = (
            (ndbi > self.thresholds['ndbi_urban']) & 
            (ndvi < self.thresholds['ndvi_urban_max']) & 
            ~water_mask
        )
        labels[urban_mask] = 3
        
        # Priority 5: BARE_SOIL
        soil_mask = (
            (bsi > self.thresholds['bsi_soil']) & 
            (ndvi < self.thresholds['ndvi_soil_max']) & 
            ~water_mask & ~urban_mask
        )
        labels[soil_mask] = 4
        
        # Count pixels per class
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = {self.CLASSES[label]: count for label, count in zip(unique, counts)}
        logger.info(f"Classification complete. Distribution: {class_distribution}")
        
        return labels
    
    def classify_raster(self, raster: np.ndarray, band_indices: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        End-to-end classification: compute indices + classify.
        
        Args:
            raster: Multi-band raster array
            band_indices: Dict mapping band names to array indices
        
        Returns:
            Tuple of (labels, indices):
                - labels: Classification result (height, width)
                - indices: Computed spectral indices (for analysis/validation)
        """
        indices = self.compute_indices(raster, band_indices)
        labels = self.classify(indices)
        return labels, indices
    
    def get_class_statistics(self, labels: np.ndarray) -> Dict[str, Dict]:
        """
        Compute statistics for classified image.
        
        Args:
            labels: Classification result from classify()
        
        Returns:
            Dict with class statistics: count, percentage, name
        """
        unique, counts = np.unique(labels, return_counts=True)
        total_pixels = labels.size
        
        stats = {}
        for label, count in zip(unique, counts):
            class_name = self.CLASSES[label]
            stats[class_name] = {
                'label': int(label),
                'count': int(count),
                'percentage': float(count / total_pixels * 100)
            }
        
        return stats
    
    def validate_bands(self, band_indices: Dict[str, int], num_bands: int) -> bool:
        """
        Validate that all required bands are present and indices are valid.
        
        Args:
            band_indices: Dict mapping band names to array indices
            num_bands: Total number of bands in raster
        
        Returns:
            True if valid, False otherwise
        
        Raises:
            ValueError: If bands are missing or indices are invalid
        """
        required_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        
        # Check all bands present
        missing = [b for b in required_bands if b not in band_indices]
        if missing:
            raise ValueError(f"Missing required bands: {missing}")
        
        # Check indices are valid
        for band, idx in band_indices.items():
            if idx < 0 or idx >= num_bands:
                raise ValueError(f"Band {band} index {idx} out of range [0, {num_bands})")
        
        logger.info(f"Band validation passed. Required bands: {required_bands}")
        return True

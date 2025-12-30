"""
Data normalization utilities.
Based on original notebook's min_max_scale function.
"""

import numpy as np


def min_max_scale(ar: np.ndarray, min_val: float = 0.0, max_val: float = 1.0, uint8: bool = False) -> np.ndarray:
    """
    Scale array values to specified range using min-max normalization.
    
    Each band is scaled independently to preserve relative differences.
    Based on original notebook implementation.
    
    Args:
        ar: Input array of shape (height, width, bands) or (n_samples, n_features)
        min_val: Minimum value of output range (default: 0.0)
        max_val: Maximum value of output range (default: 1.0)
        uint8: If True, convert to uint8 in [0, 255] range
        
    Returns:
        Scaled array with same shape as input
        
    Example:
        >>> data = np.array([[100, 200], [150, 250]])
        >>> scaled = min_max_scale(data, min_val=0, max_val=1)
        >>> print(scaled.min(), scaled.max())
        0.0 1.0
    """
    # Create output array
    res = np.zeros_like(ar, dtype=np.float32)
    
    # Scale each band/feature independently
    for i in range(ar.shape[-1]):
        band = ar[..., i].copy()
        
        # Min-max normalization
        band_min = band.min()
        band_max = band.max()
        
        if band_max > band_min:
            scale = (band - band_min) / (band_max - band_min)
        else:
            # Avoid division by zero for constant bands
            scale = np.zeros_like(band)
        
        # Store scaled values
        res[..., i] = scale
    
    # Transform to specified range
    res = res * (max_val - min_val) + min_val
    
    # Convert to uint8 if requested
    if uint8:
        res = (res * 255).astype(np.uint8)
    
    return res


def standard_scale(ar: np.ndarray) -> np.ndarray:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    This is the StandardScaler approach used in the Hejmanowska & Kramarczyk paper.
    z = (x - mean) / std
    
    Better for K-means clustering as it handles different band value ranges properly.
    
    Args:
        ar: Input array of shape (n_samples, n_features)
        
    Returns:
        Standardized array with same shape as input (mean=0, std=1 per feature)
        
    Example:
        >>> data = np.array([[100, 200], [150, 250], [120, 220]])
        >>> scaled = standard_scale(data)
        >>> print(f"Mean: {scaled.mean(axis=0)}, Std: {scaled.std(axis=0)}")
    """
    res = np.zeros_like(ar, dtype=np.float32)
    
    for i in range(ar.shape[-1]):
        feature = ar[..., i].astype(np.float32)
        mean = feature.mean()
        std = feature.std()
        
        if std > 0:
            res[..., i] = (feature - mean) / std
        else:
            # Constant feature - set to zero
            res[..., i] = 0.0
    
    return res

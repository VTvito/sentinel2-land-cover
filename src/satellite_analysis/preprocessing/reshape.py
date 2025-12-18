"""
Array reshaping utilities for image-table conversions.
Based on original notebook's reshape functions.
"""

import numpy as np


def reshape_image_to_table(ar: np.ndarray) -> np.ndarray:
    """
    Reshape image array from 3D to 2D table format.
    
    Converts (height, width, bands) → (n_pixels, n_bands)
    Each row represents one pixel with all its band values.
    
    Args:
        ar: Image array of shape (height, width, n_bands)
        
    Returns:
        Table array of shape (height * width, n_bands)
        
    Example:
        >>> image = np.random.rand(100, 100, 10)  # 100x100 image, 10 bands
        >>> table = reshape_image_to_table(image)
        >>> print(table.shape)
        (10000, 10)
    """
    # Flatten first two dimensions, keep last dimension (bands)
    data = np.reshape(ar, (-1, ar.shape[-1]))
    return data


def reshape_table_to_image(original_shape: tuple, labels: np.ndarray) -> np.ndarray:
    """
    Reshape 1D labels back to 2D image format.
    
    Converts (n_pixels,) → (height, width)
    
    Args:
        original_shape: Original image shape (height, width) or (height, width, bands)
        labels: 1D array of labels/values for each pixel
        
    Returns:
        2D image array of shape (height, width)
        
    Example:
        >>> labels = np.array([0, 1, 0, 1, 1, 0])
        >>> image = reshape_table_to_image((2, 3), labels)
        >>> print(image.shape)
        (2, 3)
    """
    # Extract height and width from original shape
    height, width = original_shape[:2]
    
    # Reshape labels to match image dimensions
    res = labels.reshape(height, width)
    return res

"""Visualization utility functions."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
from PIL.ImageOps import equalize
from typing import List, Optional, Tuple


def min_max_scale(
    array: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    as_uint8: bool = False
) -> np.ndarray:
    """Apply min-max scaling to array.
    
    Args:
        array: Input array of shape (..., channels)
        min_val: Minimum value for scaling
        max_val: Maximum value for scaling
        as_uint8: Whether to convert to uint8 (0-255)
        
    Returns:
        Scaled array
    """
    result = np.zeros_like(array, dtype=np.float32)
    
    # Scale each band independently
    for i in range(array.shape[-1]):
        band = array[..., i].copy()
        band_min, band_max = band.min(), band.max()
        
        if band_max > band_min:
            scaled = (band - band_min) / (band_max - band_min)
        else:
            scaled = np.zeros_like(band)
        
        result[..., i] = scaled
    
    # Apply range transformation
    result = result * (max_val - min_val) + min_val
    
    if as_uint8:
        result = (result * 255).astype(np.uint8)
    
    return result


def create_rgb_image(
    array: np.ndarray,
    band_indices: List[int],
    equalize_hist: bool = True
) -> Image.Image:
    """Create RGB image from array.
    
    Args:
        array: Image array with shape (height, width, channels)
        band_indices: List of 3 band indices for R, G, B
        equalize_hist: Whether to apply histogram equalization
        
    Returns:
        PIL Image object
    """
    if len(band_indices) != 3:
        raise ValueError("Must provide exactly 3 band indices for RGB")
    
    rgb_data = array[:, :, band_indices]
    
    # Ensure uint8 format
    if rgb_data.dtype != np.uint8:
        rgb_data = min_max_scale(rgb_data, as_uint8=True)
    
    image = Image.fromarray(rgb_data)
    
    if equalize_hist:
        image = equalize(image)
    
    return image


def plot_clustered_image(
    labels: np.ndarray,
    n_clusters: int,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    cmap_name: str = "tab10_r"
) -> plt.Figure:
    """Plot clustered image with legend.
    
    Args:
        labels: 2D array of cluster labels
        n_clusters: Number of clusters
        class_names: Optional list of class names
        figsize: Figure size
        cmap_name: Matplotlib colormap name
        
    Returns:
        Matplotlib Figure object
    """
    # Create colormap
    base_cmap = plt.get_cmap(cmap_name)
    cmap = ListedColormap([base_cmap(i) for i in range(n_clusters)])
    
    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_clusters)]
    
    # Create legend patches
    legend_patches = [
        Patch(color=cmap(i), label=class_names[i])
        for i in range(min(n_clusters, len(class_names)))
    ]
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    plt.imshow(labels, cmap=cmap, vmax=n_clusters - 1)
    plt.colorbar()
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.3, 1), loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    
    return fig


def plot_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Image 1",
    title2: str = "Image 2",
    figsize: Tuple[int, int] = (20, 10),
    cmap: Optional[str] = None
) -> plt.Figure:
    """Plot two images side by side for comparison.
    
    Args:
        image1: First image array
        image2: Second image array
        title1: Title for first image
        title2: Title for second image
        figsize: Figure size
        cmap: Optional colormap
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].imshow(image1, cmap=cmap)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(image2, cmap=cmap)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_elbow_curve(inertias: List[float], k_values: List[int]) -> plt.Figure:
    """Plot elbow curve for KMeans clustering.
    
    Args:
        inertias: List of inertia values
        k_values: List of k values
        
    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

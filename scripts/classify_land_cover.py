"""
Example: Land cover classification using SpectralIndicesClassifier

This script demonstrates how to:
1. Load a Sentinel-2 raster stack
2. Apply automatic classification using spectral indices
3. Visualize results
4. Export classification map
"""

import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from satellite_analysis.analyzers import SpectralIndicesClassifier


def load_sentinel_stack(tiff_path: str) -> tuple:
    """
    Load Sentinel-2 stack and extract band information.
    
    Returns:
        (raster_array, band_names, profile)
    """
    with rasterio.open(tiff_path) as src:
        # Read all bands
        raster = src.read()  # Shape: (bands, height, width)
        
        # Get band names from descriptions
        band_names = []
        for i in range(1, src.count + 1):
            tags = src.tags(i)
            if 'name' in tags:
                band_names.append(tags['name'])
        
        profile = src.profile
    
    print(f"Loaded raster: {raster.shape}")
    print(f"Bands: {band_names}")
    
    return raster, band_names, profile


def create_band_indices(band_names: list) -> dict:
    """
    Create band index mapping from band names.
    
    Args:
        band_names: List of band names (e.g., ['B02', 'B03', ...])
    
    Returns:
        Dict mapping band names to array indices
    """
    band_indices = {}
    for i, name in enumerate(band_names):
        band_indices[name] = i
    
    return band_indices


def visualize_classification(labels: np.ndarray, 
                             indices: dict,
                             rgb: np.ndarray,
                             save_path: str = None):
    """
    Visualize classification results alongside RGB and indices.
    
    Args:
        labels: Classification result
        indices: Spectral indices dict
        rgb: RGB composite for reference
        save_path: Optional path to save figure
    """
    # Define colormap for classes
    colors = ['#0066cc', '#006600', '#99cc00', '#cc3300', '#cc9966', '#999999']
    cmap = ListedColormap(colors)
    
    # Class names
    class_names = ['Water', 'Forest', 'Grassland', 'Urban', 'Bare Soil', 'Mixed']
    legend_patches = [Patch(color=colors[i], label=name) for i, name in enumerate(class_names)]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Composite', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Classification
    im1 = axes[0, 1].imshow(labels, cmap=cmap, vmin=0, vmax=5)
    axes[0, 1].set_title('Classification Result', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    axes[0, 1].legend(handles=legend_patches, loc='upper right', fontsize=8)
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / labels.size * 100
    axes[0, 2].bar([class_names[i] for i in unique], percentages, color=[colors[i] for i in unique])
    axes[0, 2].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Percentage (%)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # NDVI
    im2 = axes[1, 0].imshow(indices['NDVI'], cmap='RdYlGn', vmin=-0.5, vmax=1)
    axes[1, 0].set_title('NDVI (Vegetation)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # MNDWI
    im3 = axes[1, 1].imshow(indices['MNDWI'], cmap='Blues', vmin=-0.5, vmax=1)
    axes[1, 1].set_title('MNDWI (Water)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # NDBI
    im4 = axes[1, 2].imshow(indices['NDBI'], cmap='RdGy_r', vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title('NDBI (Urban)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def save_classification(labels: np.ndarray, 
                       output_path: str,
                       profile: dict):
    """
    Save classification result as GeoTIFF.
    
    Args:
        labels: Classification array
        output_path: Path to save GeoTIFF
        profile: Rasterio profile from input raster
    """
    # Update profile for single-band output
    profile.update(
        count=1,
        dtype=rasterio.uint8,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(labels.astype(np.uint8), 1)
        
        # Add band description
        dst.update_tags(1, name='land_cover_classification')
        
        # Add class labels as metadata
        dst.update_tags(
            class_0='Water',
            class_1='Forest',
            class_2='Grassland',
            class_3='Urban',
            class_4='Bare_Soil',
            class_5='Mixed'
        )
    
    print(f"Classification saved to: {output_path}")


def main():
    """Main execution."""
    
    # === CONFIGURATION ===
    INPUT_TIFF = 'data/milano/sentinel_stack.tiff'  # Update with your path
    OUTPUT_DIR = 'data/milano/classification/'
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # === 1. LOAD DATA ===
    print("="*60)
    print("STEP 1: Loading Sentinel-2 stack")
    print("="*60)
    
    raster, band_names, profile = load_sentinel_stack(INPUT_TIFF)
    band_indices = create_band_indices(band_names)
    
    # === 2. CLASSIFY ===
    print("\n" + "="*60)
    print("STEP 2: Spectral Indices Classification")
    print("="*60)
    
    classifier = SpectralIndicesClassifier()
    
    # Validate bands
    classifier.validate_bands(band_indices, len(band_names))
    
    # Classify
    print("\nClassifying...")
    labels, indices = classifier.classify_raster(raster, band_indices)
    
    # Get statistics
    stats = classifier.get_class_statistics(labels)
    print("\nClassification Statistics:")
    for class_name, info in stats.items():
        print(f"  {class_name:12s}: {info['percentage']:5.1f}% ({info['count']:,} pixels)")
    
    # === 3. VISUALIZE ===
    print("\n" + "="*60)
    print("STEP 3: Visualization")
    print("="*60)
    
    # Create RGB composite for reference
    # Transpose to (H, W, C) and normalize
    raster_hwc = np.transpose(raster, (1, 2, 0))
    
    # Get RGB bands (B04=Red, B03=Green, B02=Blue)
    rgb_indices = [band_names.index('B04'), band_names.index('B03'), band_names.index('B02')]
    rgb = raster_hwc[..., rgb_indices]
    
    # Min-max normalization for visualization
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band = rgb[..., i].astype(np.float32)
        rgb_norm[..., i] = (band - band.min()) / (band.max() - band.min())
    
    # Visualize
    visualize_classification(
        labels, 
        indices, 
        rgb_norm,
        save_path=f"{OUTPUT_DIR}/classification_results.png"
    )
    
    # === 4. EXPORT ===
    print("\n" + "="*60)
    print("STEP 4: Export Results")
    print("="*60)
    
    save_classification(
        labels,
        f"{OUTPUT_DIR}/land_cover_classification.tif",
        profile
    )
    
    # Save indices as multi-band GeoTIFF
    indices_stack = np.stack([
        indices['NDVI'],
        indices['MNDWI'],
        indices['NDWI'],
        indices['NDBI'],
        indices['BSI']
    ])
    
    profile.update(count=5, dtype=rasterio.float32)
    with rasterio.open(f"{OUTPUT_DIR}/spectral_indices.tif", 'w', **profile) as dst:
        dst.write(indices_stack)
        dst.update_tags(1, name='NDVI')
        dst.update_tags(2, name='MNDWI')
        dst.update_tags(3, name='NDWI')
        dst.update_tags(4, name='NDBI')
        dst.update_tags(5, name='BSI')
    
    print(f"Spectral indices saved to: {OUTPUT_DIR}/spectral_indices.tif")
    
    print("\n" + "="*60)
    print("✅ CLASSIFICATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

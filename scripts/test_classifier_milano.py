"""
Quick test of SpectralIndicesClassifier on Milano data.

This script:
1. Loads individual band files (B02, B03, B04, B08 from processed_final)
2. Creates a synthetic stack (we need B11, B12 for full classification)
3. Runs classification
4. Visualizes results
"""

import sys
sys.path.insert(0, r'c:\TEMP_1\satellite_git\src')

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
from pathlib import Path

from satellite_analysis.analyzers import SpectralIndicesClassifier


def load_bands(base_path):
    """Load available bands from processed_final."""
    bands = {}
    band_names = ['B02', 'B03', 'B04', 'B08']
    
    # Try to load B11, B12 if available
    swir_bands = []
    for file in Path(base_path).glob('*_B1[12]_*.jp2'):
        band_name = 'B11' if 'B11' in file.name else 'B12'
        swir_bands.append((band_name, file))
    
    # Load standard bands
    for band_name in band_names:
        file_path = f"{base_path}/{band_name}.jp2"
        try:
            with rasterio.open(file_path) as src:
                bands[band_name] = src.read(1)
                print(f"✅ Loaded {band_name}: {bands[band_name].shape}")
        except Exception as e:
            print(f"❌ Failed to load {band_name}: {e}")
    
    # Load SWIR bands if found
    for band_name, file_path in swir_bands:
        try:
            with rasterio.open(file_path) as src:
                bands[band_name] = src.read(1)
                print(f"✅ Loaded {band_name} (SWIR): {bands[band_name].shape} [REAL DATA]")
        except Exception as e:
            print(f"❌ Failed to load {band_name}: {e}")
    
    return bands


def create_mock_swir(nir_band, red_band, blue_band, shape):
    """
    Create mock SWIR bands for testing (since we don't have real ones).
    
    SWIR typically:
    - Low for water (absorbs SWIR strongly)
    - Medium-high for vegetation (reflects some SWIR)
    - High for bare soil and urban (reflects SWIR strongly)
    
    We'll create more realistic SWIR based on multiple bands.
    """
    # SWIR1 (B11): Use combination of NIR and RED
    # Where NIR is high and RED is low (vegetation) → medium SWIR
    # Where both are low (water) → very low SWIR
    # Where both are medium-high (urban/soil) → high SWIR
    
    # Normalize inputs
    nir_norm = nir_band / (nir_band.max() + 1e-10)
    red_norm = red_band / (red_band.max() + 1e-10)
    blue_norm = blue_band / (blue_band.max() + 1e-10)
    
    # SWIR1: Higher for bare/urban, medium for vegetation, low for water
    # Formula: weighted combination to simulate real SWIR behavior
    swir1_norm = 0.4 * red_norm + 0.3 * nir_norm + 0.3 * blue_norm
    swir1 = (swir1_norm * 2500 + 300).astype(np.uint16)  # Range: 300-2800
    
    # SWIR2: Similar but slightly lower values
    swir2_norm = 0.35 * red_norm + 0.35 * nir_norm + 0.3 * blue_norm
    swir2 = (swir2_norm * 2200 + 200).astype(np.uint16)  # Range: 200-2400
    
    return swir1, swir2


def create_stack(bands_dict):
    """Create multi-band stack from individual bands."""
    # Get shape from first band
    first_band = list(bands_dict.values())[0]
    height, width = first_band.shape
    
    # Check if we have real SWIR bands
    has_real_swir = 'B11' in bands_dict and 'B12' in bands_dict
    
    if not has_real_swir:
        # Create mock SWIR bands
        print("\n⚠️  Creating mock SWIR bands (B11, B12) for testing...")
        print("   (Real pipeline would extract these from Sentinel-2 L2A)")
        swir1, swir2 = create_mock_swir(
            bands_dict['B08'], 
            bands_dict['B04'], 
            bands_dict['B02'],
            first_band.shape
        )
    else:
        print("\n✅ Using REAL SWIR bands (B11, B12) from Sentinel-2 data")
        swir1 = bands_dict['B11']
        swir2 = bands_dict['B12']
        
        # Resample from 20m to 10m if needed
        if swir1.shape != first_band.shape:
            print(f"   Resampling SWIR from {swir1.shape} to {first_band.shape}...")
            import scipy.ndimage as ndimage
            zoom_factor = first_band.shape[0] / swir1.shape[0]
            swir1 = ndimage.zoom(swir1, zoom_factor, order=1)
            swir2 = ndimage.zoom(swir2, zoom_factor, order=1)
            print(f"   ✅ Resampled to {swir1.shape}")
    
    # Order: B02, B03, B04, B05(mock), B06(mock), B07(mock), B08, B8A(mock), B11, B12
    # We'll create a simplified 10-band stack
    stack = np.zeros((10, height, width), dtype=np.uint16)
    
    stack[0, ...] = bands_dict['B02']  # Blue
    stack[1, ...] = bands_dict['B03']  # Green
    stack[2, ...] = bands_dict['B04']  # Red
    stack[3, ...] = bands_dict['B04']  # Mock B05
    stack[4, ...] = bands_dict['B04']  # Mock B06
    stack[5, ...] = bands_dict['B04']  # Mock B07
    stack[6, ...] = bands_dict['B08']  # NIR
    stack[7, ...] = bands_dict['B08']  # Mock B8A
    stack[8, ...] = swir1              # SWIR1
    stack[9, ...] = swir2              # SWIR2
    
    print(f"✅ Created stack: {stack.shape}")
    return stack


def visualize_results(labels, indices, rgb_bands, stats):
    """Visualize classification results."""
    
    # Colors for each class
    colors = ['#0066cc', '#006600', '#99cc00', '#cc3300', '#cc9966', '#999999']
    cmap = ListedColormap(colors)
    
    # Class names
    class_names = ['Water', 'Forest', 'Grassland', 'Urban', 'Bare Soil', 'Mixed']
    legend_patches = [Patch(color=colors[i], label=name) for i, name in enumerate(class_names)]
    
    # Create RGB composite
    rgb = np.stack([rgb_bands['B04'], rgb_bands['B03'], rgb_bands['B02']], axis=-1)
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band = rgb[..., i].astype(np.float32)
        rgb_norm[..., i] = (band - band.min()) / (band.max() - band.min())
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # RGB
    axes[0, 0].imshow(rgb_norm)
    axes[0, 0].set_title('RGB Composite (Milano)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Classification
    im1 = axes[0, 1].imshow(labels, cmap=cmap, vmin=0, vmax=5)
    axes[0, 1].set_title('Spectral Indices Classification', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    axes[0, 1].legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    # Class distribution
    class_list = [stats.get(name, {'percentage': 0})['percentage'] for name in class_names]
    colors_present = [colors[i] for i, name in enumerate(class_names) if name in stats]
    names_present = [name for name in class_names if name in stats]
    percentages_present = [stats[name]['percentage'] for name in names_present]
    
    if percentages_present:  # Check if list is not empty
        axes[0, 2].bar(names_present, percentages_present, color=colors_present)
        axes[0, 2].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Percentage (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].set_ylim(0, max(percentages_present) * 1.1)
    else:
        axes[0, 2].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Class Distribution', fontsize=14, fontweight='bold')
    
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
    
    # Save
    output_path = 'data/processed/milano_classification_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved visualization to: {output_path}")
    
    plt.show()


def main():
    print("="*70)
    print("🛰️  SPECTRAL INDICES CLASSIFIER - Milano Test")
    print("="*70)
    
    # Load bands
    print("\n📂 Loading Milano bands...")
    # Using processed_final which has SWIR bands (B11, B12) already extracted
    base_path = r'c:\TEMP_1\satellite_git\data\processed_final\product_1'
    bands = load_bands(base_path)
    
    if len(bands) < 4:
        print("\n❌ Missing required bands. Cannot proceed.")
        return
    
    # Create stack
    print("\n🔧 Creating multi-band stack...")
    stack = create_stack(bands)
    
    # Band indices mapping
    band_indices = {
        'B02': 0,  # Blue
        'B03': 1,  # Green
        'B04': 2,  # Red
        'B08': 6,  # NIR
        'B11': 8,  # SWIR1 (mock)
        'B12': 9   # SWIR2 (mock)
    }
    
    # Initialize classifier
    print("\n🎯 Initializing SpectralIndicesClassifier...")
    classifier = SpectralIndicesClassifier()
    
    # Classify
    print("\n🚀 Running classification...")
    labels, indices = classifier.classify_raster(stack, band_indices)
    
    # Get statistics
    stats = classifier.get_class_statistics(labels)
    
    print("\n📊 Classification Results:")
    print("-" * 50)
    for class_name, info in stats.items():
        print(f"  {class_name:12s}: {info['percentage']:5.1f}% ({info['count']:,} pixels)")
    
    # Visual check
    print("\n🔍 Visual Validation:")
    print("-" * 50)
    
    # Check NDVI statistics
    ndvi_mean = indices['NDVI'].mean()
    ndvi_max = indices['NDVI'].max()
    print(f"  NDVI mean: {ndvi_mean:.3f} (expect 0.2-0.5 for mixed urban/vegetation)")
    print(f"  NDVI max:  {ndvi_max:.3f} (expect 0.6-0.8 for parks/vegetation)")
    
    # Check water detection
    mndwi_mean = indices['MNDWI'].mean()
    water_pixels = stats.get('WATER', {'percentage': 0})['percentage']
    print(f"  MNDWI mean: {mndwi_mean:.3f}")
    print(f"  Water pixels: {water_pixels:.1f}% (expect ~2-5% for Milano rivers)")
    
    # Check urban
    urban_pixels = stats.get('URBAN', {'percentage': 0})['percentage']
    print(f"  Urban pixels: {urban_pixels:.1f}% (expect 40-60% for Milano)")
    
    # Check vegetation
    forest = stats.get('FOREST', {'percentage': 0})['percentage']
    grassland = stats.get('GRASSLAND', {'percentage': 0})['percentage']
    veg_total = forest + grassland
    print(f"  Vegetation pixels: {veg_total:.1f}% (Forest: {forest:.1f}%, Grassland: {grassland:.1f}%)")
    print(f"                     (expect 20-40% for Milano parks/green areas)")
    
    # Visualize
    print("\n📊 Generating visualization...")
    visualize_results(labels, indices, bands, stats)
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    print("\n💡 Next Steps:")
    print("   1. Visual inspection: Does water appear blue? Vegetation green? Urban red?")
    print("   2. If results look good → proceed to Validation Suite")
    print("   3. If results need tuning → adjust thresholds in SpectralIndicesClassifier")


if __name__ == '__main__':
    main()

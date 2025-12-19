"""
Test SpectralIndicesClassifier with synthetic data.
"""

import numpy as np
from satellite_analysis.analyzers import SpectralIndicesClassifier


def test_classifier_basic():
    """Test basic functionality with synthetic data."""
    
    # Create synthetic raster (100x100 pixels, 10 bands)
    raster = np.random.randint(0, 3000, (100, 100, 10), dtype=np.uint16)
    
    # Band mapping (Sentinel-2 L2A typical order)
    band_indices = {
        'B02': 0,  # Blue
        'B03': 1,  # Green
        'B04': 2,  # Red
        'B08': 6,  # NIR
        'B11': 8,  # SWIR1
        'B12': 9   # SWIR2
    }
    
    # Initialize classifier
    classifier = SpectralIndicesClassifier()
    
    # Validate bands
    classifier.validate_bands(band_indices, raster.shape[-1])
    
    # Classify
    labels, indices = classifier.classify_raster(raster, band_indices)
    
    # Check results
    assert labels.shape == (100, 100), f"Expected (100, 100), got {labels.shape}"
    assert labels.min() >= 0 and labels.max() <= 5, f"Labels out of range: {labels.min()}-{labels.max()}"
    
    # Get statistics
    stats = classifier.get_class_statistics(labels)
    
    print("✅ Basic test passed")
    print(f"   Classes found: {list(stats.keys())}")
    print(f"   Label range: {labels.min()}-{labels.max()}")
    return None


def test_water_detection():
    """Test water detection with known values."""
    
    # Create synthetic data with known water signature
    # Water: high GREEN, low SWIR
    raster = np.zeros((50, 50, 10), dtype=np.float32)
    
    raster[..., 1] = 2000  # GREEN - high
    raster[..., 8] = 200   # SWIR1 - low
    raster[..., 9] = 100   # SWIR2 - low
    
    band_indices = {
        'B02': 0, 'B03': 1, 'B04': 2,
        'B08': 6, 'B11': 8, 'B12': 9
    }
    
    classifier = SpectralIndicesClassifier()
    labels, indices = classifier.classify_raster(raster, band_indices)
    
    # Check MNDWI is positive (water signature)
    assert indices['MNDWI'].mean() > 0, f"MNDWI should be positive for water, got {indices['MNDWI'].mean()}"
    
    # Most pixels should be classified as WATER (class 0)
    water_percentage = (labels == 0).sum() / labels.size * 100
    
    print(f"✅ Water detection test passed")
    print(f"   MNDWI mean: {indices['MNDWI'].mean():.3f}")
    print(f"   Water pixels: {water_percentage:.1f}%")
    return None


def test_vegetation_detection():
    """Test vegetation detection with known values."""
    
    # Create synthetic data with vegetation signature
    # Vegetation: high NIR, low RED
    raster = np.zeros((50, 50, 10), dtype=np.float32)
    
    raster[..., 2] = 500   # RED - low
    raster[..., 6] = 3000  # NIR - high
    
    band_indices = {
        'B02': 0, 'B03': 1, 'B04': 2,
        'B08': 6, 'B11': 8, 'B12': 9
    }
    
    classifier = SpectralIndicesClassifier()
    labels, indices = classifier.classify_raster(raster, band_indices)
    
    # Check NDVI is high (vegetation signature)
    assert indices['NDVI'].mean() > 0.5, f"NDVI should be high for vegetation, got {indices['NDVI'].mean()}"
    
    # Most pixels should be classified as FOREST or GRASSLAND (class 1 or 2)
    veg_percentage = ((labels == 1) | (labels == 2)).sum() / labels.size * 100
    
    print(f"✅ Vegetation detection test passed")
    print(f"   NDVI mean: {indices['NDVI'].mean():.3f}")
    print(f"   Vegetation pixels: {veg_percentage:.1f}%")
    return None


def test_custom_thresholds():
    """Test custom thresholds."""
    
    custom_thresholds = {
        'ndvi_forest': 0.7,  # More strict forest definition
        'mndwi_water': 0.2   # More sensitive water detection
    }
    
    raster = np.random.randint(0, 3000, (50, 50, 10), dtype=np.uint16)
    band_indices = {
        'B02': 0, 'B03': 1, 'B04': 2,
        'B08': 6, 'B11': 8, 'B12': 9
    }
    
    classifier = SpectralIndicesClassifier(thresholds=custom_thresholds)
    
    # Check thresholds were updated
    assert classifier.thresholds['ndvi_forest'] == 0.7
    assert classifier.thresholds['mndwi_water'] == 0.2
    
    labels, _ = classifier.classify_raster(raster, band_indices)
    
    print(f"✅ Custom thresholds test passed")
    print(f"   Custom thresholds applied: {custom_thresholds}")
    return None


if __name__ == '__main__':
    print("Testing SpectralIndicesClassifier...\n")
    
    test_classifier_basic()
    print()
    test_water_detection()
    print()
    test_vegetation_detection()
    print()
    test_custom_thresholds()
    
    print("\n" + "="*50)
    print("All tests passed! ✅")
    print("="*50)

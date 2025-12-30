"""Test workflow components without actual downloads.

Tests area selection, classification, and validation locally.
"""

from pathlib import Path
import numpy as np


def test_area_selector():
    """Test AreaSelector city lookup."""
    from satellite_analysis.utils import AreaSelector
    
    selector = AreaSelector()
    bbox, area_info = selector.select_by_city("Milan", radius_km=15)
    
    assert bbox is not None
    assert len(bbox) == 4
    assert area_info['center'][0] > 45  # Milan latitude
    assert area_info['center'][1] > 9   # Milan longitude
    
    print("✅ AreaSelector works")


def test_consensus_classifier():
    """Test ConsensusClassifier on synthetic data."""
    from satellite_analysis.analyzers.classification import ConsensusClassifier
    
    # Create synthetic 4-band satellite data
    np.random.seed(42)
    h, w = 100, 100
    
    # Simulate bands: B02 (blue), B03 (green), B04 (red), B08 (NIR)
    stack = np.random.randint(0, 4000, (h, w, 4), dtype=np.uint16).astype(np.float32)
    
    # Add some structure: water in corner (low reflectance)
    stack[:20, :20, :] = np.random.randint(100, 500, (20, 20, 4))
    
    # Vegetation patch (high NIR)
    stack[50:70, 50:70, 3] = np.random.randint(3000, 4000, (20, 20))
    
    classifier = ConsensusClassifier(n_clusters=6)
    band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
    labels, confidence, uncertainty, stats = classifier.classify(stack, band_indices)
    
    assert labels.shape == (h, w)
    assert confidence.shape == (h, w)
    assert labels.min() >= 0
    assert labels.max() <= 5
    assert confidence.min() >= 0
    assert confidence.max() <= 1
    
    print(f"✅ ConsensusClassifier works (found {len(np.unique(labels))} classes)")


def test_validation_metrics():
    """Test validation metrics computation."""
    from satellite_analysis.validation import compute_accuracy, compute_kappa, compute_f1_scores
    
    # Create synthetic predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 6, 1000)
    y_pred = y_true.copy()
    
    # Add 20% noise
    noise_mask = np.random.random(1000) < 0.2
    y_pred[noise_mask] = np.random.randint(0, 6, noise_mask.sum())
    
    accuracy = compute_accuracy(y_true, y_pred)
    kappa = compute_kappa(y_true, y_pred)
    f1_score = compute_f1_scores(y_true, y_pred)  # Weighted F1 score
    
    assert 0.7 < accuracy < 0.9  # Should be around 80%
    assert 0.5 < kappa < 0.9     # Good but not perfect agreement
    assert 0.7 < f1_score < 0.9  # Weighted F1 similar to accuracy
    
    print(f"✅ Validation metrics work (accuracy={accuracy:.2%}, kappa={kappa:.3f})")


def test_preprocessing():
    """Test preprocessing utilities."""
    from satellite_analysis.preprocessing import min_max_scale, reshape_image_to_table
    
    # Test min-max scaling
    data = np.array([[0, 100], [50, 200]], dtype=np.float32)
    scaled = min_max_scale(data)
    
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0
    
    # Test reshape
    image = np.random.rand(10, 10, 4)
    table = reshape_image_to_table(image)
    
    assert table.shape == (100, 4)
    
    print("✅ Preprocessing utilities work")


if __name__ == "__main__":
    print("\n🧪 Running component tests...\n")
    
    test_area_selector()
    test_preprocessing()
    test_consensus_classifier()
    test_validation_metrics()
    
    print("\n✅ All component tests passed!\n")

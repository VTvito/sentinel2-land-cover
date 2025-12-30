"""Tests for classifier registry and adapters.

Tests the Strategy pattern implementation for classifiers.
"""

import numpy as np
import pytest


class TestClassifierRegistry:
    """Test get_classifier factory function."""

    def test_get_consensus_classifier(self):
        """Factory returns consensus adapter."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("consensus", n_clusters=6)
        assert clf.name == "consensus"
        assert hasattr(clf, "required_bands")
        assert hasattr(clf, "classify")

    def test_get_kmeans_classifier(self):
        """Factory returns kmeans adapter."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("kmeans", n_clusters=8)
        assert clf.name == "kmeans"

    def test_get_spectral_classifier(self):
        """Factory returns spectral adapter."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("spectral")
        assert clf.name == "spectral"

    def test_invalid_classifier_raises(self):
        """Factory raises on unknown classifier."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        with pytest.raises(ValueError, match="Unknown classifier"):
            get_classifier("invalid")


class TestClassifierProtocol:
    """Test all classifiers implement ClassifierPort protocol."""

    @pytest.fixture(params=["consensus", "kmeans", "spectral"])
    def classifier(self, request):
        """Parametrized fixture for all classifiers."""
        from satellite_analysis.analyzers.classification import get_classifier
        return get_classifier(request.param, n_clusters=6)

    def test_has_name(self, classifier):
        """Classifier has name attribute."""
        assert hasattr(classifier, "name")
        assert isinstance(classifier.name, str)

    def test_has_required_bands(self, classifier):
        """Classifier has required_bands method."""
        bands = classifier.required_bands()
        assert isinstance(bands, list)
        assert len(bands) > 0
        assert all(isinstance(b, str) for b in bands)

    def test_has_classify_method(self, classifier):
        """Classifier has classify method."""
        assert hasattr(classifier, "classify")
        assert callable(classifier.classify)


class TestConsensusAdapter:
    """Test ConsensusClassifierAdapter."""

    def test_required_bands(self):
        """Returns correct bands."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("consensus")
        assert clf.required_bands() == ["B02", "B03", "B04", "B08"]

    def test_classify_output_shape(self, sample_bands_dict):
        """Classify returns correct shapes."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("consensus", n_clusters=6)
        labels, confidence, stats = clf.classify(sample_bands_dict)
        
        expected_shape = sample_bands_dict["B02"].shape
        assert labels.shape == expected_shape
        assert confidence.shape == expected_shape

    def test_classify_label_range(self, sample_bands_dict):
        """Labels are in valid range [0, 5]."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("consensus")
        labels, _, _ = clf.classify(sample_bands_dict)
        
        assert labels.min() >= 0
        assert labels.max() <= 5

    def test_classify_confidence_range(self, sample_bands_dict):
        """Confidence is in [0, 1]."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("consensus")
        _, confidence, _ = clf.classify(sample_bands_dict)
        
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0

    def test_classify_returns_stats(self, sample_bands_dict):
        """Stats dict has required keys."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("consensus")
        _, _, stats = clf.classify(sample_bands_dict)
        
        assert isinstance(stats, dict)
        assert "avg_confidence" in stats
        assert "class_distribution" in stats


class TestKMeansAdapter:
    """Test KMeansClassifierAdapter."""

    def test_required_bands(self):
        """Returns correct bands."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("kmeans")
        assert clf.required_bands() == ["B02", "B03", "B04", "B08"]

    def test_classify_output_shape(self, sample_bands_dict):
        """Classify returns correct shapes."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("kmeans", n_clusters=6)
        labels, confidence, stats = clf.classify(sample_bands_dict)
        
        expected_shape = sample_bands_dict["B02"].shape
        assert labels.shape == expected_shape
        assert confidence.shape == expected_shape

    def test_n_clusters_in_stats(self, sample_bands_dict):
        """Stats includes n_clusters."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("kmeans", n_clusters=8)
        _, _, stats = clf.classify(sample_bands_dict)
        
        assert stats["n_clusters"] == 8


class TestSpectralAdapter:
    """Test SpectralClassifierAdapter."""

    def test_required_bands(self):
        """Returns correct bands (includes SWIR)."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("spectral")
        bands = clf.required_bands()
        
        assert "B11" in bands
        assert "B12" in bands
        assert len(bands) == 6

    def test_missing_swir_raises(self, sample_bands_dict):
        """Raises if SWIR bands missing."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        clf = get_classifier("spectral")
        
        with pytest.raises(FileNotFoundError, match="B11"):
            clf.classify(sample_bands_dict)

    def test_classify_with_swir(self, sample_bands_6, band_indices_6):
        """Classify works with SWIR bands."""
        from satellite_analysis.analyzers.classification import get_classifier
        
        bands_dict = {
            "B02": sample_bands_6[:, :, 0],
            "B03": sample_bands_6[:, :, 1],
            "B04": sample_bands_6[:, :, 2],
            "B08": sample_bands_6[:, :, 3],
            "B11": sample_bands_6[:, :, 4],
            "B12": sample_bands_6[:, :, 5],
        }
        
        clf = get_classifier("spectral")
        labels, confidence, stats = clf.classify(bands_dict)
        
        assert labels.shape == sample_bands_6.shape[:2]
        assert "note" in stats

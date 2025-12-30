"""Tests for api.py facade.

Tests the public API: analyze(), analyze_batch(), exports.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestAnalyzeFunction:
    """Test analyze() facade."""

    @pytest.mark.integration
    def test_analyze_milan(self):
        """Full analysis on Milan data."""
        from satellite_analysis import analyze
        
        result = analyze("Milan")
        
        assert result is not None
        assert hasattr(result, "labels")
        assert hasattr(result, "confidence")
        assert isinstance(result.labels, np.ndarray)

    def test_analyze_invalid_city_raises(self):
        """Raises for unknown city without data."""
        from satellite_analysis import analyze
        
        with pytest.raises((ValueError, FileNotFoundError)):
            analyze("NonExistentCity123")

    @pytest.mark.integration
    def test_analyze_with_classifier(self):
        """Analysis with specific classifier."""
        from satellite_analysis import analyze
        
        result = analyze("Milan", classifier="kmeans")
        
        assert result is not None
        # Classifier choice reflected in config_summary
        assert result.config_summary.get("classifier") == "kmeans"

    @pytest.mark.integration
    def test_analyze_returns_stats(self):
        """Result includes statistics."""
        from satellite_analysis import analyze
        
        result = analyze("Milan")
        
        assert hasattr(result, "classification")
        assert hasattr(result.classification, "statistics")


class TestAnalyzeBatch:
    """Test analyze_batch() for multiple cities."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_batch_single_city(self):
        """Batch with one city."""
        from satellite_analysis import analyze_batch
        
        results = analyze_batch(["Milan"])
        
        assert len(results) == 1
        assert "Milan" in results

    @pytest.mark.slow
    @pytest.mark.integration
    def test_batch_returns_dict(self):
        """Returns dict of results."""
        from satellite_analysis import analyze_batch
        
        results = analyze_batch(["Milan"])
        
        assert isinstance(results, dict)


class TestQuickPreview:
    """Test quick_preview() function."""

    @pytest.mark.integration
    def test_quick_preview_returns_result(self):
        """Preview returns AnalysisResult."""
        from satellite_analysis import quick_preview
        from satellite_analysis.core.types import AnalysisResult
        
        result = quick_preview("Milan")
        
        assert isinstance(result, AnalysisResult)
        assert result.labels is not None


class TestAPIExports:
    """Test export functions from api facade."""

    @pytest.fixture
    def sample_result(self, project_root):
        """Get a real analysis result."""
        from satellite_analysis import analyze
        return analyze("Milan", max_size=500)  # smaller for speed

    @pytest.mark.integration
    def test_export_json(self, sample_result, temp_output_dir):
        """Export to JSON."""
        from satellite_analysis import export_json
        
        output = export_json(sample_result, str(temp_output_dir / "result.json"))
        
        assert output is not None
        assert output.exists()

    @pytest.mark.integration
    def test_export_image(self, sample_result, temp_output_dir):
        """Export to PNG."""
        from satellite_analysis import export_image
        
        output = export_image(sample_result, str(temp_output_dir / "result.png"))
        
        assert output is not None
        assert output.exists()

    @pytest.mark.integration
    def test_export_geotiff(self, sample_result, temp_output_dir):
        """Export to GeoTIFF."""
        from satellite_analysis import export_geotiff
        
        output = export_geotiff(sample_result, str(temp_output_dir / "result.tif"))
        
        assert output is not None

    @pytest.mark.integration
    def test_export_report(self, sample_result, temp_output_dir):
        """Export to HTML."""
        from satellite_analysis import export_report
        
        output = export_report(sample_result, str(temp_output_dir / "result.html"))
        
        assert output is not None


class TestLandCoverClasses:
    """Test LAND_COVER_CLASSES constant."""

    def test_classes_exported(self):
        """LAND_COVER_CLASSES is accessible."""
        from satellite_analysis import LAND_COVER_CLASSES
        
        assert isinstance(LAND_COVER_CLASSES, dict)

    def test_six_classes(self):
        """Contains 6 classes."""
        from satellite_analysis import LAND_COVER_CLASSES
        
        assert len(LAND_COVER_CLASSES) == 6

    def test_class_structure(self):
        """Each class has name, color, rgb."""
        from satellite_analysis import LAND_COVER_CLASSES
        
        for class_id, class_info in LAND_COVER_CLASSES.items():
            assert isinstance(class_id, int)
            assert "name" in class_info
            assert "color" in class_info
            assert "rgb" in class_info

    def test_expected_class_names(self):
        """Contains expected land cover types."""
        from satellite_analysis import LAND_COVER_CLASSES
        
        names = [info["name"] for info in LAND_COVER_CLASSES.values()]
        assert "Water" in names
        assert "Vegetation" in names
        assert "Urban" in names


class TestCompareFunction:
    """Test compare() function."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_compare_requires_dates(self):
        """Compare needs date_before and date_after."""
        from satellite_analysis import compare
        
        # compare() requires date_before and date_after params
        with pytest.raises(TypeError):
            compare("Milan")  # Missing required args

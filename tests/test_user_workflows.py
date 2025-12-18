"""
🧪 User Workflow Tests - Simulates Real User Experience

Tests all documented workflows from README.md as a real user would execute them.
Catches integration bugs, API mismatches, and missing imports.

Run with: pytest tests/test_user_workflows.py -v
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import pytest
import tempfile
import shutil

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestImports:
    """Test all imports work correctly - catches missing __init__.py files."""
    
    def test_core_imports(self):
        """Test core module imports."""
        from satellite_analysis.utils import AreaSelector
        from satellite_analysis.analyzers.classification import ConsensusClassifier
        from satellite_analysis.analyzers.classification import SpectralIndicesClassifier
        from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
        assert True
    
    def test_preprocessing_imports(self):
        """Test preprocessing module imports."""
        from satellite_analysis.preprocessing import min_max_scale
        from satellite_analysis.preprocessing import reshape_image_to_table
        from satellite_analysis.preprocessing import reshape_table_to_image
        assert True
    
    def test_validation_imports(self):
        """Test validation module imports."""
        from satellite_analysis.validation import compute_accuracy
        from satellite_analysis.validation import compute_kappa
        from satellite_analysis.validation import compute_f1_scores
        assert True


class TestConsensusClassifierAPI:
    """Test ConsensusClassifier API matches documentation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create realistic test data."""
        np.random.seed(42)
        h, w = 100, 100
        
        # Simulate 4-band Sentinel-2 data (B02, B03, B04, B08)
        stack = np.random.randint(500, 3000, (h, w, 4), dtype=np.uint16).astype(np.float32)
        
        # Water region (low reflectance across all bands)
        stack[:20, :20, :] = np.random.randint(100, 400, (20, 20, 4))
        
        # Vegetation region (high NIR, moderate visible)
        stack[40:60, 40:60, :3] = np.random.randint(300, 800, (20, 20, 3))  # Low visible
        stack[40:60, 40:60, 3] = np.random.randint(2500, 4000, (20, 20))    # High NIR
        
        # Urban region (similar reflectance, low NIR)
        stack[70:90, 70:90, :] = np.random.randint(1500, 2500, (20, 20, 4))
        stack[70:90, 70:90, 3] = np.random.randint(800, 1500, (20, 20))
        
        return stack
    
    def test_classify_method_exists(self, sample_data):
        """Test that classify() method exists and works."""
        from satellite_analysis.analyzers.classification import ConsensusClassifier
        
        classifier = ConsensusClassifier(n_clusters=6)
        band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
        
        # Must return 4 values: labels, confidence, uncertainty_mask, stats
        result = classifier.classify(sample_data, band_indices, has_swir=False)
        
        assert len(result) == 4, "classify() must return 4 values"
        labels, confidence, uncertainty_mask, stats = result
        
        assert labels.shape == sample_data.shape[:2]
        assert confidence.shape == sample_data.shape[:2]
        assert uncertainty_mask.shape == sample_data.shape[:2]
        assert isinstance(stats, dict)
    
    def test_classify_returns_valid_labels(self, sample_data):
        """Test labels are in valid range 0-5."""
        from satellite_analysis.analyzers.classification import ConsensusClassifier
        
        classifier = ConsensusClassifier(n_clusters=6)
        band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
        
        labels, _, _, _ = classifier.classify(sample_data, band_indices, has_swir=False)
        
        assert labels.min() >= 0, "Labels must be >= 0"
        assert labels.max() <= 5, "Labels must be <= 5"
    
    def test_classify_confidence_range(self, sample_data):
        """Test confidence values are in [0, 1]."""
        from satellite_analysis.analyzers.classification import ConsensusClassifier
        
        classifier = ConsensusClassifier(n_clusters=6)
        band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
        
        _, confidence, _, _ = classifier.classify(sample_data, band_indices, has_swir=False)
        
        assert confidence.min() >= 0.0, "Confidence must be >= 0"
        assert confidence.max() <= 1.0, "Confidence must be <= 1"
    
    def test_stats_contains_required_keys(self, sample_data):
        """Test stats dict has all required keys."""
        from satellite_analysis.analyzers.classification import ConsensusClassifier
        
        classifier = ConsensusClassifier(n_clusters=6)
        band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
        
        _, _, _, stats = classifier.classify(sample_data, band_indices, has_swir=False)
        
        required_keys = ['agreement_pct', 'avg_confidence', 'uncertain_pct', 'class_distribution']
        for key in required_keys:
            assert key in stats, f"Stats must contain '{key}'"
    
    def test_no_fit_predict_method(self, sample_data):
        """Ensure fit_predict() does NOT exist (was a bug in notebook)."""
        from satellite_analysis.analyzers.classification import ConsensusClassifier
        
        classifier = ConsensusClassifier(n_clusters=6)
        assert not hasattr(classifier, 'fit_predict'), \
            "fit_predict() should NOT exist - use classify() instead"


class TestSpectralClassifierAPI:
    """Test SpectralIndicesClassifier API."""
    
    @pytest.fixture
    def sample_data_with_swir(self):
        """Create test data with SWIR bands."""
        np.random.seed(42)
        h, w = 50, 50
        # 6 bands: B02, B03, B04, B08, B11, B12
        return np.random.randint(500, 3000, (h, w, 6), dtype=np.uint16).astype(np.float32)
    
    def test_requires_swir_bands(self, sample_data_with_swir):
        """Test that full classifier requires SWIR bands."""
        from satellite_analysis.analyzers.classification import SpectralIndicesClassifier
        
        classifier = SpectralIndicesClassifier()
        
        # Should fail without SWIR bands
        band_indices_no_swir = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
        
        # The compute_indices method requires B11 and B12
        with pytest.raises(KeyError):
            classifier.compute_indices(sample_data_with_swir[:, :, :4], band_indices_no_swir)
    
    def test_classify_takes_indices_not_raster(self):
        """Test classify() takes pre-computed indices, not raw raster."""
        from satellite_analysis.analyzers.classification import SpectralIndicesClassifier
        import inspect
        
        classifier = SpectralIndicesClassifier()
        sig = inspect.signature(classifier.classify)
        params = list(sig.parameters.keys())
        
        # Should only take 'indices' parameter (plus self)
        assert len(params) == 1, \
            f"classify() should take 1 param (indices), got {params}"
        assert params[0] == 'indices', \
            f"classify() param should be 'indices', got '{params[0]}'"


class TestKMeansClustererAPI:
    """Test KMeansPlusPlusClusterer API."""
    
    def test_fit_predict_workflow(self):
        """Test standard fit-predict workflow."""
        from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
        
        np.random.seed(42)
        data = np.random.rand(1000, 4)
        
        clusterer = KMeansPlusPlusClusterer(n_clusters=6, random_state=42)
        clusterer.fit(data)
        labels = clusterer.predict(data)
        
        assert labels.shape == (1000,)
        assert len(np.unique(labels)) <= 6


class TestCLIHelp:
    """Test CLI scripts load without errors."""
    
    def test_analyze_city_help(self):
        """Test analyze_city.py --help works."""
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "analyze_city.py"), "--help"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "city" in result.stdout.lower()
    
    def test_validate_classification_help(self):
        """Test validate_classification.py --help works."""
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        script = PROJECT_ROOT / "scripts" / "validate_classification.py"
        if script.exists():
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"


class TestAnalyzeCityMethods:
    """Test all analysis methods work with real data."""
    
    @pytest.fixture
    def rome_data_path(self):
        """Path to Rome test data."""
        path = PROJECT_ROOT / "data" / "cities" / "rome" / "bands"
        if not path.exists():
            pytest.skip("Rome test data not available")
        return path
    
    def test_load_bands(self, rome_data_path):
        """Test loading satellite bands."""
        import rasterio
        
        bands = {}
        for band_name in ['B02', 'B03', 'B04', 'B08']:
            band_path = rome_data_path / f"{band_name}.tif"
            assert band_path.exists(), f"Missing {band_name}.tif"
            
            with rasterio.open(band_path) as src:
                bands[band_name] = src.read(1).astype(np.float32)
        
        # All bands should have same shape
        shapes = [b.shape for b in bands.values()]
        assert len(set(shapes)) == 1, "All bands must have same shape"
    
    def test_consensus_on_real_data(self, rome_data_path):
        """Test ConsensusClassifier on real Rome data."""
        import rasterio
        from satellite_analysis.analyzers.classification import ConsensusClassifier
        
        # Load bands
        bands = {}
        for band_name in ['B02', 'B03', 'B04', 'B08']:
            with rasterio.open(rome_data_path / f"{band_name}.tif") as src:
                bands[band_name] = src.read(1).astype(np.float32)
        
        stack = np.stack([bands['B02'], bands['B03'], bands['B04'], bands['B08']], axis=-1)
        band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
        
        classifier = ConsensusClassifier(n_clusters=6)
        labels, confidence, uncertainty, stats = classifier.classify(stack, band_indices, has_swir=False)
        
        assert labels.shape == stack.shape[:2]
        assert stats['agreement_pct'] > 0


class TestNotebookAPI:
    """Test that notebook code matches actual API."""
    
    def test_notebook_classification_code(self):
        """Verify notebook uses correct API."""
        notebook_path = PROJECT_ROOT / "notebooks" / "city_analysis.ipynb"
        
        if not notebook_path.exists():
            pytest.skip("Notebook not found")
        
        import json
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Find classification cell
        found_correct_api = False
        found_wrong_api = False
        
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                if 'classifier.classify(' in source:
                    found_correct_api = True
                
                if 'classifier.fit_predict(' in source:
                    found_wrong_api = True
        
        assert not found_wrong_api, "Notebook still uses deprecated fit_predict() method"
        assert found_correct_api, "Notebook should use classify() method"


class TestDataOrganization:
    """Test data output is properly organized."""
    
    def test_output_structure(self):
        """Verify expected output structure exists."""
        rome_path = PROJECT_ROOT / "data" / "cities" / "rome"
        
        if not rome_path.exists():
            pytest.skip("Rome data not available")
        
        expected_structure = [
            "bands/B02.tif",
            "bands/B03.tif", 
            "bands/B04.tif",
            "bands/B08.tif",
        ]
        
        for path in expected_structure:
            full_path = rome_path / path
            assert full_path.exists(), f"Missing: {path}"


class TestValidationMetrics:
    """Test validation metrics are computed correctly."""
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        from satellite_analysis.validation import compute_accuracy
        
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        
        assert compute_accuracy(y_true, y_pred) == 1.0
    
    def test_accuracy_zero(self):
        """Test accuracy with all wrong predictions."""
        from satellite_analysis.validation import compute_accuracy
        
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        
        assert compute_accuracy(y_true, y_pred) == 0.0
    
    def test_kappa_range(self):
        """Test kappa is in valid range [-1, 1]."""
        from satellite_analysis.validation import compute_kappa
        
        np.random.seed(42)
        y_true = np.random.randint(0, 6, 100)
        y_pred = np.random.randint(0, 6, 100)
        
        kappa = compute_kappa(y_true, y_pred)
        assert -1 <= kappa <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Shared fixtures and configuration for all tests."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests requiring real data")
    config.addinivalue_line("markers", "network: marks tests requiring network access")


# =============================================================================
# Synthetic data fixtures
# =============================================================================

@pytest.fixture
def sample_bands_4():
    """4-band synthetic satellite data (B02, B03, B04, B08)."""
    np.random.seed(42)
    h, w = 100, 100
    
    stack = np.random.randint(500, 3000, (h, w, 4), dtype=np.uint16).astype(np.float32)
    
    # Water region (low reflectance)
    stack[:20, :20, :] = np.random.randint(100, 400, (20, 20, 4))
    
    # Vegetation region (high NIR)
    stack[40:60, 40:60, :3] = np.random.randint(300, 800, (20, 20, 3))
    stack[40:60, 40:60, 3] = np.random.randint(2500, 4000, (20, 20))
    
    # Urban region (similar reflectance, low NIR)
    stack[70:90, 70:90, :] = np.random.randint(1500, 2500, (20, 20, 4))
    stack[70:90, 70:90, 3] = np.random.randint(800, 1500, (20, 20))
    
    return stack


@pytest.fixture
def sample_bands_dict(sample_bands_4):
    """4-band data as dict (for registry adapters)."""
    return {
        "B02": sample_bands_4[:, :, 0],
        "B03": sample_bands_4[:, :, 1],
        "B04": sample_bands_4[:, :, 2],
        "B08": sample_bands_4[:, :, 3],
    }


@pytest.fixture
def sample_bands_6():
    """6-band synthetic data with SWIR (B02, B03, B04, B08, B11, B12)."""
    np.random.seed(42)
    h, w = 50, 50
    return np.random.randint(500, 3000, (h, w, 6), dtype=np.uint16).astype(np.float32)


@pytest.fixture
def band_indices_4():
    """Standard 4-band indices mapping."""
    return {"B02": 0, "B03": 1, "B04": 2, "B08": 3}


@pytest.fixture
def band_indices_6():
    """Standard 6-band indices mapping (with SWIR)."""
    return {"B02": 0, "B03": 1, "B04": 2, "B08": 3, "B11": 4, "B12": 5}


# =============================================================================
# Path fixtures
# =============================================================================

@pytest.fixture
def project_root():
    """Project root path."""
    return PROJECT_ROOT


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output

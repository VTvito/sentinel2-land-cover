"""Tests for ProjectPaths utility.

Tests centralized path resolution from project root.
"""

from pathlib import Path

import pytest


class TestProjectPathsClass:
    """Test ProjectPaths class initialization and resolution."""

    def test_init_creates_instance(self):
        """ProjectPaths initializes correctly."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        
        paths = ProjectPaths()
        assert paths is not None

    def test_project_root_is_absolute(self):
        """Project root is an absolute path."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        
        paths = ProjectPaths()
        assert paths.project_root.is_absolute()

    def test_project_root_exists(self):
        """Project root directory exists."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        
        paths = ProjectPaths()
        assert paths.project_root.exists()


class TestPathResolution:
    """Test resolve() method."""

    @pytest.fixture
    def paths(self):
        """ProjectPaths instance."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        return ProjectPaths()

    def test_resolve_simple_path(self, paths):
        """Resolves simple relative path."""
        resolved = paths.resolve("data")
        
        assert resolved.is_absolute()
        assert resolved.name == "data"

    def test_resolve_nested_path(self, paths):
        """Resolves nested path."""
        resolved = paths.resolve("data", "cities")
        
        assert "data" in resolved.parts
        assert "cities" in resolved.parts

    def test_resolve_is_consistent(self, paths):
        """Same input gives same output."""
        path1 = paths.resolve("data", "cities", "milan")
        path2 = paths.resolve("data", "cities", "milan")
        
        assert path1 == path2


class TestConvenienceMethods:
    """Test convenience methods for common paths."""

    @pytest.fixture
    def paths(self):
        """ProjectPaths instance."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        return ProjectPaths()

    def test_data_method(self, paths):
        """data() method works."""
        data_dir = paths.data()
        
        assert data_dir.name == "data"
        assert data_dir.is_absolute()

    def test_config_method(self, paths):
        """config() method works."""
        config_path = paths.config()
        
        assert "config" in str(config_path)
        assert config_path.is_absolute()

    def test_data_with_parts(self, paths):
        """data() with nested parts works."""
        cities_dir = paths.data("cities")
        
        assert cities_dir.name == "cities"
        assert "data" in cities_dir.parts


class TestCityPathMethods:
    """Test city-specific path methods."""

    @pytest.fixture
    def paths(self):
        """ProjectPaths instance."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        return ProjectPaths()

    def test_city_method(self, paths):
        """city() method returns city directory."""
        city_dir = paths.city("milan")
        
        assert "milan" in str(city_dir).lower()
        assert city_dir.is_absolute()

    def test_city_with_subpath(self, paths):
        """city() with subpath returns bands directory."""
        bands_dir = paths.city("milan", "bands")
        
        assert "bands" in str(bands_dir)
        assert bands_dir.is_absolute()

    def test_latest_method(self, paths):
        """latest() returns latest directory."""
        latest_dir = paths.latest("milan")
        
        assert "latest" in str(latest_dir)
        assert latest_dir.is_absolute()


class TestRunsDirectory:
    """Test runs directory management."""

    @pytest.fixture
    def paths(self):
        """ProjectPaths instance."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        return ProjectPaths()

    def test_runs_base(self, paths):
        """runs_base() returns runs directory."""
        runs_dir = paths.runs_base("milan")
        
        assert "runs" in str(runs_dir)


class TestPathAvoidance:
    """Test we avoid relative path patterns."""

    def test_no_relative_data_path(self):
        """Never use relative paths like Path('data/...')."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        
        paths = ProjectPaths()
        data_dir = paths.data()
        
        # Should NOT start with just 'data'
        assert data_dir.is_absolute()
        assert data_dir.parts[0] != "data"

    def test_no_hardcoded_city_path(self):
        """Never use hardcoded paths like Path('data/cities/milan')."""
        from satellite_analysis.utils.project_paths import ProjectPaths
        
        paths = ProjectPaths()
        city_dir = paths.city("milan")
        
        # Path should be absolute and resolved
        assert city_dir.is_absolute()
        # Should contain proper path segments
        assert len(city_dir.parts) > 3

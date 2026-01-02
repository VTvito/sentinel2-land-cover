# Changelog

## [2.3.0] - 2025-12-30

### Added
- **`raw_clusters` mode**: Keep distinct cluster IDs (0 to N-1) without semantic mapping. Essential for KMeans visualization where clusters shouldn't collapse to semantic classes.
- **`export_rgb()` function**: Generate publication-quality RGB True Color, False Color Composite (NIR-R-G), and NDVI visualizations at 300 DPI.
- **RGB GeoTIFF export**: Georeferenced RGB True Color for GIS integration.
- **Manual product selection**: Notebook workflow now requires explicit product selection (Cells 4-5) for better data quality control.

### Changed
- **Improved color palette**: High-contrast colors for better visualization. Urban changed from gray (#808080) to crimson (#DC143C).
- **Notebook updated**: Configuration cell now includes `RAW_CLUSTERS` parameter; visualization cell auto-detects mode and uses distinct palettes.
- **Auto-download disabled**: `auto_download=False` by default in notebook workflow. Users must explicitly select products via Cell 5.

### Fixed
- **KMeans cluster collapse bug**: Fixed issue where 6 clusters would collapse to 3 semantic classes due to NDVI/NDWI threshold mapping. Use `raw_clusters=True` to preserve all clusters.

---

## [2.2.0] - 2025-12-30

### Added
- **Classifier registry**: Strategy pattern with pluggable adapters (`registry.py`).
- **Ports/contracts**: `ClassifierPort`, `AreaSelectorPort`, `OutputManagerPort` in `core/ports.py`.
- **ProjectPaths**: Centralized path resolution from project root (`utils/project_paths.py`).
- **Pipeline options**: `resample_method` (bilinear/cubic), `crop_fail_raises` (strict cropping).
- **py.typed marker**: PEP 561 typed package support.
- **PNG image export**: `export_image(result)` for shareable summaries.
- **Test infrastructure**: `conftest.py` with shared fixtures, pytest markers (slow, integration, network).
- **New test suites**: `test_registry.py`, `test_api.py`, `test_paths.py` covering new architecture.

### Changed
- Version synced across `__init__.py` and `pyproject.toml`.
- Documentation trimmed (~60% reduction) for lower maintenance overhead.
- Legacy `Config` (yaml-based) clearly marked; prefer `AnalysisConfig` for runtime.
- Test count: **85 tests** (71 fast unit tests + 14 integration tests).

### Fixed
- Removed TODO comments; clarified date-specific data notes.
- Exports `Classifier`, `ProjectPaths` from public API.

## [2.1.0] - 2025-12-19

### Added
- **Batch processing**: `analyze_batch(["Milan", "Rome", "Florence"])`
- **Change detection**: `compare("Milan", "2023-06", "2024-06")`
- **Export functions**: `export_geotiff()`, `export_report()`, `export_json()`
- **REST API**: FastAPI server at `scripts/api_server.py`
- **CLI enhancements**: `--cities`, `--export`, `--compare`, `--lang` flags
- **i18n**: HTML reports in English and Italian

### Changed
- CLI now supports multi-city batch operations
- All exports save to analysis output directory

---

## [2.0.1] - 2025-12-19

### Added
- **Simple API**: `analyze("city")` as main entry point
- **AnalysisResult** with convenience methods: `summary()`, `class_distribution()`
- **Path resolution**: Works from any directory (notebooks, CLI, scripts)

### Fixed
- FileNotFoundError when running from notebooks

---

## [2.0.0] - 2025-12-19

### Added
- **CompletePipeline**: One-line analysis with automatic download
- **Smart downsampling**: Handles 10980x10980 → configurable size
- **AnalysisResult dataclass**: Structured output

### Changed
- `analyze_city.py` rewritten (693 → 140 lines)
- Removed obsolete scripts

---

## [1.0.0] - 2025-12-18

### Added
- **ConsensusClassifier**: K-Means + Spectral combined
- **Validation suite**: Accuracy, Kappa, F1-score metrics
- **SCL validator**: ESA Scene Classification validation
- **Web UI**: Streamlit app

---

## Earlier Versions

See git history for v0.9.0 (K-Means), v0.3.0 (Spectral), v0.2.0 (Preprocessing).

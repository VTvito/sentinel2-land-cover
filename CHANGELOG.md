# Changelog

## [2.2.0] - 2025-12-19

### Added
- **PNG image export**: `export_image(result)` for a shareable summary.
- **Classifier selection** end-to-end: `classifier="consensus"|"kmeans"|"spectral"` (with band checks).
- **Maintainer docs**: architecture + maintenance guides in `docs/`.
- **GitHub community scaffolding**: issue templates, PR template, CONTRIBUTING, SECURITY.
- **CI smoke test**: quick offline analyze run after pytest.

### Changed
- Notebook is the recommended "golden path" in docs.
- Output provenance now records key run parameters (classifier, bands, dates/cloud cover where applicable).

### Fixed
- Path resolution and `.gitignore` hygiene (avoid ignoring source code; ignore local caches).
- Tests emit no pytest-return warnings; avoids divide-by-zero warning in consensus heuristics.

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

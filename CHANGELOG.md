# 📜 CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2025-12-18 - Demo Mode & Improved UX 🎯

### Added

#### 🎮 Demo Mode (NEW)
- **`--demo` flag** for instant testing without satellite data
  - Sample GeoTIFF bands included in `data/demo/milan_sample/`
  - Run `python scripts/analyze_city.py --demo` to try immediately
  - No Copernicus credentials required

#### 🧙 Setup Wizard (NEW)
- **`scripts/setup.py`** interactive configuration wizard
  - Dependency checking
  - Copernicus credentials configuration
  - First-time user onboarding

#### 📖 Quick Start Guide (NEW)
- **`QUICKSTART.md`** - 2-minute getting started guide
  - Demo mode instructions
  - Common commands reference table

### Changed

#### 🎨 UX Improvements
- `-v/--verbose` flag for progress output with tqdm
- `--city` now optional when using `--demo` mode
- Improved error messages with actionable suggestions

---

## [1.0.1] - 2025-12-18 - Bug Fixes 🐛

### Fixed
- Notebook using deprecated `fit_predict()` instead of `classify()`
- Spectral method failing without SWIR bands (now uses simplified RGB+NIR)
- CLI encoding issues on Windows

### Added
- **OutputManager** for timestamped, organized results storage
- Results now saved to `runs/<timestamp>_<method>/` with metadata
- `run_info.json` with parameters, duration, and statistics
- `latest/` folder pointing to most recent run

### Changed
- Updated README with new output structure

### Testing
- Added comprehensive `test_user_workflows.py` (20 new tests)
- All 31 tests now passing

---

## [1.0.0] - 2025-12-18 - Production Release 🚀

### 🎉 First Stable Release!

This is the first production-ready release of the Satellite Analysis Toolkit, featuring complete land cover classification with consensus logic and validation suite.

### Added

#### 🔮 Consensus Classifier (NEW - Priority P1)
- **ConsensusClassifier** combining K-Means + Spectral classification
  - Pixel-level agreement computation
  - Confidence scoring (0.0 = disagree, 1.0 = full agreement)
  - Uncertainty flagging for manual review
  - Automatic cluster-to-class mapping
  - File: `src/satellite_analysis/analyzers/classification/consensus_classifier.py`

- **Unified 6-Class Scheme**:
  - 0: WATER
  - 1: VEGETATION (merged forest + grassland)
  - 2: BARE_SOIL
  - 3: URBAN
  - 4: BRIGHT_SURFACES
  - 5: SHADOWS_MIXED

#### 🔍 Validation Suite (NEW - Priority P0)
- **Validation Metrics Module**
  - Overall Accuracy (OA)
  - Cohen's Kappa coefficient with interpretation
  - F1-score (per-class, weighted, macro)
  - Producer's and User's accuracy
  - Comprehensive ValidationReport class
  - File: `src/satellite_analysis/validation/metrics.py`

- **Confusion Matrix Visualization**
  - Normalized and raw confusion matrices
  - Classification comparison plots
  - Confidence map visualization
  - Consensus analysis multi-panel plots
  - File: `src/satellite_analysis/validation/confusion_matrix.py`

- **ESA SCL Validator**
  - Mapping from ESA Scene Classification Layer to our classes
  - Cloud/shadow pixel filtering
  - SCL statistics computation
  - Full validation pipeline
  - File: `src/satellite_analysis/validation/scl_validator.py`

#### 📊 New Scripts
- **validate_classification.py**: Complete validation workflow
  ```bash
  python scripts/validate_classification.py --city Milan --method consensus --report
  ```
  
- **Enhanced analyze_city.py**: Now supports consensus method
  ```bash
  python scripts/analyze_city.py --city Milan --method consensus
  ```

### Changed

#### 🎯 Default Analysis Method
- Changed default method from `kmeans` to `consensus` in `analyze_city.py`
- Consensus classification is now the recommended approach

#### 📁 New Validation Output Structure
```
data/cities/<city>/validation/
├── consensus_analysis.png    # 4-panel comparison
├── confidence_map.png        # Confidence heatmap
├── confusion_matrix.png      # Normalized confusion matrix
└── validation_report.txt     # Full text report
```

#### 📝 Documentation
- Updated `AI_AGENT_INSTRUCTIONS.md` for v1.0.0
- Added comprehensive docstrings to all new modules
- Updated QUICKSTART guide (pending)

### Technical Details

#### Consensus Classification Algorithm
```
1. Run K-Means++ clustering (6 clusters, 2M sample training)
2. Run Spectral Indices classification (simplified RGB+NIR)
3. Learn cluster→class mapping (most common spectral class per cluster)
4. Compute pixel-level agreement
5. Generate confidence map:
   - 1.0: Both methods agree
   - 0.5: Same category (natural/built), different class
   - 0.0: Complete disagreement
6. Flag uncertain pixels (confidence < 0.5)
```

#### Validation Metrics Interpretation
```
Kappa Score:
  < 0.00: Less than chance agreement
  0.01–0.20: Slight agreement
  0.21–0.40: Fair agreement
  0.41–0.60: Moderate agreement
  0.61–0.80: Substantial agreement
  0.81–1.00: Almost perfect agreement

Target for v1.0.0:
  - Overall Accuracy: > 75%
  - Kappa: > 0.65 (substantial agreement)
```

#### Performance (Milano Centro)
```
Consensus Classification:
  - Total time: ~45 seconds
  - Memory peak: ~2.1 GB
  - Agreement: ~70-80% (K-Means vs Spectral)
  - Average confidence: 0.7-0.9
  - Uncertain pixels: 10-20%
```

### Breaking Changes
- None (backward compatible with v0.9.0)

### Removed (Cleanup)

#### 📄 Obsolete Documentation
- `OPTIMIZATION_REPORT.md` - Consolidated in README
- `PREPROCESSING_REPORT.md` - Consolidated in README
- `AREA_SELECTION_REPORT.md` - Consolidated in README
- `QUICK_PREVIEW_REPORT.md` - Consolidated in README
- `CITY_CROPPING_METHODOLOGY.md` - Consolidated in README

#### 📜 Legacy Scripts
- `kmeans_milano_optimized.py` - Superseded by analyze_city.py
- `test_classifier_milano.py` - Superseded by analyze_city.py
- `classify_land_cover.py` - Superseded by analyze_city.py
- `select_area.py` - Functionality in analyze_city.py
- `extract_swir_bands.py` - Rarely used
- `quick_rgb_preview.py` - Functionality in analyze_city.py

### Added (New)

#### 🌐 Web UI
- **app.py**: Streamlit web interface for interactive analysis
  ```bash
  streamlit run scripts/app.py
  ```

#### 📓 Jupyter Notebooks
- `notebooks/city_analysis.ipynb` - Complete analysis tutorial
- `notebooks/clustering_example.ipynb` - K-Means tutorial
- `notebooks/download_example.ipynb` - Download API guide

### Migration Guide
No migration needed. Existing scripts will continue to work.
To use new features:
```bash
# Use consensus classification (recommended)
python scripts/analyze_city.py --city Milan --method consensus

# Validate results
python scripts/validate_classification.py --city Milan --report
```

---

## [0.9.0] - 2025-10-27 - K-Means Clustering Release ✨

### Added

#### 🎯 K-Means Clustering (Complete Implementation)
- **K-Means algorithm** from scratch with memory optimization
  - Chunked distance calculation (10K samples per chunk)
  - Avoids memory overflow on large datasets (120M+ pixels)
  - File: `src/satellite_analysis/analyzers/clustering/kmeans.py`

- **K-Means++ algorithm** with smart initialization
  - Improved centroid selection for faster convergence
  - Chunked computation for memory efficiency
  - File: `src/satellite_analysis/analyzers/clustering/kmeans_plus_plus.py`

- **Sklearn wrapper** for comparison and validation
  - File: `src/satellite_analysis/analyzers/clustering/sklearn_kmeans.py`

- **Factory pattern** for easy algorithm selection
  - File: `src/satellite_analysis/analyzers/clustering/base.py`

#### 🗺️ City Cropping Methodology (Structural Workflow)
- **Structural cropping script** for any city
  - Automatic city center detection via `AreaSelector`
  - Tile coverage verification (checks if city is in tile)
  - Cropping with proper CRS transformation
  - RGB preview generation for visual verification
  - File: `scripts/crop_city_area.py`
  - **Usage**: `python scripts/crop_city_area.py --city "Milan" --radius 15 --preview`

- **Complete methodology documentation**
  - Step-by-step workflow for any city
  - Troubleshooting guide for common issues
  - Performance comparison (full tile vs cropped)
  - File: `CITY_CROPPING_METHODOLOGY.md`

#### 🚀 Optimized K-Means Workflow
- **Optimized Milano clustering script**
  - Smart sampling strategy: train on 2M, predict on all
  - Elbow method with 500K sample (fast K optimization)
  - Support for both .jp2 (original) and .tif (cropped) files
  - Full resolution output with statistics
  - File: `scripts/kmeans_milano_optimized.py`

#### 🛠️ Preprocessing Utilities
- **Normalization module**
  - Min-max scaling from original notebook
  - Per-band independent scaling
  - File: `src/satellite_analysis/preprocessing/normalization.py`

- **Reshape utilities**
  - Image ↔ table conversions for clustering
  - `reshape_image_to_table()`: (H,W,C) → (N,C)
  - `reshape_table_to_image()`: (N,) → (H,W)
  - File: `src/satellite_analysis/preprocessing/reshape.py`

#### 📊 Band Extraction Scripts
- **Extract all bands script**
  - Extracts 10m bands (B02, B03, B04, B08) from ZIP
  - Supports both .jp2 and .tif output
  - File: `scripts/extract_all_bands.py`

### Changed

#### ⚡ Performance Improvements
- **Memory optimization**:
  - K-Means: Chunked distance calculation prevents OOM on large datasets
  - Reduction: ~25GB RAM → ~2GB RAM for 120M pixels

- **Speed optimization**:
  - K-Means: Train on sample (2M pixels), predict on all (8.9M)
  - Speedup: ~10x faster (5 minutes → 30 seconds)

- **Spatial optimization**:
  - City cropping: 92.6% size reduction
  - From: 10980×10980 = 120M pixels
  - To: 3006×2980 = 8.9M pixels (Milano 15km radius)

#### 📝 Documentation Updates
- Updated `README.md` with K-Means features and city cropping workflow
- Updated `private_docs/GAP_ANALYSIS.md` with completion status
- Added performance benchmarks for K-Means operations

### Fixed
- **Wrong area issue**: Created structural methodology to ensure correct city center
  - Problem: Images showed lakes/countryside instead of city
  - Solution: City cropping with verification step (preview mandatory)
  - Documented in: `CITY_CROPPING_METHODOLOGY.md`

- **Memory issues**: K-Means OOM on large datasets
  - Problem: 120M pixels × 6 clusters = 720M distance calculations
  - Solution: Chunked processing (10K samples per iteration)

- **Speed issues**: K-Means taking hours on full tile
  - Problem: Elbow method + clustering on 120M pixels
  - Solution: Smart sampling + prediction strategy (10x faster)

### Technical Details

#### Test Results (Milano Centro)
```
Area: 3006×2980 pixels (8.9M pixels, 15km radius)
K-Means Performance:
  - Elbow method: ~10 seconds (K=3-8, 500K sample)
  - Training: ~5 seconds (K=6, 2M sample)
  - Prediction: ~25 seconds (8.9M pixels)
  - Total: ~40 seconds (vs 5+ minutes on full tile)

Cluster Distribution:
  - Cluster 1: 29.3% (urban/vegetation mix)
  - Cluster 2: 24.9% (secondary urban)
  - Cluster 0: 17.0% (tertiary)
  - Cluster 5: 14.5% (parks/green areas)
  - Cluster 3: 12.3% (mixed areas)
  - Cluster 4: 2.0% (water/shadows)

Output Files:
  - data/processed/milano_centro/clustering/kmeans_result.png
  - data/processed/milano_centro/clustering/elbow_curve.png
  - data/processed/milano_centro/clustering/kmeans_labels_full.npy
  - data/processed/milano_centro/clustering/kmeans_centroids.npy
```

#### Structural Workflow
```bash
# Complete workflow for any city:
1. Download full tile → data/raw/product_1.zip
2. Extract bands → data/processed/product_1/*.jp2
3. Crop to city → data/processed/<city>_centro/*.tif (92.6% reduction)
4. Verify preview → <city>_centro/rgb_preview.png (MANDATORY)
5. Run K-Means → <city>_centro/clustering/*.png + *.npy
```

---

## [0.3.0] - 2024-10-17 - Spectral Classification Release

### Added
- SpectralIndicesClassifier with 6-class decision tree
- NDVI, MNDWI, NDBI, BSI spectral indices
- Automatic preprocessing and validation

### Changed
- Updated area selection with verified coordinates
- Improved preprocessing pipeline

---

## [0.2.0] - 2024-10-16 - Preprocessing Pipeline

### Added
- Automatic cropping with CRS transformation
- Band extraction and resampling
- RGB/FCC/NDVI composite generation
- PreprocessingPipeline class

---

## [0.1.0] - 2024-10-15 - Initial Release

### Added
- Area selection by city or coordinates
- OAuth2 authentication for Copernicus Data Space
- STAC catalog search
- Product download with progress bar
- Basic band extraction

---

## Version Numbering

- **MAJOR** version: Incompatible API changes
- **MINOR** version: New features (backward compatible)
- **PATCH** version: Bug fixes (backward compatible)

Current: **0.9.0** - Major feature addition (K-Means clustering)
Next: **1.0.0** - Production release with consensus logic + validation

---

## Upcoming Features (Roadmap)

### [0.10.0] - Consensus Logic (Planned)
- ConsensusClassifier combining Spectral + K-Means
- Agreement computation and confidence scoring
- Uncertainty flagging for manual review

### [1.0.0] - Production Release (Planned)
- Complete validation suite vs ESA Scene Classification
- Classification pipeline with unified interface
- End-to-end CLI automation
- Performance optimization and polish

# 📜 CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

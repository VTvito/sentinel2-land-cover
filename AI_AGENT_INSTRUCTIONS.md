# 🤖 AI Agent Instructions - Satellite Analysis Project

**Project**: Sentinel-2 Satellite Imagery Analysis Toolkit  
**Version**: 0.9.0 (K-Means Clustering Release)  
**Last Updated**: December 18, 2025  
**Status**: Active Development - Ready for v1.0.0

---

## 📋 Project Overview

This is a **professional toolkit** for processing and analyzing Sentinel-2 satellite imagery, specifically focused on **urban area classification**. The project provides:

- **Download**: Sentinel-2 tile download from Copernicus Data Space
- **Preprocessing**: Band extraction, city cropping, RGB visualization
- **Analysis**: K-Means clustering + Spectral indices classification
- **One-Command Workflow**: Simplified UX with `analyze_city.py`

**Key Achievement**: Successfully optimized K-Means from 5+ minutes → 40 seconds through:
- Memory optimization (chunked processing: 25GB → 2GB RAM)
- Speed optimization (smart sampling: train on 2M, predict on all)
- Spatial optimization (city cropping: 92.6% size reduction)

---

## 🎯 Current Status (v0.9.0)

### ✅ Completed Features

**Core Analysis** (Priority P1 - DONE):
- ✅ K-Means clustering (custom implementation with memory optimization)
- ✅ K-Means++ initialization (smart centroid selection)
- ✅ Sklearn wrapper (for comparison)
- ✅ Spectral Indices Classifier (NDVI, MNDWI, NDBI, BSI)
- ✅ City cropping methodology (structural workflow)
- ✅ One-command analysis script (`analyze_city.py`)

**Preprocessing** (DONE):
- ✅ Band extraction from ZIP
- ✅ City area cropping with CRS transformation
- ✅ RGB True Color visualization with histogram equalization
- ✅ Min-max normalization (per-band scaling)
- ✅ Image ↔ table reshaping for ML

**Infrastructure** (DONE):
- ✅ AreaSelector (city coordinates database)
- ✅ Factory pattern for clustering algorithms
- ✅ Comprehensive documentation (README, QUICKSTART, CHANGELOG)

### 🔴 Pending Features (Roadmap to v1.0.0)

**Priority P0 - Validation Suite** (CRITICAL):
- ❌ Validation against ESA Scene Classification Layer (SCL)
- ❌ Accuracy metrics (Overall Accuracy, Kappa, F1-score)
- ❌ Confusion matrix visualization
- ❌ Per-class accuracy analysis

**Priority P1 - Consensus Logic** (NEXT):
- ❌ ConsensusClassifier (combines K-Means + Spectral)
- ❌ Agreement computation (pixel-level consensus)
- ❌ Confidence scoring (0-1 scale)
- ❌ Uncertainty flagging (for manual review)

**Priority P2 - Automation**:
- ❌ End-to-end CLI (download → analyze → validate)
- ❌ Batch processing for multiple cities
- ❌ Report generation (PDF with maps + statistics)

**Priority P3 - Advanced Features**:
- ❌ Temporal analysis (multi-date comparison)
- ❌ Change detection
- ❌ Advanced classifiers (Random Forest, SVM)

---

## 📁 Project Structure

### Critical Files (Must Understand)

```
satellite_git/
├── scripts/
│   ├── analyze_city.py           # 🎯 ONE-COMMAND analysis (NEW - v0.9.0)
│   ├── kmeans_milano_optimized.py # K-Means workflow (optimized)
│   ├── crop_city_area.py          # City cropping utility
│   ├── test_classifier_milano.py  # Spectral classification test
│   └── extract_all_bands.py       # Band extraction from ZIP
│
├── src/satellite_analysis/
│   ├── analyzers/
│   │   ├── clustering/
│   │   │   ├── base.py            # Abstract base + Factory
│   │   │   ├── kmeans.py          # Custom K-Means (memory optimized)
│   │   │   ├── kmeans_plus_plus.py # K-Means++ initialization
│   │   │   └── sklearn_kmeans.py  # Sklearn wrapper
│   │   └── classification/
│   │       └── spectral_classifier.py # Spectral indices (6 classes)
│   │
│   ├── preprocessing/
│   │   ├── normalization.py       # min_max_scale (per-band)
│   │   └── reshape.py             # image ↔ table conversion
│   │
│   ├── utils/
│   │   └── area_selector.py       # City coordinates database
│   │
│   ├── downloaders/               # Sentinel-2 download (OAuth2)
│   ├── preprocessors/             # Band extraction, cropping
│   └── pipelines/                 # High-level workflows
│
├── data/
│   ├── raw/                       # Downloaded ZIPs (ignored in git)
│   ├── processed/                 # Extracted bands (ignored)
│   │   └── milano_centro/         # Cropped Milano area
│   │       ├── B02.tif, B03.tif, B04.tif, B08.tif
│   │       ├── rgb_preview.png    # Visual verification
│   │       └── clustering/        # K-Means outputs
│   └── cities/                    # Organized by city (NEW structure)
│
├── config/
│   ├── config.yaml                # Sentinel Hub credentials
│   └── area_cache.json            # Cached city coordinates
│
├── private_docs/
│   └── GAP_ANALYSIS.md            # Internal roadmap + status
│
├── README.md                      # Project overview
├── QUICKSTART.md                  # 5-minute getting started
├── CHANGELOG.md                   # Version history (v0.9.0)
└── CITY_CROPPING_METHODOLOGY.md   # Structural cropping guide
```

### Ignored in Git (.gitignore)

```
data/raw/          # Large ZIPs (~1.2GB each)
data/processed/    # Extracted bands
**/clustering/     # Output files (.png, .npy)
*.zip, *.tif, *.jp2
private_docs/      # Internal documentation
```

---

## 🛠️ Key Technologies

### Core Stack
- **Python 3.10+** (required)
- **rasterio 1.3.9** - Geospatial raster I/O, CRS transformation
- **numpy 1.26.0** - Numerical computing, array operations
- **scikit-learn 1.3.0** - Sklearn K-Means for comparison
- **matplotlib 3.8.0** - Visualization, RGB composites
- **shapely 2.0.2** - Geometric operations for bbox
- **geopy 2.4.0** - Geocoding, city coordinate lookup

### Specialized Libraries
- **pyproj** - CRS transformations (WGS84 ↔ UTM)
- **PIL (Pillow)** - Image processing, histogram equalization
- **requests-oauthlib** - OAuth2 for Copernicus Data Space

### Data Format
- **Input**: Sentinel-2 L2A products (ZIP format, ~1.2GB)
- **Bands**: B02, B03, B04, B08 (10m resolution, uint16)
- **Output**: .tif (GeoTIFF), .png (visualization), .npy (labels/centroids)

---

## 🧪 Testing & Validation

### Current Test Data
- **City**: Milano (Milan, Italy)
- **Coordinates**: 45.464°N, 9.190°E
- **Radius**: 15 km
- **Area**: 3006×2980 pixels = 8.9M pixels
- **Tile**: T32TNR (Sentinel-2 tile ID)
- **Date**: 2024-06-01 to 2024-08-31 (summer, low cloud cover)

### Performance Benchmarks (Milano Centro)
```
K-Means Clustering (v0.9.0):
  - Elbow method: ~10 seconds (K=3-8, 500K sample)
  - Training: ~5 seconds (K=6, 2M sample, K-Means++)
  - Prediction: ~25 seconds (8.9M pixels, chunked)
  - Total: ~40 seconds (vs 5+ minutes on full tile)
  - Memory: ~2GB peak (vs 25GB on full tile)

Cluster Distribution:
  - Cluster 1: 29.3% (dominant urban/vegetation)
  - Cluster 2: 24.9% (secondary urban)
  - Cluster 0: 17.0% (tertiary)
  - Cluster 5: 14.5% (parks/green areas)
  - Cluster 3: 12.3% (mixed)
  - Cluster 4: 2.0% (water/shadows)
```

### Known Issues
- **Automatic download**: Not yet implemented in `analyze_city.py` (requires manual workflow)
- **Tile coverage**: Must verify tile contains city center (see `crop_city_area.py`)
- **Cloud cover**: Best results with <10% cloud cover (use date filter)

---

## 💡 Development Guidelines

### Code Style
- **PEP 8** compliance (use `black` formatter if available)
- **Type hints** for all public functions
- **Docstrings** in Google style:
  ```python
  def function(arg1: type, arg2: type) -> return_type:
      """Short description.
      
      Longer description if needed.
      
      Args:
          arg1: Description
          arg2: Description
          
      Returns:
          Description of return value
          
      Example:
          >>> result = function(val1, val2)
          >>> print(result)
          expected_output
      """
  ```

### Testing Strategy
- **Manual testing** on Milano centro (standard dataset)
- **Visual validation** mandatory before accepting results (check rgb_preview.png)
- **Performance tracking** (time, memory usage for large datasets)
- **Comparison testing** (K-Means vs Spectral vs sklearn)

### Git Workflow
- **Branch**: `develop` (active development)
- **Main branch**: `main` (stable releases only)
- **Commit messages**: Follow Conventional Commits
  ```
  feat: Add consensus classifier implementation
  fix: Correct memory leak in K-Means prediction
  docs: Update QUICKSTART with new examples
  refactor: Simplify analyze_city.py script
  perf: Optimize chunked distance calculation
  ```

### Documentation Priority
- **CRITICAL**: Always update CHANGELOG.md when releasing features
- **IMPORTANT**: Keep README.md in sync with major changes
- **RECOMMENDED**: Update QUICKSTART.md for UX changes
- **OPTIONAL**: Expand GAP_ANALYSIS.md for planning

---

## 🚀 Next Steps for AI Agent

### Immediate Tasks (v1.0.0 Preparation)

#### 1. Implement Consensus Logic (Priority P1) - CRITICAL

**Goal**: Combine K-Means + Spectral classification with confidence scoring.

**Files to Create**:
- `src/satellite_analysis/analyzers/classification/consensus_classifier.py`

**Requirements**:
```python
class ConsensusClassifier:
    """
    Combines multiple classification methods with confidence scoring.
    
    Approach:
    1. Run K-Means clustering (6 clusters)
    2. Run Spectral Indices classification (6 classes)
    3. Map K-Means clusters to semantic classes (manual mapping)
    4. Compute agreement at pixel level
    5. Assign confidence score (0.0 = disagree, 1.0 = full agreement)
    6. Flag uncertain pixels for manual review
    
    Output:
    - Final classification map (6 classes)
    - Confidence map (0-1 per pixel)
    - Uncertainty mask (boolean, True = needs review)
    - Statistics (agreement %, average confidence)
    """
    
    def __init__(self):
        self.kmeans = KMeansPlusPlusClusterer(n_clusters=6)
        self.spectral = SpectralIndicesClassifier()
        self.cluster_to_class_map = None  # To be learned or defined
    
    def fit(self, data, labels_kmeans, labels_spectral):
        """Learn cluster → class mapping based on agreement."""
        # TODO: Implement mapping logic
        # Approach: For each K-Means cluster, find most common Spectral class
        pass
    
    def compute_consensus(self, labels_kmeans, labels_spectral):
        """Compute pixel-level agreement and confidence."""
        # TODO: Implement consensus logic
        pass
    
    def classify(self, raster, band_indices):
        """Full pipeline: K-Means + Spectral + Consensus."""
        # TODO: Implement
        pass
```

**Testing**:
- Test on Milano centro (existing data)
- Compare with individual methods (K-Means only, Spectral only)
- Visualize confidence map (heatmap showing uncertain areas)
- Validate that uncertain areas are actually ambiguous (visual check)

**Success Criteria**:
- ✅ Consensus classifier returns final map + confidence + uncertainty
- ✅ Agreement > 70% on Milano centro
- ✅ Uncertain pixels < 20%
- ✅ Visual validation shows uncertainty in expected areas (e.g., parks/urban boundary)

---

#### 2. Implement Validation Suite (Priority P0) - CRITICAL

**Goal**: Compare classifications against ESA Scene Classification Layer (SCL).

**Files to Create**:
- `src/satellite_analysis/validation/metrics.py`
- `src/satellite_analysis/validation/confusion_matrix.py`
- `scripts/validate_classification.py`

**Requirements**:
```python
# ESA SCL Classes (must download SCL band from Sentinel-2 product)
SCL_CLASSES = {
    0: 'No Data',
    1: 'Saturated/Defective',
    2: 'Dark Area Pixels',
    3: 'Cloud Shadows',
    4: 'Vegetation',
    5: 'Not Vegetated',
    6: 'Water',
    7: 'Unclassified',
    8: 'Cloud Medium Probability',
    9: 'Cloud High Probability',
    10: 'Thin Cirrus',
    11: 'Snow/Ice'
}

# Map our classes to ESA classes
OUR_CLASSES = {
    0: 'Water',
    1: 'Vegetation',
    2: 'Bare Soil',
    3: 'Urban',
    4: 'Bright Surfaces',
    5: 'Shadows'
}

# Validation metrics
def compute_accuracy(y_true, y_pred):
    """Overall accuracy."""
    pass

def compute_kappa(y_true, y_pred):
    """Cohen's Kappa coefficient."""
    pass

def compute_f1_scores(y_true, y_pred, average='weighted'):
    """F1-score per class."""
    pass

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Visualize confusion matrix."""
    pass
```

**Testing**:
- Download SCL band for Milano test tile
- Map our 6 classes to ESA SCL classes (manual mapping)
- Compute metrics for:
  - K-Means only
  - Spectral only
  - Consensus classifier
- Generate report with:
  - Overall Accuracy
  - Kappa score
  - F1-score per class
  - Confusion matrix heatmap

**Success Criteria**:
- ✅ Overall Accuracy > 75% (good for unsupervised)
- ✅ Kappa > 0.65 (substantial agreement)
- ✅ Confusion matrix shows expected confusions (e.g., urban ↔ bare soil)
- ✅ Validation report generated automatically

---

#### 3. Enhance `analyze_city.py` (UX Improvement)

**Goal**: Make the one-command script even more user-friendly.

**Enhancements**:
1. **Progress bars** for long operations (e.g., using `tqdm`)
2. **Automatic download** integration (if credentials available)
3. **Result summary table** (cluster statistics, performance metrics)
4. **Error handling** with helpful messages (e.g., "City not found, did you mean Milan?")
5. **Interactive preview** (open image automatically, ask for confirmation)

**Example Enhanced Output**:
```
======================================================================
🛰️  SATELLITE CITY ANALYZER
======================================================================

🌍 City: Milan
   Center: 45.4640°N, 9.1900°E
   Radius: 15 km
   Area: 894.8 km²

✅ Using existing data in: data/processed/milano_centro

📂 Loading bands...
   ✅ B02: (3006, 2980)
   ✅ B03: (3006, 2980)
   ✅ B04: (3006, 2980)
   ✅ B08: (3006, 2980)
   Stack shape: (3006, 2980, 4)

🎨 Creating preview...
   ✅ Saved: data/processed/milano_centro/preview.png
   ⚠️  Opening preview... [Press Enter after verification]

🎯 K-Means Clustering Analysis
────────────────────────────────────────────────────────────────────
   Data: 8,957,880 pixels × 4 bands
   Training on 2,000,000 samples...
   [████████████████████████████████████████] 100% ETA: 0s
   ✅ Training complete (inertia: 2,927)
   
   Classifying all 8,957,880 pixels...
   [████████████████████████████████████████] 100% ETA: 0s

   Cluster Distribution:
      Cluster 0:  29.8% ████████████████████████████████
      Cluster 1:   3.0% ███
      Cluster 2:  15.9% ███████████████
      Cluster 3:  23.3% ███████████████████████
      Cluster 4:   0.1% █
      Cluster 5:  27.9% ███████████████████████████

   ✅ Saved: data/processed/milano_centro/analysis/kmeans.png

======================================================================
✅ ANALYSIS COMPLETE
======================================================================

📁 Output directory: data/processed/milano_centro
📊 Results:
   • Kmeans: data/processed/milano_centro/analysis/kmeans.png

⏱️  Total time: 42 seconds
💾 Memory peak: 2.1 GB
```

---

### Medium-Term Tasks (v1.1.0)

#### 4. Implement Batch Processing

**Goal**: Analyze multiple cities in one command.

**Example**:
```bash
python scripts/analyze_cities.py \
    --cities Milan Rome Florence Venice \
    --method consensus \
    --validate
```

**Output**:
```
results/
├── milan/
│   ├── kmeans.png
│   ├── spectral.png
│   ├── consensus.png
│   └── validation_report.pdf
├── rome/
│   └── ...
└── summary.html  # Comparative report
```

---

#### 5. Add Temporal Analysis

**Goal**: Compare city changes over time (2 or more dates).

**Use Case**:
- Urban expansion detection
- Vegetation change monitoring
- Seasonal variations

**Example**:
```python
analyzer = TemporalAnalyzer()
results = analyzer.compare_dates(
    city="Milan",
    dates=["2023-06-01", "2024-06-01"],
    method="consensus"
)

# Output: Change map, statistics, trend analysis
```

---

## 🐛 Known Issues & Limitations

### Technical Debt
1. **Automatic download not implemented** in `analyze_city.py`
   - Currently requires manual ZIP placement in `data/raw/`
   - Fix: Integrate `CroppingDownloader` with OAuth2

2. **No cloud masking**
   - User must manually select low-cloud-cover dates
   - Fix: Use SCL band for automatic cloud/shadow masking

3. **Single-tile limitation**
   - Can only process one Sentinel-2 tile at a time
   - Fix: Implement tile mosaicking for large areas

4. **No GPU acceleration**
   - K-Means runs on CPU only
   - Potential: Use PyTorch/CuPy for GPU K-Means

### User Experience Issues
1. **Path complexity**
   - User must understand `data/processed/milano_centro` vs `data/cities/milan`
   - Fix: Standardize on `data/cities/<city>/` structure

2. **Manual preview verification**
   - User must manually check `rgb_preview.png`
   - Improvement: Auto-open preview, ask for confirmation

3. **No error recovery**
   - If one step fails, must restart from beginning
   - Fix: Implement checkpointing (save intermediate results)

---

## 📞 Communication Guidelines

### When Working with User

**DO**:
- ✅ Ask clarifying questions if requirements are ambiguous
- ✅ Propose multiple solutions with trade-offs
- ✅ Provide concrete examples and code snippets
- ✅ Explain technical decisions in simple terms
- ✅ Test thoroughly before presenting results
- ✅ Update documentation alongside code changes

**DON'T**:
- ❌ Make assumptions about user's technical level
- ❌ Implement features without discussing approach first
- ❌ Skip testing on Milano centro dataset
- ❌ Break existing functionality without warning
- ❌ Create new files without updating documentation
- ❌ Use overly complex solutions when simple ones exist

### Reporting Progress

**Format for Status Updates**:
```markdown
## 📊 Progress Update

**Task**: [Task name]
**Status**: [In Progress / Completed / Blocked]
**Time Spent**: [X hours]

### What was done:
- ✅ Item 1
- ✅ Item 2
- 🔄 Item 3 (in progress)

### What's next:
- [ ] Item 4
- [ ] Item 5

### Blockers / Questions:
- [Issue description] → [Proposed solution]
```

---

## 🔑 Critical Success Factors

### For v1.0.0 Release

**Must Have**:
1. ✅ Consensus classifier (P1) - combines K-Means + Spectral
2. ✅ Validation suite (P0) - accuracy metrics vs ESA SCL
3. ✅ Comprehensive tests on Milano + 2 other cities
4. ✅ Documentation update (README, QUICKSTART, CHANGELOG)
5. ✅ Performance validation (no regression from v0.9.0)

**Should Have**:
- 🔶 Automatic download in `analyze_city.py`
- 🔶 Cloud masking using SCL band
- 🔶 Batch processing for multiple cities

**Nice to Have**:
- 💡 GPU acceleration
- 💡 Temporal analysis
- 💡 PDF report generation

---

## 📚 Reference Documentation

### Internal Docs (Must Read)
- `private_docs/GAP_ANALYSIS.md` - Detailed roadmap + feature status
- `CITY_CROPPING_METHODOLOGY.md` - Spatial cropping workflow
- `CHANGELOG.md` - Version history + technical details

### External Resources
- [Sentinel-2 User Guide](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi)
- [ESA Scene Classification](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview)
- [Copernicus Data Space API](https://documentation.dataspace.copernicus.eu/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)

---

## 🎓 Learning Resources

### For Understanding the Codebase
1. Start with `scripts/analyze_city.py` - high-level workflow
2. Read `src/satellite_analysis/analyzers/clustering/kmeans.py` - core algorithm
3. Study `scripts/crop_city_area.py` - spatial operations
4. Review `src/satellite_analysis/analyzers/classification/spectral_classifier.py` - decision tree

### Key Concepts
- **K-Means clustering**: Unsupervised learning, centroid-based
- **Spectral indices**: NDVI (vegetation), MNDWI (water), NDBI (urban)
- **CRS transformation**: WGS84 (geographic) ↔ UTM (projected)
- **Histogram equalization**: Contrast enhancement for visualization

---

## 💬 Final Notes

This project was built iteratively with a strong focus on:
1. **Simplicity** - One command should do everything
2. **Performance** - 10x speedup through optimization
3. **Correctness** - Always verify with RGB preview before trusting results
4. **Documentation** - Every feature is documented with examples

The **biggest lesson learned**: Over-engineering is a problem. The refactoring from v0.3 → v0.9.0 initially created too much complexity (5 scripts, 4 documentation files). The cleanup in v0.9.0 consolidated everything into ONE script (`analyze_city.py`) while keeping technical quality.

**Remember**: Always test on Milano centro before claiming success. Visual validation is MANDATORY - satellite imagery can be tricky!

Good luck! 🚀

# 🛰️ Satellite Analysis - Sentinel-2 Processing Pipeline

**Version 1.0.0** - Professional toolkit for Sentinel-2 satellite imagery analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![UV Package Manager](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)

---

## 🚀 **Quick Start → [QUICKSTART.md](QUICKSTART.md)**

**Analyze any city in ONE command:**

```bash
# Recommended: Consensus classification (K-Means + Spectral combined)
python scripts/analyze_city.py --city Milan --method consensus

# Or K-Means only
python scripts/analyze_city.py --city Milan --method kmeans
```

**That's it!** Results in `data/cities/<city>/analysis/` 🎉

See **[QUICKSTART.md](QUICKSTART.md)** for full getting-started guide (5 minutes).

---

## ✨ What's New in v1.0.0

### 🔮 **Consensus Classification** (NEW!)
- Combines K-Means clustering + Spectral indices for robust results
- **Confidence scoring**: Know how reliable each pixel classification is
- **Uncertainty flagging**: Automatically identify ambiguous areas
- **Automatic mapping**: Learns cluster-to-class relationships

### 🔍 **Validation Suite** (NEW!)
- Compare classifications against ESA Scene Classification Layer (SCL)
- Full metrics: Overall Accuracy, Kappa, F1-score (per-class and weighted)
- Confusion matrix visualization
- Comprehensive validation reports

---

## ✨ What This Does

### 🎯 **One-Command Analysis**
- **Consensus Classification**: Best of K-Means + Spectral (recommended)
- **K-Means Clustering**: Automatic land cover classification (6 clusters)
- **Spectral Indices**: Water, vegetation, urban, bare soil detection
- **City Cropping**: Extract 15km radius around any city center
- **Visualization**: Multi-panel comparison + confidence maps

### 🔧 **Key Features**
- 🌍 **Area Selection**: By city name or coordinates
- 📥 **Smart Download**: Sentinel-2 tiles with cloud filtering
- ⚡ **Performance**: 10x faster K-Means (memory optimized)
- 🎨 **RGB True Color**: Natural-looking previews with histogram equalization
- 📊 **Validation**: Compare against ESA reference data

### 📊 **Performance**
- **Memory**: 25GB → 2GB RAM (chunked processing)
- **Speed**: 5min → 45sec (smart sampling: train on 2M, predict on all)
- **Space**: 92% reduction (10980² → 3000² pixels for cropped area)

---

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone repository
git clone https://github.com/VTvito/satellite_git.git
cd satellite_git

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure (optional for auto-download)

Create `config/config.yaml` with your Sentinel Hub credentials:

```yaml
sentinel:
  client_id: "your_client_id"
  client_secret: "your_client_secret"
```

### 3. Run Analysis

**See [QUICKSTART.md](QUICKSTART.md) for complete guide.**

```bash
# Analyze any city
python scripts/analyze_city.py --city Milan --method kmeans

# Or use manual workflow (if needed)
python scripts/kmeans_milano_optimized.py
```

---

## 📊 Architecture

```
src/satellite_analysis/
├── utils/              # AreaSelector, geocoding, visualization
├── downloaders/        # Sentinel-2 download (OAuth2)
├── preprocessors/      # Band extraction, cropping
├── analyzers/          # Analysis algorithms ✨
│   ├── classification/ # SpectralIndicesClassifier
│   └── clustering/     # KMeans, KMeans++, Sklearn wrapper
├── preprocessing/      # Normalization, reshaping
├── pipelines/          # High-level workflows
└── config/             # Settings management

scripts/
├── analyze_city.py             # 🎯 ONE-COMMAND analysis (consensus default)
├── app.py                      # 🌐 Web UI (Streamlit)
├── validate_classification.py  # 🔍 Validation suite (NEW v1.0.0)
├── crop_city_area.py           # City cropping utility
├── kmeans_milano_optimized.py  # K-Means workflow
└── test_classifier_milano.py   # Spectral classification

notebooks/
├── city_analysis.ipynb         # 📓 Complete analysis workflow
├── clustering_example.ipynb    # K-Means tutorial
└── download_example.ipynb      # Download API example
```

### Key Modules

**Analyzers**:
- `ConsensusClassifier` - **NEW** Combined K-Means + Spectral with confidence
- `KMeansClusterer` - Custom K-Means with chunked processing
- `KMeansPlusPlusClusterer` - Smart initialization
- `SklearnKMeansClusterer` - Sklearn wrapper for comparison
- `SpectralIndicesClassifier` - Rule-based classification (NDVI, MNDWI, NDBI, BSI)

**Validation** (NEW in v1.0.0):
- `ValidationReport` - Comprehensive accuracy assessment
- `compute_accuracy()`, `compute_kappa()`, `compute_f1_scores()` - Metrics
- `plot_confusion_matrix()` - Visualization
- `SCLValidator` - Compare against ESA Scene Classification Layer

**Preprocessing**:
- `min_max_scale()` - Normalize bands to [0, 1]
- `reshape_image_to_table()` - Convert (H,W,C) → (N, C) for ML
- `reshape_table_to_image()` - Convert back to (H,W) for visualization

---

## 🧪 Development

### Run Tests

```bash
# Area selection tests
python tests/test_area_selection.py

# Preprocessing tests
python tests/test_preprocessing_pipeline.py

# Complete workflow
python tests/test_complete_workflow.py
```

### Code Structure

```python
# Example: Custom analysis pipeline
from satellite_analysis.utils import AreaSelector
from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
from satellite_analysis.analyzers.classification import ConsensusClassifier
from satellite_analysis.preprocessing import min_max_scale, reshape_image_to_table
from satellite_analysis.validation import ValidationReport

# 1. Select area
selector = AreaSelector()
bbox, info = selector.select_by_city("Milan", radius_km=15)

# 2. Load bands (manual or with pipeline)
# ... load B02, B03, B04, B08 ...

# 3. Run consensus classification (recommended)
classifier = ConsensusClassifier(n_clusters=6)
labels, confidence, uncertainty, stats = classifier.classify(
    stack, band_indices={'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
)

# 4. Or run K-Means directly
data = reshape_image_to_table(stack)  # (H*W, 4)
data_scaled = min_max_scale(data)
clusterer = KMeansPlusPlusClusterer(n_clusters=6)
clusterer.fit(data_scaled)
labels = clusterer.predict(data_scaled)
```

---
## 🌐 Web UI (Streamlit)

**Interactive web interface** for analyzing cities without command line:

```bash
# Install UI dependencies
pip install streamlit

# Launch the web app
streamlit run scripts/app.py
```

**Features:**
- 🏙️ City selection with automatic geocoding
- 📊 Real-time classification visualization
- 🗺️ Confidence maps and uncertainty analysis
- 📥 Export results

---

## 📓 Jupyter Notebooks

Interactive tutorials in `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `city_analysis.ipynb` | Complete analysis workflow (recommended) |
| `clustering_example.ipynb` | K-Means clustering tutorial |
| `download_example.ipynb` | Sentinel-2 download API guide |

```bash
# Install notebook dependencies
pip install jupyter ipykernel

# Launch Jupyter
jupyter notebook notebooks/
```

---
## � Performance Benchmarks

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Download (1.2GB) | ~5 min | - | Sentinel-2 tile |
| Crop to city | ~30s | <1GB | 92% reduction |
| K-Means (full tile) | 5+ min | 25GB | 120M pixels → OOM |
| **Consensus (cropped)** | **45s** | **2GB** | **Recommended** ✨ |
| K-Means (cropped) | ~40s | ~2GB | Clustering only |
| Spectral classification | <10s | <1GB | Rule-based |
| Validation report | <5s | <1GB | Metrics + plots |

**Optimization Highlights**:
- ✅ Chunked distance calculation (10K samples/chunk) → No OOM
- ✅ Smart sampling (train 2M, predict all) → 10x speedup
- ✅ City cropping (92% reduction) → Focused analysis
- ✅ Consensus validation → Confidence in results

---

## 📝 Documentation

**For Users**:
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute getting started guide ⭐
- `ARCHITECTURE.md` - System design
- `PREPROCESSING_REPORT.md` - Pipeline details
- `AREA_SELECTION_REPORT.md` - Area selection guide

**For Developers**:
- `private_docs/GAP_ANALYSIS.md` - Feature roadmap
- Code docstrings - Full API documentation

---

## 🛠️ Key Dependencies

- **rasterio** (1.3.9): Geospatial raster I/O
- **numpy** (1.26.0): Numerical computing
- **scikit-learn** (1.3.0): Machine learning algorithms
- **matplotlib** (3.8.0): Visualization
- **shapely** (2.0.2): Geometric operations
- **geopy** (2.4.0): Geocoding

---

## 📜 License

MIT License - See LICENSE file

---

## 🗺️ Roadmap

**v1.0.0** (Current) - Production Release ✅
- ✅ Consensus classification (K-Means + Spectral)
- ✅ Validation suite (OA, Kappa, F1, confusion matrix)
- ✅ Confidence scoring and uncertainty flagging
- ✅ One-command analysis with consensus default

**v1.1.0** (Next) - Batch Processing
- Batch analysis for multiple cities
- PDF report generation
- Temporal analysis (multi-date comparison)

See `CHANGELOG.md` for detailed version history.

---

## 🐛 Issues

Found a bug? Have a feature request? Please open an issue on GitHub.

---

**Version**: 1.0.0 - **Production Release** 🚀  
**Last Updated**: December 18, 2025

**Made with ❤️ for satellite imagery analysis**

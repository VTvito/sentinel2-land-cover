# 🛰️ Satellite Analysis - Sentinel-2 Processing Pipeline

**Version 0.9.0** - Professional toolkit for Sentinel-2 satellite imagery analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![UV Package Manager](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)

---

## 🚀 **Quick Start → [QUICKSTART.md](QUICKSTART.md)**

**Analyze any city in ONE command:**

```bash
python scripts/analyze_city.py --city Milan --method kmeans
```

**That's it!** Results in `data/cities/<city>/analysis/` 🎉

See **[QUICKSTART.md](QUICKSTART.md)** for full getting-started guide (5 minutes).

---

## ✨ What This Does

### 🎯 **One-Command Analysis**
- **K-Means Clustering**: Automatic land cover classification (6 clusters)
- **Spectral Indices**: Water, vegetation, urban, bare soil detection
- **City Cropping**: Extract 15km radius around any city center
- **Visualization**: Side-by-side RGB + classification results

### 🔧 **Key Features**
- 🌍 **Area Selection**: By city name or coordinates
- 📥 **Smart Download**: Sentinel-2 tiles with cloud filtering
- ⚡ **Performance**: 10x faster K-Means (memory optimized)
- 🎨 **RGB True Color**: Natural-looking previews with histogram equalization

### 📊 **Performance**
- **Memory**: 25GB → 2GB RAM (chunked processing)
- **Speed**: 5min → 40sec (smart sampling: train on 2M, predict on all)
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
├── analyze_city.py             # 🎯 ONE-COMMAND analysis ✨ NEW
├── crop_city_area.py           # City cropping utility
├── kmeans_milano_optimized.py  # K-Means workflow
└── test_classifier_milano.py   # Spectral classification
```

### Key Modules

**Analyzers** (NEW in v0.9.0):
- `KMeansClusterer` - Custom K-Means with chunked processing
- `KMeansPlusPlusClusterer` - Smart initialization
- `SklearnKMeansClusterer` - Sklearn wrapper for comparison
- `SpectralIndicesClassifier` - Rule-based classification (NDVI, MNDWI, NDBI, BSI)

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
from satellite_analysis.preprocessing import min_max_scale, reshape_image_to_table

# 1. Select area
selector = AreaSelector()
bbox, info = selector.select_by_city("Milan", radius_km=15)

# 2. Load bands (manual or with pipeline)
# ... load B02, B03, B04, B08 ...

# 3. Prepare data
data = reshape_image_to_table(stack)  # (H*W, 4)
data_scaled = min_max_scale(data)

# 4. Cluster
clusterer = KMeansPlusPlusClusterer(n_clusters=6)
clusterer.fit(data_scaled)
labels = clusterer.predict(data_scaled)
```

---

## � Performance Benchmarks

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Download (1.2GB) | ~5 min | - | Sentinel-2 tile |
| Crop to city | ~30s | <1GB | 92% reduction |
| K-Means (full tile) | 5+ min | 25GB | 120M pixels → OOM |
| **K-Means (cropped)** | **40s** | **2GB** | **10x faster** ✨ |
| Spectral classification | <10s | <1GB | Rule-based |

**Optimization Highlights**:
- ✅ Chunked distance calculation (10K samples/chunk) → No OOM
- ✅ Smart sampling (train 2M, predict all) → 10x speedup
- ✅ City cropping (92% reduction) → Focused analysis

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

## � Roadmap

**v0.9.0** (Current) - K-Means Clustering ✅
- Custom K-Means with memory optimization
- City cropping methodology
- One-command analysis script

**v1.0.0** (Next) - Consensus Logic
- Multi-method consensus classification
- Confidence scores
- Enhanced visualization

See `private_docs/GAP_ANALYSIS.md` for detailed roadmap.

---

## 🐛 Issues

Found a bug? Have a feature request? Please open an issue on GitHub.

---

**Version**: 0.9.0 - **K-Means Clustering Release** ✨  
**Last Updated**: October 15, 2025

```bash
# Install UV and dependencies
pip install uv
git clone https://github.com/VTvito/satellite_git.git
cd satellite_git
uv venv
.venv\Scripts\activate  # Windows
uv pip install -e .
```

### Configure OAuth2
Register at [Copernicus Data Space](https://dataspace.copernicus.eu/) and edit `config/config.yaml`:
```yaml
**Made with ❤️ for satellite imagery analysis**

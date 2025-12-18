# 🤖 AI Agent Instructions - Satellite Analysis Project

**Project**: Sentinel-2 Satellite Imagery Analysis Toolkit  
**Version**: 1.0.0 (Production Release)  
**Last Updated**: December 18, 2025  
**Status**: Production Ready

---

## 📋 Project Overview

Professional toolkit for **Sentinel-2 satellite imagery analysis** with:
- **One-Command Analysis**: `python scripts/analyze_city.py --city Milan --method consensus`
- **Web UI**: `streamlit run scripts/app.py`
- **Jupyter Notebooks**: Interactive tutorials in `notebooks/`

**Key Features**:
- Consensus Classification (K-Means + Spectral combined)
- Validation Suite (accuracy vs ESA SCL reference)
- Confidence Scoring and Uncertainty Flagging
- Memory-Optimized Processing (25GB → 2GB RAM)

---

## 🎯 Current Status (v1.0.0)

### ✅ Completed Features

| Feature | Status | Files |
|---------|--------|-------|
| K-Means Clustering | ✅ Done | `analyzers/clustering/` |
| Spectral Indices | ✅ Done | `analyzers/classification/spectral_indices.py` |
| Consensus Classifier | ✅ Done | `analyzers/classification/consensus_classifier.py` |
| Validation Suite | ✅ Done | `validation/` |
| Web UI (Streamlit) | ✅ Done | `scripts/app.py` |
| Jupyter Notebooks | ✅ Done | `notebooks/` |

### 🔴 Pending (v1.1.0 Roadmap)

- ❌ Batch processing for multiple cities
- ❌ PDF report generation
- ❌ Temporal analysis (multi-date comparison)
- ❌ Automatic download in analyze_city.py

---

## 📁 Project Structure

```
satellite_git/
├── scripts/                    # Entry points
│   ├── analyze_city.py         # 🎯 Main CLI (one-command)
│   ├── app.py                  # 🌐 Web UI (Streamlit)
│   ├── validate_classification.py  # Validation tool
│   ├── crop_city_area.py       # Crop utility
│   ├── download_products.py    # Download utility
│   └── extract_all_bands.py    # Band extraction
│
├── src/satellite_analysis/     # Core library
│   ├── analyzers/
│   │   ├── clustering/         # KMeans, KMeans++
│   │   └── classification/     # Spectral, Consensus
│   ├── validation/             # Metrics, SCL validator
│   ├── preprocessing/          # Normalization, reshape
│   ├── utils/                  # AreaSelector, visualization
│   ├── downloaders/            # Sentinel-2 download
│   └── pipelines/              # High-level workflows
│
├── notebooks/                  # Interactive tutorials
│   ├── city_analysis.ipynb     # Complete workflow
│   ├── clustering_example.ipynb
│   └── download_example.ipynb
│
├── tests/                      # Unit tests
├── config/                     # Configuration files
├── data/                       # Data directory (gitignored)
│
├── README.md                   # Main documentation
├── QUICKSTART.md               # Getting started guide
├── CHANGELOG.md                # Version history
└── pyproject.toml              # Dependencies
```

### Key Entry Points

| Use Case | Command |
|----------|---------|
| CLI Analysis | `python scripts/analyze_city.py --city Milan --method consensus` |
| Web Interface | `streamlit run scripts/app.py` |
| Validation | `python scripts/validate_classification.py --city Milan --report` |
| Jupyter | `jupyter notebook notebooks/city_analysis.ipynb` |

---

## 🛠️ Core Components

### Analyzers

```python
# Consensus Classifier (recommended)
from satellite_analysis.analyzers.classification import ConsensusClassifier

classifier = ConsensusClassifier(n_clusters=6)
labels, confidence, uncertainty, stats = classifier.classify(
    stack, band_indices={'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
)

# K-Means Clustering
from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer

clusterer = KMeansPlusPlusClusterer(n_clusters=6)
clusterer.fit(data_scaled)
labels = clusterer.predict(data_scaled)

# Spectral Classification
from satellite_analysis.analyzers.classification import SpectralIndicesClassifier

classifier = SpectralIndicesClassifier()
labels, indices = classifier.classify(raster, band_indices)
```

### Validation

```python
from satellite_analysis.validation import (
    compute_accuracy, compute_kappa, compute_f1_scores,
    ValidationReport, SCLValidator
)

report = ValidationReport(y_true, y_pred, class_names)
print(f"Accuracy: {report.accuracy:.2%}")
print(f"Kappa: {report.kappa:.3f}")
```

### Preprocessing

```python
from satellite_analysis.preprocessing import min_max_scale
from satellite_analysis.preprocessing import reshape_image_to_table, reshape_table_to_image

# Normalize bands
data_scaled = min_max_scale(data)

# Image to table for ML
table = reshape_image_to_table(image)  # (H,W,C) → (N,C)
```

---

## 📊 Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Consensus (cropped) | ~45s | ~2GB |
| K-Means (cropped) | ~40s | ~2GB |
| Spectral only | <10s | <1GB |
| Validation | <5s | <1GB |

**Optimizations**:
- Chunked processing (no OOM)
- Smart sampling (train 2M, predict all)
- City cropping (92% size reduction)

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Specific tests
python tests/test_area_selection.py
python tests/test_spectral_classifier.py
python tests/test_complete_workflow.py
```

**Standard Test Data**: Milano centro (45.464°N, 9.190°E, 15km radius)

---

## 💡 Development Guidelines

### Code Style
- **PEP 8** with Black formatter
- **Type hints** for public functions
- **Google-style docstrings**

### Git Workflow
```bash
# Branch: develop (active development)
# Commits: Conventional Commits format
feat: Add new feature
fix: Bug fix
docs: Documentation
refactor: Code refactoring
```

### When Adding Features
1. Create feature in `src/satellite_analysis/`
2. Add CLI script in `scripts/` if user-facing
3. Update `__init__.py` exports
4. Add tests in `tests/`
5. Update CHANGELOG.md

---

## 🔑 Class Definitions

```python
# 6-class land cover classification
CLASSES = {
    0: 'WATER',
    1: 'VEGETATION',
    2: 'BARE_SOIL', 
    3: 'URBAN',
    4: 'BRIGHT_SURFACES',
    5: 'SHADOWS_MIXED'
}
```

---

## 📞 Communication Guidelines

**DO**:
- ✅ Test on Milano centro before claiming success
- ✅ Update CHANGELOG.md for features
- ✅ Visual validation is MANDATORY
- ✅ Keep README.md in sync

**DON'T**:
- ❌ Skip testing on standard dataset
- ❌ Create redundant documentation files
- ❌ Over-engineer simple solutions

---

## 📚 Files Reference

### Documentation (Essential)
| File | Purpose |
|------|---------|
| README.md | Main project documentation |
| QUICKSTART.md | 5-minute getting started |
| CHANGELOG.md | Version history |

### Configuration
| File | Purpose |
|------|---------|
| pyproject.toml | Dependencies & metadata |
| config/config.yaml | Sentinel Hub credentials |
| config/area_cache.json | Cached city coordinates |

---

**Remember**: Visual validation is MANDATORY - always check preview images before trusting results!

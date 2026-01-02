# Release Notes v2.3.0 🚀

**Release Date**: December 30, 2025

## 🎯 Highlights

This release focuses on **visualization quality** and **data quality control** with raw cluster preservation, publication-ready RGB exports, and manual product selection workflow.

---

## ✨ What's New

### Raw Clusters Mode
- **`raw_clusters=True`** preserves distinct cluster IDs (0 to N-1) without semantic mapping
- Essential for KMeans visualization where 6 clusters shouldn't collapse to semantic classes
- Fixes cluster collapse bug in previous versions

```python
result = analyze("Milan", raw_clusters=True)  # Keep all 6 distinct clusters
```

### RGB Export Functions
- **`export_rgb()`**: Generate publication-quality visualizations at 300 DPI
  - True Color (R-G-B)
  - False Color Composite (NIR-R-G)
  - NDVI vegetation index
- **RGB GeoTIFF export**: Georeferenced True Color for GIS integration

```python
export_rgb(result, "Milan", output_dir="exports/")
```

### Manual Product Selection
- **Improved notebook workflow** (Cells 4-5): explicit product selection for better data quality control
- Users can now review cloud cover, acquisition date, and other metadata before downloading
- `auto_download=False` by default in notebook

### Visual Improvements
- **High-contrast color palette** for better visualization
- Urban class changed from gray to crimson (#DC143C)
- Auto-detection of raw vs semantic mode with distinct color schemes

---

## 🔧 Configuration

New notebook parameters:
```python
RAW_CLUSTERS = False  # Set to True for distinct cluster visualization
SELECTED_PRODUCT_INDEX = 0  # Index of product to download from search results
```

---

## 📦 Installation

```bash
git clone https://github.com/VTvito/sentinel2-land-cover.git
cd sentinel2-land-cover
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[notebooks]"
```

---

## 🔗 Quick Links

- **Notebook**: [notebooks/city_analysis.ipynb](notebooks/city_analysis.ipynb)
- **Documentation**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## 🐛 Bug Fixes

- Fixed KMeans cluster collapse issue where 6 clusters would map to 3 semantic classes
- Improved path resolution consistency

---

## 💡 API Examples

```python
from satellite_analysis import analyze, export_rgb, export_geotiff

# Basic analysis
result = analyze("Florence")

# Raw clusters (for KMeans visualization)
result = analyze("Milan", raw_clusters=True)

# Export visualizations
export_rgb(result, "Milan", output_dir="exports/")
export_geotiff(result, "Milan", output_dir="exports/")
```

---

## 🔄 Upgrade Notes

- **Breaking change**: Notebook now requires manual product selection (Cell 5)
- `auto_download` parameter defaults to `False` in notebook workflow
- Update existing notebooks to include `SELECTED_PRODUCT_INDEX` configuration

---

## 📊 Statistics

- **Test Coverage**: 85 tests (71 unit + 14 integration)
- **Python Support**: 3.10, 3.11, 3.12, 3.13
- **Land Cover Classes**: 6 semantic classes (Water, Vegetation, Bare Soil, Urban, Bright Surfaces, Shadows/Mixed)

---

## 🙏 Contributors

Thanks to all contributors and users for feedback and suggestions!

---

**Full Changelog**: [CHANGELOG.md](CHANGELOG.md)

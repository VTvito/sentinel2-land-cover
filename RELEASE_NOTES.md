# Release Notes v2.3.0

## 🎯 Highlights

**Raw Clusters Mode** is the marquee feature of this release, solving the cluster collapse bug that plagued KMeans visualizations. Now you can preserve distinct cluster IDs without semantic mapping.

**RGB Export Suite** brings publication-quality visualization tools (300 DPI) with three modes: True Color, False Color Composite (NIR-R-G), and NDVI heatmaps - all georeferenced for GIS integration.

## ✨ New Features

### `raw_clusters=True` Parameter
Keep distinct cluster IDs (0 to N-1) without semantic mapping. Essential for KMeans where 6 clusters shouldn't collapse to 3 semantic classes.

```python
result = analyze("Milan", raw_clusters=True)  # Preserves all 6 KMeans clusters
```

### `export_rgb()` Function
Generate publication-quality RGB visualizations at 300 DPI:

```python
from satellite_analysis import export_rgb

# True Color RGB (like a natural photo)
export_rgb(result, mode="true_color", output_path="true_color.png")

# False Color Composite (NIR-R-G, vegetation in red)
export_rgb(result, mode="false_color", output_path="false_color.png")

# NDVI heatmap (vegetation health)
export_rgb(result, mode="ndvi", output_path="ndvi.png")
```

### RGB GeoTIFF Export
Georeferenced RGB True Color for GIS integration:

```python
from satellite_analysis import export_geotiff

export_geotiff(result, output_path="milan_rgb.tif", rgb=True)
```

## 🔧 Improvements

### Enhanced Color Palette
High-contrast colors for better visualization:
- Urban: Gray → **Crimson** (#DC143C) for better visibility
- All classes now use distinct, vibrant colors

### Manual Product Selection
Notebook workflow now requires explicit product selection (Cells 4-5) for better data quality control:
1. Cell 4: Search available products
2. Cell 5: Select `SELECTED_PRODUCT_INDEX` → download
3. Cell 6: Preview RGB (optional)
4. Cell 7: `analyze()` with `auto_download=False`

### Configuration Cell Update
Notebook configuration cell now includes `RAW_CLUSTERS` parameter for toggling between semantic and raw modes.

## 🐛 Bug Fixes

### KMeans Cluster Collapse Fixed
Fixed issue where 6 KMeans clusters would collapse to 3 semantic classes due to NDVI/NDWI threshold mapping. Use `raw_clusters=True` to preserve all clusters and see the full clustering output.

**Before:** 6 clusters → 3 semantic classes (data loss)  
**After:** 6 clusters → 6 distinct visualizations with `raw_clusters=True`

## 📊 Workflow Changes

- **Auto-download disabled by default**: Users must explicitly select products via Cell 5 for better data quality control
- **Visualization auto-detection**: Notebook visualization cell automatically detects mode and uses appropriate color palettes

## 🔄 Migration Guide

### From v2.2.0

No breaking changes. All existing code continues to work:

```python
# Still works exactly as before
result = analyze("Milan")  # Uses semantic mapping (default)

# New option for raw clustering
result = analyze("Milan", raw_clusters=True)  # Keeps distinct cluster IDs
```

### Notebook Users

Update your notebook configuration cell to include:

```python
RAW_CLUSTERS = False  # or True for raw mode
```

And update Cell 7 to use `auto_download=False`:

```python
result = analyze(
    CITY_NAME,
    auto_download=False,  # Must select products manually in Cell 5
    raw_clusters=RAW_CLUSTERS
)
```

## 📚 Documentation

- Updated notebook with new workflow (manual product selection)
- Enhanced API documentation for `raw_clusters` parameter
- New examples for RGB export functions

## 🙏 Acknowledgments

Thanks to all users who reported the cluster collapse issue and helped test the new `raw_clusters` mode!

---

**Full Changelog**: [CHANGELOG.md](CHANGELOG.md)  
**Issues Fixed**: Cluster collapse bug in KMeans visualization  
**Download**: See [Releases](https://github.com/yourusername/satellite_git/releases/tag/v2.3.0)

# 🛰️ Satellite Analysis - Sentinel-2 Processing Pipeline

Professional toolkit for Sentinel-2 satellite imagery: download, preprocessing with automatic cropping, and analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![UV Package Manager](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)

---

## ✨ Key Features

### 🌍 **Area Selection**
- Select by city name (geopy integration)
- Select by coordinates (lat/lon + radius)
- Automatic bbox generation (rectangle/circle)

### 📥 **Download**
- OAuth2 authentication with Copernicus Data Space
- STAC catalog search by date, cloud cover, area
- Progress bar for large downloads (~1.2 GB in 5 min)

### 🔧 **Preprocessing**
- **Automatic Cropping** with CRS transformation ⭐
- Band extraction (B02, B03, B04, B08)
- RGB, False Color Composite (FCC), NDVI generation
- 92.5% data reduction (10980² → 3007×2994 pixels for 15km area)

### 🎨 **Visualization**
- 3-panel display (RGB / FCC / NDVI)
- Automatic save and open
- Custom NDVI colormap

---

## 🚀 Quick Start

### 1. Installation

```powershell
# Install UV package manager
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone repository
git clone https://github.com/VTvito/satellite_git.git
cd satellite_git

# Install dependencies
uv sync
```

### 2. Configuration

Create `config/config.yaml`:

```yaml
copernicus:
  client_id: "your_client_id"
  client_secret: "your_client_secret"
  username: "your_username"
  password: "your_password"
```

### 3. Complete Workflow

```python
from satellite_analysis.utils import AreaSelector
from satellite_analysis.downloaders import SentinelDownloader
from satellite_analysis.pipelines import PreprocessingPipeline

# 1. Select area
selector = AreaSelector()
bbox, info = selector.select_by_city("Milan", radius_km=15)
print(f"Area: {info['area_km2']:.1f} km²")

# 2. Download Sentinel-2 product
downloader = SentinelDownloader()
products = downloader.search(
    bbox=bbox,
    date_start="2024-10-01",
    date_end="2024-10-15",
    max_cloud_cover=20
)

zip_file = downloader.download(
    product_id=products[0]['id'],
    output_dir="data/raw"
)

# 3. Preprocessing with automatic crop
pipeline = PreprocessingPipeline()
result = pipeline.run(
    zip_path=str(zip_file),
    bbox=bbox,  # Automatic crop to area!
    save_visualization=True,
    open_visualization=True
)

print(f"RGB: {result.rgb.shape}")
print(f"NDVI range: [{result.metadata['ndvi_range'][0]:.3f}, {result.metadata['ndvi_range'][1]:.3f}]")
```

---

## 📊 Architecture

```
src/satellite_analysis/
├── utils/              # Area selection, geocoding
├── downloaders/        # Sentinel-2 download with OAuth2
├── preprocessors/      # Band extraction, crop with CRS transform
├── pipelines/          # High-level workflows
└── config/             # Configuration management
```

---

## 🧪 Testing

```powershell
# Test area selection
uv run python tests/test_area_selection.py

# Test preprocessing with crop
uv run python tests/test_preprocessing_pipeline.py

# Test complete workflow (download + preprocessing)
uv run python tests/test_complete_workflow.py
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Geocoding | <1s | geopy API |
| Product search | 2-3s | STAC catalog |
| Download 1.2GB | 5 min | 4 MB/s bandwidth |
| Preprocessing (crop) | 30s | Includes NDVI |
| **Data reduction** | **92.5%** | 10980² → 3007×2994 px |

---

## 🔧 Advanced: Automatic Cropping

The preprocessing pipeline includes **automatic cropping with CRS transformation**:

1. Input bbox in WGS84 (EPSG:4326) - standard geographic coordinates
2. Auto-detect raster CRS (typically UTM 32N - EPSG:32632)
3. Transform bbox to raster CRS using `rasterio.warp.transform_geom`
4. Crop with `rasterio.mask` and shapely polygon
5. Output: only requested area, preserving quality

**Before crop**: 110 km × 110 km tile (10980 × 10980 pixels = 1.2 GB)  
**After crop**: 30 km × 30 km area (3007 × 2994 pixels = 35 MB)  
**Benefit**: Faster processing, lower memory, focused output

---

## 📝 Documentation

- `ARCHITECTURE.md` - System design and module structure
- `PREPROCESSING_REPORT.md` - Preprocessing pipeline details
- `AREA_SELECTION_REPORT.md` - Area selection and geocoding

---

## 🛠️ Dependencies

- **rasterio** (1.3.9): Geospatial raster I/O
- **shapely** (2.0.2): Geometric operations
- **geopy** (2.4.0): Geocoding
- **matplotlib** (3.8.0): Visualization
- **requests-oauthlib**: OAuth2 authentication

---

## 📜 License

MIT License - See LICENSE file for details

---

## 👥 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 🐛 Issues

Found a bug? Have a feature request? Please open an issue on GitHub.

---

**Version**: 0.3.0  
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
auth:
  client_id: "your_client_id"
  client_secret: "your_client_secret"
```

### Download Example
```python
from satellite_analysis.utils import AreaSelector
from satellite_analysis.pipelines import DownloadPipeline

# Select area by city name (automatic coordinates!)
selector = AreaSelector()
bbox, metadata = selector.select_by_city("Milan", radius_km=15)

print(f"Area: {metadata['area_km2']:.1f} km² centered on Milan")

# Download
pipeline = DownloadPipeline.from_config("config/config.yaml")
result = pipeline.run(
    bbox=bbox,  # Use correct coordinates!
    start_date="2023-03-01",
    end_date="2023-03-15",
    max_cloud_cover=20
)

print(f"Downloaded {result.downloaded_count} products")
# Files saved in: data/raw/product_*.zip
# Previews saved in: data/previews/product_*_preview.png
```

### Preprocessing Example
```python
from satellite_analysis.pipelines import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    bands=['B02', 'B03', 'B04', 'B08'],  # RGB + NIR
    resolution="10m"
)

result = pipeline.run("data/raw/product_1.zip")

# Access processed data
rgb_image = result.rgb          # RGB composite (uint8)
fcc_image = result.fcc          # False Color Composite
ndvi_map = result.ndvi          # Vegetation index
bands = result.band_data        # Raw band data (uint16)
```
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete system design.

### 3 Usage Levels

1. **Simple** (Notebooks): High-level pipelines for interactive analysis
2. **Modular** (Scripts): Individual components for custom workflows
3. **Advanced** (Extensions): Implement custom strategies (Auth, Clustering, Classification)

### Key Components

- `downloaders/`: OAuth2 auth + STAC catalog + product download
- `preprocessors/`: Band extraction (TODO)
- `analyzers/`: KMeans++ clustering, classification (TODO)
- `pipelines/`: High-level orchestration
- `utils/`: Geospatial + visualization helpers

## Status

✅ **Implemented**: OAuth2 auth, catalog search, product download (tested with 1.15 GB file), band extraction, RGB/FCC/NDVI composites, KMeans++  
🔜 **TODO**: Classification (Random Forest, SVM)

## Performance Notes

### Download Speed ⚠️

Downloading Sentinel-2 data is **intentionally slow** (~1.2 GB per product, ~160 minutes at 1 Mbps):

- **Why?** Copernicus Data Space provides full tiles (~110 km × 110 km), not area subsets
- **Your area (15 km)** is only ~7% of the tile, but you download 100%
- **Inevitable**: Free API doesn't support spatial cropping during download

**Solutions**:
1. **Use CroppingDownloader** (saves 97% disk space after download):
   ```python
   from satellite_analysis.downloaders import CroppingDownloader
   # Downloads 1.2 GB, saves only 35 MB on disk
   ```

2. **Sentinel Hub API** (€0.01-0.05 per request, 120x faster):
   - Downloads only requested area (~10 MB vs 1.2 GB)
   - Time: ~1 minute vs 160 minutes

3. **Google Earth Engine** (free with quotas):
   - Cloud processing + area cropping
   - Ideal for research projects

📖 See [doc/DOWNLOAD_SPEED_EXPLANATION.md](doc/DOWNLOAD_SPEED_EXPLANATION.md) for details.

## Troubleshooting

- **OAuth2 failed**: Check credentials in `config/config.yaml`
- **No products found**: Increase `max_cloud_cover` or extend date range
- **Download failed**: Refresh token with `auth.refresh()`
- **Wrong area downloaded**: The catalog now filters tiles that contain the center point of your bbox (fixed!)

## License

MIT License

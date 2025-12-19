# 🛰️ Satellite City Analyzer

**Notebook-first land cover analysis from Sentinel-2 imagery.**

Open and run the notebook:

- [notebooks/city_analysis.ipynb](notebooks/city_analysis.ipynb)

Alternative entry points:

```bash
python scripts/analyze_city.py --city Milan --export image report
```

```python
from satellite_analysis import analyze
result = analyze("Milan")
print(result.summary())
```

---

## Features

- **Notebook-first UX**: one place to configure + run + visualize
- **One-line API**: `analyze("city")` does everything
- **Batch processing**: Analyze multiple cities at once
- **Change detection**: Compare land cover across time periods
- **Multiple exports**: GeoTIFF, HTML reports, JSON, PNG image
- **CLI & Python API**: Use from terminal or code
- **REST API**: FastAPI server for integrations

**Land Cover Classes**: Water, Vegetation, Bare Soil, Urban, Bright Surfaces, Shadows/Mixed

---

## Installation

```bash
git clone https://github.com/VTvito/sentinel2-land-cover.git
cd sentinel2-land-cover

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -e .

# Notebook dependencies (recommended)
pip install -e ".[notebooks]"
```

Notes:

- Package name is `satellite-image-analysis`, import name is `satellite_analysis`.

For REST API:
```bash
pip install -e ".[api]"
```

---

## Quick Start

### Notebook (recommended)

1) Install deps (above)

2) Open the notebook and run top-to-bottom:

- [notebooks/city_analysis.ipynb](notebooks/city_analysis.ipynb)

### Python API

```python
from satellite_analysis import analyze, export_report, export_geotiff, export_image

# Analyze
result = analyze("Florence", max_size=2000)
print(f"Confidence: {result.avg_confidence:.1%}")

# Export
export_geotiff(result)
export_report(result, language="en")
export_image(result)  # writes a shareable PNG summary
```

### Command Line

```bash
# Single city
python scripts/analyze_city.py --city Milan

# Batch + export
python scripts/analyze_city.py --cities Milan Rome Florence --export report geotiff image

# Change detection
python scripts/analyze_city.py --city Milan --compare 2023-06 2024-06
```

### REST API

```bash
python scripts/api_server.py  # Starts on localhost:8000
```

```bash
curl -X POST "http://localhost:8000/analyze?city=Milan&max_size=1000"
```
---
## Maintainers

- Architecture overview: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Maintenance playbook: [docs/MAINTENANCE_GUIDE.md](docs/MAINTENANCE_GUIDE.md)

## Configuration

Add your [Copernicus Data Space](https://dataspace.copernicus.eu/) credentials to `config/config.yaml`:

```yaml
sentinel:
  client_id: "your-client-id"
  client_secret: "your-secret"
```

---

## Troubleshooting (fast)

- **Auth / downloads fail**: verify `config/config.yaml` (start from `config/config.yaml.example`).
- **No products found**: widen date range or increase `MAX_CLOUD_COVER`.
- **Too slow**: set `MAX_SIZE=500–1500` and try `CLASSIFIER="kmeans"`.
- **Disk usage**: downloads can be large; delete `data/cities/<city>/previews/` and old `runs/`.
- **Notebook path issues**: run from repo root; paths resolve from project root.

---

## Output

Results saved to `data/cities/{city}/runs/{timestamp}/`:

```
├── labels.npy           # Classification array
├── confidence.npy       # Confidence scores (0-1)
├── run_info.json        # Metadata
├── {city}_classification.tif  # GeoTIFF (if exported)
└── {city}_report.html         # HTML report (if exported)
└── {city}_summary.png         # PNG image summary (if exported)
```

---

## API Reference

```python
from satellite_analysis import (
    # Core
    analyze,           # Single city analysis
    quick_preview,     # Fast low-res preview
    analyze_batch,     # Multiple cities
    
    # Change Detection
    compare,           # Compare two time periods
    
    # Exports
    export_geotiff,    # GIS-ready raster
    export_report,     # HTML report (en/it)
    export_image,      # PNG image summary
    export_json,       # Machine-readable
    
    # Types
    LAND_COVER_CLASSES,
    ChangeResult,
)
```

---

## Requirements

- Python 3.10+
- ~2GB RAM
- Copernicus Data Space credentials (for downloads)

---

## License

MIT License

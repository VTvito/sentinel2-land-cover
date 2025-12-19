# 🛰️ Satellite City Analyzer

**Classify land cover from Sentinel-2 satellite imagery in one command.**

```bash
python scripts/analyze_city.py --city Milan
```

```python
from satellite_analysis import analyze
result = analyze("Milan")
```

---

## Features

- **One-line API**: `analyze("city")` does everything
- **Batch processing**: Analyze multiple cities at once
- **Change detection**: Compare land cover across time periods
- **Multiple exports**: GeoTIFF, HTML reports, JSON
- **CLI & Python API**: Use from terminal or code
- **REST API**: FastAPI server for integrations

**Land Cover Classes**: Water, Vegetation, Urban, Bare Soil, Shadows, Bright Surfaces

---

## Installation

```bash
git clone https://github.com/VTvito/sentinel2-land-cover.git
cd sentinel2-land-cover

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -e .
```

For REST API:
```bash
pip install -e ".[api]"
```

---

## Quick Start

### Python API

```python
from satellite_analysis import analyze, export_report, export_geotiff

# Analyze
result = analyze("Florence", max_size=2000)
print(f"Confidence: {result.avg_confidence:.1%}")

# Export
export_geotiff(result)
export_report(result, language="en")
```

### Command Line

```bash
# Single city
python scripts/analyze_city.py --city Milan

# Batch + export
python scripts/analyze_city.py --cities Milan Rome Florence --export report geotiff

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

## Configuration

Add your [Copernicus Data Space](https://dataspace.copernicus.eu/) credentials to `config/config.yaml`:

```yaml
sentinel:
  client_id: "your-client-id"
  client_secret: "your-secret"
```

---

## Output

Results saved to `data/cities/{city}/runs/{timestamp}/`:

```
├── labels.npy           # Classification array
├── confidence.npy       # Confidence scores (0-1)
├── run_info.json        # Metadata
├── {city}_classification.tif  # GeoTIFF (if exported)
└── {city}_report.html         # HTML report (if exported)
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

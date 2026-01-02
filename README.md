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

Notebook-first UX · one-line API · batch & change detection · GeoTIFF/HTML/JSON/PNG exports · CLI & REST

**Classes**: Water, Vegetation, Bare Soil, Urban, Bright Surfaces, Shadows/Mixed

---

## Installation

```bash
git clone https://github.com/VTvito/sentinel2-land-cover.git && cd sentinel2-land-cover
python -m venv .venv && .venv\Scripts\activate  # Windows (source .venv/bin/activate on Linux/Mac)
pip install -e ".[notebooks]"   # or [api] for REST server
```

---

## Quick Start

**Notebook** → [notebooks/city_analysis.ipynb](notebooks/city_analysis.ipynb)
- **Manual workflow**: Search products → select by index → preview → analyze
- Full control over data quality (cloud cover, acquisition date)

**Python**
```python
from satellite_analysis import analyze, export_geotiff, export_report
result = analyze("Milan", max_size=2000)  # resample_method="cubic", crop_fail_raises=True
export_geotiff(result); export_report(result)
```

**CLI**
```bash
python scripts/analyze_city.py --city Milan --export report geotiff image
python scripts/analyze_city.py --cities Milan Rome --compare 2023-06 2024-06
```

**REST** → `python scripts/api_server.py` then `POST /analyze?city=Milan`
---

## Configuration

Add your [Copernicus Data Space](https://dataspace.copernicus.eu/) credentials to `config/config.yaml`:

```yaml
sentinel:
  client_id: "your-client-id"
  client_secret: "your-secret"
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Auth fails | Check `config/config.yaml` |
| No products | Widen dates / raise cloud cover |
| Slow | `max_size=1000`, `classifier="kmeans"` |
| Disk full | Delete old runs: `data/cities/*/runs/`, clear cache: `data/previews/`, `data/raw/` |

---

## Output

**Analysis results**: `data/cities/{city}/runs/{timestamp}/` → `labels.npy`, `confidence.npy`, `run_info.json`, plus exports (`*.tif`, `*.html`, `*.png`).

**Temporary cache** (safe to delete):
- `data/previews/` → RGB previews from notebook Cell 6
- `data/raw/` → Downloaded Sentinel-2 ZIP files (extracted to `bands/`)

---

## API

`analyze`, `analyze_batch`, `quick_preview`, `compare`, `export_geotiff`, `export_rgb`, `export_report`, `export_image`, `export_json`, `LAND_COVER_CLASSES`

**New in v2.3**: `raw_clusters=True` keeps distinct cluster IDs, `export_rgb()` for publication-quality images.

---

## Roadmap

- [ ] **Temporal comparison** – Change detection between two time periods
- [ ] **Multi-city batch** – Process multiple cities in parallel
- [ ] **Custom classifiers** – Plugin system for ML models

---

Python 3.10+ · MIT License

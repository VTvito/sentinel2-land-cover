# Copilot Instructions - Satellite City Analyzer

## Architecture

Sentinel-2 land cover classification toolkit. Simple API wraps internal pipelines.

```
analyze("Milan") → api.py → CompletePipeline → ConsensusClassifier → AnalysisResult
```

## Public API (always use these)

```python
from satellite_analysis import (
    analyze,           # Main analysis
    analyze_batch,     # Multiple cities
    compare,           # Change detection
    export_geotiff,    # GIS export
    export_report,     # HTML report
    export_image,      # PNG image summary
    export_json,       # Machine-readable
)

result = analyze("Milan", max_size=3000)
print(result.summary())
```

## Key Files

| File | Purpose |
|------|---------|
| `api.py` | Public functions - entry point |
| `exports.py` | GeoTIFF, HTML, JSON exports |
| `change_detection.py` | Temporal comparison |
| `pipelines/complete_pipeline.py` | Orchestrator (internal) |
| `analyzers/classification/` | ConsensusClassifier, SpectralIndices |
| `scripts/analyze_city.py` | CLI |
| `scripts/download_products.py` | Sentinel-2 download |

## Critical Pattern: Path Resolution

```python
# ✅ Always resolve from project_root
path = self._resolve_path("data", "cities", city, "bands")

# ❌ Never use relative paths (breaks from notebooks/)
path = Path("data/cities/milan")
```

## Land Cover Classes

```python
{0: "Water", 1: "Vegetation", 2: "Bare Soil", 3: "Urban", 4: "Bright Surfaces", 5: "Shadows/Mixed"}
```

## Testing

```bash
pytest tests/ -v  # 27 tests
python -c "from satellite_analysis import analyze; print(analyze('Milan', max_size=500).summary())"
```

## CLI

```bash
python scripts/analyze_city.py --city Milan --export report geotiff --lang it
python scripts/analyze_city.py --cities Milan Rome --export json
python scripts/download_products.py --city Milan --cloud-cover 15
```

## Adding Features

1. **Export format** → `exports.py` → expose in `api.py` and `__init__.py`
2. **New classifier** → Add to `analyzers/classification/`, update `CompletePipeline`
3. **New CLI option** → Update `scripts/analyze_city.py` argparse AND document in `__init__.py`

## Dependencies

Core: `numpy`, `rasterio`, `scikit-learn`, `scipy`, `matplotlib`
Optional: `fastapi`, `uvicorn` (for REST API, install with `pip install -e ".[api]"`)
Optional: `fastapi`, `uvicorn` (for REST API, install with `pip install -e ".[api]"`)

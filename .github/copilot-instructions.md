# Copilot Instructions

## Flow

`analyze("Milan")` → `CompletePipeline` → `registry.get_classifier()` → `AnalysisResult`

## Key modules

| Module | Purpose |
|--------|--------|
| `api.py` | Facade: `analyze()`, `export_*()` |
| `exports.py` | GeoTIFF/HTML/JSON/PNG/RGB + `LAND_COVER_CLASSES` |
| `analyzers/classification/registry.py` | Classifier adapters (Strategy pattern) |
| `utils/project_paths.py` | Path resolution from project root |

## Path rule (CRITICAL)

```python
# ✅ Always from project_root
path = self._resolve_path("data", "cities", city)
# ❌ Never relative
path = Path("data/cities/milan")
```

## Classes

`{0:Water, 1:Vegetation, 2:Bare Soil, 3:Urban, 4:Bright Surfaces, 5:Shadows/Mixed}`

## Adding features

- **Export**: `exports.py` → `api.py` → `__init__.py`
- **Classifier**: `analyzers/` → `registry.py` → pipeline

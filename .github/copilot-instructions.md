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

Semantic: `{0:Water, 1:Vegetation, 2:Bare Soil, 3:Urban, 4:Bright Surfaces, 5:Shadows/Mixed}`

Raw mode: `raw_clusters=True` keeps distinct IDs (0 to N-1) without semantic mapping

## Notebook workflow (v2.3+)

**Manual only** (no auto-download):
1. Cell 4: Search products
2. Cell 5: Select `SELECTED_PRODUCT_INDEX` → download
3. Cell 6: Preview RGB (optional)
4. Cell 7: `analyze()` with `auto_download=False`

## Adding features

- **Export**: `exports.py` → `api.py` → `__init__.py`
- **Classifier**: `analyzers/` → `registry.py` → pipeline

# Contributing

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev,notebooks]"
pytest tests/ -v
```

## Rules

- Parameters must affect behavior (no silent ignoring)
- Use `LAND_COVER_CLASSES` from `exports.py`
- Resolve paths from project root (not CWD)

## Module Structure

```
api.py              → Public facade (analyze, exports)
core/               → AnalysisConfig, AnalysisResult, ports
pipelines/          → CompletePipeline, DownloadPipeline
analyzers/          → classification/ (registry, consensus, spectral)
exports.py          → GeoTIFF/HTML/JSON/PNG/RGB + LAND_COVER_CLASSES
utils/              → ProjectPaths, AreaSelector, OutputManager
downloaders/        → Copernicus auth + catalog
validation/         → Metrics, confusion matrix
```

## Common Tasks

| Task | Files |
|------|-------|
| Add export | `exports.py` → `api.py` → `__init__.py` → CLI |
| Add classifier | `analyzers/` → `registry.py` → pipeline → tests |
| Add parameter | `AnalysisConfig` → pipeline → `run_info.json` |

## Tests

```bash
pytest tests/ -v
python -c "from satellite_analysis import analyze; print(analyze('Milan', max_size=300).summary())"
```

## Sharp Edge

Two `AnalysisResult` types: `core.types.AnalysisResult` (public) wraps `pipelines...AnalysisResult` (internal).

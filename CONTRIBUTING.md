# Contributing

Thanks for helping improve Satellite City Analyzer.

## Quick rules (keep the project stable)

- If `analyze(...)` accepts a parameter, it must affect behavior (no silent ignoring).
- Never duplicate land-cover class IDs: use `LAND_COVER_CLASSES` from `src/satellite_analysis/exports.py`.
- Never rely on CWD for paths (must work from notebooks/CLI/API/CI).

## Dev setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e .
pip install -e ".[dev,notebooks]"
```

## Tests

```bash
python -m pytest tests/ -v
```

Smoke run (no credentials required if demo data is present):

```bash
python -c "from satellite_analysis import analyze; print(analyze('Milan', max_size=200, classifier='kmeans').summary())"
```

## Where to change things

- Public API: `src/satellite_analysis/api.py`
- Pipeline orchestration: `src/satellite_analysis/pipelines/complete_pipeline.py`
- Exports: `src/satellite_analysis/exports.py`
- Notebook UX: `notebooks/city_analysis.ipynb`
- CLI: `scripts/analyze_city.py`

## Adding features

- New export format: implement in `exports.py` → re-export in `api.py` and `__init__.py` → wire CLI → update docs.
- New classifier: add required bands + pipeline support → update notebook toggle → add/adjust tests.

## Documentation

- User-facing: `README.md`, `QUICKSTART.md`, `notebooks/city_analysis.ipynb`
- Maintainer-facing: `docs/ARCHITECTURE.md`, `docs/MAINTENANCE_GUIDE.md`

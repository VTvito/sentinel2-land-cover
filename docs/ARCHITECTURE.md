# Architecture (for maintainers and agents)

This project is **notebook-first**: the Jupyter notebook is the primary UX, and all other entry points (CLI, API server) should behave equivalently.

## Primary flow

- `satellite_analysis.analyze("Milan", ...)` (public API)
  - implemented in `src/satellite_analysis/api.py`
  - creates an `AnalysisConfig` (typed, notebook-friendly)
  - runs the internal pipeline (`CompletePipeline`)
  - returns a `core.types.AnalysisResult` (stable result type for exports)

## Entry points

- Notebook: `notebooks/city_analysis.ipynb`
- Python API: `src/satellite_analysis/api.py`
- CLI: `scripts/analyze_city.py`
- REST API (optional): `scripts/api_server.py`

## Internal pipeline

`CompletePipeline` lives in `src/satellite_analysis/pipelines/complete_pipeline.py` and orchestrates:

1. **Area selection**: geocode city + compute bbox (`utils/area_selector.py`)
2. **Data availability**: check local bands for the classifier’s required band set
3. **Download** (if needed): select a suitable Sentinel-2 product + download
4. **Band extraction**: extract required bands into `data/cities/<city>/bands/`
5. **Load & crop**: load bands and optionally crop to the city bbox
6. **Downsample**: reduce resolution if above `max_size`
7. **Classify**: run classifier (`consensus`, `kmeans`, or `spectral`)
8. **Persist outputs**: write `labels.npy`, `confidence.npy`, and `run_info.json` via `utils/output_manager.py`

## Classifiers

The pipeline supports these classifier modes:

- `consensus` (default): `ConsensusClassifier`
- `kmeans`: clustering + heuristics mapping to canonical land cover classes
- `spectral`: indices-based classification; **requires SWIR bands** and resampling/alignment

### Band requirements

- `consensus`, `kmeans`: `B02`, `B03`, `B04`, `B08`
- `spectral`: `B02`, `B03`, `B04`, `B08`, plus `B11`, `B12` (SWIR; requires 20m→10m alignment)

## Canonical land cover classes (single source of truth)

**Do not duplicate class IDs anywhere**.

Canonical mapping is defined in `src/satellite_analysis/exports.py` as `LAND_COVER_CLASSES`:

- `0`: Water
- `1`: Vegetation
- `2`: Bare Soil
- `3`: Urban
- `4`: Bright Surfaces
- `5`: Shadows/Mixed

Everything (notebook legends, exports, plots) must use this mapping.

## Output layout

All outputs are organized under `data/cities/<city>/`.

- `bands/`: extracted bands
- `runs/<timestamp>_<classifier>/`:
  - `labels.npy`
  - `confidence.npy`
  - `run_info.json` (parameters + statistics + output file list)
- `latest/`: copy of the most recent run

## Stability invariants (do not regress)

1. **No silent ignoring of public parameters**
   - If `analyze(...)` accepts a parameter, it must affect pipeline behavior.
2. **Project-root path resolution**
   - Any relative-looking path must resolve from project root, not CWD.
3. **Classifier-specific band checks**
   - Data existence checks must validate the full required band set.
4. **Exports must be consistent**
   - All export formats must use canonical class mapping and include classifier/provenance metadata.

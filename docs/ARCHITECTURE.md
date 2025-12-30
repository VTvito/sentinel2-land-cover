# Architecture

## Entry points

| Entry | File | Usage |
|-------|------|-------|
| Python | `api.py` | `analyze()`, `analyze_batch()`, `compare()` |
| Notebook | `notebooks/city_analysis.ipynb` | Primary UX |
| CLI | `scripts/analyze_city.py` | `--city`, `--export`, `--compare` |
| REST | `scripts/api_server.py` | `POST /analyze` (FastAPI) |

## Flow

`analyze("Milan")` → `AnalysisConfig` → `CompletePipeline` → classifier → `AnalysisResult`

## Internal pipeline

`CompletePipeline` lives in `src/satellite_analysis/pipelines/complete_pipeline.py` and orchestrates:

1. **Area selection**: geocode city + compute bbox (`utils/area_selector.py`)
2. **Data availability**: check local bands for the classifier’s required band set
3. **Download** (if needed): select a suitable Sentinel-2 product + download
4. **Band extraction**: extract required bands into `data/cities/<city>/bands/`
5. **Load & crop**: load bands and optionally crop to the city bbox (`crop_fail_raises=True` by default to avoid silent fallbacks)
6. **Downsample**: reduce resolution if above `max_size` using `resample_method` (`bilinear` default, `cubic` for sharper SWIR alignment)
7. **Classify**: run classifier (`consensus`, `kmeans`, or `spectral`)
8. **Persist outputs**: write `labels.npy`, `confidence.npy`, and `run_info.json` via `utils/output_manager.py`

## Classifiers

| Mode | Bands | Notes |
|------|-------|-------|
| consensus | B02-04, B08 | default, combines kmeans + spectral indices |
| kmeans | B02-04, B08 | fast, supports `raw_clusters=True` for distinct IDs |
| spectral | +B11, B12 | SWIR-based, requires 20m bands |

Registry: `analyzers/classification/registry.py` → implements `core.ports.ClassifierPort`.

### Raw Clusters Mode

`analyze("Milan", classifier="kmeans", raw_clusters=True)` keeps cluster IDs (0 to n-1) without semantic mapping. Useful for exploratory analysis.

## Internals

- **Ports**: `core/ports.py` (classifier + output manager contracts)
- **Paths**: `utils/project_paths.py` (always resolve from project root)

## Classes (canonical)

`0:Water, 1:Vegetation, 2:Bare Soil, 3:Urban, 4:Bright Surfaces, 5:Shadows/Mixed` → `exports.LAND_COVER_CLASSES`

## Invariants

1. No ignored parameters
2. Paths resolve from project root
3. Band checks match classifier
4. Exports use canonical classes

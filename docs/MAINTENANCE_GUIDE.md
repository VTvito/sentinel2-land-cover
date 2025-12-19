# Maintenance Guide (for future AI agents)

This guide is a practical checklist for extending and maintaining the project without reintroducing past problems.

## Where to start

- Public API facade: `src/satellite_analysis/api.py`
- Pipeline orchestrator: `src/satellite_analysis/pipelines/complete_pipeline.py`
- Exports (GeoTIFF / HTML / JSON / PNG): `src/satellite_analysis/exports.py`
- Constants (canonical class IDs): `exports.py::LAND_COVER_CLASSES`
- Output writing + run history: `src/satellite_analysis/utils/output_manager.py`

## Common tasks

### Add a new export format

1. Implement exporter in `src/satellite_analysis/exports.py`.
2. Re-export it via:
   - `src/satellite_analysis/api.py` (thin wrapper)
   - `src/satellite_analysis/__init__.py` (`__all__` + import)
3. If CLI should support it, add it to `scripts/analyze_city.py --export` choices and invoke it.
4. Update docs: README + QUICKSTART + this file.

### Add a new classifier

1. Implement classifier logic under `src/satellite_analysis/analyzers/`.
2. In `CompletePipeline`:
   - Add the classifier name to the allowed `classifier` set.
   - Define required bands in `_bands_needed_for_classifier()`.
   - Ensure download/extract includes the required bands.
   - Ensure load/crop supports band alignment/resampling if resolutions differ.
   - Include classifier name + `bands_used` in run metadata.
3. Ensure notebook exposes the classifier (single variable toggle).
4. Add/adjust tests where appropriate.

### Change or add a parameter to `analyze(...)`

Rules:

- Do not add parameters that are ignored.
- Thread parameters into:
  - `AnalysisConfig` (so exports can read provenance)
  - pipeline invocation
  - `run_info.json` parameters (so the run is reproducible)

## Tests

From repo root:

- `pytest tests/ -v`

Sanity smoke run:

- `python -c "from satellite_analysis import analyze; print(analyze('Milan', max_size=300).summary())"`

## Known sharp edges

- There are currently **two different `AnalysisResult` definitions**:
  - `core.types.AnalysisResult` (public/stable result type returned by `analyze()`)
  - `pipelines.complete_pipeline.AnalysisResult` (internal pipeline result)

This is intentional for now (API wraps pipeline results). If refactoring, avoid breaking exports and notebook expectations.

## Documentation policy

- User-facing: README + QUICKSTART + notebook.
- Maintainer/agent-facing: `docs/ARCHITECTURE.md` + this guide.
- If behavior changes, update docs in the same PR.

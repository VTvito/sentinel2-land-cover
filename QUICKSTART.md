# ⚡ Quick Start

## 1. Install

```bash
pip install -e .

# Notebook dependencies (recommended)
pip install -e ".[notebooks]"
```

## 2. Notebook (recommended)

Open and run:

- [notebooks/city_analysis.ipynb](notebooks/city_analysis.ipynb)

## 3. Analyze (Python API)

```python
from satellite_analysis import analyze

result = analyze("Milan", max_size=2000)
print(result.summary())
```

Or via CLI:
```bash
python scripts/analyze_city.py --city Milan
```

## 4. Export

```python
from satellite_analysis import export_geotiff, export_report, export_image

export_geotiff(result)
export_report(result, language="it")
export_image(result)
```

## 5. Results

Output in `data/cities/milan/runs/{timestamp}/`:
- `labels.npy` - Classification
- `confidence.npy` - Confidence map
- `milan_classification.tif` - GeoTIFF
- `milan_report.html` - HTML report
- `milan_summary.png` - PNG image summary

---

📚 Full docs: [README.md](README.md)
📓 Notebook: [notebooks/city_analysis.ipynb](notebooks/city_analysis.ipynb)

Maintainers: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/MAINTENANCE_GUIDE.md](docs/MAINTENANCE_GUIDE.md)

# 🛰️ Satellite City Analyzer

[![CI](https://github.com/VTvito/satellite_git/actions/workflows/ci.yml/badge.svg)](https://github.com/VTvito/satellite_git/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Classify land cover from Sentinel-2 satellite imagery in one command.**

> 🔄 **Looking for a sentinelsat alternative?** This toolkit works with the new [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) API after the [deprecation of the old Copernicus Open Access Hub](https://github.com/sentinelsat/sentinelsat/issues/607).

Analyze any city using Sentinel-2 data: detect water, vegetation, urban areas, and more.

```bash
python scripts/analyze_city.py --city Milan
```

![Milan Land Cover Classification](docs/example_output.png)
*Sample output: Milan city center land cover classification*

---

## What It Does

| Input | Output |
|-------|--------|
| City name (e.g., "Milan") | Land cover classification map |
| Sentinel-2 satellite bands | Confidence scores per pixel |
| | Validation report |

**6 Land Cover Classes:**
- 🌊 Water
- 🌲 Vegetation  
- 🏜️ Bare Soil
- 🏙️ Urban
- ☀️ Bright Surfaces
- 🌑 Shadows/Mixed

---

## Quick Start (5 minutes)

### 1. Install

```bash
git clone https://github.com/VTvito/satellite_git.git
cd satellite_git

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e .
```

### 2. Run

```bash
# Analyze Milan (requires satellite data - see "Download Your Own Data" below)
python scripts/analyze_city.py --city Milan

# Results in: data/cities/milan/
```

### 3. See Results

```
data/cities/Milan/
├── preview.png              # RGB satellite image
├── analysis/
│   ├── consensus.png        # Classification map
│   └── confidence_map.png   # Confidence heatmap
└── validation/
    └── validation_report.txt
```

---

## Three Ways to Use

### 🖥️ Command Line (Recommended)

```bash
# Basic analysis
python scripts/analyze_city.py --city Rome

# With options
python scripts/analyze_city.py --city Florence --radius 20 --method kmeans
```

### 🌐 Web Interface

```bash
pip install streamlit
streamlit run scripts/app.py
```
Interactive dashboard with multi-city comparison.

### 📓 Jupyter Notebook

```bash
jupyter notebook notebooks/city_analysis.ipynb
```
Step-by-step tutorial.

---

## How It Works

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ City Name   │───▶│ Sentinel-2   │───▶│ Classification  │
│ "Milan"     │    │ Bands        │    │ (6 classes)     │
└─────────────┘    └──────────────┘    └─────────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │ Consensus Classifier │
              │ • K-Means clustering │
              │ • Spectral indices   │
              │ • Confidence scoring │
              └──────────────────────┘
```

**Methods Available:**
- `consensus` (default) - Best accuracy, combines K-Means + Spectral
- `kmeans` - Fast clustering
- `spectral` - Rule-based (water, vegetation, urban detection)

---

## Download Your Own Data

To analyze new cities, you need Sentinel-2 imagery:

### Option A: Copernicus Data Space (Recommended)

1. Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu)
2. Add credentials to `config/config.yaml`:
   ```yaml
   sentinel:
     client_id: "your_client_id"
     client_secret: "your_client_secret"
   ```
3. Download:
   ```bash
   python scripts/download_products.py --city Rome --cloud-cover 10
   ```

### Option B: Manual Download

1. Download from [Copernicus Browser](https://browser.dataspace.copernicus.eu)
2. Extract bands:
   ```bash
   python scripts/extract_all_bands.py your_download.zip data/cities/rome/bands
   ```

---

## Project Structure

```
satellite_git/
├── scripts/           # Command-line tools
│   ├── analyze_city.py        # Main analysis script
│   ├── app.py                 # Web UI (Streamlit)
│   └── validate_classification.py
│
├── notebooks/         # Interactive tutorials
│   └── city_analysis.ipynb
│
├── src/satellite_analysis/    # Core library
│   ├── analyzers/     # Classification algorithms
│   ├── validation/    # Accuracy metrics
│   └── utils/         # Helpers
│
├── data/              # Your data (gitignored)
│   └── cities/
│       └── milan/
│
└── config/            # Configuration
    └── config.yaml    # API credentials
```

---

## Requirements

- Python 3.10+
- ~2GB RAM for analysis
- ~1GB disk per city

**Dependencies:** numpy, rasterio, scikit-learn, matplotlib

---

## Why This Project?

In October 2023, the **Copernicus Open Access Hub was retired**, breaking the widely-used `sentinelsat` library ([see discussion](https://github.com/sentinelsat/sentinelsat/issues/607)). Thousands of researchers and developers lost their workflows overnight.

This toolkit was built to:
1. ✅ Work with the **new Copernicus Data Space Ecosystem API**
2. ✅ Provide **one-command analysis** (no manual band stacking)
3. ✅ Include **validation metrics** (accuracy, kappa, F1-score)
4. ✅ Offer multiple interfaces (CLI, Web UI, Jupyter)

### Migration from Sentinelsat

| Old (sentinelsat) | New (this toolkit) |
|-------------------|-------------------|
| `api.query()` | `python scripts/download_products.py --city Milan` |
| Manual band extraction | Automatic with `extract_all_bands.py` |
| No classification | Built-in Consensus Classifier |
| No validation | Full validation suite |

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - Free for personal and commercial use.

---

## Links

- [Full Documentation](CHANGELOG.md)
- [API Reference](src/satellite_analysis/)
- [Report Issues](https://github.com/VTvito/satellite_git/issues)

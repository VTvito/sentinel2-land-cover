# 🛰️ Satellite City Analyzer

**Classify land cover from satellite imagery in one command.**

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

## License

MIT License - Free for personal and commercial use.

---

## Links

- [Full Documentation](CHANGELOG.md)
- [API Reference](src/satellite_analysis/)
- [Report Issues](https://github.com/VTvito/satellite_git/issues)

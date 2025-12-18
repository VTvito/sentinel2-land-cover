# 🚀 Quick Start Guide

Get started with satellite analysis in **5 minutes**.

**Version 1.0.0** - Production Release

## 📦 Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd satellite_git

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials (optional, for automatic download)
# Edit config/config.yaml with your Sentinel Hub credentials
```

## 🎯 Analyze a City (ONE Command)

```bash
# RECOMMENDED: Consensus classification (K-Means + Spectral combined)
python scripts/analyze_city.py --city Milan --method consensus

# K-Means clustering only
python scripts/analyze_city.py --city Milan --method kmeans

# Spectral classification only
python scripts/analyze_city.py --city Milan --method spectral

# Compare K-Means and Spectral methods
python scripts/analyze_city.py --city Milan --method both
```

**That's it!** Results in `data/cities/<city>/`

## 🔍 Validate Results (NEW in v1.0.0)

```bash
# Generate validation report
python scripts/validate_classification.py --city Milan --report

# Compare all methods
python scripts/validate_classification.py --city Milan --compare
```

## 📊 What You Get

```
data/cities/milan/
├── preview.png              # RGB image (verify correct area!)
├── analysis/
│   ├── consensus.png        # Consensus result (4-panel comparison)
│   ├── confidence_map.png   # Confidence heatmap
│   ├── kmeans.png           # K-Means clustering result
│   └── spectral.png         # Spectral classification result
└── validation/              # NEW in v1.0.0
    ├── consensus_analysis.png
    ├── confusion_matrix.png
    └── validation_report.txt
```

## 🔧 Advanced Usage

### Custom Area

```bash
# Larger radius
python scripts/analyze_city.py --city Rome --radius 20 --method consensus

# Different city
python scripts/analyze_city.py --city Florence --method spectral
```

### Manual Workflow (if automatic download fails)

```bash
# 1. Download Sentinel-2 tile manually
#    → Place in: data/raw/

# 2. Extract bands
python scripts/extract_all_bands.py data/raw/your_tile.zip data/cities/milan/bands

# 3. Crop to city (optional but recommended)
python scripts/crop_city_area.py --city Milan --input data/cities/milan/bands --output data/cities/milan/bands

# 4. Analyze
python scripts/analyze_city.py --city Milan --method consensus
```

## 📚 Methods Available

| Method | Description | Speed | Use Case |
|--------|-------------|-------|----------|
| **consensus** | K-Means + Spectral combined | ⚡ Fast | Best accuracy (RECOMMENDED) |
| **kmeans** | K-Means clustering (6 clusters) | ⚡ Fast | General land cover |
| **spectral** | Rule-based spectral indices | ⚡⚡ Very fast | Water/vegetation/urban |
| **both** | Compare kmeans and spectral | ⚡ Medium | Method comparison |

## 🛠️ Troubleshooting

### "No data found"
→ Use `--download` flag or follow manual workflow

### "City not in tile"
→ Check `preview.png` - you may need a different Sentinel-2 tile

### "Out of memory"
→ Reduce radius: `--radius 10` (default is 15 km)

## 📖 What's Happening Under the Hood?

```python
# 1. Get city coordinates from AreaSelector
bbox, metadata = selector.select_by_city("Milan", radius_km=15)

# 2. Load cropped Sentinel-2 bands
stack = load_bands(['B02', 'B03', 'B04', 'B08'])  # (H, W, 4)

# 3. K-Means clustering
data = stack.reshape(-1, 4)  # Flatten to (N_pixels, 4_bands)
labels = kmeans.fit_predict(data)  # Cluster assignment

# 4. Visualize results
labels_image = labels.reshape(H, W)
plt.imshow(labels_image)
```

**Memory optimization**: Train on sample (2M pixels), predict on all.  
**Speed optimization**: Chunked processing (no OOM, 10x faster).

## 🎓 Next Steps

- **Web UI**: Run `streamlit run scripts/app.py` for interactive interface
- **Jupyter notebooks**: See `notebooks/city_analysis.ipynb` for exploratory analysis
- **Custom analysis**: Check `src/satellite_analysis/analyzers/` for more algorithms
- **Gap analysis**: Read `private_docs/GAP_ANALYSIS.md` for roadmap

## 🌐 Web Interface (Alternative)

Prefer a visual interface? Use the Streamlit web app:

```bash
# Install Streamlit
pip install streamlit

# Launch web UI
streamlit run scripts/app.py
```

## 📓 Interactive Notebooks

For learning and exploration, try Jupyter notebooks:

```bash
# Launch Jupyter
jupyter notebook notebooks/city_analysis.ipynb
```

## 📞 Need Help?

- Check output `preview.png` - does it show the correct city?
- Read error messages - they usually explain what's wrong
- For development: see `README.md` for full documentation

---

**Pro Tip**: Always check `preview.png` before trusting analysis results! 🎨

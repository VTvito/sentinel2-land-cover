# Release Notes - v1.1.0

**Release Date**: December 19, 2025  
**Version**: 1.1.0 - Demo Mode & Improved UX 🎯

---

## 🎉 What's New

### 🎮 Demo Mode

Try the toolkit instantly without downloading satellite data:

```bash
python scripts/analyze_city.py --demo
```

- Sample Milan GeoTIFF bands included
- No Copernicus credentials required
- Perfect for testing and learning

**Files added:**
- `data/demo/milan_sample/bands/` - Sample satellite imagery
- `data/demo/milan_sample/preview.png` - RGB preview

### 🧙 Setup Wizard

Interactive first-time setup for new users:

```bash
python scripts/setup.py
```

**Features:**
- Dependency checking
- Copernicus credentials configuration
- Guided onboarding process

### 📖 Quick Start Guide

New `QUICKSTART.md` - Get started in 2 minutes:
- Demo mode instructions
- Installation steps
- Common commands reference

### 🎨 UX Improvements

- **`-v/--verbose` flag** - Progress output with tqdm
- **`--city` optional** when using `--demo` mode
- **Better error messages** with actionable suggestions

---

## 🐛 Bug Fixes (from v1.0.1)

- Fixed notebook using deprecated `fit_predict()` → `classify()`
- Fixed spectral method failing without SWIR bands
- Fixed CLI encoding issues on Windows
- Removed duplicate `tqdm` import

---

## 📦 Installation

```bash
git clone https://github.com/VTvito/sentinel2-land-cover.git
cd sentinel2-land-cover
pip install -e .

# Try demo mode immediately
python scripts/analyze_city.py --demo
```

---

## 📊 Performance

- **Demo mode**: ~15 seconds to classify
- **Memory**: ~500MB for demo data
- **All tests**: 31/31 passing ✅

---

## 🔧 Technical Details

### New Files
- `scripts/setup.py` (207 lines) - Interactive setup wizard
- `QUICKSTART.md` - 2-minute quick start guide
- `data/demo/milan_sample/bands/*.tif` - Sample GeoTIFF bands

### Modified Files
- `scripts/analyze_city.py` - Added `--demo` and `-v` flags
- `pyproject.toml` - Version bump to 1.1.0
- `CHANGELOG.md` - Added v1.1.0 and v1.0.1 sections
- `README.md` - Added demo mode section
- `docs/AI_AGENT_INSTRUCTIONS.md` - Updated to v1.1.0

---

## 📝 Changelog

**Full changelog:** [CHANGELOG.md](CHANGELOG.md)

### [1.1.0] - Added
- Demo mode with sample data
- Setup wizard (`scripts/setup.py`)
- QUICKSTART.md guide
- `-v/--verbose` flag
- Better error messages

### [1.1.0] - Changed
- `--city` now optional with `--demo`
- Improved user onboarding flow

---

## 🚀 Upgrade Guide

No breaking changes - fully backward compatible.

**To use new features:**

```bash
# Update to latest
git pull origin main
pip install -e .

# Try demo mode
python scripts/analyze_city.py --demo

# Run setup wizard
python scripts/setup.py
```

---

## 🔗 Links

- **Repository**: https://github.com/VTvito/sentinel2-land-cover
- **Issues**: https://github.com/VTvito/sentinel2-land-cover/issues
- **Full Documentation**: [README.md](README.md)

---

## 👥 Contributors

**Vito D'Elia** - [@VTvito](https://github.com/VTvito)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

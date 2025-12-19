# 🎯 v1.1.0 - Demo Mode & Improved UX

## ✨ Highlights

🎮 **Demo Mode** - Try instantly without downloading satellite data  
🧙 **Setup Wizard** - Interactive first-time configuration  
📖 **Quick Start Guide** - Get running in 2 minutes  
🎨 **Better UX** - Verbose mode, improved errors

---

## 🚀 Try It Now

```bash
git clone https://github.com/VTvito/sentinel2-land-cover.git
cd sentinel2-land-cover
pip install -e .

# Run demo (no data download required)
python scripts/analyze_city.py --demo
```

---

## 📦 What's Included

### New Features

- **Demo Mode**: `--demo` flag runs with included sample data
- **Setup Wizard**: `python scripts/setup.py` for guided configuration
- **QUICKSTART.md**: 2-minute getting started guide
- **Verbose Output**: `-v` flag for progress bars
- **Better Errors**: Actionable error messages with suggestions

### Bug Fixes (from v1.0.1)

- Fixed notebook API (`fit_predict()` → `classify()`)
- Fixed spectral classifier without SWIR bands
- Fixed Windows CLI encoding issues
- Removed duplicate imports

---

## 📊 Stats

- **31/31 tests passing** ✅
- **Demo runtime**: ~15 seconds
- **Sample data**: 4 GeoTIFF bands (Milan)
- **No breaking changes** - fully backward compatible

---

## 📝 Full Changelog

See [CHANGELOG.md](https://github.com/VTvito/sentinel2-land-cover/blob/main/CHANGELOG.md) for complete details.

---

## 🙏 Thanks

Thanks to everyone testing and providing feedback!

**Full Release Notes**: [RELEASE_NOTES_v1.1.0.md](https://github.com/VTvito/sentinel2-land-cover/blob/main/RELEASE_NOTES_v1.1.0.md)

# 🗺️ STRUCTURAL METHODOLOGY: City Area Analysis

## Problem Statement

When analyzing satellite data for **urban areas**, using the **full Sentinel-2 tile** (~110km × 110km) causes:
- ❌ **Wrong area**: Image shows lakes, mountains, or countryside instead of city center
- ❌ **Decentered**: City is in a corner, not centered
- ❌ **Wasted resources**: Processing 120M pixels when only 9M are needed (~92% waste)
- ❌ **Slow processing**: K-Means on 120M pixels is much slower than 9M

## Solution: Structural Cropping Methodology

### Overview
```
Full Tile (10980×10980)  →  Crop  →  City Area (3000×3000)
   120M pixels           →  92.6%  →    9M pixels
   ~110km × 110km        → reduction→   ~30km × 30km
```

---

## STEP-BY-STEP METHODOLOGY

### 📋 Prerequisites

1. **AreaSelector configured** with correct city coordinates
   - File: `src/satellite_analysis/utils/area_selector.py`
   - Predefined cities: Milan, Rome, Florence, Venice, etc.
   - Coordinates verified in: `config/area_cache.json`

2. **Full tile downloaded and extracted**
   - Raw ZIP in: `data/raw/product_1.zip`
   - Extracted bands in: `data/processed/product_1/*.jp2`

3. **Cropping script available**
   - Script: `scripts/crop_city_area.py`
   - Dependencies: rasterio, pyproj, numpy

---

### 🎯 WORKFLOW

#### **STEP 1: Verify City Coordinates**

Check if city is in AreaSelector database:

```python
from satellite_analysis.utils import AreaSelector

selector = AreaSelector()
cities = selector.list_predefined_cities()
print(cities)  # ['Milan', 'Rome', 'Florence', ...]
```

If city is **not** in database:
1. Find correct coordinates (e.g., from Google Maps)
2. Add to `src/satellite_analysis/utils/area_selector.py`
3. Test: `selector.select_by_city("NewCity", radius_km=15)`

**For Milan** (example):
```python
PREDEFINED_CITIES = {
    'Milan': {
        'lat': 45.464,
        'lon': 9.190,
        'radius_km': 15,
        'description': 'Milan city center'
    }
}
```

✅ **Verification**: Check that `area_cache.json` has correct coordinates.

---

#### **STEP 2: Download Full Tile**

Use existing download pipeline:

```bash
python scripts/download_products.py \
    --bbox 8.998 45.329 9.382 45.599 \
    --start 2024-06-01 \
    --end 2024-08-31 \
    --limit 10 \
    --max-downloads 1
```

**Output**:
- `data/raw/product_1.zip` (~1.2 GB)

---

#### **STEP 3: Extract Bands**

Extract required bands from ZIP:

```bash
python scripts/extract_all_bands.py \
    data/raw/product_1.zip \
    data/processed/product_1
```

**Output**:
- `data/processed/product_1/B02.jp2` (10980×10980)
- `data/processed/product_1/B03.jp2`
- `data/processed/product_1/B04.jp2`
- `data/processed/product_1/B08.jp2`

---

#### **STEP 4: Crop to City Area** ⭐ **CRITICAL STEP**

Use structural cropping script:

```bash
python scripts/crop_city_area.py \
    --city "Milan" \
    --radius 15 \
    --input-dir "data/processed/product_1" \
    --output-dir "data/processed/milano_centro" \
    --preview \
    --force
```

**What happens**:
1. ✅ Loads city coordinates from AreaSelector (45.464°N, 9.190°E)
2. ✅ Calculates bbox (15km radius around center)
3. ✅ **Verifies tile contains city** (checks if center is inside tile bounds)
4. ✅ Crops each band to bbox
5. ✅ Saves cropped .tif files (3006×2980 pixels, ~92.6% smaller)
6. ✅ Creates RGB preview for visual verification

**Output**:
- `data/processed/milano_centro/B02.tif` (3006×2980)
- `data/processed/milano_centro/B03.tif`
- `data/processed/milano_centro/B04.tif`
- `data/processed/milano_centro/B08.tif`
- `data/processed/milano_centro/rgb_preview.png` ⚠️ **VERIFY THIS!**

---

#### **STEP 5: Verify Cropped Area** ⚠️ **MANDATORY**

Open the preview:

```bash
# Windows
Start-Process "data/processed/milano_centro/rgb_preview.png"

# Linux/Mac
open data/processed/milano_centro/rgb_preview.png
```

**Visual Checklist**:
- [ ] Image shows **city center** (not lakes, mountains, countryside)
- [ ] City is **centered** in image (yellow crosshair at center)
- [ ] Image shows **urban area** (buildings, streets, parks visible)
- [ ] No large water bodies in center (unless city is on coast/lake)

**If image is WRONG**:
→ Go to **TROUBLESHOOTING** section below

---

#### **STEP 6: Run Analysis on Cropped Area**

Update analysis script to use cropped data:

```python
# In scripts/kmeans_milano_optimized.py

base_path = r'c:\TEMP_1\satellite_git\data\processed\milano_centro'
# NOT: data/processed/product_1  ← WRONG (full tile)
```

Run analysis:

```bash
python scripts/kmeans_milano_optimized.py
```

**Expected**:
- ✅ Loads 3006×2980 = 8.9M pixels (not 120M)
- ✅ Faster processing (~10x speedup)
- ✅ Different cluster distribution (city-specific patterns)
- ✅ Output in `data/processed/milano_centro/clustering/`

---

## 🔧 TROUBLESHOOTING

### Problem: "City center is OUTSIDE this tile"

**Cause**: Downloaded tile doesn't cover the city.

**Solution**:
1. Check tile coverage:
   ```python
   # In Python
   import rasterio
   from pyproj import Transformer
   
   with rasterio.open('data/processed/product_1/B02.jp2') as src:
       bounds = src.bounds
       print(f"Tile bounds (UTM): {bounds}")
       
       # Transform city center to UTM
       transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
       x, y = transformer.transform(9.190, 45.464)  # Milan
       print(f"City center (UTM): ({x:.2f}, {y:.2f})")
   ```

2. If city is outside:
   - Download different tile
   - Check Sentinel-2 tile grid: https://maps.esa.int/sentinel2/
   - Find correct tile ID (e.g., T32TNR, T32TMR)
   - Adjust download bbox to cover city

---

### Problem: Image shows lakes instead of city

**Cause**: Wrong coordinates in AreaSelector OR wrong bbox in download.

**Solution**:
1. Verify coordinates:
   ```python
   from satellite_analysis.utils import AreaSelector
   
   selector = AreaSelector()
   bbox, meta = selector.select_by_city("Milan", radius_km=15)
   
   print(f"Center: {meta['center']}")  # Should be (45.464, 9.190)
   print(f"Bbox: {bbox}")
   ```

2. If coordinates are wrong:
   - Find correct coordinates (Google Maps: right-click → "What's here?")
   - Update in `area_selector.py`
   - Re-run cropping

---

### Problem: City is decentered (in corner of image)

**Cause**: Radius too small OR incorrect center coordinates.

**Solution**:
1. Increase radius:
   ```bash
   python scripts/crop_city_area.py --city "Milan" --radius 20  # Larger area
   ```

2. Or use manual coordinates:
   ```bash
   python scripts/crop_city_area.py --lat 45.464 --lon 9.190 --radius 15
   ```

---

### Problem: Cluster distribution looks wrong

**Cause**: Using full tile instead of cropped area.

**Solution**:
1. Check which path is used:
   ```python
   # In analysis script
   base_path = r'data\processed\milano_centro'  # ✅ CORRECT
   # NOT: data\processed\product_1  # ❌ WRONG
   ```

2. Verify file sizes:
   ```bash
   # Cropped (correct): ~3000×3000 = 9M pixels
   # Full (wrong): 10980×10980 = 120M pixels
   ```

---

## 📊 VALIDATION CHECKLIST

Before proceeding to analysis, verify:

- [ ] **Step 1**: ✅ City coordinates verified in `area_cache.json`
- [ ] **Step 2**: ✅ Full tile downloaded (`data/raw/product_1.zip`)
- [ ] **Step 3**: ✅ Bands extracted (`data/processed/product_1/*.jp2`)
- [ ] **Step 4**: ✅ Cropping completed successfully
- [ ] **Step 5**: ✅ Preview shows **CORRECT city center**
- [ ] **Step 6**: ✅ Analysis uses cropped data path

**Only proceed if ALL checks pass!**

---

## 🎯 EXPECTED RESULTS

### Correct Milan Center Analysis

**Cluster Distribution** (from cropped Milano centro):
```
Cluster 1: 29.3%  (dominant urban/vegetation)
Cluster 2: 24.9%  (secondary urban)
Cluster 0: 17.0%  (tertiary)
Cluster 5: 14.5%  (parks/green)
Cluster 3: 12.3%  (mixed)
Cluster 4:  2.0%  (water/shadows)
```

**Visual Features**:
- ✅ Dense urban patterns in center
- ✅ Park areas (Parco Sempione visible)
- ✅ River/canals (Navigli)
- ✅ Symmetrical urban layout

**Wrong Results** (full tile with lakes):
```
Cluster 3: 35.0%  (water dominates!)
Cluster 1: 27.4%
Cluster 2: 17.2%
...
```
→ Distribution dominated by water/countryside → **WRONG AREA**

---

## 🔄 REUSABILITY FOR OTHER CITIES

This methodology works for **ANY city**:

### Example: Florence

```bash
# Step 1: Verify coordinates
python -c "from satellite_analysis.utils import AreaSelector; \
    s = AreaSelector(); \
    bbox, meta = s.select_by_city('Florence', radius_km=12); \
    print(meta)"

# Step 2-3: Download and extract (adjust bbox from Step 1)
python scripts/download_products.py --bbox <bbox> ...
python scripts/extract_all_bands.py ...

# Step 4: Crop
python scripts/crop_city_area.py \
    --city "Florence" \
    --radius 12 \
    --input-dir "data/processed/product_1" \
    --output-dir "data/processed/firenze_centro" \
    --preview --force

# Step 5: Verify preview
Start-Process "data/processed/firenze_centro/rgb_preview.png"

# Step 6: Run analysis
# Update script: base_path = 'data/processed/firenze_centro'
python scripts/kmeans_milano_optimized.py
```

---

## 📈 PERFORMANCE COMPARISON

### Full Tile (BEFORE)
```
Size: 10980×10980 = 120M pixels
Memory: ~480 MB per band
Processing: ~5 minutes K-Means
Disk: ~500 MB (4 bands)
Coverage: Includes lakes, mountains, countryside
```

### Cropped City (AFTER)
```
Size: 3006×2980 = 9M pixels
Memory: ~36 MB per band (92.5% reduction)
Processing: ~30 seconds K-Means (90% faster)
Disk: ~37 MB (4 bands) (92.6% reduction)
Coverage: Only city center (relevant area)
```

**Benefits**:
- ✅ **10x faster** processing
- ✅ **92% less** disk space
- ✅ **Correct** area analyzed
- ✅ **Centered** on city

---

## 🚀 AUTOMATION SCRIPT

For frequent use, create a wrapper script:

```python
# scripts/analyze_city.py

import sys
from pathlib import Path

def analyze_city(city_name, radius_km=15):
    """Complete workflow: download → crop → analyze."""
    
    # 1. Get bbox
    from satellite_analysis.utils import AreaSelector
    selector = AreaSelector()
    bbox, meta = selector.select_by_city(city_name, radius_km=radius_km)
    
    # 2. Download (if needed)
    # ...
    
    # 3. Crop
    import subprocess
    subprocess.run([
        "python", "scripts/crop_city_area.py",
        "--city", city_name,
        "--radius", str(radius_km),
        "--preview", "--force"
    ])
    
    # 4. Verify
    print(f"\n⚠️  VERIFY: Check preview in data/processed/{city_name.lower()}_centro/rgb_preview.png")
    input("Press Enter when verified...")
    
    # 5. Analyze
    # Update paths and run analysis
    # ...

if __name__ == "__main__":
    analyze_city("Milan", radius_km=15)
```

---

## 📚 DOCUMENTATION REFERENCES

- **Area Selection**: `AREA_SELECTION_REPORT.md`
- **City Coordinates**: `config/area_cache.json`
- **Cropping Script**: `scripts/crop_city_area.py`
- **AreaSelector Class**: `src/satellite_analysis/utils/area_selector.py`

---

## ✅ SUCCESS CRITERIA

Analysis is correct when:

1. ✅ Preview shows **city center** (not lakes/countryside)
2. ✅ City is **centered** in preview image
3. ✅ Cluster distribution is **city-specific** (not water-dominated)
4. ✅ Analysis path uses **cropped directory** (`milano_centro`)
5. ✅ File sizes are **~3000×3000** (~9M pixels), not 10980×10980

**If ANY criterion fails → Go to TROUBLESHOOTING**

---

## 🎓 KEY TAKEAWAYS

1. **ALWAYS crop to city area** before analysis
2. **ALWAYS verify preview** before processing
3. **NEVER use full tile** for city-specific analysis
4. **Document coordinates** for reproducibility
5. **Reuse methodology** for any city

**This methodology is STRUCTURAL and REUSABLE for all cities! 🌍**

"""
STRUCTURAL CROP CITY AREA SCRIPT

This script provides a DOCUMENTED METHODOLOGY to crop any city area from Sentinel-2 data.

WORKFLOW:
1. Load full tile (10980x10980 pixels, ~109km x 109km)
2. Select city center using AreaSelector (verified coordinates)
3. Crop to specified radius (e.g., 15km around center)
4. Save cropped bands to new location
5. Verify result with preview

USAGE:
    python scripts/crop_city_area.py --city "Milan" --radius 15
    python scripts/crop_city_area.py --lat 45.464 --lon 9.190 --radius 15

TROUBLESHOOTING:
    If image is wrong/decentered:
    1. Check city coordinates in AreaSelector
    2. Verify tile contains the city (check bounds)
    3. Adjust radius if needed
    4. Use preview to validate before processing
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from satellite_analysis.utils import AreaSelector


def get_city_bbox(city_name: str = None, lat: float = None, lon: float = None, 
                  radius_km: float = 15):
    """
    Get bbox for city using AreaSelector.
    
    Args:
        city_name: City name (e.g., "Milan")
        lat, lon: Manual coordinates (if city not in database)
        radius_km: Radius around center
        
    Returns:
        tuple: (bbox, center_lat, center_lon, metadata)
    """
    selector = AreaSelector()
    
    if city_name:
        bbox, metadata = selector.select_by_city(city_name, radius_km=radius_km)
        print(f"✓ Using predefined coordinates for {city_name}")
    elif lat and lon:
        bbox, metadata = selector.select_by_coordinates(lat, lon, radius_km=radius_km)
        print(f"✓ Using manual coordinates: {lat}°N, {lon}°E")
    else:
        raise ValueError("Must provide either city_name or (lat, lon)")
    
    center = metadata['center']
    
    print(f"\n📍 Area Information:")
    print(f"   City/Area: {metadata.get('city', 'Custom')}")
    print(f"   Center: {center[0]:.4f}°N, {center[1]:.4f}°E")
    print(f"   Radius: {radius_km} km")
    print(f"   Bbox (WGS84): {bbox}")
    print(f"   Area: {metadata['area_km2']:.1f} km²")
    
    return bbox, center[0], center[1], metadata


def check_tile_contains_city(tile_path: str, center_lat: float, center_lon: float):
    """
    Verify that the tile contains the city center.
    
    Args:
        tile_path: Path to any band file (e.g., B02.jp2)
        center_lat, center_lon: City center coordinates
        
    Returns:
        bool: True if tile contains city, False otherwise
    """
    with rasterio.open(tile_path) as src:
        # Transform city coordinates to tile CRS
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(center_lon, center_lat)
        
        # Check if point is within tile bounds
        bounds = src.bounds
        contains = (bounds.left <= x <= bounds.right and 
                   bounds.bottom <= y <= bounds.top)
        
        print(f"\n🗺️  Tile Verification:")
        print(f"   Tile bounds (UTM): {bounds}")
        print(f"   City center (UTM): ({x:.2f}, {y:.2f})")
        print(f"   Status: {'✓ INSIDE TILE' if contains else '✗ OUTSIDE TILE'}")
        
        if not contains:
            print(f"\n⚠️  WARNING: City center is OUTSIDE this tile!")
            print(f"   You need to download a different tile that covers this area.")
            
        return contains


def crop_band(input_path: str, output_path: str, bbox: list, 
              center_lat: float, center_lon: float):
    """
    Crop a single band to the specified bbox.
    
    Args:
        input_path: Input band file
        output_path: Output cropped band file
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        center_lat, center_lon: City center (for verification)
        
    Returns:
        dict: Crop statistics
    """
    with rasterio.open(input_path) as src:
        # Transform bbox to tile CRS
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        
        min_x, min_y = transformer.transform(bbox[0], bbox[1])
        max_x, max_y = transformer.transform(bbox[2], bbox[3])
        
        # Get window for cropping
        window = from_bounds(min_x, min_y, max_x, max_y, src.transform)
        
        # Read cropped data
        cropped_data = src.read(1, window=window)
        
        # Update transform for cropped window
        cropped_transform = src.window_transform(window)
        
        # Write cropped band
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=cropped_data.shape[0],
            width=cropped_data.shape[1],
            count=1,
            dtype=cropped_data.dtype,
            crs=src.crs,
            transform=cropped_transform
        ) as dst:
            dst.write(cropped_data, 1)
        
        return {
            'shape': cropped_data.shape,
            'original_shape': (src.height, src.width),
            'reduction': (1 - (cropped_data.size / (src.height * src.width))) * 100
        }


def create_rgb_preview(output_dir: Path, bands: list):
    """
    Create RGB preview of cropped area for visual verification.
    
    Args:
        output_dir: Directory containing cropped bands
        bands: List of band names to use
    """
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    
    print(f"\n🎨 Creating RGB preview...")
    
    # Load RGB bands
    with rasterio.open(output_dir / 'B04.tif') as src:
        b04 = src.read(1).astype(np.float32)
    with rasterio.open(output_dir / 'B03.tif') as src:
        b03 = src.read(1).astype(np.float32)
    with rasterio.open(output_dir / 'B02.tif') as src:
        b02 = src.read(1).astype(np.float32)
    
    # Stack and normalize
    rgb = np.dstack([b04, b03, b02])
    for i in range(3):
        band = rgb[:, :, i]
        rgb[:, :, i] = (band - band.min()) / (band.max() - band.min() + 1e-10)
    
    # Convert to uint8 and equalize
    rgb = (rgb * 255).astype(np.uint8)
    rgb_pil = Image.fromarray(rgb)
    rgb_pil = ImageOps.equalize(rgb_pil)
    rgb = np.array(rgb_pil)
    
    # Create figure
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb)
    plt.title(f'Cropped Area - RGB True Color\n{output_dir.name}', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Add center crosshair
    h, w = rgb.shape[:2]
    plt.axhline(y=h//2, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=w//2, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    plt.text(w//2, h//2, 'CENTER', color='yellow', fontsize=10,
            ha='center', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    preview_path = output_dir / 'rgb_preview.png'
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {preview_path}")
    
    return preview_path


def main():
    """Main cropping workflow."""
    parser = argparse.ArgumentParser(
        description="Crop Sentinel-2 tile to city area using documented methodology"
    )
    
    # City selection
    parser.add_argument('--city', type=str, help='City name (e.g., "Milan")')
    parser.add_argument('--lat', type=float, help='Manual latitude')
    parser.add_argument('--lon', type=float, help='Manual longitude')
    parser.add_argument('--radius', type=float, default=15, 
                       help='Radius around center in km (default: 15)')
    
    # Paths
    parser.add_argument('--input-dir', type=str, 
                       default='data/processed/product_1',
                       help='Input directory with full tile bands')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed/city_cropped',
                       help='Output directory for cropped bands')
    
    # Bands to process
    parser.add_argument('--bands', type=str, nargs='+',
                       default=['B02', 'B03', 'B04', 'B08'],
                       help='Bands to crop (default: B02 B03 B04 B08)')
    
    # Options
    parser.add_argument('--preview', action='store_true',
                       help='Create RGB preview after cropping')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing output')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🗺️  STRUCTURAL CITY AREA CROPPING")
    print("=" * 70)
    
    # Step 1: Get city bbox and coordinates
    print("\n📍 STEP 1: Area Selection")
    print("-" * 70)
    
    bbox, center_lat, center_lon, metadata = get_city_bbox(
        city_name=args.city,
        lat=args.lat,
        lon=args.lon,
        radius_km=args.radius
    )
    
    # Step 2: Verify tile contains city
    print("\n🔍 STEP 2: Tile Verification")
    print("-" * 70)
    
    input_dir = Path(args.input_dir)
    sample_band = input_dir / f'{args.bands[0]}.jp2'
    
    if not sample_band.exists():
        print(f"❌ Error: Band file not found: {sample_band}")
        print(f"   Make sure you have extracted bands to {input_dir}")
        return False
    
    if not check_tile_contains_city(str(sample_band), center_lat, center_lon):
        print("\n❌ ABORT: City is not in this tile!")
        print("   Solution: Download a different tile that covers this area.")
        return False
    
    # Step 3: Crop bands
    print("\n✂️  STEP 3: Cropping Bands")
    print("-" * 70)
    
    output_dir = Path(args.output_dir)
    
    if output_dir.exists() and not args.force:
        print(f"⚠️  Output directory already exists: {output_dir}")
        print(f"   Use --force to overwrite")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for band in args.bands:
        input_file = input_dir / f'{band}.jp2'
        output_file = output_dir / f'{band}.tif'
        
        if not input_file.exists():
            print(f"   ⚠️  Skipping {band}: file not found")
            continue
        
        print(f"   Processing {band}...", end=' ')
        stats = crop_band(str(input_file), str(output_file), bbox, 
                         center_lat, center_lon)
        
        print(f"✓ {stats['original_shape']} → {stats['shape']} "
              f"({stats['reduction']:.1f}% reduction)")
    
    # Step 4: Create preview
    if args.preview:
        print("\n🎨 STEP 4: Creating Preview")
        print("-" * 70)
        preview_path = create_rgb_preview(output_dir, args.bands)
    
    # Step 5: Summary
    print("\n" + "=" * 70)
    print("✅ CROPPING COMPLETE")
    print("=" * 70)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📦 Cropped bands: {len(args.bands)}")
    print(f"📏 Cropped area: {metadata['area_km2']:.1f} km²")
    
    if args.preview:
        print(f"\n🖼️  Preview: {preview_path}")
        print(f"   ⚠️  IMPORTANT: Verify that the image shows the CORRECT city center!")
        print(f"   If image is wrong, check:")
        print(f"   1. City coordinates in AreaSelector")
        print(f"   2. Downloaded tile coverage")
        print(f"   3. Bbox calculation")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Verify preview image shows correct city area")
    print(f"   2. Run analysis on cropped data: {output_dir}")
    print(f"   3. If wrong, adjust coordinates and re-run")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

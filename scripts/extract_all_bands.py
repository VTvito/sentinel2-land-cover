"""
Extract all bands from Sentinel-2 ZIP product.
"""
import zipfile
from pathlib import Path
import sys

def extract_bands(zip_path: str, output_dir: str):
    """Extract 10m bands (B02, B03, B04, B08) from ZIP."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Extracting bands from: {zip_path}")
    
    bands_10m = ['B02', 'B03', 'B04', 'B08']
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List all files
        all_files = zf.namelist()
        
        # Find 10m bands
        for band in bands_10m:
            band_files = [f for f in all_files if f'_{band}_10m.jp2' in f and 'IMG_DATA/R10m' in f]
            
            if band_files:
                source_file = band_files[0]
                target_file = output_dir / f"{band}.jp2"
                
                # Extract
                with zf.open(source_file) as source:
                    with open(target_file, 'wb') as target:
                        target.write(source.read())
                
                print(f"   ✅ Extracted: {band}.jp2")
            else:
                print(f"   ⚠️  Not found: {band}")
    
    print(f"\n✅ Bands extracted to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_all_bands.py <zip_file> <output_dir>")
        sys.exit(1)
    
    extract_bands(sys.argv[1], sys.argv[2])

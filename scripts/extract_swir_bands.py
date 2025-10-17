"""
Extract B11 and B12 (SWIR) bands from Sentinel-2 ZIP for classification testing.
"""

import sys
sys.path.insert(0, r'c:\TEMP_1\satellite_git\src')

import zipfile
import re
from pathlib import Path


def extract_swir_bands(zip_path, output_dir):
    """Extract B11 and B12 bands from Sentinel-2 ZIP."""
    
    print(f"📦 Opening ZIP: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zp:
        # List all files
        all_files = zp.namelist()
        
        # Find B11 and B12 files (20m resolution)
        swir_files = []
        for file in all_files:
            # Look for B11_20m.jp2 or B12_20m.jp2
            if re.search(r'B1[12]_20m\.jp2$', file):
                swir_files.append(file)
                print(f"   Found: {Path(file).name}")
        
        if not swir_files:
            print("❌ No SWIR bands found in ZIP!")
            return False
        
        # Extract
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for file in swir_files:
            # Extract to output directory
            zp.extract(file, output_path)
            
            # Get just the filename and move to output root
            extracted_path = output_path / file
            final_name = Path(file).name
            final_path = output_path / final_name
            
            # Move file to root if in subdirectory
            if extracted_path != final_path:
                if final_path.exists():
                    print(f"   ⚠️  File already exists, skipping: {final_name}")
                else:
                    extracted_path.rename(final_path)
                    print(f"   ✅ Extracted: {final_name}")
            
            # Clean up empty directories
            try:
                parent = extracted_path.parent
                while parent != output_path and parent.exists():
                    if not any(parent.iterdir()):
                        parent.rmdir()
                    parent = parent.parent
            except:
                pass
        
        print(f"\n✅ SWIR bands extracted to: {output_path}")
        return True


if __name__ == '__main__':
    zip_path = r'c:\TEMP_1\satellite_git\data\raw\product_1.zip'
    output_dir = r'c:\TEMP_1\satellite_git\data\processed_final\product_1'
    
    success = extract_swir_bands(zip_path, output_dir)
    
    if success:
        print("\n💡 Next: Re-run test_classifier_milano.py with real SWIR bands")
    else:
        print("\n❌ Extraction failed")

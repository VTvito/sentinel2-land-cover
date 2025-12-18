"""
🛰️ Satellite City Analyzer - All-in-One Script

ONE COMMAND to analyze any city with Sentinel-2 data.

USAGE:
    # Quick analysis (uses existing data if available)
    python analyze_city.py --city Milan --method kmeans
    
    # Full workflow (download + crop + analyze)
    python analyze_city.py --city Milan --method kmeans --download
    
    # Compare methods
    python analyze_city.py --city Milan --method both

METHODS:
    - kmeans: K-Means clustering (6 clusters, fast)
    - spectral: Spectral indices classification (rule-based)
    - both: Run both and compare

OUTPUT:
    data/cities/<city>/
        ├── bands/              # Cropped bands
        ├── preview.png         # RGB preview
        └── analysis/
            ├── kmeans.png      # K-Means result
            ├── spectral.png    # Spectral result
            └── comparison.png  # Side-by-side (if both)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from satellite_analysis.utils import AreaSelector
from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
from satellite_analysis.analyzers.classification import SpectralIndicesClassifier
from satellite_analysis.preprocessing.normalization import min_max_scale
from satellite_analysis.preprocessing.reshape import reshape_image_to_table, reshape_table_to_image


class CityAnalyzer:
    """All-in-one city analyzer."""
    
    def __init__(self, city_name: str, radius_km: float = 15, use_existing_dir: str = None):
        self.city = city_name
        self.radius = radius_km
        
        # Allow using existing directory (for milano_centro)
        if use_existing_dir:
            self.base_dir = Path(use_existing_dir)
            print(f"\n🌍 City: {city_name}")
            print(f"   Using existing data: {self.base_dir}")
            self.bbox = None
            self.metadata = None
        else:
            self.base_dir = Path(f"data/cities/{city_name.lower()}")
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            # Get city coordinates
            selector = AreaSelector()
            self.bbox, self.metadata = selector.select_by_city(city_name, radius_km=radius_km)
            
            print(f"\n🌍 City: {city_name}")
            print(f"   Center: {self.metadata['center'][0]:.4f}°N, {self.metadata['center'][1]:.4f}°E")
            print(f"   Radius: {radius_km} km")
            print(f"   Area: {self.metadata['area_km2']:.1f} km²")
    
    def check_data_available(self) -> bool:
        """Check if cropped data exists."""
        # Try both locations: base_dir and base_dir/bands
        possible_dirs = [self.base_dir, self.base_dir / "bands"]
        
        required_bands = ['B02', 'B03', 'B04', 'B08']
        
        for check_dir in possible_dirs:
            if not check_dir.exists():
                continue
            
            all_found = True
            for band in required_bands:
                if not (check_dir / f"{band}.tif").exists():
                    all_found = False
                    break
            
            if all_found:
                # Store the correct bands directory
                self.bands_dir = check_dir
                return True
        
        return False
    
    def download_and_prepare(self):
        """Download full tile and crop to city area."""
        print("\n📥 Step 1: Download and Crop")
        print("-" * 60)
        
        print("   ⚠️  Automatic download not yet available.")
        print("   Please use manual workflow:")
        print(f"   1. Download Sentinel-2 tile covering {self.city}")
        print(f"   2. Extract bands: python scripts/extract_all_bands.py <zip> {self.base_dir / 'bands'}")
        print(f"   3. Crop if needed: python scripts/crop_city_area.py --city {self.city}")
        return None
    
    def load_bands(self):
        """Load cropped bands."""
        # Use the bands_dir found by check_data_available
        if not hasattr(self, 'bands_dir'):
            # Fallback: try both locations
            for check_dir in [self.base_dir, self.base_dir / "bands"]:
                if (check_dir / "B02.tif").exists():
                    self.bands_dir = check_dir
                    break
        
        print("\n📂 Loading bands...")
        print(f"   From: {self.bands_dir}")
        
        stack = []
        band_names = ['B02', 'B03', 'B04', 'B08']
        
        for band in band_names:
            band_file = self.bands_dir / f"{band}.tif"
            with rasterio.open(band_file) as src:
                data = src.read(1)
                stack.append(data)
                print(f"   ✅ {band}: {data.shape}")
        
        stack = np.stack(stack, axis=-1)  # (H, W, 4)
        print(f"   Stack shape: {stack.shape}")
        
        return stack, band_names
    
    def create_preview(self, stack):
        """Create RGB preview for visual verification."""
        print("\n🎨 Creating preview...")
        
        # Extract RGB
        rgb = stack[:, :, [2, 1, 0]]  # B04, B03, B02
        
        # Normalize
        rgb_norm = np.zeros_like(rgb, dtype=np.float32)
        for i in range(3):
            band = rgb[:, :, i].astype(np.float32)
            rgb_norm[:, :, i] = (band - band.min()) / (band.max() - band.min() + 1e-10)
        
        # Convert to uint8 and equalize
        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
        from PIL import Image, ImageOps
        rgb_pil = Image.fromarray(rgb_uint8)
        rgb_pil = ImageOps.equalize(rgb_pil)
        rgb_eq = np.array(rgb_pil)
        
        # Save
        preview_path = self.base_dir / "preview.png"
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_eq)
        plt.title(f'{self.city} - RGB True Color', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Add center marker
        h, w = rgb_eq.shape[:2]
        plt.axhline(y=h//2, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
        plt.axvline(x=w//2, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {preview_path}")
        
        return rgb_eq, preview_path
    
    def analyze_kmeans(self, stack, rgb):
        """Run K-Means clustering analysis."""
        print("\n🎯 K-Means Clustering Analysis")
        print("-" * 60)
        
        # Prepare data
        data = reshape_image_to_table(stack)
        data_scaled = min_max_scale(data)
        
        print(f"   Data: {data.shape[0]:,} pixels × {data.shape[1]} bands")
        
        # Sample for training (2M pixels)
        n_total = len(data_scaled)
        sample_size = min(2_000_000, n_total)
        step = max(1, n_total // sample_size)
        sample_indices = np.arange(0, n_total, step)[:sample_size]
        sample = data_scaled[sample_indices]
        
        print(f"   Training on {len(sample):,} samples...")
        
        # Train K-Means++
        clusterer = KMeansPlusPlusClusterer(n_clusters=6, max_iterations=30)
        clusterer.fit(sample)
        
        print(f"   ✅ Training complete (inertia: {clusterer.inertia_:,.0f})")
        
        # Predict all pixels
        print(f"   Classifying all {n_total:,} pixels...")
        labels = clusterer.predict(data_scaled)
        labels_image = reshape_table_to_image(stack.shape, labels)
        
        # Statistics
        unique, counts = np.unique(labels_image, return_counts=True)
        print(f"\n   Cluster Distribution:")
        for cluster_id, count in zip(unique, counts):
            pct = count / labels_image.size * 100
            print(f"      Cluster {cluster_id}: {pct:>5.1f}%")
        
        # Visualize
        output_dir = self.base_dir / "analysis"
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        
        # RGB
        axes[0].imshow(rgb)
        axes[0].set_title(f'{self.city} - RGB True Color', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # K-Means
        from matplotlib.colors import ListedColormap
        colors = plt.cm.tab10(np.linspace(0, 1, 6))
        cmap = ListedColormap(colors)
        
        im = axes[1].imshow(labels_image, cmap=cmap, interpolation='nearest')
        axes[1].set_title(f'{self.city} - K-Means Clustering (K=6)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=f'Cluster {i}') for i in range(6)]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        result_path = output_dir / "kmeans.png"
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   ✅ Saved: {result_path}")
        
        return labels_image, result_path
    
    def analyze_spectral(self, stack, rgb):
        """Run Spectral Indices classification."""
        print("\n🌈 Spectral Indices Classification")
        print("-" * 60)
        
        # Create raster-like dict for classifier
        raster = {
            'B02': stack[:, :, 0],
            'B03': stack[:, :, 1],
            'B04': stack[:, :, 2],
            'B08': stack[:, :, 3]
        }
        
        band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
        
        # Classify
        classifier = SpectralIndicesClassifier()
        labels = classifier.classify(raster, band_indices)
        
        # Statistics
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n   Class Distribution:")
        class_names = classifier.class_names
        for class_id, count in zip(unique, counts):
            pct = count / labels.size * 100
            print(f"      {class_names.get(class_id, f'Class {class_id}')}: {pct:>5.1f}%")
        
        # Visualize
        output_dir = self.base_dir / "analysis"
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        
        # RGB
        axes[0].imshow(rgb)
        axes[0].set_title(f'{self.city} - RGB True Color', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Classification
        from matplotlib.colors import ListedColormap
        colors = [
            '#0000FF',  # Water
            '#90EE90',  # Vegetation
            '#8B4513',  # Bare soil
            '#808080',  # Urban
            '#FFFF00',  # Bright surfaces
            '#000000'   # Shadows
        ]
        cmap = ListedColormap(colors)
        
        im = axes[1].imshow(labels, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
        axes[1].set_title(f'{self.city} - Spectral Classification', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[i], label=class_names[i]) 
            for i in range(6)
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        result_path = output_dir / "spectral.png"
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   ✅ Saved: {result_path}")
        
        return labels, result_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🛰️ Satellite City Analyzer - One command to analyze any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick K-Means analysis (uses existing data)
  python analyze_city.py --city Milan --method kmeans
  
  # Download fresh data and analyze
  python analyze_city.py --city Milan --method kmeans --download
  
  # Compare both methods
  python analyze_city.py --city Milan --method both
  
  # Different city
  python analyze_city.py --city Rome --radius 20 --method spectral --download
        """
    )
    
    parser.add_argument('--city', type=str, required=True,
                       help='City name (e.g., Milan, Rome, Florence)')
    parser.add_argument('--radius', type=float, default=15,
                       help='Radius around city center in km (default: 15)')
    parser.add_argument('--method', type=str, choices=['kmeans', 'spectral', 'both'],
                       default='kmeans', help='Analysis method (default: kmeans)')
    parser.add_argument('--download', action='store_true',
                       help='Force download fresh data (otherwise uses existing)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Use existing data directory (e.g., data/processed/milano_centro)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🛰️  SATELLITE CITY ANALYZER")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = CityAnalyzer(args.city, args.radius, use_existing_dir=args.data_dir)
    
    # Check if data exists
    has_data = analyzer.check_data_available()
    
    if not has_data or args.download:
        if not has_data:
            print("\n⚠️  No existing data found. Will download...")
        else:
            print("\n♻️  Downloading fresh data...")
        
        try:
            analyzer.download_and_prepare()
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("   Using manual workflow instead:")
            print(f"   1. Download tile covering {args.city}")
            print(f"   2. Extract bands to: {analyzer.base_dir / 'bands'}")
            return 1
    else:
        print(f"\n✅ Using existing data in: {analyzer.base_dir / 'bands'}")
    
    # Load bands
    stack, band_names = analyzer.load_bands()
    
    # Create preview
    rgb, preview_path = analyzer.create_preview(stack)
    
    print(f"\n⚠️  VERIFY: Check preview image to confirm correct city area")
    print(f"   Preview: {preview_path}")
    
    # Run analysis
    results = {}
    
    if args.method in ['kmeans', 'both']:
        labels_km, path_km = analyzer.analyze_kmeans(stack, rgb)
        results['kmeans'] = (labels_km, path_km)
    
    if args.method in ['spectral', 'both']:
        labels_sp, path_sp = analyzer.analyze_spectral(stack, rgb)
        results['spectral'] = (labels_sp, path_sp)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output directory: {analyzer.base_dir}")
    print(f"📊 Results:")
    for method, (labels, path) in results.items():
        print(f"   • {method.capitalize()}: {path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

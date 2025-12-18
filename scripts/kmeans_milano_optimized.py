"""
K-Means Clustering on Milano Sentinel-2 Data (OPTIMIZED)
Based on original notebook workflow but with smart optimizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
from satellite_analysis.preprocessing.normalization import min_max_scale
from satellite_analysis.preprocessing.reshape import reshape_image_to_table, reshape_table_to_image


def load_milano_stack(base_path: str):
    """Load Milano Sentinel-2 bands and create stack."""
    print("📂 Loading Milano bands...")
    
    base_path = Path(base_path)
    
    # Check which bands are available
    bands_10m = ['B02', 'B03', 'B04', 'B08']
    
    # Try to find 20m bands (SWIR)
    b11_candidates = list(base_path.glob("*B11*.jp2"))
    b12_candidates = list(base_path.glob("*B12*.jp2"))
    
    has_swir = len(b11_candidates) > 0 and len(b12_candidates) > 0
    
    if has_swir:
        print("   ℹ️  SWIR bands (B11, B12) found - using 6-band stack")
        bands_20m = ['B11', 'B12']
    else:
        print("   ℹ️  SWIR bands not found - using 4-band stack (B02, B03, B04, B08)")
        bands_20m = []
    
    # Read 10m bands (support both .jp2 and .tif)
    stack_10m = []
    for band in bands_10m:
        # Try .tif first (cropped files), then .jp2 (original files)
        band_file = base_path / f"{band}.tif"
        if not band_file.exists():
            band_file = base_path / f"{band}.jp2"
        
        with rasterio.open(band_file) as src:
            data = src.read(1)
            stack_10m.append(data)
            print(f"   ✅ Loaded {band}: {data.shape}")
    
    # Stack 10m bands
    stack_10m = np.stack(stack_10m, axis=-1)  # (H, W, 4)
    
    if has_swir:
        target_shape = stack_10m.shape[:2]
        
        # Read and resample 20m bands
        stack_20m_list = []
        for band_file in [b11_candidates[0], b12_candidates[0]]:
            with rasterio.open(band_file) as src:
                data = src.read(1)
                band_name = 'B11' if 'B11' in str(band_file) else 'B12'
                print(f"   ✅ Loaded {band_name} (SWIR): {data.shape}")
                
                # Resample to 10m resolution
                from scipy.ndimage import zoom
                zoom_factor = target_shape[0] / data.shape[0]
                print(f"   Resampling SWIR from {data.shape} to {target_shape}...")
                data_resampled = zoom(data, zoom_factor, order=1)
                print(f"   ✅ Resampled to {data_resampled.shape}")
                stack_20m_list.append(data_resampled)
        
        stack_20m = np.stack(stack_20m_list, axis=-1)  # (H, W, 2)
        
        # Combine all bands
        stack = np.concatenate([stack_10m, stack_20m], axis=-1)  # (H, W, 6)
        band_names = bands_10m + bands_20m
    else:
        # Only 10m bands
        stack = stack_10m  # (H, W, 4)
        band_names = bands_10m
    
    print(f"   ✅ Final stack shape: {stack.shape}")
    print(f"   ✅ Bands: {band_names}")
    
    return stack, band_names


def smart_sample_for_training(data: np.ndarray, sample_size: int = 1_000_000):
    """
    Smart sampling: Take every N-th pixel to maintain spatial structure.
    Better than random sampling for K-Means training.
    """
    n_total = len(data)
    
    if n_total <= sample_size:
        print(f"   Using all {n_total:,} pixels (no sampling needed)")
        return data, None
    
    # Take every N-th pixel
    step = n_total // sample_size
    indices = np.arange(0, n_total, step)[:sample_size]
    
    print(f"   ⚡ Smart sampling: {len(indices):,} pixels (every {step}-th pixel)")
    
    return data[indices], indices


def main():
    """Main execution - following original notebook workflow."""
    print("=" * 70)
    print("🛰️  K-MEANS CLUSTERING - Milano (Optimized)")
    print("=" * 70)
    
    # Configuration - Using CROPPED city area (not full tile!)
    base_path = r'c:\TEMP_1\satellite_git\data\processed\milano_centro'
    output_dir = Path(base_path) / "clustering"
    output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load and prepare data (like original notebook)
    # ========================================================================
    print("\n📊 Step 1: Loading data...")
    stack, band_names = load_milano_stack(base_path)
    
    # Reshape to table
    print("\n🔧 Step 2: Preparing data...")
    data = reshape_image_to_table(stack)
    print(f"   Original data shape: {data.shape}")
    print(f"   Data range: [{data.min():.0f}, {data.max():.0f}]")
    
    # Scale to [0, 1]
    data_scaled = min_max_scale(data)
    print(f"   Scaled range: [{data_scaled.min():.3f}, {data_scaled.max():.3f}]")
    
    original_shape = stack.shape[:2]
    
    # ========================================================================
    # STEP 3: Elbow Method (on sample)
    # ========================================================================
    print("\n📈 Step 3: Elbow Method...")
    
    # Sample for elbow (very aggressive)
    sample_elbow, _ = smart_sample_for_training(data_scaled, sample_size=500_000)
    
    k_values = [3, 4, 5, 6, 7, 8]
    inertias = []
    
    print("   Testing K values:")
    for k in k_values:
        clusterer = KMeansPlusPlusClusterer(n_clusters=k, max_iterations=20, random_state=42)
        clusterer.fit(sample_elbow)
        inertias.append(clusterer.inertia_)
        print(f"   K={k}: Inertia = {clusterer.inertia_:,.0f}")
    
    # Plot elbow
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title('Elbow Method - Finding Optimal K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    elbow_path = output_dir / 'elbow_curve.png'
    plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: {elbow_path}")
    plt.close()
    
    # ========================================================================
    # STEP 4: K-Means++ clustering (like original notebook)
    # ========================================================================
    optimal_k = 6  # Based on elbow and original notebook
    
    print(f"\n🎯 Step 4: K-Means++ Clustering (K={optimal_k})...")
    print("   Strategy: Train on 2M sample, predict on ALL pixels")
    
    # Train on sample (2M pixels = manageable)
    sample_train, _ = smart_sample_for_training(data_scaled, sample_size=2_000_000)
    
    print(f"   Training K-Means++ on {len(sample_train):,} pixels...")
    clusterer = KMeansPlusPlusClusterer(n_clusters=optimal_k, max_iterations=30, random_state=42)
    clusterer.fit(sample_train)
    
    print(f"   ✅ Training complete!")
    print(f"   Final inertia: {clusterer.inertia_:,.0f}")
    
    # ========================================================================
    # STEP 5: Predict on ALL pixels (using centroids found)
    # ========================================================================
    print(f"\n🔮 Step 5: Classifying ALL pixels...")
    print(f"   Total pixels to classify: {len(data_scaled):,}")
    
    # Predict in chunks to avoid memory issues
    chunk_size = 5_000_000  # 5M pixels per chunk
    labels = np.zeros(len(data_scaled), dtype=np.int32)
    
    for start_idx in range(0, len(data_scaled), chunk_size):
        end_idx = min(start_idx + chunk_size, len(data_scaled))
        chunk = data_scaled[start_idx:end_idx]
        
        # Predict labels for chunk
        labels[start_idx:end_idx] = clusterer.predict(chunk)
        
        progress = end_idx / len(data_scaled) * 100
        print(f"   Progress: {progress:.1f}%", end='\r')
    
    print(f"\n   ✅ Classification complete!")
    
    # Reshape to image
    labels_image = reshape_table_to_image(original_shape, labels)
    
    # ========================================================================
    # STEP 6: Visualization (like original notebook)
    # ========================================================================
    print(f"\n📊 Step 6: Creating visualization...")
    
    # Create RGB True Color for visualization
    # RGB = B04 (Red), B03 (Green), B02 (Blue) - natural colors
    stack_scaled = min_max_scale(stack, uint8=True)
    
    # Check if we have 4 or 6 bands
    n_bands = stack.shape[-1]
    if n_bands == 6:
        rgb = stack_scaled[:, :, [2, 1, 0]]  # B04, B03, B02 (indices in 6-band stack)
    else:  # 4 bands: B02, B03, B04, B08
        rgb = stack_scaled[:, :, [2, 1, 0]]  # B04, B03, B02 (indices: 2=B04, 1=B03, 0=B02)
    
    # Apply histogram equalization for better contrast
    from PIL.ImageOps import equalize
    from PIL import Image
    rgb_pil = Image.fromarray(rgb)
    rgb_pil = equalize(rgb_pil)
    rgb = np.array(rgb_pil)
    
    # Statistics
    print("\n📋 Cluster Statistics:")
    print("-" * 60)
    unique, counts = np.unique(labels_image, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = count / labels_image.size * 100
        print(f"   Cluster {cluster_id}: {count:>12,} pixels ({pct:>5.1f}%)")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # RGB True Color (natural colors)
    axes[0].imshow(rgb)
    axes[0].set_title('RGB True Color (B04-B03-B02)\nNatural Colors', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # K-Means result
    cmap = ListedColormap(plt.cm.tab10.colors[:optimal_k])
    im = axes[1].imshow(labels_image, cmap=cmap, vmin=0, vmax=optimal_k-1)
    axes[1].set_title(f'K-Means++ Clustering (K={optimal_k})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Cluster ID', fontsize=12)
    
    plt.tight_layout()
    
    result_path = output_dir / 'kmeans_result.png'
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Saved: {result_path}")
    plt.close()
    
    # ========================================================================
    # STEP 7: Save results
    # ========================================================================
    print(f"\n💾 Step 7: Saving results...")
    
    # Save labels
    np.save(output_dir / 'kmeans_labels_full.npy', labels_image)
    print(f"   ✅ Labels: kmeans_labels_full.npy")
    
    # Save centroids
    np.save(output_dir / 'kmeans_centroids.npy', clusterer.centroids_)
    print(f"   ✅ Centroids: kmeans_centroids.npy ({clusterer.centroids_.shape})")
    
    # ========================================================================
    # FINAL
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ K-MEANS CLUSTERING COMPLETE")
    print("=" * 70)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Open {result_path.name} to see the clustering result")
    print(f"   2. Compare with SpectralIndicesClassifier output")
    print(f"   3. Implement Consensus Logic")
    print(f"\n🎯 Strategy used:")
    print(f"   - Train on 2M sampled pixels (fast)")
    print(f"   - Predict on ALL 120M pixels (accurate)")
    print(f"   - Result: Full resolution clustering map!")


if __name__ == '__main__':
    main()

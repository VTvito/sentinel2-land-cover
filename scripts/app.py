"""
🛰️ Satellite City Analyzer - Web UI
Simple web interface for analyzing cities with Sentinel-2 imagery.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import warnings

from satellite_analysis.utils import AreaSelector
from satellite_analysis.analyzers.classification import ConsensusClassifier

# Page config
st.set_page_config(
    page_title="🛰️ Satellite City Analyzer",
    page_icon="🛰️",
    layout="wide"
)

# Class definitions
CLASS_NAMES = {
    0: 'Water', 1: 'Vegetation', 2: 'Bare Soil',
    3: 'Urban', 4: 'Bright Surfaces', 5: 'Shadows/Mixed'
}
CLASS_COLORS = ['#0066CC', '#228B22', '#CD853F', '#808080', '#FFD700', '#2F2F2F']


def find_data_directory(city: str, project_root: Path):
    """Find data directory for a city."""
    city_key = city.lower().replace(' ', '_')
    possible_dirs = [
        project_root / f"data/cities/{city_key}/bands",
        project_root / f"data/cities/{city_key}",
        project_root / f"data/processed/{city_key}_centro",
        project_root / f"data/processed/{city_key}",
        project_root / "data/processed/milano_centro",  # Fallback for Milan
    ]
    for d in possible_dirs:
        if d.exists() and (d / "B02.tif").exists():
            return d
    return None


def load_bands(data_dir: Path):
    """Load Sentinel-2 bands."""
    bands = {}
    for band_name in ['B02', 'B03', 'B04', 'B08']:
        with rasterio.open(data_dir / f"{band_name}.tif") as src:
            bands[band_name] = src.read(1)
    return np.stack([bands['B02'], bands['B03'], bands['B04'], bands['B08']], axis=-1)


def create_rgb(stack):
    """Create RGB image from bands."""
    rgb = stack[:, :, [2, 1, 0]].astype(np.float32)
    for i in range(3):
        channel = rgb[:, :, i]
        rgb[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-10)
    return np.clip(rgb * 1.5, 0, 1)


def run_classification(stack):
    """Run consensus classification."""
    classifier = ConsensusClassifier(n_clusters=6, confidence_threshold=0.5, random_state=42)
    band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels, confidence, uncertainty, stats = classifier.classify(stack, band_indices)
    return labels, confidence, stats


def create_classification_figure(rgb, labels, confidence, city):
    """Create visualization figure."""
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title('RGB True Color', fontweight='bold')
    axes[0].axis('off')
    
    # Classification
    axes[1].imshow(labels, cmap=cmap, vmin=0, vmax=5)
    axes[1].set_title('Land Cover Classification', fontweight='bold')
    axes[1].axis('off')
    
    # Confidence
    im = axes[2].imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title('Confidence Map', fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    # Legend
    legend_elements = [Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(6)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9, 
               bbox_to_anchor=(0.5, -0.05))
    
    plt.suptitle(f'{city} - Satellite Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

st.title("🛰️ Satellite City Analyzer")
st.markdown("Analyze land cover in any city using Sentinel-2 satellite imagery.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    city = st.text_input("City Name", value="Milan")
    radius = st.slider("Radius (km)", 5, 30, 15)
    
    st.markdown("---")
    st.markdown("**Available Cities:**")
    
    # Show available data
    data_cities = []
    for d in (PROJECT_ROOT / "data/cities").glob("*/bands"):
        data_cities.append(d.parent.name.title())
    for d in (PROJECT_ROOT / "data/processed").glob("*_centro"):
        data_cities.append(d.name.replace("_centro", "").title())
    
    if data_cities:
        st.markdown("\n".join([f"- {c}" for c in set(data_cities)]))
    else:
        st.markdown("_No data downloaded yet_")

# Main content
if st.button("🔍 Analyze City", type="primary"):
    
    # Find location
    with st.spinner("📍 Finding city location..."):
        selector = AreaSelector()
        try:
            bbox, info = selector.select_by_city(city, radius_km=radius)
            st.success(f"Found **{city}** at {info['center'][0]:.4f}°N, {info['center'][1]:.4f}°E")
        except Exception as e:
            st.error(f"Could not find city: {e}")
            st.stop()
    
    # Find data
    data_dir = find_data_directory(city, PROJECT_ROOT)
    
    if data_dir is None:
        st.warning("⚠️ No satellite data found for this city.")
        st.info(f"""
        **To download data, run in terminal:**
        ```bash
        python scripts/analyze_city.py --city "{city}" --method consensus
        ```
        """)
        st.stop()
    
    st.info(f"📂 Using data from: `{data_dir.relative_to(PROJECT_ROOT)}`")
    
    # Load data
    with st.spinner("📡 Loading satellite bands..."):
        stack = load_bands(data_dir)
        rgb = create_rgb(stack)
    
    st.success(f"Loaded {stack.shape[0]:,} × {stack.shape[1]:,} pixels ({stack.shape[0]*stack.shape[1]:,} total)")
    
    # Run classification
    with st.spinner("🔄 Running classification (this takes ~30 seconds)..."):
        labels, confidence, stats = run_classification(stack)
    
    st.success("✅ Classification complete!")
    
    # Show results
    st.header("📊 Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Agreement", f"{stats['agreement_pct']:.1f}%")
    col2.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")
    col3.metric("Uncertain Pixels", f"{stats['uncertain_pct']:.1f}%")
    
    # Class distribution
    st.subheader("Land Cover Distribution")
    dist_data = {k: v['percentage'] for k, v in stats['class_distribution'].items()}
    st.bar_chart(dist_data)
    
    # Visualization
    st.subheader("Visualization")
    fig = create_classification_figure(rgb, labels, confidence, city)
    st.pyplot(fig)
    
    # Download options
    st.subheader("💾 Download Results")
    col1, col2 = st.columns(2)
    
    # Save labels as npy file for download
    import io
    labels_buffer = io.BytesIO()
    np.save(labels_buffer, labels)
    labels_buffer.seek(0)
    
    col1.download_button(
        "Download Labels (NPY)",
        labels_buffer.getvalue(),
        file_name=f"{city.lower()}_classification.npy",
        mime="application/octet-stream"
    )

st.markdown("---")
st.markdown("*Built with Streamlit • Satellite Analysis Toolkit v1.0.0*")

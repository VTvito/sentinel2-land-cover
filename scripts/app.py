"""
🛰️ Satellite City Analyzer - Web UI
Professional web interface for land cover classification.

Run: streamlit run scripts/app.py
"""

import streamlit as st
import sys
from pathlib import Path
import io

# Setup
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

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Satellite City Analyzer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Land cover classes
CLASSES = {
    0: ('Water', '#0066CC', '🌊'),
    1: ('Vegetation', '#228B22', '🌲'),
    2: ('Bare Soil', '#CD853F', '🏜️'),
    3: ('Urban', '#808080', '🏙️'),
    4: ('Bright Surfaces', '#FFD700', '☀️'),
    5: ('Shadows/Mixed', '#2F2F2F', '🌑'),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def get_available_cities():
    """Get list of cities with available data."""
    cities = set()
    
    # Check data/cities
    cities_dir = PROJECT_ROOT / "data/cities"
    if cities_dir.exists():
        for d in cities_dir.iterdir():
            if d.is_dir() and any(d.glob("**/B02.tif")):
                cities.add(d.name.replace('_', ' ').title())
    
    # Check data/processed
    processed_dir = PROJECT_ROOT / "data/processed"
    if processed_dir.exists():
        for d in processed_dir.iterdir():
            if d.is_dir() and (d / "B02.tif").exists():
                name = d.name.replace('_centro', '').replace('_', ' ').title()
                cities.add(name)
    
    return sorted(cities) if cities else ["Milan"]


def find_data_dir(city: str) -> Path | None:
    """Find data directory for a city."""
    city_key = city.lower().replace(' ', '_')
    
    search_paths = [
        PROJECT_ROOT / f"data/cities/{city_key}/bands",
        PROJECT_ROOT / f"data/cities/{city_key}",
        PROJECT_ROOT / f"data/processed/{city_key}_centro",
        PROJECT_ROOT / f"data/processed/{city_key}",
    ]
    
    # Special case: Milan -> Milano
    if city_key == "milan":
        search_paths.append(PROJECT_ROOT / "data/processed/milano_centro")
    
    for path in search_paths:
        if path.exists() and (path / "B02.tif").exists():
            return path
    return None


@st.cache_data
def load_satellite_data(data_dir_str: str):
    """Load and cache satellite bands."""
    data_dir = Path(data_dir_str)
    bands = {}
    for name in ['B02', 'B03', 'B04', 'B08']:
        with rasterio.open(data_dir / f"{name}.tif") as src:
            bands[name] = src.read(1)
    
    stack = np.stack([bands['B02'], bands['B03'], bands['B04'], bands['B08']], axis=-1)
    
    # Create RGB preview
    rgb = stack[:, :, [2, 1, 0]].astype(np.float32)
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)
    
    return stack, rgb


def run_classification(stack: np.ndarray):
    """Run consensus classification."""
    classifier = ConsensusClassifier(n_clusters=6, random_state=42)
    band_indices = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classifier.classify(stack, band_indices)


def create_results_figure(rgb, labels, confidence, city):
    """Create visualization figure."""
    colors = [CLASSES[i][1] for i in range(6)]
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title('Satellite Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Classification
    axes[1].imshow(labels, cmap=cmap, vmin=0, vmax=5)
    axes[1].set_title('Land Cover', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Confidence
    im = axes[2].imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title('Confidence', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Legend
    legend = [Patch(facecolor=CLASSES[i][1], label=f"{CLASSES[i][2]} {CLASSES[i][0]}") 
              for i in range(6)]
    fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    
    plt.suptitle(f'{city}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render sidebar with city selection."""
    with st.sidebar:
        st.header("🎯 Select City")
        
        available = get_available_cities()
        city = st.selectbox("City", available, index=0)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool classifies land cover from 
        Sentinel-2 satellite imagery into 6 classes:
        
        - 🌊 Water
        - 🌲 Vegetation
        - 🏜️ Bare Soil
        - 🏙️ Urban
        - ☀️ Bright Surfaces
        - 🌑 Shadows
        """)
        
        st.markdown("---")
        st.markdown("**Add New City:**")
        st.code(f"python scripts/analyze_city.py --city <name>", language="bash")
        
        return city


def render_metrics(stats):
    """Render metrics row."""
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "🎯 Agreement",
        f"{stats.get('agreement_pct', stats.get('agreement', 0)*100):.1f}%",
        help="Agreement between K-Means and Spectral methods"
    )
    col2.metric(
        "📊 Confidence",
        f"{stats.get('avg_confidence', stats.get('average_confidence', 0)):.2f}",
        help="Average confidence score (0-1)"
    )
    col3.metric(
        "⚠️ Uncertain",
        f"{stats.get('uncertain_pct', stats.get('uncertain_pixels', 0)*100):.1f}%",
        help="Pixels with low confidence"
    )
    col4.metric(
        "📐 Pixels",
        f"{stats.get('total_pixels', 0):,}",
        help="Total pixels analyzed"
    )


def render_distribution(labels):
    """Render class distribution."""
    st.subheader("📊 Land Cover Distribution")
    
    total = labels.size
    data = {}
    for i in range(6):
        count = np.sum(labels == i)
        name = f"{CLASSES[i][2]} {CLASSES[i][0]}"
        data[name] = count / total * 100
    
    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = [CLASSES[i][1] for i in range(6)]
    bars = ax.barh(list(data.keys()), list(data.values()), color=colors)
    ax.set_xlabel('Percentage (%)')
    ax.set_xlim(0, max(data.values()) * 1.1)
    
    # Add percentage labels
    for bar, pct in zip(bars, data.values()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.title("🛰️ Satellite City Analyzer")
    st.markdown("Land cover classification from Sentinel-2 imagery")
    
    # Sidebar
    city = render_sidebar()
    
    # Find data
    data_dir = find_data_dir(city)
    
    if data_dir is None:
        st.warning(f"⚠️ No satellite data found for **{city}**")
        st.info("""
        **To add this city, run in terminal:**
        ```bash
        python scripts/analyze_city.py --city "{}" --method consensus
        ```
        """.format(city))
        return
    
    # Load data
    with st.spinner("Loading satellite data..."):
        stack, rgb = load_satellite_data(str(data_dir))
    
    st.success(f"✅ Loaded data for **{city}** ({stack.shape[0]:,} × {stack.shape[1]:,} pixels)")
    
    # Preview
    st.subheader("🖼️ Satellite Preview")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_preview, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_title(f'{city} - RGB True Color', fontsize=12)
        st.pyplot(fig_preview)
        plt.close()
    
    with col2:
        st.markdown("### Quick Info")
        st.markdown(f"**Dimensions:** {stack.shape[0]:,} × {stack.shape[1]:,}")
        st.markdown(f"**Total Pixels:** {stack.shape[0]*stack.shape[1]:,}")
        st.markdown(f"**Bands:** B02, B03, B04, B08")
        st.markdown(f"**Source:** {data_dir.relative_to(PROJECT_ROOT)}")
    
    # Classification button
    st.markdown("---")
    
    if st.button("🔬 Run Classification", type="primary", use_container_width=True):
        
        with st.spinner("🔄 Running classification (30-60 seconds)..."):
            labels, confidence, uncertainty, stats = run_classification(stack)
        
        st.success("✅ Classification complete!")
        
        # Results
        st.header("📊 Results")
        
        # Metrics
        stats['total_pixels'] = labels.size
        render_metrics(stats)
        
        # Visualization
        st.subheader("🗺️ Classification Maps")
        fig = create_results_figure(rgb, labels, confidence, city)
        st.pyplot(fig)
        plt.close()
        
        # Distribution
        render_distribution(labels)
        
        # Download
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Labels
        labels_buffer = io.BytesIO()
        np.save(labels_buffer, labels)
        col1.download_button(
            "📥 Labels (NPY)",
            labels_buffer.getvalue(),
            file_name=f"{city.lower()}_labels.npy",
            mime="application/octet-stream"
        )
        
        # Confidence
        conf_buffer = io.BytesIO()
        np.save(conf_buffer, confidence)
        col2.download_button(
            "📥 Confidence (NPY)",
            conf_buffer.getvalue(),
            file_name=f"{city.lower()}_confidence.npy",
            mime="application/octet-stream"
        )
        
        # Figure
        fig_buffer = io.BytesIO()
        fig.savefig(fig_buffer, format='png', dpi=150, bbox_inches='tight')
        col3.download_button(
            "📥 Figure (PNG)",
            fig_buffer.getvalue(),
            file_name=f"{city.lower()}_analysis.png",
            mime="image/png"
        )
    
    # Footer
    st.markdown("---")
    st.caption("Satellite City Analyzer v1.0.0 • Built with Streamlit")


if __name__ == "__main__":
    main()

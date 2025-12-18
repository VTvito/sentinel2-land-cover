"""
🛰️ Satellite City Analyzer - Web UI

Run: streamlit run scripts/app.py
"""

import streamlit as st
import sys
from pathlib import Path
import io
import yaml

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

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Satellite City Analyzer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CLASSES = {
    0: ('Water', '#0066CC', '🌊'),
    1: ('Vegetation', '#228B22', '🌲'),
    2: ('Bare Soil', '#CD853F', '🏜️'),
    3: ('Urban', '#808080', '🏙️'),
    4: ('Bright Surfaces', '#FFD700', '☀️'),
    5: ('Shadows/Mixed', '#2F2F2F', '🌑'),
}

# All Italian cities in AreaSelector
ALL_CITIES = [
    "Milan", "Rome", "Florence", "Venice", "Turin", 
    "Naples", "Bologna", "Genoa", "Palermo", "Verona"
]

# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def get_available_cities() -> list:
    """Get cities that have downloaded data."""
    cities = set()
    
    for data_dir in [PROJECT_ROOT / "data/cities", PROJECT_ROOT / "data/processed"]:
        if data_dir.exists():
            for d in data_dir.iterdir():
                if d.is_dir() and any(d.rglob("B02.tif")):
                    name = d.name.replace('_centro', '').replace('_', ' ').title()
                    # Fix Milano -> Milan
                    if name == "Milano":
                        name = "Milan"
                    cities.add(name)
    
    return sorted(cities)


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
        search_paths.append(PROJECT_ROOT / "data/cities/milano/bands")
    
    for path in search_paths:
        if path.exists() and (path / "B02.tif").exists():
            return path
    return None


def has_credentials() -> bool:
    """Check if API credentials are configured."""
    config_file = PROJECT_ROOT / "config/config.yaml"
    if not config_file.exists():
        return False
    
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        client_id = config.get('sentinel', {}).get('client_id', '')
        return client_id and 'your_' not in client_id.lower()
    except:
        return False


@st.cache_data
def load_satellite_data(data_dir_str: str):
    """Load and cache satellite bands."""
    data_dir = Path(data_dir_str)
    bands = {}
    for name in ['B02', 'B03', 'B04', 'B08']:
        with rasterio.open(data_dir / f"{name}.tif") as src:
            bands[name] = src.read(1)
    
    stack = np.stack([bands['B02'], bands['B03'], bands['B04'], bands['B08']], axis=-1)
    
    # Create RGB
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


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.header("🎯 City Selection")
        
        # Get available cities
        available = get_available_cities()
        
        # Selection mode
        mode = st.radio(
            "Selection Mode",
            ["Single City", "Compare Cities"],
            horizontal=True
        )
        
        if mode == "Single City":
            # Show available cities first, then all cities
            all_options = available + [c for c in ALL_CITIES if c not in available]
            city = st.selectbox(
                "Select City",
                all_options,
                format_func=lambda x: f"✅ {x}" if x in available else f"⬜ {x} (no data)"
            )
            cities = [city]
        else:
            # Multi-select from available only
            if available:
                cities = st.multiselect(
                    "Select Cities to Compare",
                    available,
                    default=available[:2] if len(available) >= 2 else available
                )
            else:
                st.warning("No cities with data available")
                cities = []
        
        st.markdown("---")
        
        # Data status
        st.markdown("### 📊 Data Status")
        for city in ALL_CITIES[:5]:
            status = "✅" if city in available else "⬜"
            st.markdown(f"{status} {city}")
        
        st.markdown("---")
        
        # Credentials status
        st.markdown("### 🔐 API Status")
        if has_credentials():
            st.success("✅ Credentials configured")
        else:
            st.warning("⚠️ No credentials")
            st.caption("Add to config/config.yaml")
        
        return cities, mode


def render_no_data_message(city: str):
    """Show message when no data is available."""
    st.warning(f"⚠️ No satellite data for **{city}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Option 1: CLI Download")
        st.code(f"python scripts/analyze_city.py --city \"{city}\"", language="bash")
    
    with col2:
        st.markdown("### 🌐 Option 2: Manual Download")
        st.markdown("""
        1. Go to [Copernicus Browser](https://browser.dataspace.copernicus.eu)
        2. Search for your city
        3. Download Sentinel-2 L2A product
        4. Extract bands with:
        """)
        st.code(f"python scripts/extract_all_bands.py <file.zip> data/cities/{city.lower()}/bands")


def render_analysis(city: str, data_dir: Path):
    """Render analysis for a single city."""
    # Load data
    with st.spinner(f"Loading {city}..."):
        stack, rgb = load_satellite_data(str(data_dir))
    
    st.success(f"✅ **{city}**: {stack.shape[0]:,} × {stack.shape[1]:,} pixels")
    
    # Show preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb)
        ax.set_title(f'{city}', fontsize=12)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### Info")
        st.markdown(f"**Size:** {stack.shape[0]:,} × {stack.shape[1]:,}")
        st.markdown(f"**Pixels:** {stack.shape[0]*stack.shape[1]:,}")
        st.markdown(f"**Source:** `{data_dir.name}`")
    
    # Classification button
    if st.button(f"🔬 Classify {city}", key=f"btn_{city}", use_container_width=True):
        with st.spinner(f"Classifying {city} (30-60s)..."):
            labels, confidence, uncertainty, stats = run_classification(stack)
        
        st.session_state[f'results_{city}'] = {
            'labels': labels,
            'confidence': confidence,
            'stats': stats,
            'rgb': rgb
        }
    
    # Show results if available
    if f'results_{city}' in st.session_state:
        results = st.session_state[f'results_{city}']
        render_results(city, results)


def render_results(city: str, results: dict):
    """Render classification results."""
    labels = results['labels']
    confidence = results['confidence']
    stats = results['stats']
    rgb = results['rgb']
    
    st.markdown("---")
    st.subheader(f"📊 Results: {city}")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Agreement", f"{stats.get('agreement', 0)*100:.1f}%")
    col2.metric("Confidence", f"{stats.get('average_confidence', 0):.2f}")
    col3.metric("Uncertain", f"{stats.get('uncertain_pixels', 0)*100:.1f}%")
    
    # Visualization
    colors = [CLASSES[i][1] for i in range(6)]
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(rgb)
    axes[0].set_title('Satellite')
    axes[0].axis('off')
    
    axes[1].imshow(labels, cmap=cmap, vmin=0, vmax=5)
    axes[1].set_title('Classification')
    axes[1].axis('off')
    
    im = axes[2].imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title('Confidence')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    legend = [Patch(facecolor=CLASSES[i][1], label=f"{CLASSES[i][2]} {CLASSES[i][0]}") 
              for i in range(6)]
    fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=8, frameon=False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Distribution
    st.markdown("**Land Cover Distribution:**")
    dist_data = {}
    for i in range(6):
        pct = np.sum(labels == i) / labels.size * 100
        dist_data[f"{CLASSES[i][2]} {CLASSES[i][0]}"] = pct
    
    st.bar_chart(dist_data)
    
    # Export
    col1, col2 = st.columns(2)
    
    labels_buf = io.BytesIO()
    np.save(labels_buf, labels)
    col1.download_button(
        "📥 Download Labels",
        labels_buf.getvalue(),
        f"{city.lower()}_labels.npy"
    )
    
    fig_buf = io.BytesIO()
    fig.savefig(fig_buf, format='png', dpi=150, bbox_inches='tight')
    col2.download_button(
        "📥 Download Figure",
        fig_buf.getvalue(),
        f"{city.lower()}_analysis.png"
    )


def render_comparison(cities: list):
    """Render comparison view for multiple cities."""
    st.subheader("🔄 City Comparison")
    
    # Check all cities have data
    valid_cities = []
    for city in cities:
        data_dir = find_data_dir(city)
        if data_dir:
            valid_cities.append((city, data_dir))
        else:
            st.warning(f"⚠️ No data for {city}")
    
    if len(valid_cities) < 2:
        st.info("Select at least 2 cities with data to compare")
        return
    
    # Load all data
    all_data = {}
    for city, data_dir in valid_cities:
        with st.spinner(f"Loading {city}..."):
            stack, rgb = load_satellite_data(str(data_dir))
            all_data[city] = {'stack': stack, 'rgb': rgb, 'dir': data_dir}
    
    # Show previews side by side
    cols = st.columns(len(valid_cities))
    for i, (city, data_dir) in enumerate(valid_cities):
        with cols[i]:
            st.markdown(f"**{city}**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(all_data[city]['rgb'])
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
    
    # Classify all button
    if st.button("🔬 Classify All Cities", use_container_width=True):
        results = {}
        
        for city, data_dir in valid_cities:
            with st.spinner(f"Classifying {city}..."):
                labels, confidence, _, stats = run_classification(all_data[city]['stack'])
                results[city] = {
                    'labels': labels,
                    'confidence': confidence,
                    'stats': stats
                }
        
        # Show comparison table
        st.markdown("### 📊 Comparison Results")
        
        comparison_data = []
        for city in results:
            stats = results[city]['stats']
            labels = results[city]['labels']
            
            row = {'City': city}
            for i in range(6):
                pct = np.sum(labels == i) / labels.size * 100
                row[CLASSES[i][0]] = f"{pct:.1f}%"
            comparison_data.append(row)
        
        st.table(comparison_data)


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.title("🛰️ Satellite City Analyzer")
    st.markdown("Land cover classification from Sentinel-2 satellite imagery")
    
    # Sidebar
    cities, mode = render_sidebar()
    
    if not cities:
        st.info("👈 Select a city from the sidebar")
        return
    
    # Main content
    if mode == "Single City":
        city = cities[0]
        data_dir = find_data_dir(city)
        
        if data_dir is None:
            render_no_data_message(city)
        else:
            render_analysis(city, data_dir)
    else:
        render_comparison(cities)
    
    # Footer
    st.markdown("---")
    st.caption("Satellite City Analyzer v1.0.0")


if __name__ == "__main__":
    main()

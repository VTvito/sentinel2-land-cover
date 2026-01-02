"""Export utilities for satellite analysis results.

Provides export to various formats:
- GeoTIFF: Georeferenced raster (for GIS software)
- HTML Report: Interactive summary with maps and statistics
- PNG: Quick visualization images

Example:
    >>> from satellite_analysis import analyze
    >>> from satellite_analysis.exports import export_geotiff, export_report
    >>> 
    >>> result = analyze("Florence")
    >>> export_geotiff(result, "florence_classification.tif")
    >>> export_report(result, "florence_report.html")
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
import numpy as np
import json

# Land cover class definitions - High-contrast palette for visualization
# Designed for maximum distinguishability and academic publication
LAND_COVER_CLASSES = {
    0: {"name": "Water", "color": "#1E90FF", "rgb": (30, 144, 255)},       # Dodger blue - clear water
    1: {"name": "Vegetation", "color": "#228B22", "rgb": (34, 139, 34)},   # Forest green - natural vegetation
    2: {"name": "Bare Soil", "color": "#CD853F", "rgb": (205, 133, 63)},   # Peru - exposed earth
    3: {"name": "Urban", "color": "#DC143C", "rgb": (220, 20, 60)},        # Crimson - built-up (high contrast)
    4: {"name": "Bright Surfaces", "color": "#FFD700", "rgb": (255, 215, 0)},  # Gold - high reflectance
    5: {"name": "Shadows/Mixed", "color": "#4B0082", "rgb": (75, 0, 130)}, # Indigo - uncertain/shadow
}


def export_geotiff(
    result: "AnalysisResult",
    output_path: Optional[Union[str, Path]] = None,
    *,
    include_confidence: bool = True,
    crs: str = "EPSG:32632",  # UTM zone 32N (default for Italy)
) -> Path:
    """Export classification result as georeferenced GeoTIFF.
    
    Creates a GeoTIFF file that can be opened in QGIS, ArcGIS, Google Earth, etc.
    
    Args:
        result: AnalysisResult from analyze()
        output_path: Output file path. If None, saves to result.output_dir
        include_confidence: If True, creates a second band with confidence values
        crs: Coordinate reference system (default: UTM 32N for Italy)
        
    Returns:
        Path to the created GeoTIFF file
        
    Example:
        >>> result = analyze("Florence")
        >>> path = export_geotiff(result, "output/florence.tif")
        >>> print(f"Saved to: {path}")
    """
    import rasterio
    from rasterio.transform import from_bounds
    
    # Determine output path
    if output_path is None:
        output_path = result.output_dir / f"{result.city.lower()}_classification.tif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get labels and confidence
    labels = result.labels
    confidence = result.confidence
    height, width = labels.shape
    
    # Try to get bounds from original data, otherwise use dummy bounds
    bounds = _get_bounds_from_result(result)
    
    # Create transform
    transform = from_bounds(
        bounds['west'], bounds['south'], 
        bounds['east'], bounds['north'],
        width, height
    )
    
    # Determine number of bands
    count = 2 if include_confidence else 1
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype='uint8' if count == 1 else 'float32',
        crs=crs,
        transform=transform,
        compress='lzw',
    ) as dst:
        # Band 1: Classification labels
        dst.write(labels.astype('uint8'), 1)
        dst.set_band_description(1, 'Land Cover Classification')
        
        # Band 2: Confidence (optional)
        if include_confidence:
            # Scale confidence to 0-255 for smaller file
            conf_scaled = (confidence * 255).astype('uint8')
            dst.write(conf_scaled, 2)
            dst.set_band_description(2, 'Classification Confidence (0-255)')
        
        # Add metadata
        dst.update_tags(
            city=result.city,
            classifier=result.config_summary.get('classifier', 'consensus'),
            timestamp=datetime.now().isoformat(),
            avg_confidence=f"{result.avg_confidence:.2%}",
            software="Satellite City Analyzer"
        )
        
        # Add class descriptions
        for cls_id, cls_info in LAND_COVER_CLASSES.items():
            dst.update_tags(1, **{f"class_{cls_id}": cls_info['name']})
    
    return output_path


def export_colored_geotiff(
    result: "AnalysisResult",
    output_path: Optional[Union[str, Path]] = None,
    crs: str = "EPSG:32632",
) -> Path:
    """Export classification as RGB colored GeoTIFF.
    
    Creates a visually appealing GeoTIFF with colors for each land cover class.
    
    Args:
        result: AnalysisResult from analyze()
        output_path: Output file path
        crs: Coordinate reference system
        
    Returns:
        Path to the created GeoTIFF file
    """
    import rasterio
    from rasterio.transform import from_bounds
    
    # Determine output path
    if output_path is None:
        output_path = result.output_dir / f"{result.city.lower()}_classification_rgb.tif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    labels = result.labels
    height, width = labels.shape
    bounds = _get_bounds_from_result(result)
    
    # Create RGB image
    rgb = np.zeros((3, height, width), dtype='uint8')
    for cls_id, cls_info in LAND_COVER_CLASSES.items():
        mask = labels == cls_id
        rgb[0, mask] = cls_info['rgb'][0]  # R
        rgb[1, mask] = cls_info['rgb'][1]  # G
        rgb[2, mask] = cls_info['rgb'][2]  # B
    
    # Create transform
    transform = from_bounds(
        bounds['west'], bounds['south'], 
        bounds['east'], bounds['north'],
        width, height
    )
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype='uint8',
        crs=crs,
        transform=transform,
        compress='lzw',
    ) as dst:
        dst.write(rgb)
        dst.set_band_description(1, 'Red')
        dst.set_band_description(2, 'Green')
        dst.set_band_description(3, 'Blue')
    
    return output_path


def export_rgb(
    city: str,
    output_path: Optional[Union[str, Path]] = None,
    *,
    bands_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    stretch_percentile: int = 2,
    include_fcc: bool = True,
    include_ndvi: bool = True,
    dpi: int = 300,
) -> Dict[str, Path]:
    """Export RGB True Color and other band composites from satellite data.
    
    Creates publication-quality images from the original satellite bands:
    - RGB True Color (B04, B03, B02)
    - False Color Composite (B08, B04, B03) - highlights vegetation
    - NDVI visualization
    
    Args:
        city: City name
        output_path: Base output path (without extension). If None, uses city data dir
        bands_dir: Directory with band files. If None, auto-detected
        project_root: Project root override
        stretch_percentile: Percentile for histogram stretch (default: 2%)
        include_fcc: Include False Color Composite
        include_ndvi: Include NDVI visualization
        dpi: Output resolution (default: 300 for publication)
        
    Returns:
        Dictionary with paths to created files
        
    Example:
        >>> paths = export_rgb("Milan")
        >>> print(paths["rgb"])  # Path to RGB image
        >>> print(paths["fcc"])  # Path to FCC image
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import rasterio
    
    # Find project root
    if project_root is None:
        from satellite_analysis.utils.project_paths import ProjectPaths
        paths = ProjectPaths()
        project_root = paths.project_root
    
    # Find bands directory
    if bands_dir is None:
        bands_dir = project_root / "data" / "cities" / city.lower() / "bands"
    bands_dir = Path(bands_dir)
    
    if not bands_dir.exists():
        raise FileNotFoundError(f"Bands directory not found: {bands_dir}")
    
    # Determine output directory
    if output_path is None:
        output_dir = bands_dir.parent / "exports"
    else:
        output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = city.lower()
    
    # Load bands
    def load_band(band_name: str) -> np.ndarray:
        # Try direct files first
        for ext in [".jp2", ".tif", ".TIF"]:
            path = bands_dir / f"{band_name}{ext}"
            if path.exists():
                with rasterio.open(path) as src:
                    return src.read(1).astype(np.float32)
        
        # Try recursive search (files might be in subdirectories like .SAFE)
        for ext in [".jp2", ".tif", ".TIF"]:
            matches = list(bands_dir.rglob(f"**/{band_name}{ext}"))
            if matches:
                path = matches[0]  # Use first match
                with rasterio.open(path) as src:
                    return src.read(1).astype(np.float32)
        
        raise FileNotFoundError(f"Band {band_name} not found in {bands_dir}")
    
    # Histogram stretch function
    def stretch(arr: np.ndarray, percentile: int = 2) -> np.ndarray:
        p_low = np.nanpercentile(arr, percentile)
        p_high = np.nanpercentile(arr, 100 - percentile)
        stretched = np.clip((arr - p_low) / (p_high - p_low + 1e-6), 0, 1)
        return (stretched * 255).astype(np.uint8)
    
    outputs = {}
    
    # Load RGB bands (B04=Red, B03=Green, B02=Blue)
    b04 = load_band("B04")  # Red
    b03 = load_band("B03")  # Green
    b02 = load_band("B02")  # Blue
    
    # Create RGB True Color
    rgb = np.stack([stretch(b04), stretch(b03), stretch(b02)], axis=-1)
    
    rgb_path = output_dir / f"{base_name}_rgb.png"
    plt.figure(figsize=(12, 12), dpi=dpi)
    plt.imshow(rgb)
    plt.title(f"{city} - RGB True Color (Sentinel-2)", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(rgb_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    outputs["rgb"] = rgb_path
    
    # Also save as high-res GeoTIFF if possible
    try:
        rgb_tif_path = output_dir / f"{base_name}_rgb.tif"
        # Get transform from one of the bands
        with rasterio.open(bands_dir / "B04.jp2") as src:
            transform = src.transform
            crs = src.crs
        
        with rasterio.open(
            rgb_tif_path, 'w', driver='GTiff',
            height=rgb.shape[0], width=rgb.shape[1], count=3,
            dtype='uint8', crs=crs, transform=transform, compress='lzw'
        ) as dst:
            for i in range(3):
                dst.write(rgb[:, :, i], i + 1)
        outputs["rgb_geotiff"] = rgb_tif_path
    except Exception:
        pass  # Skip GeoTIFF if bands don't have georef info
    
    # False Color Composite (vegetation emphasis)
    if include_fcc:
        try:
            b08 = load_band("B08")  # NIR
            fcc = np.stack([stretch(b08), stretch(b04), stretch(b03)], axis=-1)
            
            fcc_path = output_dir / f"{base_name}_fcc.png"
            plt.figure(figsize=(12, 12), dpi=dpi)
            plt.imshow(fcc)
            plt.title(f"{city} - False Color Composite (NIR-R-G)", fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(fcc_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            outputs["fcc"] = fcc_path
        except FileNotFoundError:
            pass
    
    # NDVI visualization
    if include_ndvi:
        try:
            b08 = load_band("B08")  # NIR
            eps = 1e-6
            ndvi = (b08 - b04) / (b08 + b04 + eps)
            
            # NDVI colormap: brown → yellow → green
            colors = ['#8B4513', '#CD853F', '#F4A460', '#FFFFE0', '#90EE90', '#228B22', '#006400']
            cmap = LinearSegmentedColormap.from_list('ndvi', colors, N=256)
            
            ndvi_path = output_dir / f"{base_name}_ndvi.png"
            fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
            im = ax.imshow(ndvi, cmap=cmap, vmin=-0.5, vmax=0.9)
            ax.set_title(f"{city} - NDVI (Vegetation Index)", fontsize=14, fontweight='bold')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI', fontsize=12)
            plt.tight_layout()
            plt.savefig(ndvi_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            outputs["ndvi"] = ndvi_path
        except FileNotFoundError:
            pass
    
    return outputs


def export_report(
    result: "AnalysisResult",
    output_path: Optional[Union[str, Path]] = None,
    *,
    title: Optional[str] = None,
    include_map: bool = True,
    include_confidence_map: bool = True,
    language: str = "en",
) -> Path:
    """Generate HTML report with analysis results.
    
    Creates a self-contained HTML file with:
    - Land cover classification map
    - Confidence map
    - Class distribution chart
    - Statistics summary
    
    Args:
        result: AnalysisResult from analyze()
        output_path: Output file path. If None, saves to result.output_dir
        title: Report title (default: "Land Cover Analysis - {city}")
        include_map: Include classification map image
        include_confidence_map: Include confidence heatmap
        language: Report language ("en" or "it")
        
    Returns:
        Path to the created HTML file
        
    Example:
        >>> result = analyze("Milan")
        >>> export_report(result, "milan_report.html")
    """
    import base64
    from io import BytesIO
    
    # Determine output path
    if output_path is None:
        output_path = result.output_dir / f"{result.city.lower()}_report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate title
    if title is None:
        title = f"Land Cover Analysis - {result.city}"
    
    # Get class distribution
    distribution = result.class_distribution()
    
    # Generate map images as base64
    map_img_b64 = ""
    conf_img_b64 = ""
    
    if include_map:
        map_img_b64 = _generate_classification_image_b64(result.labels)
    
    if include_confidence_map:
        conf_img_b64 = _generate_confidence_image_b64(result.confidence)
    
    # Generate HTML
    html = _generate_html_report(
        title=title,
        city=result.city,
        result=result,
        distribution=distribution,
        map_img_b64=map_img_b64,
        conf_img_b64=conf_img_b64,
        language=language,
    )
    
    # Write HTML
    output_path.write_text(html, encoding='utf-8')
    
    return output_path


def export_image(
    result: "AnalysisResult",
    output_path: Optional[Union[str, Path]] = None,
    *,
    title: Optional[str] = None,
    include_confidence_map: bool = False,
    dpi: int = 150,
) -> Path:
    """Export a quick-look PNG image (classification + stats).

    The image is intended for sharing in docs/issues/chats where an interactive
    HTML report is too heavy.

    Args:
        result: AnalysisResult from analyze()
        output_path: Output file path. If None, saves to result.output_dir
        title: Optional image title
        include_confidence_map: If True, includes a confidence heatmap panel
        dpi: PNG resolution

    Returns:
        Path to the created PNG file
    """
    import matplotlib

    # Ensure headless-safe rendering (e.g. CI, servers)
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if output_path is None:
        output_path = result.output_dir / f"{result.city.lower()}_summary.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"Land Cover Summary - {result.city}"

    labels = result.labels
    confidence = result.confidence
    distribution = result.class_distribution()

    # Create colormap (ensure at least the canonical 0..5 classes exist)
    max_class = int(max(labels.max(), 5))
    colors = [
        LAND_COVER_CLASSES.get(i, {"color": "#808080"})["color"]
        for i in range(max_class + 1)
    ]
    cmap = ListedColormap(colors)

    # Layout: map (left) + distribution + stats (right)
    if include_confidence_map:
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1, 1])
        ax_map = fig.add_subplot(gs[:, 0])
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 1.0])
        ax_map = fig.add_subplot(gs[0, 0])
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[0, 2])

    fig.suptitle(title, fontsize=16)

    # Classification map
    ax_map.imshow(labels, cmap=cmap, interpolation="nearest")
    ax_map.set_title("Classification", fontsize=12)
    ax_map.axis("off")

    # Distribution bar chart (only classes present)
    present = [(cls_id, data) for cls_id, data in sorted(distribution.items())]
    cls_names = [LAND_COVER_CLASSES.get(cls_id, {"name": f"Class {cls_id}"})["name"] for cls_id, _ in present]
    percents = [data["percentage"] for _, data in present]
    bar_colors = [LAND_COVER_CLASSES.get(cls_id, {"color": "#808080"})["color"] for cls_id, _ in present]

    ax_dist.barh(range(len(present)), percents, color=bar_colors)
    ax_dist.set_yticks(range(len(present)), labels=cls_names)
    ax_dist.invert_yaxis()
    ax_dist.set_xlim(0, 100)
    ax_dist.set_xlabel("%")
    ax_dist.set_title("Land Cover Distribution", fontsize=12)
    ax_dist.grid(axis="x", alpha=0.2)

    # Stats panel
    ax_stats.axis("off")
    classifier = result.config_summary.get("classifier", "consensus")
    lines = [
        f"City: {result.city}",
        f"Classifier: {classifier}",
        f"Processed shape: {result.processed_shape[0]}×{result.processed_shape[1]}",
        f"Average confidence: {result.avg_confidence:.1%}",
        f"Total pixels: {result.total_pixels:,}",
    ]
    if "start_date" in result.config_summary and "end_date" in result.config_summary:
        lines.insert(2, f"Date range: {result.config_summary.get('start_date')} → {result.config_summary.get('end_date')}")
    ax_stats.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)

    if include_confidence_map:
        # Add confidence map as an inset inside the map axis (bottom-right)
        inset = ax_map.inset_axes([0.62, 0.02, 0.36, 0.36])
        im = inset.imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
        inset.set_title("Confidence", fontsize=10)
        inset.axis("off")
        cax = ax_map.inset_axes([0.62, 0.0, 0.36, 0.02])
        fig.colorbar(im, cax=cax, orientation="horizontal")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def export_json(
    result: "AnalysisResult",
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Export analysis results as JSON (metadata only, no raster data).
    
    Useful for programmatic access to statistics and metadata.
    
    Args:
        result: AnalysisResult from analyze()
        output_path: Output file path
        
    Returns:
        Path to the created JSON file
    """
    if output_path is None:
        output_path = result.output_dir / f"{result.city.lower()}_results.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "city": result.city,
        "timestamp": datetime.now().isoformat(),
        "shape": {
            "original": result.original_shape,
            "processed": result.processed_shape,
            "was_downsampled": result.was_downsampled,
        },
        "statistics": {
            "total_pixels": result.total_pixels,
            "avg_confidence": result.avg_confidence,
            "execution_time_seconds": result.execution_time,
        },
        "class_distribution": {
            str(cls_id): {
                "name": LAND_COVER_CLASSES.get(cls_id, {}).get("name", f"Class {cls_id}"),
                **data
            }
            for cls_id, data in result.class_distribution().items()
        },
        "config": result.config_summary,
    }
    
    output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    
    return output_path


# =============================================================================
# Helper Functions
# =============================================================================

def _get_bounds_from_result(result: "AnalysisResult") -> Dict[str, float]:
    """Get geographic bounds from result or use defaults."""
    # Try to get from config or metadata
    config = result.config_summary
    
    # Default bounds for common Italian cities (UTM 32N approximate)
    city_bounds = {
        "florence": {"west": 667000, "south": 4843000, "east": 697000, "north": 4873000},
        "milan": {"west": 503000, "south": 5023000, "east": 533000, "north": 5053000},
        "rome": {"west": 280000, "south": 4617000, "east": 310000, "north": 4647000},
        "venice": {"west": 758000, "south": 5023000, "east": 788000, "north": 5053000},
    }
    
    city_key = result.city.lower()
    if city_key in city_bounds:
        return city_bounds[city_key]
    
    # Default: use pixel coordinates scaled
    h, w = result.labels.shape
    return {"west": 0, "south": 0, "east": w * 10, "north": h * 10}  # 10m per pixel


def _generate_classification_image_b64(labels: np.ndarray) -> str:
    """Generate base64-encoded PNG of classification map."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from io import BytesIO
    import base64
    
    # Create colormap
    colors = [LAND_COVER_CLASSES.get(i, {"color": "#808080"})["color"] 
              for i in range(max(labels.max() + 1, 6))]
    cmap = ListedColormap(colors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(labels, cmap=cmap, interpolation='nearest')
    ax.axis('off')
    
    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _generate_confidence_image_b64(confidence: np.ndarray) -> str:
    """Generate base64-encoded PNG of confidence map."""
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Confidence')
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _generate_html_report(
    title: str,
    city: str,
    result: "AnalysisResult",
    distribution: Dict,
    map_img_b64: str,
    conf_img_b64: str,
    language: str = "en",
) -> str:
    """Generate HTML report content."""
    
    # Labels based on language
    labels = {
        "en": {
            "title": title,
            "summary": "Analysis Summary",
            "city": "City",
            "date": "Analysis Date",
            "resolution": "Resolution",
            "pixels": "Total Pixels",
            "confidence": "Average Confidence",
            "time": "Processing Time",
            "distribution": "Land Cover Distribution",
            "class_name": "Class",
            "percentage": "Percentage",
            "pixel_count": "Pixels",
            "classification_map": "Land Cover Classification Map",
            "confidence_map": "Classification Confidence Map",
            "legend": "Legend",
            "generated": "Generated by Satellite City Analyzer",
        },
        "it": {
            "title": title,
            "summary": "Riepilogo Analisi",
            "city": "Città",
            "date": "Data Analisi",
            "resolution": "Risoluzione",
            "pixels": "Pixel Totali",
            "confidence": "Confidenza Media",
            "time": "Tempo Elaborazione",
            "distribution": "Distribuzione Copertura del Suolo",
            "class_name": "Classe",
            "percentage": "Percentuale",
            "pixel_count": "Pixel",
            "classification_map": "Mappa Classificazione Copertura del Suolo",
            "confidence_map": "Mappa Confidenza Classificazione",
            "legend": "Legenda",
            "generated": "Generato da Satellite City Analyzer",
        }
    }
    
    L = labels.get(language, labels["en"])
    
    # Generate distribution table rows
    dist_rows = ""
    for cls_id, data in sorted(distribution.items()):
        cls_info = LAND_COVER_CLASSES.get(cls_id, {"name": f"Class {cls_id}", "color": "#808080"})
        bar_width = data['percentage']
        dist_rows += f"""
        <tr>
            <td>
                <span class="color-box" style="background-color: {cls_info['color']}"></span>
                {cls_info['name']}
            </td>
            <td>
                <div class="bar-container">
                    <div class="bar" style="width: {bar_width}%; background-color: {cls_info['color']}"></div>
                </div>
            </td>
            <td class="num">{data['percentage']:.1f}%</td>
            <td class="num">{data['count']:,}</td>
        </tr>
        """
    
    # Generate legend items
    legend_items = ""
    for cls_id, cls_info in LAND_COVER_CLASSES.items():
        legend_items += f"""
        <div class="legend-item">
            <span class="color-box" style="background-color: {cls_info['color']}"></span>
            {cls_info['name']}
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{L['title']}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin: 20px 0 15px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .num {{ text-align: right; font-family: monospace; }}
        .color-box {{
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 8px;
            vertical-align: middle;
        }}
        .bar-container {{
            width: 100%;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
        }}
        .bar {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }}
        .map-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .map-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 0.9em;
        }}
        .maps-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
        }}
        @media (max-width: 600px) {{
            .maps-grid {{ grid-template-columns: 1fr; }}
            .stat-value {{ font-size: 1.4em; }}
        }}
    </style>
</head>
<body>
    <h1>🛰️ {L['title']}</h1>
    
    <div class="card">
        <h2>{L['summary']}</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{city}</div>
                <div class="stat-label">{L['city']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{datetime.now().strftime('%Y-%m-%d')}</div>
                <div class="stat-label">{L['date']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{result.processed_shape[0]}×{result.processed_shape[1]}</div>
                <div class="stat-label">{L['resolution']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{result.total_pixels:,}</div>
                <div class="stat-label">{L['pixels']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{result.avg_confidence:.1%}</div>
                <div class="stat-label">{L['confidence']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{result.execution_time:.1f}s</div>
                <div class="stat-label">{L['time']}</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>{L['distribution']}</h2>
        <table>
            <thead>
                <tr>
                    <th>{L['class_name']}</th>
                    <th style="width: 40%"></th>
                    <th class="num">{L['percentage']}</th>
                    <th class="num">{L['pixel_count']}</th>
                </tr>
            </thead>
            <tbody>
                {dist_rows}
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>{L['legend']}</h2>
        <div class="legend">
            {legend_items}
        </div>
    </div>
    
    <div class="maps-grid">
        {"<div class='card'><h2>" + L['classification_map'] + "</h2><div class='map-container'><img src='data:image/png;base64," + map_img_b64 + "' alt='Classification Map'></div></div>" if map_img_b64 else ""}
        {"<div class='card'><h2>" + L['confidence_map'] + "</h2><div class='map-container'><img src='data:image/png;base64," + conf_img_b64 + "' alt='Confidence Map'></div></div>" if conf_img_b64 else ""}
    </div>
    
    <footer>
        {L['generated']} • {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </footer>
</body>
</html>
"""
    
    return html

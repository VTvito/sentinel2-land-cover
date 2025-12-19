"""Change Detection Module - Compare satellite imagery across time.

Detects land cover changes between two time periods for the same area.
Essential for monitoring:
- Urban expansion
- Deforestation
- Flooding/water changes
- Construction projects

Example:
    >>> from satellite_analysis import compare
    >>> changes = compare("Milan", "2023-06", "2024-06")
    >>> print(f"Changed area: {changes.changed_percentage:.1%}")
    >>> changes.export_report("milan_changes.html")
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, date
import numpy as np


@dataclass
class ChangeResult:
    """Result of change detection between two time periods."""
    
    city: str
    date_before: str
    date_after: str
    
    # Classification results
    labels_before: np.ndarray
    labels_after: np.ndarray
    
    # Change analysis
    change_mask: np.ndarray  # Boolean: True where changed
    change_matrix: np.ndarray  # Transition matrix (from_class x to_class)
    
    # Confidence
    confidence_before: np.ndarray
    confidence_after: np.ndarray
    
    # Metadata
    output_dir: Path
    execution_time: float
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.labels_before.shape
    
    @property
    def total_pixels(self) -> int:
        return self.labels_before.size
    
    @property
    def changed_pixels(self) -> int:
        return int(self.change_mask.sum())
    
    @property
    def changed_percentage(self) -> float:
        return self.changed_pixels / self.total_pixels
    
    @property
    def unchanged_percentage(self) -> float:
        return 1.0 - self.changed_percentage
    
    def get_transitions(self) -> Dict[Tuple[int, int], int]:
        """Get count of transitions between classes.
        
        Returns:
            Dict mapping (from_class, to_class) -> pixel count
        """
        transitions = {}
        for i in range(self.change_matrix.shape[0]):
            for j in range(self.change_matrix.shape[1]):
                if i != j and self.change_matrix[i, j] > 0:
                    transitions[(i, j)] = int(self.change_matrix[i, j])
        return transitions
    
    def get_major_changes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the most significant class transitions.
        
        Args:
            top_n: Number of top transitions to return
            
        Returns:
            List of dicts with from_class, to_class, count, percentage
        """
        from .exports import LAND_COVER_CLASSES
        
        transitions = self.get_transitions()
        sorted_trans = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for (from_cls, to_cls), count in sorted_trans[:top_n]:
            from_name = LAND_COVER_CLASSES.get(from_cls, {}).get("name", f"Class {from_cls}")
            to_name = LAND_COVER_CLASSES.get(to_cls, {}).get("name", f"Class {to_cls}")
            
            results.append({
                "from_class": from_cls,
                "to_class": to_cls,
                "from_name": from_name,
                "to_name": to_name,
                "count": count,
                "percentage": count / self.total_pixels * 100,
            })
        
        return results
    
    def summary(self) -> str:
        """Human-readable summary of changes."""
        lines = [
            f"Change Detection: {self.city}",
            f"  Period: {self.date_before} → {self.date_after}",
            f"  Shape: {self.shape}",
            f"  Changed: {self.changed_percentage:.1%} ({self.changed_pixels:,} pixels)",
            f"  Unchanged: {self.unchanged_percentage:.1%}",
            "",
            "  Top Changes:",
        ]
        
        for change in self.get_major_changes(3):
            lines.append(
                f"    {change['from_name']} → {change['to_name']}: "
                f"{change['percentage']:.2f}%"
            )
        
        return "\n".join(lines)


def compare(
    city: str,
    date_before: Union[str, date],
    date_after: Union[str, date],
    *,
    max_size: Optional[int] = 2000,
    classifier: str = "consensus",
    on_progress: Optional[callable] = None,
    project_root: Optional[Path] = None,
) -> ChangeResult:
    """Compare land cover between two time periods.
    
    Analyzes the same area at two different dates and detects changes.
    
    Args:
        city: City name
        date_before: Earlier date (YYYY-MM or YYYY-MM-DD)
        date_after: Later date (YYYY-MM or YYYY-MM-DD)
        max_size: Max image dimension
        classifier: Classification method
        on_progress: Progress callback
        project_root: Project root override
        
    Returns:
        ChangeResult with change analysis
        
    Example:
        >>> changes = compare("Milan", "2023-06", "2024-06")
        >>> print(f"Urban expansion: {changes.changed_percentage:.1%}")
        
        >>> # Check specific transitions
        >>> for t in changes.get_major_changes():
        ...     print(f"{t['from_name']} → {t['to_name']}: {t['percentage']:.2f}%")
    """
    import time
    from .api import analyze
    from .utils import OutputManager
    
    start_time = time.time()
    
    def progress(msg: str):
        if on_progress:
            on_progress(msg)
    
    # Parse dates
    date_before_str = _parse_date_to_month(date_before)
    date_after_str = _parse_date_to_month(date_after)
    
    # Analyze both periods
    progress(f"Analyzing {city} for {date_before_str}...")
    
    # For now, we use existing data (future: download specific dates)
    # This is a simplified version that compares the same data twice
    # TODO: Implement proper date-specific data retrieval
    
    result_before = analyze(
        city,
        max_size=max_size,
        classifier=classifier,
        project_root=project_root,
    )
    
    progress(f"Analyzing {city} for {date_after_str}...")
    
    result_after = analyze(
        city,
        max_size=max_size,
        classifier=classifier,
        project_root=project_root,
    )
    
    # Compute change detection
    progress("Computing changes...")
    
    change_mask, change_matrix = _compute_changes(
        result_before.labels,
        result_after.labels,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = result_before.output_dir.parent / f"change_{date_before_str}_to_{date_after_str}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save change results
    np.save(output_dir / "change_mask.npy", change_mask)
    np.save(output_dir / "change_matrix.npy", change_matrix)
    np.save(output_dir / "labels_before.npy", result_before.labels)
    np.save(output_dir / "labels_after.npy", result_after.labels)
    
    execution_time = time.time() - start_time
    
    return ChangeResult(
        city=city,
        date_before=date_before_str,
        date_after=date_after_str,
        labels_before=result_before.labels,
        labels_after=result_after.labels,
        change_mask=change_mask,
        change_matrix=change_matrix,
        confidence_before=result_before.confidence,
        confidence_after=result_after.confidence,
        output_dir=output_dir,
        execution_time=execution_time,
    )


def export_change_report(
    changes: ChangeResult,
    output_path: Optional[Union[str, Path]] = None,
    language: str = "en",
) -> Path:
    """Export change detection results as HTML report.
    
    Args:
        changes: ChangeResult from compare()
        output_path: Output file path
        language: Report language ("en" or "it")
        
    Returns:
        Path to the created HTML file
    """
    import base64
    from io import BytesIO
    from .exports import LAND_COVER_CLASSES
    
    if output_path is None:
        output_path = changes.output_dir / f"change_report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    before_img = _generate_classification_image_b64(changes.labels_before)
    after_img = _generate_classification_image_b64(changes.labels_after)
    change_img = _generate_change_map_b64(changes.change_mask, changes.labels_before, changes.labels_after)
    
    # Generate HTML
    html = _generate_change_html_report(
        changes=changes,
        before_img=before_img,
        after_img=after_img,
        change_img=change_img,
        language=language,
    )
    
    output_path.write_text(html, encoding='utf-8')
    
    return output_path


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_date_to_month(d: Union[str, date]) -> str:
    """Parse date to YYYY-MM format."""
    if isinstance(d, date):
        return d.strftime("%Y-%m")
    
    # Try parsing various formats
    for fmt in ["%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%Y/%m"]:
        try:
            parsed = datetime.strptime(d, fmt)
            return parsed.strftime("%Y-%m")
        except ValueError:
            continue
    
    return d  # Return as-is if can't parse


def _compute_changes(
    labels_before: np.ndarray,
    labels_after: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute change mask and transition matrix.
    
    Returns:
        (change_mask, change_matrix)
    """
    # Change mask: True where class changed
    change_mask = labels_before != labels_after
    
    # Transition matrix: count of pixels going from class i to class j
    n_classes = max(labels_before.max(), labels_after.max()) + 1
    change_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    for i in range(n_classes):
        for j in range(n_classes):
            change_matrix[i, j] = np.sum((labels_before == i) & (labels_after == j))
    
    return change_mask, change_matrix


def _generate_classification_image_b64(labels: np.ndarray) -> str:
    """Generate base64-encoded PNG of classification map."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from .exports import LAND_COVER_CLASSES
    
    colors = [LAND_COVER_CLASSES.get(i, {"color": "#808080"})["color"] 
              for i in range(max(labels.max() + 1, 6))]
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(labels, cmap=cmap, interpolation='nearest')
    ax.axis('off')
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    buf.seek(0)
    import base64
    return base64.b64encode(buf.read()).decode('utf-8')


def _generate_change_map_b64(
    change_mask: np.ndarray,
    labels_before: np.ndarray,
    labels_after: np.ndarray,
) -> str:
    """Generate change visualization as base64 PNG."""
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    # Create RGB image showing changes
    h, w = change_mask.shape
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 200  # Gray background (unchanged)
    
    # Color changes: red = urbanization, green = vegetation gain, blue = water
    from .exports import LAND_COVER_CLASSES
    
    # Highlight changes with bright colors
    rgb[change_mask] = [255, 100, 100]  # Red for any change
    
    # Specific change types
    # Vegetation loss (1 -> anything else)
    veg_loss = (labels_before == 1) & (labels_after != 1)
    rgb[veg_loss] = [255, 50, 50]  # Bright red
    
    # Vegetation gain (anything -> 1)
    veg_gain = (labels_before != 1) & (labels_after == 1)
    rgb[veg_gain] = [50, 255, 50]  # Bright green
    
    # Urbanization (anything -> 2)
    urban_gain = (labels_before != 2) & (labels_after == 2)
    rgb[urban_gain] = [255, 165, 0]  # Orange
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.axis('off')
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


from io import BytesIO  # Import at module level


def _generate_change_html_report(
    changes: ChangeResult,
    before_img: str,
    after_img: str,
    change_img: str,
    language: str = "en",
) -> str:
    """Generate HTML report for change detection."""
    from .exports import LAND_COVER_CLASSES
    
    labels = {
        "en": {
            "title": f"Change Detection Report - {changes.city}",
            "summary": "Change Summary",
            "period": "Analysis Period",
            "total_area": "Total Area",
            "changed": "Changed",
            "unchanged": "Unchanged",
            "before": "Before",
            "after": "After",
            "changes": "Changes",
            "top_changes": "Top Land Cover Transitions",
            "from": "From",
            "to": "To",
            "pixels": "Pixels",
            "percent": "Percent",
            "legend": "Change Map Legend",
            "legend_unchanged": "Unchanged",
            "legend_veg_loss": "Vegetation Loss",
            "legend_veg_gain": "Vegetation Gain",
            "legend_urban": "Urbanization",
            "legend_other": "Other Changes",
        },
        "it": {
            "title": f"Report Rilevamento Cambiamenti - {changes.city}",
            "summary": "Riepilogo Cambiamenti",
            "period": "Periodo Analisi",
            "total_area": "Area Totale",
            "changed": "Modificato",
            "unchanged": "Invariato",
            "before": "Prima",
            "after": "Dopo",
            "changes": "Cambiamenti",
            "top_changes": "Principali Transizioni Copertura del Suolo",
            "from": "Da",
            "to": "A",
            "pixels": "Pixel",
            "percent": "Percentuale",
            "legend": "Legenda Mappa Cambiamenti",
            "legend_unchanged": "Invariato",
            "legend_veg_loss": "Perdita Vegetazione",
            "legend_veg_gain": "Guadagno Vegetazione",
            "legend_urban": "Urbanizzazione",
            "legend_other": "Altri Cambiamenti",
        }
    }
    
    L = labels.get(language, labels["en"])
    
    # Generate transitions table
    trans_rows = ""
    for change in changes.get_major_changes(10):
        trans_rows += f"""
        <tr>
            <td>{change['from_name']}</td>
            <td>→</td>
            <td>{change['to_name']}</td>
            <td class="num">{change['count']:,}</td>
            <td class="num">{change['percentage']:.2f}%</td>
        </tr>
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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f0f0f0;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; margin-bottom: 20px; }}
        h2 {{ color: #34495e; margin: 20px 0 15px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .stat-item {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 6px; }}
        .stat-value {{ font-size: 1.6em; font-weight: bold; }}
        .stat-value.changed {{ color: #e74c3c; }}
        .stat-value.unchanged {{ color: #27ae60; }}
        .stat-label {{ font-size: 0.9em; color: #666; }}
        .maps-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
        .map-container {{ text-align: center; }}
        .map-container img {{ max-width: 100%; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15); }}
        .map-label {{ font-weight: bold; margin-bottom: 10px; color: #555; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; }}
        .num {{ text-align: right; font-family: monospace; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}
        footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.85em; }}
        @media (max-width: 900px) {{ .maps-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <h1>📊 {L['title']}</h1>
    
    <div class="card">
        <h2>{L['summary']}</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{changes.date_before} → {changes.date_after}</div>
                <div class="stat-label">{L['period']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{changes.total_pixels:,}</div>
                <div class="stat-label">{L['total_area']} (px)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value changed">{changes.changed_percentage:.1%}</div>
                <div class="stat-label">{L['changed']}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value unchanged">{changes.unchanged_percentage:.1%}</div>
                <div class="stat-label">{L['unchanged']}</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="maps-grid">
            <div class="map-container">
                <div class="map-label">{L['before']} ({changes.date_before})</div>
                <img src="data:image/png;base64,{before_img}" alt="Before">
            </div>
            <div class="map-container">
                <div class="map-label">{L['after']} ({changes.date_after})</div>
                <img src="data:image/png;base64,{after_img}" alt="After">
            </div>
            <div class="map-container">
                <div class="map-label">{L['changes']}</div>
                <img src="data:image/png;base64,{change_img}" alt="Changes">
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #c8c8c8"></div>{L['legend_unchanged']}</div>
            <div class="legend-item"><div class="legend-color" style="background: #ff3232"></div>{L['legend_veg_loss']}</div>
            <div class="legend-item"><div class="legend-color" style="background: #32ff32"></div>{L['legend_veg_gain']}</div>
            <div class="legend-item"><div class="legend-color" style="background: #ffa500"></div>{L['legend_urban']}</div>
            <div class="legend-item"><div class="legend-color" style="background: #ff6464"></div>{L['legend_other']}</div>
        </div>
    </div>
    
    <div class="card">
        <h2>{L['top_changes']}</h2>
        <table>
            <thead>
                <tr>
                    <th>{L['from']}</th>
                    <th></th>
                    <th>{L['to']}</th>
                    <th class="num">{L['pixels']}</th>
                    <th class="num">{L['percent']}</th>
                </tr>
            </thead>
            <tbody>
                {trans_rows}
            </tbody>
        </table>
    </div>
    
    <footer>
        Generated by Satellite City Analyzer • {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </footer>
</body>
</html>
"""
    
    return html

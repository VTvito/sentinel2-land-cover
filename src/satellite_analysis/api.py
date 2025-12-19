"""Simple public API for satellite analysis.

This module provides a single, easy-to-use entry point for analysis.

Example (from notebook):
    >>> from satellite_analysis import analyze
    >>> result = analyze("Florence")
    
Example (batch analysis):
    >>> from satellite_analysis import analyze_batch
    >>> results = analyze_batch(["Milan", "Rome", "Florence"])
    
Example (with export):
    >>> from satellite_analysis import analyze, export_geotiff, export_report
    >>> result = analyze("Milan")
    >>> export_geotiff(result, "milan.tif")
    >>> export_report(result, "milan_report.html")
"""

from pathlib import Path
from typing import Optional, Callable, Literal, Union, List, Dict
from datetime import date

from .core.config import AnalysisConfig
from .core.types import AnalysisResult


def analyze(
    city: str,
    *,
    # Date range
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
    
    # Download options
    cloud_cover: int = 20,
    radius_km: float = 15.0,
    
    # Processing options
    max_size: Optional[int] = None,
    classifier: Literal["kmeans", "spectral", "consensus"] = "consensus",
    n_clusters: int = 6,
    
    # Output options
    save_preview: bool = True,
    output_dir: Optional[Path] = None,
    
    # Callbacks
    on_progress: Optional[Callable[[str], None]] = None,
    
    # Advanced
    project_root: Optional[Path] = None,
) -> AnalysisResult:
    """Analyze satellite imagery for a city.
    
    This is the main entry point for satellite analysis. It handles:
    - Finding or downloading data
    - Loading and preprocessing bands
    - Running classification
    - Saving results
    
    Args:
        city: City name (e.g., "Florence", "Milan", "Rome")
        
        start_date: Start of date range (YYYY-MM-DD or date object)
        end_date: End of date range (default: today)
        
        cloud_cover: Maximum cloud coverage % (default: 20)
        radius_km: Search radius in km (default: 15)
        
        max_size: Max pixels per dimension (None = full resolution)
        classifier: Classification method ("kmeans", "spectral", "consensus")
        n_clusters: Number of clusters (default: 6)
        
        save_preview: Save PNG preview (default: True)
        output_dir: Custom output directory (default: data/cities/{city}/latest)
        
        on_progress: Callback for progress updates
        
        project_root: Override project root detection
        
    Returns:
        AnalysisResult with labels, confidence, statistics, etc.
        
    Raises:
        FileNotFoundError: If no data found and download fails
        ValueError: If city not found and no coordinates provided
        
    Example:
        >>> result = analyze("Florence")
        >>> print(f"Found {len(result.classes)} land cover types")
        >>> print(f"Average confidence: {result.avg_confidence:.1%}")
    """
    from datetime import datetime
    import time
    
    def progress(msg: str):
        if on_progress:
            on_progress(msg)
    
    start_time = time.time()
    
    # Parse dates if strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Create config
    config = AnalysisConfig.for_notebook(
        city=city,
        start_date=start_date,
        end_date=end_date,
        cloud_cover=cloud_cover,
        radius_km=radius_km,
        max_size=max_size,
        classifier=classifier,
        n_clusters=n_clusters,
        save_preview=save_preview,
    )
    
    if project_root:
        object.__setattr__(config, 'project_root', Path(project_root).resolve())
    
    progress(f"Analyzing {city}...")
    
    # Use CompletePipeline for now (will be replaced with new steps)
    from .pipelines.complete_pipeline import CompletePipeline
    
    pipeline = CompletePipeline(
        project_root=config.project_root,
        max_size=max_size or 5000,
        n_clusters=n_clusters,
    )
    
    # Run analysis
    pipeline_result = pipeline.run(
        city=city,
        max_size=max_size,
    )
    
    execution_time = time.time() - start_time
    
    # Convert to new result type
    from .core.types import ClassificationResult
    
    classification = ClassificationResult(
        labels=pipeline_result.labels,
        confidence=pipeline_result.confidence,
        uncertainty_mask=None,  # Not in pipeline result
        statistics=pipeline_result.metadata.get('statistics', {}),
    )
    
    return AnalysisResult(
        city=city,
        classification=classification,
        output_dir=Path(pipeline_result.output_dir),
        execution_time=execution_time,
        original_shape=tuple(pipeline_result.metadata.get('original_shape', pipeline_result.image_shape)),
        processed_shape=pipeline_result.image_shape,
        config_summary=config.to_dict(),
    )


def quick_preview(
    city: str,
    *,
    max_size: int = 500,
    on_progress: Optional[Callable[[str], None]] = None,
) -> AnalysisResult:
    """Quick preview analysis with reduced resolution.
    
    Same as analyze() but with:
    - max_size=500 (faster processing)
    - classifier="kmeans" (fastest method)
    
    Great for testing data availability and pipeline.
    
    Args:
        city: City name
        max_size: Max pixels (default: 500)
        on_progress: Progress callback
        
    Returns:
        AnalysisResult
        
    Example:
        >>> result = quick_preview("Florence")  # ~10 seconds
    """
    return analyze(
        city,
        max_size=max_size,
        classifier="kmeans",
        on_progress=on_progress,
    )


def analyze_batch(
    cities: List[str],
    *,
    max_size: Optional[int] = 2000,
    classifier: Literal["kmeans", "spectral", "consensus"] = "consensus",
    on_progress: Optional[Callable[[str, str], None]] = None,
    stop_on_error: bool = False,
    **kwargs,
) -> Dict[str, Union[AnalysisResult, Exception]]:
    """Analyze multiple cities in batch.
    
    Processes multiple cities sequentially and returns all results.
    Errors are captured but don't stop processing (unless stop_on_error=True).
    
    Args:
        cities: List of city names
        max_size: Max pixels per dimension (default: 2000)
        classifier: Classification method
        on_progress: Callback with (city, status) for progress updates
        stop_on_error: If True, stop on first error
        **kwargs: Additional arguments passed to analyze()
        
    Returns:
        Dictionary mapping city name to AnalysisResult or Exception
        
    Example:
        >>> results = analyze_batch(["Milan", "Rome", "Florence"])
        >>> for city, result in results.items():
        ...     if isinstance(result, Exception):
        ...         print(f"{city}: ERROR - {result}")
        ...     else:
        ...         print(f"{city}: {result.avg_confidence:.1%} confidence")
        Milan: 87.3% confidence
        Rome: 84.1% confidence
        Florence: 89.2% confidence
    """
    results: Dict[str, Union[AnalysisResult, Exception]] = {}
    
    total = len(cities)
    for i, city in enumerate(cities, 1):
        if on_progress:
            on_progress(city, f"Processing {i}/{total}")
        
        try:
            result = analyze(
                city,
                max_size=max_size,
                classifier=classifier,
                **kwargs,
            )
            results[city] = result
            
            if on_progress:
                on_progress(city, f"Done - {result.avg_confidence:.1%} confidence")
                
        except Exception as e:
            results[city] = e
            if on_progress:
                on_progress(city, f"Error: {e}")
            
            if stop_on_error:
                break
    
    return results


# =============================================================================
# Export functions (re-exported for convenience)
# =============================================================================

def export_geotiff(
    result: AnalysisResult,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Path:
    """Export result as GeoTIFF. See exports.export_geotiff for full docs."""
    from .exports import export_geotiff as _export_geotiff
    return _export_geotiff(result, output_path, **kwargs)


def export_report(
    result: AnalysisResult,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Path:
    """Export result as HTML report. See exports.export_report for full docs."""
    from .exports import export_report as _export_report
    return _export_report(result, output_path, **kwargs)


def export_json(
    result: AnalysisResult,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Export result as JSON. See exports.export_json for full docs."""
    from .exports import export_json as _export_json
    return _export_json(result, output_path)

"""Complete end-to-end pipeline for satellite city analysis.

This pipeline integrates:
- Download (if needed)
- Band extraction
- Automatic downsampling for large images
- Classification
- Result management
"""

from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import rasterio
from datetime import datetime, timedelta
import logging

from satellite_analysis.utils import AreaSelector, OutputManager
from satellite_analysis.analyzers.classification import ConsensusClassifier
from satellite_analysis.pipelines.download_pipeline import DownloadPipeline
from satellite_analysis.preprocessors.band_extractor import BandExtractor

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of complete analysis pipeline."""
    
    city: str
    labels: np.ndarray
    confidence: np.ndarray
    image_shape: Tuple[int, int]
    output_dir: Path
    metadata: dict
    
    @property
    def total_pixels(self) -> int:
        """Total number of pixels analyzed."""
        return self.labels.size
    
    @property
    def class_distribution(self) -> dict:
        """Distribution of classes."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = counts.sum()
        return {
            int(cls): {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
            for cls, count in zip(unique, counts)
        }


class CompletePipeline:
    """Complete end-to-end pipeline for satellite city analysis.
    
    This pipeline handles the entire workflow:
    1. Check if data exists locally
    2. Download if needed (optional)
    3. Extract bands from archive
    4. Load and preprocess bands
    5. Automatic downsampling for large images
    6. Land cover classification
    7. Save results with OutputManager
    
    Example:
        >>> pipeline = CompletePipeline.from_config("config/config.yaml")
        >>> result = pipeline.run(
        ...     city="Florence",
        ...     download=True,
        ...     max_size=5000
        ... )
        >>> print(f"Analyzed {result.total_pixels:,} pixels")
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        max_size: int = 5000,
        n_clusters: int = 6,
        project_root: Optional[Path] = None
    ):
        """Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
            max_size: Maximum image dimension (auto-downsample if larger)
            n_clusters: Number of clusters for K-Means
            project_root: Project root directory (auto-detected if None)
        """
        # Auto-detect project root if not provided
        if project_root is None:
            # Go up from this file: pipelines -> satellite_analysis -> src -> project_root
            self.project_root = Path(__file__).parent.parent.parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
        
        self.config_path = self.project_root / config_path
        self.max_size = max_size
        self.n_clusters = n_clusters
        
        # Initialize components
        self.area_selector = AreaSelector()
        self.classifier = ConsensusClassifier(n_clusters=n_clusters)
    
    def _resolve_path(self, *parts: str) -> Path:
        """Resolve path relative to project root."""
        return self.project_root / Path(*parts)
        
    @classmethod
    def from_config(cls, config_path: str) -> 'CompletePipeline':
        """Create pipeline from configuration file."""
        return cls(config_path=config_path)
    
    def run(
        self,
        city: str,
        download: bool = False,
        max_size: Optional[int] = None,
        radius_km: float = 15.0
    ) -> AnalysisResult:
        """Run complete analysis pipeline.
        
        Args:
            city: City name (e.g., "Florence", "Milan")
            download: If True, download data if not available
            max_size: Override max image dimension
            radius_km: Radius around city center in km
            
        Returns:
            AnalysisResult with labels, confidence, and metadata
        """
        if max_size:
            self.max_size = max_size
        
        logger.info(f"Starting analysis for {city}")
        
        # Step 1: Check if data exists
        data_dir, needs_download = self._check_data_exists(city)
        
        # Step 2: Download if needed
        if needs_download:
            if not download:
                raise FileNotFoundError(
                    f"No data found for {city}. Run with download=True to download automatically."
                )
            
            logger.info(f"Downloading data for {city}...")
            data_dir = self._download_data(city, radius_km)
        
        # Step 3: Load bands
        logger.info(f"Loading bands from {data_dir}")
        bands = self._load_bands(data_dir)
        
        # Step 4: Downsample if needed
        original_shape = bands['B02'].shape
        if max(original_shape) > self.max_size:
            logger.info(f"Downsampling from {original_shape} to fit {self.max_size}x{self.max_size}")
            bands = self._downsample_bands(bands)
        
        # Step 5: Classify
        logger.info(f"Running classification on {bands['B02'].shape}")
        labels, confidence, stats = self._classify(bands)
        
        # Step 6: Save results (use absolute path for base_path)
        base_path = str(self._resolve_path("data", "cities"))
        output_manager = OutputManager(city, base_path=base_path)
        
        with output_manager.create_run('consensus', {
            'n_clusters': self.n_clusters,
            'original_shape': list(original_shape),
            'processed_shape': list(bands['B02'].shape),
            'downsampled': max(original_shape) > self.max_size
        }) as run:
            run.save_labels(labels)
            run.save_confidence(confidence)
            run.set_statistics(stats)
            
            output_dir = run.path
        
        # Create result
        result = AnalysisResult(
            city=city,
            labels=labels,
            confidence=confidence,
            image_shape=bands['B02'].shape,
            output_dir=output_dir,
            metadata={
                'original_shape': original_shape,
                'downsampled': max(original_shape) > self.max_size,
                'statistics': stats
            }
        )
        
        logger.info(f"✅ Analysis complete. Results saved to: {output_dir}")
        
        return result
    
    def _check_data_exists(self, city: str) -> Tuple[Optional[Path], bool]:
        """Check if data exists for the given city.
        
        Returns:
            (data_dir, needs_download) where data_dir is None if not found
        """
        # Possible data locations (all resolved to absolute paths)
        possible_paths = [
            self._resolve_path("data", "cities", city.lower(), "bands"),
            self._resolve_path("data", "processed", f"{city.lower()}_centro"),
            self._resolve_path("data", "demo", f"{city.lower()}_sample", "bands"),
        ]
        
        for path in possible_paths:
            if path.exists():
                # Check if has any image files (search recursively)
                if any(path.rglob("*.tif")) or any(path.rglob("*.jp2")):
                    return path, False
        
        return None, True
    
    def _download_data(self, city: str, radius_km: float) -> Path:
        """Download satellite data for the city.
        
        Returns:
            Path to bands directory
        """
        # Get city coordinates
        bbox, metadata = self.area_selector.select_by_city(city, radius_km=radius_km)
        
        # Setup download pipeline
        download_pipeline = DownloadPipeline.from_config(str(self.config_path))
        
        # Date range: last 30 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Download
        output_dir = self._resolve_path("data", "cities", city.lower(), "raw")
        download_pipeline.output_dir = str(output_dir)
        
        result = download_pipeline.run(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            limit=10,
            max_downloads=1  # Just one product
        )
        
        if result.downloaded_count == 0:
            raise RuntimeError(f"Failed to download data for {city}")
        
        # Extract bands
        bands_dir = self._resolve_path("data", "cities", city.lower(), "bands")
        bands_dir.mkdir(parents=True, exist_ok=True)
        
        extractor = BandExtractor(output_dir=str(bands_dir))
        for product_file in result.downloaded_files:
            extractor.extract_bands(
                zip_path=str(product_file),
                bands=['B02', 'B03', 'B04', 'B08'],
                resolution='10m'
            )
        
        logger.info(f"✅ Data downloaded and extracted to {bands_dir}")
        
        return bands_dir
    
    def _load_bands(self, data_dir: Path) -> dict:
        """Load satellite bands from directory.
        
        Supports both .tif and .jp2 formats.
        Searches recursively if bands not in root.
        
        Returns:
            Dictionary of band_name -> numpy array
        """
        bands_needed = ['B02', 'B03', 'B04', 'B08']
        bands = {}
        
        for band_name in bands_needed:
            # Try .tif first, then .jp2 in current directory
            band_path = data_dir / f"{band_name}.tif"
            if not band_path.exists():
                band_path = data_dir / f"{band_name}.jp2"
            
            # If not found, search recursively
            if not band_path.exists():
                tif_matches = list(data_dir.rglob(f"{band_name}.tif"))
                jp2_matches = list(data_dir.rglob(f"{band_name}.jp2"))
                
                if tif_matches:
                    band_path = tif_matches[0]
                elif jp2_matches:
                    band_path = jp2_matches[0]
                else:
                    raise FileNotFoundError(f"Band {band_name} not found in {data_dir}")
            
            with rasterio.open(band_path) as src:
                bands[band_name] = src.read(1).astype(np.float32)
        
        logger.info(f"Loaded {len(bands)} bands: {bands['B02'].shape}")
        
        return bands
    
    def _downsample_bands(self, bands: dict) -> dict:
        """Downsample bands to fit within max_size.
        
        Uses bilinear interpolation to preserve spatial relationships.
        """
        from scipy.ndimage import zoom
        
        h, w = bands['B02'].shape
        scale = self.max_size / max(h, w)
        
        if scale >= 1.0:
            return bands  # No downsampling needed
        
        downsampled = {}
        for band_name, data in bands.items():
            # Use zoom for fast downsampling
            downsampled[band_name] = zoom(data, scale, order=1).astype(np.float32)
        
        new_shape = downsampled['B02'].shape
        logger.info(f"Downsampled from {(h, w)} to {new_shape} (scale={scale:.3f})")
        
        return downsampled
    
    def _classify(self, bands: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Run classification on loaded bands.
        
        Returns:
            (labels, confidence, statistics)
        """
        # Stack bands
        band_stack = np.stack([
            bands['B02'],
            bands['B03'],
            bands['B04'],
            bands['B08']
        ], axis=-1)
        
        # Band indices
        band_indices = {
            'B02': 0,  # Blue
            'B03': 1,  # Green
            'B04': 2,  # Red
            'B08': 3   # NIR
        }
        
        # Classify
        labels, confidence, uncertainty_mask, stats = self.classifier.classify(
            band_stack,
            band_indices,
            has_swir=False
        )
        
        return labels, confidence, stats

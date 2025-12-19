"""Complete end-to-end pipeline for satellite city analysis.

This pipeline integrates:
- Download (if needed)
- Band extraction
- Automatic downsampling for large images
- Classification
- Result management
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Literal
from dataclasses import dataclass
import numpy as np
import rasterio
from datetime import datetime, date, timedelta
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
        project_root: Optional[Path] = None,
        classifier: Literal["kmeans", "spectral", "consensus"] = "consensus",
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

        if classifier not in {"kmeans", "spectral", "consensus"}:
            raise ValueError(f"Unknown classifier: {classifier}")
        self.classifier_type: Literal["kmeans", "spectral", "consensus"] = classifier
        
        # Initialize components
        self.area_selector = AreaSelector()
        self._consensus_classifier = ConsensusClassifier(n_clusters=n_clusters)
    
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
        radius_km: float = 15.0,
        crop_to_city: bool = True,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        cloud_cover: Optional[float] = None,
    ) -> AnalysisResult:
        """Run complete analysis pipeline.
        
        Args:
            city: City name (e.g., "Florence", "Milan")
            download: If True, download data if not available
            max_size: Override max image dimension
            radius_km: Radius around city center in km
            crop_to_city: If True, crop image to city bounding box (centered on city)
            
        Returns:
            AnalysisResult with labels, confidence, and metadata
        """
        if max_size:
            self.max_size = max_size
        
        logger.info(f"Starting analysis for {city}")

        bands_needed = self._bands_needed_for_classifier()
        
        # Step 1: Get city bbox for cropping
        bbox, city_metadata = self.area_selector.select_by_city(city, radius_km=radius_km)
        logger.info(f"City center: {city_metadata['center']}, radius: {radius_km}km")
        
        # Step 2: Check if data exists
        data_dir, needs_download = self._check_data_exists(city, bands_needed=bands_needed)
        
        # Step 3: Download if needed
        if needs_download:
            if not download:
                raise FileNotFoundError(
                    f"No data found for {city}. Run with download=True to download automatically."
                )
            
            logger.info(f"Downloading data for {city}...")
            data_dir = self._download_data(
                city,
                radius_km,
                start_date=start_date,
                end_date=end_date,
                cloud_cover=cloud_cover,
            )
        
        # Step 4: Load bands (with optional cropping to city bbox)
        logger.info(f"Loading bands from {data_dir}")
        if crop_to_city:
            logger.info(f"Cropping to city bbox: {bbox}")
            bands = self._load_bands_cropped(data_dir, bbox, bands_needed=bands_needed)
        else:
            bands = self._load_bands(data_dir, bands_needed=bands_needed)
        
        # Step 5: Downsample if needed
        original_shape = bands['B02'].shape
        if max(original_shape) > self.max_size:
            logger.info(f"Downsampling from {original_shape} to fit {self.max_size}x{self.max_size}")
            bands = self._downsample_bands(bands)
        
        # Step 6: Classify
        logger.info(f"Running classification on {bands['B02'].shape}")
        labels, confidence, stats = self._classify(bands)
        
        # Step 7: Save results (use absolute path for base_path)
        base_path = str(self._resolve_path("data", "cities"))
        output_manager = OutputManager(city, base_path=base_path)
        
        def _to_iso(dt: Optional[Union[str, date, datetime]]) -> Optional[str]:
            if dt is None:
                return None
            if isinstance(dt, (date, datetime)):
                return dt.isoformat()
            return str(dt)

        with output_manager.create_run(self.classifier_type, {
            'classifier': self.classifier_type,
            'bands_used': bands_needed,
            'n_clusters': self.n_clusters if self.classifier_type in {"kmeans", "consensus"} else None,
            'max_size': self.max_size,
            'start_date': _to_iso(start_date),
            'end_date': _to_iso(end_date),
            'cloud_cover': cloud_cover,
            'original_shape': list(original_shape),
            'processed_shape': list(bands['B02'].shape),
            'downsampled': max(original_shape) > self.max_size,
            'cropped_to_city': crop_to_city,
            'bbox': bbox,
            'city_center': city_metadata['center'],
            'radius_km': radius_km
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
                'statistics': stats,
                'bbox': bbox,
                'city_center': city_metadata['center'],
                'radius_km': radius_km,
                'cropped_to_city': crop_to_city,
                'classifier': self.classifier_type,
                'bands_used': bands_needed,
            }
        )
        
        logger.info(f"✅ Analysis complete. Results saved to: {output_dir}")
        
        return result
    
    def _check_data_exists(self, city: str, *, bands_needed: List[str]) -> Tuple[Optional[Path], bool]:
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
                if self._has_required_bands(path, bands_needed=bands_needed):
                    return path, False
        
        return None, True

    def _has_required_bands(self, data_dir: Path, *, bands_needed: List[str]) -> bool:
        for band_name in bands_needed:
            if (data_dir / f"{band_name}.tif").exists() or (data_dir / f"{band_name}.jp2").exists():
                continue
            if any(data_dir.rglob(f"{band_name}.tif")) or any(data_dir.rglob(f"{band_name}.jp2")):
                continue
            return False
        return True

    def _bands_needed_for_classifier(self) -> List[str]:
        if self.classifier_type == "spectral":
            return ["B02", "B03", "B04", "B08", "B11", "B12"]
        return ["B02", "B03", "B04", "B08"]
    
    def _download_data(
        self,
        city: str,
        radius_km: float,
        *,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        cloud_cover: Optional[float] = None,
    ) -> Path:
        """Download satellite data for the city.
        
        Returns:
            Path to bands directory
        """
        # Get city coordinates
        bbox, metadata = self.area_selector.select_by_city(city, radius_km=radius_km)
        
        # Setup download pipeline
        download_pipeline = DownloadPipeline.from_config(str(self.config_path))
        
        def _to_ymd(d: Optional[Union[str, date, datetime]]) -> Optional[str]:
            if d is None:
                return None
            if isinstance(d, str):
                return d
            return d.strftime("%Y-%m-%d")

        # Date range defaults: last 30 days
        end_date_str = _to_ymd(end_date) or datetime.now().strftime("%Y-%m-%d")
        start_date_str = _to_ymd(start_date) or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        if cloud_cover is not None:
            download_pipeline.max_cloud_cover = float(cloud_cover)
        
        # Download
        output_dir = self._resolve_path("data", "cities", city.lower(), "raw")
        download_pipeline.output_dir = str(output_dir)
        # Keep the underlying downloader in sync with the updated output dir.
        try:
            download_pipeline.downloader.output_dir = Path(output_dir)
            download_pipeline.downloader.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort; download will still work if downloader already points to a valid dir.
            pass
        
        result = download_pipeline.run(
            bbox=bbox,
            start_date=start_date_str,
            end_date=end_date_str,
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
            # Always extract the 10m core bands.
            extractor.extract_bands(
                zip_path=str(product_file),
                bands=['B02', 'B03', 'B04', 'B08'],
                resolution='10m'
            )

            # Spectral mode requires SWIR bands, which are 20m.
            if self.classifier_type == "spectral":
                extractor.extract_bands(
                    zip_path=str(product_file),
                    bands=['B11', 'B12'],
                    resolution='20m'
                )
        
        logger.info(f"✅ Data downloaded and extracted to {bands_dir}")
        
        return bands_dir
    
    def _load_bands(self, data_dir: Path, *, bands_needed: List[str]) -> dict:
        """Load satellite bands from directory.
        
        Supports both .tif and .jp2 formats.
        Searches recursively if bands not in root.
        
        Returns:
            Dictionary of band_name -> numpy array
        """
        bands: Dict[str, np.ndarray] = {}

        # Load B02 first as the canonical grid (10m) for optional resampling.
        if 'B02' not in bands_needed:
            raise ValueError("bands_needed must include 'B02'")

        b02_path = self._find_band_path(data_dir, 'B02')
        with rasterio.open(b02_path) as src:
            target_crs = src.crs
            target_transform = src.transform
            target_shape = src.shape
            bands['B02'] = src.read(1).astype(np.float32)

        for band_name in bands_needed:
            if band_name == 'B02':
                continue

            # Try .tif first, then .jp2 in current directory
            band_path = self._find_band_path(data_dir, band_name)
            with rasterio.open(band_path) as src:
                data = src.read(1).astype(np.float32)
                if src.shape != target_shape or src.transform != target_transform or src.crs != target_crs:
                    data = self._resample_to_target(
                        data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        dst_shape=target_shape,
                    )
                bands[band_name] = data
        
        logger.info(f"Loaded {len(bands)} bands: {bands['B02'].shape}")
        
        return bands

    def _find_band_path(self, data_dir: Path, band_name: str) -> Path:
        """Find a band file path under data_dir, searching recursively."""
        band_path = data_dir / f"{band_name}.tif"
        if not band_path.exists():
            band_path = data_dir / f"{band_name}.jp2"

        if band_path.exists():
            return band_path

        tif_matches = list(data_dir.rglob(f"{band_name}.tif"))
        if tif_matches:
            return tif_matches[0]
        jp2_matches = list(data_dir.rglob(f"{band_name}.jp2"))
        if jp2_matches:
            return jp2_matches[0]

        raise FileNotFoundError(f"Band {band_name} not found in {data_dir}")

    def _resample_to_target(
        self,
        src_data: np.ndarray,
        *,
        src_transform,
        src_crs,
        dst_transform,
        dst_crs,
        dst_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Reproject/resample src_data onto the target grid."""
        from rasterio.warp import reproject
        from rasterio.enums import Resampling

        dst = np.empty(dst_shape, dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        return dst
    
    def _load_bands_cropped(self, data_dir: Path, bbox: list, *, bands_needed: List[str]) -> dict:
        """Load satellite bands and crop to bounding box.
        
        Args:
            data_dir: Directory containing band files
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84
            
        Returns:
            Dictionary of band_name -> cropped numpy array
        """
        from rasterio.mask import mask as rasterio_mask
        from rasterio.warp import transform_geom
        from shapely.geometry import box, shape
        
        bands: Dict[str, np.ndarray] = {}
        
        if 'B02' not in bands_needed:
            raise ValueError("bands_needed must include 'B02'")

        band_paths: Dict[str, Path] = {band_name: self._find_band_path(data_dir, band_name) for band_name in bands_needed}
        
        # Get CRS from first band
        first_band = list(band_paths.values())[0]
        with rasterio.open(first_band) as src:
            raster_crs = src.crs
        
        # Create bbox polygon in WGS84 and transform to raster CRS
        bbox_wgs84 = box(bbox[0], bbox[1], bbox[2], bbox[3])
        bbox_geojson = {
            "type": "Polygon",
            "coordinates": [list(bbox_wgs84.exterior.coords)]
        }
        
        # Transform to raster CRS (typically UTM)
        bbox_transformed_geojson = transform_geom(
            src_crs="EPSG:4326",  # WGS84
            dst_crs=raster_crs,
            geom=bbox_geojson
        )
        bbox_transformed = shape(bbox_transformed_geojson)
        
        logger.info(f"Cropping to bbox in {raster_crs}: {bbox_transformed.bounds}")
        
        # Load and crop each band (also capture per-band crop transforms)
        transforms: Dict[str, object] = {}
        crs_by_band: Dict[str, object] = {}
        for band_name, band_path in band_paths.items():
            with rasterio.open(band_path) as src:
                crs_by_band[band_name] = src.crs
                try:
                    out_image, out_transform = rasterio_mask(
                        src,
                        [bbox_transformed],
                        crop=True,
                        all_touched=True  # Include pixels touched by bbox edge
                    )
                    
                    # Remove band dimension if present
                    if out_image.ndim == 3:
                        out_image = out_image[0]

                    bands[band_name] = out_image.astype(np.float32)
                    transforms[band_name] = out_transform
                    
                except Exception as e:
                    logger.warning(f"Crop failed for {band_name}: {e}. Falling back to full image.")
                    bands[band_name] = src.read(1).astype(np.float32)
                    transforms[band_name] = src.transform

        # If any band ended up on a different grid, resample to B02 cropped grid.
        target_shape = bands['B02'].shape
        target_transform = transforms['B02']
        target_crs = crs_by_band['B02']

        for band_name, data in list(bands.items()):
            if band_name == 'B02':
                continue
            if data.shape != target_shape or transforms.get(band_name) != target_transform or crs_by_band.get(band_name) != target_crs:
                bands[band_name] = self._resample_to_target(
                    data,
                    src_transform=transforms[band_name],
                    src_crs=crs_by_band[band_name],
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_shape=target_shape,
                )
        
        logger.info(f"Cropped {len(bands)} bands: {bands['B02'].shape}")
        
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
        if self.classifier_type == "consensus":
            band_stack = np.stack([
                bands['B02'],
                bands['B03'],
                bands['B04'],
                bands['B08']
            ], axis=-1)

            band_indices = {
                'B02': 0,  # Blue
                'B03': 1,  # Green
                'B04': 2,  # Red
                'B08': 3   # NIR
            }

            labels, confidence, _, stats = self._consensus_classifier.classify(
                band_stack,
                band_indices,
                has_swir=False,
            )
            return labels, confidence, stats

        if self.classifier_type == "kmeans":
            return self._classify_kmeans(bands)

        if self.classifier_type == "spectral":
            return self._classify_spectral(bands)

        raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

    def _classify_kmeans(self, bands: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        """KMeans++ segmentation + heuristic cluster labeling into canonical land-cover classes."""
        from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
        from satellite_analysis.preprocessing.reshape import reshape_image_to_table, reshape_table_to_image
        from satellite_analysis.preprocessing.normalization import min_max_scale

        b02 = bands['B02'].astype(np.float32)
        b03 = bands['B03'].astype(np.float32)
        b04 = bands['B04'].astype(np.float32)
        b08 = bands['B08'].astype(np.float32)

        eps = 1e-6
        ndvi = (b08 - b04) / (b08 + b04 + eps)
        ndwi = (b03 - b08) / (b03 + b08 + eps)
        brightness = (b02 + b03 + b04) / 3.0

        # Use global brightness quantiles for robust thresholds across different scaling.
        p10 = float(np.nanpercentile(brightness, 10))
        p90 = float(np.nanpercentile(brightness, 90))

        stack = np.stack([b02, b03, b04, b08], axis=-1)
        data = reshape_image_to_table(stack)
        data_scaled = min_max_scale(data)

        clusterer = KMeansPlusPlusClusterer(
            n_clusters=self.n_clusters,
            max_iterations=30,
            random_state=42,
        )
        cluster_labels_1d = clusterer.fit_predict(data_scaled)
        cluster_img = reshape_table_to_image(b02.shape, cluster_labels_1d).astype(np.int32)

        # Label each cluster into the canonical 0-5 land cover classes.
        cluster_to_class: Dict[int, int] = {}
        cluster_to_conf: Dict[int, float] = {}

        for k in range(self.n_clusters):
            mask = cluster_img == k
            if not np.any(mask):
                cluster_to_class[k] = 5
                cluster_to_conf[k] = 0.5
                continue

            mean_ndvi = float(np.nanmean(ndvi[mask]))
            mean_ndwi = float(np.nanmean(ndwi[mask]))
            mean_brightness = float(np.nanmean(brightness[mask]))
            mean_rgb = float(np.nanmean(((b02 + b03 + b04) / 3.0)[mask]))
            mean_nir = float(np.nanmean(b08[mask]))

            # Water
            if mean_ndwi > 0.25 and mean_ndvi < 0.2:
                cluster_to_class[k] = 0
                cluster_to_conf[k] = float(np.clip((mean_ndwi - 0.25) / 0.25 + 0.5, 0.5, 1.0))
                continue

            # Vegetation
            if mean_ndvi > 0.5:
                cluster_to_class[k] = 1
                cluster_to_conf[k] = float(np.clip((mean_ndvi - 0.5) / 0.3 + 0.5, 0.5, 1.0))
                continue

            # Bright surfaces / shadows by brightness
            if mean_brightness >= p90 and mean_ndvi < 0.2:
                cluster_to_class[k] = 4
                cluster_to_conf[k] = 0.7
                continue
            if mean_brightness <= p10:
                cluster_to_class[k] = 5
                cluster_to_conf[k] = 0.7
                continue

            # Built vs soil split (heuristic, 10m-only)
            visible_to_nir = mean_rgb / (mean_nir + eps)
            if mean_ndvi < 0.2 and visible_to_nir > 0.9:
                cluster_to_class[k] = 3  # Urban
                cluster_to_conf[k] = 0.6
            else:
                cluster_to_class[k] = 2  # Bare soil
                cluster_to_conf[k] = 0.6

        labels = np.zeros_like(cluster_img, dtype=np.uint8)
        confidence = np.zeros_like(brightness, dtype=np.float32)
        for k in range(self.n_clusters):
            mask = cluster_img == k
            labels[mask] = np.uint8(cluster_to_class[k])
            confidence[mask] = np.float32(cluster_to_conf[k])

        stats = self._basic_stats(labels, confidence)
        stats['n_clusters'] = self.n_clusters
        return labels, confidence, stats

    def _classify_spectral(self, bands: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Spectral indices classification (requires SWIR bands B11/B12)."""
        from satellite_analysis.analyzers.classification import SpectralIndicesClassifier

        for b in ('B11', 'B12'):
            if b not in bands:
                raise FileNotFoundError(
                    f"Spectral classifier requires {b}. Ensure SWIR bands are available (download/extract)."
                )

        # Build raster stack and indices.
        band_order = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        band_stack = np.stack([bands[b].astype(np.float32) for b in band_order], axis=-1)
        band_indices = {b: i for i, b in enumerate(band_order)}

        spectral = SpectralIndicesClassifier()
        spectral.validate_bands(band_indices, num_bands=band_stack.shape[-1])
        spectral_labels, _ = spectral.classify_raster(band_stack, band_indices)

        # Map spectral classes -> canonical land cover classes used by exports.
        # Spectral: 0 Water, 1 Forest, 2 Grassland, 3 Urban, 4 Bare Soil, 5 Mixed
        mapping = {
            0: 0,
            1: 1,
            2: 1,
            3: 3,
            4: 2,
            5: 5,
        }
        labels = np.vectorize(lambda x: mapping.get(int(x), 5), otypes=[np.uint8])(spectral_labels)

        # Simple confidence: mixed is lower-confidence.
        confidence = np.where(spectral_labels == 5, 0.5, 0.9).astype(np.float32)

        stats = self._basic_stats(labels, confidence)
        stats['note'] = 'spectral indices classifier (SWIR required)'
        return labels, confidence, stats

    def _basic_stats(self, labels: np.ndarray, confidence: np.ndarray) -> dict:
        unique, counts = np.unique(labels, return_counts=True)
        total = int(counts.sum())
        return {
            'avg_confidence': float(np.mean(confidence)),
            'class_distribution': {
                int(cls): {
                    'count': int(count),
                    'percentage': float(count / total * 100.0),
                }
                for cls, count in zip(unique, counts)
            }
        }

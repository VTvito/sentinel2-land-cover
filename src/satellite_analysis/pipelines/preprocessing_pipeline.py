"""Complete preprocessing pipeline for Sentinel-2 data."""
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from dataclasses import dataclass

from satellite_analysis.preprocessors import BandExtractor, BandComposer


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline."""
    product_name: str
    bands: Dict[str, Path]
    band_data: Dict[str, np.ndarray]
    rgb: np.ndarray
    fcc: np.ndarray
    ndvi: np.ndarray
    metadata: Dict
    visualization_path: Optional[Path] = None
    

class PreprocessingPipeline:
    """High-level preprocessing pipeline: Extract → Crop → Compose → Visualize.
    
    Supports automatic cropping to area of interest using bbox parameter.
    
    Example:
        >>> pipeline = PreprocessingPipeline()
        >>> result = pipeline.run(
        ...     zip_path="data/raw/product.zip",
        ...     bbox=[9.0, 45.3, 9.4, 45.6]  # Crop to Milano area
        ... )
        >>> print(f"RGB shape: {result.rgb.shape}")
    """
    
    def __init__(
        self,
        output_dir: str = "data/processed",
        bands: Optional[List[str]] = None,
        resolution: str = "10m"
    ):
        """Initialize preprocessing pipeline.
        
        Args:
            output_dir: Directory to save processed data
            bands: Bands to extract (default: RGB+NIR)
            resolution: Band resolution (10m, 20m, 60m)
        """
        self.output_dir = Path(output_dir)
        self.extractor = BandExtractor(str(output_dir))
        self.composer = BandComposer()
        self.bands = bands or ['B02', 'B03', 'B04', 'B08']
        self.resolution = resolution
    
    def run(
        self,
        zip_path: str,
        bbox: Optional[List[float]] = None,
        save_visualization: bool = True,
        open_visualization: bool = False
    ) -> PreprocessingResult:
        """Run complete preprocessing pipeline with optional cropping.
        
        Args:
            zip_path: Path to Sentinel-2 ZIP file
            bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84.
                  If provided, data will be cropped to this area.
            save_visualization: Save RGB/FCC/NDVI visualization
            open_visualization: Open visualization after saving
        
        Returns:
            PreprocessingResult with all processed data
        """
        zip_path = Path(zip_path)
        product_name = zip_path.stem
        
        # Extract bands
        bands = self.extractor.extract_bands(
            zip_path=str(zip_path),
            bands=self.bands,
            resolution=self.resolution
        )
        
        # Read bands (with optional crop)
        if bbox:
            band_data = self.extractor.read_bands(bands, bbox=bbox)
        else:
            band_data = self.extractor.read_bands(bands)
        
        # Create composites
        rgb = self.composer.create_rgb(bands=band_data)
        fcc = self.composer.create_fcc(bands=band_data)
        ndvi = self.composer.create_ndvi(bands=band_data)
        
        # Metadata
        metadata = {
            'product_name': product_name,
            'bands_extracted': list(bands.keys()),
            'rgb_shape': rgb.shape,
            'fcc_shape': fcc.shape,
            'ndvi_shape': ndvi.shape,
            'ndvi_range': (float(ndvi.min()), float(ndvi.max())),
            'cropped': bbox is not None,
            'bbox': bbox if bbox else None
        }
        
        # Save visualization
        visualization_path = None
        if save_visualization:
            output_image = self.output_dir / product_name / "analysis_visualization.png"
            output_image.parent.mkdir(parents=True, exist_ok=True)
            
            self.composer.visualize_composites(
                rgb=rgb,
                fcc=fcc,
                ndvi=ndvi,
                output_path=str(output_image)
            )
            
            visualization_path = output_image
            
            # Open if requested
            if open_visualization:
                import subprocess
                try:
                    subprocess.run(["start", str(output_image)], shell=True, check=True)
                except:
                    pass  # Silent fail
        
        return PreprocessingResult(
            product_name=product_name,
            bands=bands,
            band_data=band_data,
            rgb=rgb,
            fcc=fcc,
            ndvi=ndvi,
            metadata=metadata,
            visualization_path=visualization_path
        )

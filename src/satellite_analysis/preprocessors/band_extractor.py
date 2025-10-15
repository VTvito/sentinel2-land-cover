"""Band extraction from Sentinel-2 ZIP products."""

import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import rasterio
import numpy as np


class BandExtractor:
    """Extract and read bands from Sentinel-2 .SAFE ZIP archives.
    
    Sentinel-2 L2A structure:
        *.SAFE/GRANULE/*/IMG_DATA/
            R10m/ - 10m resolution (B02, B03, B04, B08)
            R20m/ - 20m resolution (B01, B05, B06, B07, B8A, B11, B12)
            R60m/ - 60m resolution (B01, B09)
    """
    
    # Band definitions
    BANDS_10M = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
    BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # RedEdge, NIR narrow, SWIR
    BANDS_60M = ['B01', 'B09']  # Coastal aerosol, Water vapor
    
    BAND_NAMES = {
        'B01': 'Coastal aerosol',
        'B02': 'Blue',
        'B03': 'Green',
        'B04': 'Red',
        'B05': 'Red Edge 1',
        'B06': 'Red Edge 2',
        'B07': 'Red Edge 3',
        'B08': 'NIR',
        'B8A': 'NIR narrow',
        'B09': 'Water vapor',
        'B11': 'SWIR 1',
        'B12': 'SWIR 2'
    }
    
    def __init__(self, output_dir: str = "data/processed"):
        """Initialize band extractor.
        
        Args:
            output_dir: Directory to save extracted bands
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_bands(
        self,
        zip_path: str,
        bands: Optional[List[str]] = None,
        resolution: str = "10m"
    ) -> Dict[str, Path]:
        """Extract specified bands from Sentinel-2 ZIP.
        
        Args:
            zip_path: Path to Sentinel-2 ZIP file
            bands: List of bands to extract (e.g., ['B02', 'B03', 'B04'])
                   If None, extracts all bands for the resolution
            resolution: Target resolution ('10m', '20m', '60m')
        
        Returns:
            Dictionary mapping band name to extracted file path
        """
        zip_path = Path(zip_path)
        
        if bands is None:
            bands = self._get_default_bands(resolution)
        
        # Find .SAFE directory in ZIP
        with zipfile.ZipFile(zip_path, 'r') as zf:
            safe_dir = self._find_safe_dir(zf)
            if not safe_dir:
                raise ValueError(f"No .SAFE directory found in {zip_path}")
            
            # Extract bands
            extracted = {}
            for band in bands:
                band_file = self._find_band_file(zf, safe_dir, band, resolution)
                if band_file:
                    output_path = self._extract_band(zf, band_file, zip_path.stem, band)
                    extracted[band] = output_path
        
        return extracted
    
    def read_band(self, band_path: Path) -> np.ndarray:
        """Read band data from JP2 file.
        
        Args:
            band_path: Path to band JP2 file
        
        Returns:
            Band data as numpy array
        """
        with rasterio.open(band_path) as src:
            return src.read(1)
    
    def read_bands(
        self, 
        band_paths: Dict[str, Path],
        bbox: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Read multiple bands into memory, optionally cropped to bbox.
        
        Args:
            band_paths: Dictionary mapping band name to file path
            bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84.
                  If provided, bands will be cropped to this area.
        
        Returns:
            Dictionary mapping band name to numpy array (cropped if bbox provided)
        """
        if bbox is None:
            return {band: self.read_band(path) for band, path in band_paths.items()}
        else:
            return self._read_bands_cropped(band_paths, bbox)
    
    def get_band_metadata(self, band_path: Path) -> Dict:
        """Get band metadata (CRS, transform, shape, etc).
        
        Args:
            band_path: Path to band JP2 file
        
        Returns:
            Dictionary with metadata
        """
        with rasterio.open(band_path) as src:
            return {
                'crs': src.crs,
                'transform': src.transform,
                'shape': src.shape,
                'bounds': src.bounds,
                'dtype': src.dtypes[0],
                'nodata': src.nodata
            }
    
    def _get_default_bands(self, resolution: str) -> List[str]:
        """Get default bands for resolution."""
        if resolution == "10m":
            return self.BANDS_10M
        elif resolution == "20m":
            return self.BANDS_20M
        elif resolution == "60m":
            return self.BANDS_60M
        else:
            raise ValueError(f"Invalid resolution: {resolution}")
    
    def _find_safe_dir(self, zf: zipfile.ZipFile) -> Optional[str]:
        """Find .SAFE directory in ZIP."""
        for name in zf.namelist():
            if '.SAFE/' in name:
                return name.split('.SAFE/')[0] + '.SAFE'
        return None
    
    def _find_band_file(
        self,
        zf: zipfile.ZipFile,
        safe_dir: str,
        band: str,
        resolution: str
    ) -> Optional[str]:
        """Find band file in ZIP."""
        # Pattern: GRANULE/*/IMG_DATA/R{resolution}/{tile}_{date}_{band}_{resolution}.jp2
        # Example: .../R10m/T32TMR_20230312T101729_B04_10m.jp2
        resolution_dir = f"R{resolution}"
        
        for name in zf.namelist():
            if (f"{safe_dir}/GRANULE/" in name and
                f"/IMG_DATA/{resolution_dir}/" in name and
                f"_{band}_{resolution}" in name and
                name.endswith('.jp2')):
                return name
        
        return None
    
    def _extract_band(
        self,
        zf: zipfile.ZipFile,
        band_file: str,
        product_name: str,
        band: str
    ) -> Path:
        """Extract band file from ZIP."""
        # Output: data/processed/{product_name}/{band}.jp2
        output_dir = self.output_dir / product_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{band}.jp2"
        
        # Extract
        with zf.open(band_file) as source:
            with open(output_path, 'wb') as target:
                target.write(source.read())
        
        return output_path
    
    def _read_bands_cropped(
        self,
        band_paths: Dict[str, Path],
        bbox: List[float]
    ) -> Dict[str, np.ndarray]:
        """Read bands and crop to bounding box.
        
        Args:
            band_paths: Dictionary mapping band name to file path
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84
        
        Returns:
            Dictionary mapping band name to cropped numpy array
        """
        from rasterio.mask import mask as rasterio_mask
        from rasterio.warp import transform_geom
        from shapely.geometry import box, shape
        
        # Get CRS from first band
        first_band = list(band_paths.values())[0]
        with rasterio.open(first_band) as src:
            raster_crs = src.crs
        
        # Create bbox polygon in WGS84
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
        
        # Crop each band
        cropped = {}
        for band_name, band_path in band_paths.items():
            with rasterio.open(band_path) as src:
                try:
                    out_image, out_transform = rasterio_mask(
                        src,
                        [bbox_transformed],
                        crop=True,
                        all_touched=False
                    )
                    
                    # Remove band dimension if present
                    if out_image.ndim == 3:
                        out_image = out_image[0]
                    
                    cropped[band_name] = out_image
                    
                except Exception as e:
                    # Fallback to full image if crop fails
                    cropped[band_name] = src.read(1)
        
        return cropped


class BandComposer:
    """Create band composites (RGB, FCC, etc) from extracted bands."""
    
    def __init__(self):
        """Initialize band composer."""
        pass
    
    def create_rgb(
        self,
        bands: Dict[str, np.ndarray],
        stretch: bool = True
    ) -> np.ndarray:
        """Create RGB composite from B04, B03, B02.
        
        Args:
            bands: Dictionary with at least B02, B03, B04
            stretch: Apply histogram stretch
        
        Returns:
            RGB array (height, width, 3)
        """
        if not all(b in bands for b in ['B02', 'B03', 'B04']):
            raise ValueError("RGB requires bands B02, B03, B04")
        
        rgb = np.stack([bands['B04'], bands['B03'], bands['B02']], axis=-1)
        
        if stretch:
            rgb = self._histogram_stretch(rgb)
        
        return rgb
    
    def create_fcc(
        self,
        bands: Dict[str, np.ndarray],
        stretch: bool = True
    ) -> np.ndarray:
        """Create False Color Composite (NIR, Red, Green).
        
        Args:
            bands: Dictionary with at least B03, B04, B08
            stretch: Apply histogram stretch
        
        Returns:
            FCC array (height, width, 3)
        """
        if not all(b in bands for b in ['B03', 'B04', 'B08']):
            raise ValueError("FCC requires bands B03, B04, B08")
        
        fcc = np.stack([bands['B08'], bands['B04'], bands['B03']], axis=-1)
        
        if stretch:
            fcc = self._histogram_stretch(fcc)
        
        return fcc
    
    def create_ndvi(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate NDVI (Normalized Difference Vegetation Index).
        
        Args:
            bands: Dictionary with at least B04, B08
        
        Returns:
            NDVI array (height, width)
        """
        if not all(b in bands for b in ['B04', 'B08']):
            raise ValueError("NDVI requires bands B04, B08")
        
        nir = bands['B08'].astype(float)
        red = bands['B04'].astype(float)
        
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        return ndvi
    
    def visualize_composites(
        self,
        rgb: np.ndarray,
        fcc: np.ndarray,
        ndvi: np.ndarray,
        output_path: str
    ) -> None:
        """Create visualization with RGB, FCC and NDVI side by side.
        
        Args:
            rgb: RGB composite (H, W, 3)
            fcc: False Color Composite (H, W, 3)
            ndvi: NDVI array (H, W)
            output_path: Path to save the output image
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RGB
        axes[0].imshow(rgb)
        axes[0].set_title('RGB (True Color)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # FCC
        axes[1].imshow(fcc)
        axes[1].set_title('FCC (False Color)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # NDVI con colormap
        # Verde scuro = alta vegetazione, rosso/marrone = bassa vegetazione/suolo
        colors = ['#8B4513', '#D2691E', '#F4A460', '#FFFFE0', '#90EE90', '#00FF00', '#006400']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('ndvi', colors, N=n_bins)
        
        im = axes[2].imshow(ndvi, cmap=cmap, vmin=-1, vmax=1)
        axes[2].set_title('NDVI (Vegetation Index)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Colorbar per NDVI
        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('NDVI Value', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _histogram_stretch(self, array: np.ndarray, percentile: int = 2) -> np.ndarray:
        """Apply percentile histogram stretch."""
        stretched = np.zeros_like(array, dtype=np.uint8)
        
        for i in range(array.shape[-1]):
            band = array[..., i]
            p_low = np.percentile(band, percentile)
            p_high = np.percentile(band, 100 - percentile)
            
            band_stretched = np.clip((band - p_low) / (p_high - p_low) * 255, 0, 255)
            stretched[..., i] = band_stretched.astype(np.uint8)
        
        return stretched

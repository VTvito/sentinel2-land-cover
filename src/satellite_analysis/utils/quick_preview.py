"""Quick preview generation from Sentinel-2 products."""

import zipfile
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np


class QuickPreview:
    """Generate quick previews from Sentinel-2 TCI (True Color Image).
    
    TCI is a pre-rendered RGB composite already in the product at 10m resolution,
    perfect for quick visual inspection without heavy processing.
    """
    
    def __init__(self, output_dir: str = "data/previews"):
        """Initialize quick preview generator.
        
        Args:
            output_dir: Directory to save preview images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_preview(
        self,
        zip_path: str,
        product_info: Optional[dict] = None,
        thumbnail_size: int = 1200
    ) -> Path:
        """Generate quick preview from TCI band.
        
        Args:
            zip_path: Path to Sentinel-2 ZIP file
            product_info: Optional metadata (date, cloud cover, etc)
            thumbnail_size: Max dimension for thumbnail (pixels)
        
        Returns:
            Path to saved preview image
        """
        zip_path = Path(zip_path)
        
        # Extract TCI
        tci_array = self._extract_tci(zip_path)
        
        if tci_array is None:
            raise ValueError(f"No TCI found in {zip_path}")
        
        # Create thumbnail
        thumbnail = self._create_thumbnail(tci_array, thumbnail_size)
        
        # Create figure with metadata overlay
        fig = self._create_preview_figure(thumbnail, zip_path.stem, product_info)
        
        # Save
        output_path = self.output_dir / f"{zip_path.stem}_preview.png"
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def generate_batch_preview(
        self,
        zip_paths: list,
        products_info: Optional[list] = None
    ) -> list:
        """Generate previews for multiple products.
        
        Args:
            zip_paths: List of ZIP file paths
            products_info: Optional list of metadata dicts
        
        Returns:
            List of preview image paths
        """
        previews = []
        
        for idx, zip_path in enumerate(zip_paths):
            info = products_info[idx] if products_info else None
            try:
                preview_path = self.generate_preview(zip_path, info)
                previews.append(preview_path)
            except Exception as e:
                print(f"Warning: Could not generate preview for {zip_path}: {e}")
                continue
        
        return previews
    
    def _extract_tci(self, zip_path: Path) -> Optional[np.ndarray]:
        """Extract TCI (True Color Image) from ZIP.
        
        TCI path pattern: GRANULE/*/IMG_DATA/R10m/*_TCI_10m.jp2
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for TCI extraction. Install with: pip install rasterio")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find TCI file
            tci_file = None
            for name in zf.namelist():
                if '/IMG_DATA/R10m/' in name and '_TCI_10m.jp2' in name:
                    tci_file = name
                    break
            
            if not tci_file:
                return None
            
            # Extract to temp location
            temp_dir = self.output_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            tci_path = zf.extract(tci_file, temp_dir)
            
            # Read with rasterio
            with rasterio.open(tci_path) as src:
                # TCI is RGB, read all 3 bands
                tci = src.read()  # (3, H, W)
                tci = np.transpose(tci, (1, 2, 0))  # (H, W, 3)
            
            # Cleanup temp
            Path(tci_path).unlink()
            
            return tci
    
    def _create_thumbnail(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """Create thumbnail preserving aspect ratio."""
        h, w = image.shape[:2]
        
        # Calculate new size
        if max(h, w) <= max_size:
            return image
        
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use PIL for high-quality resize
        pil_image = Image.fromarray(image)
        pil_image.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
        
        return np.array(pil_image)
    
    def _create_preview_figure(
        self,
        thumbnail: np.ndarray,
        product_name: str,
        product_info: Optional[dict] = None
    ) -> plt.Figure:
        """Create figure with thumbnail and metadata overlay."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display thumbnail
        ax.imshow(thumbnail)
        ax.axis('off')
        
        # Title
        title = f"Preview: {product_name}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Metadata overlay
        if product_info:
            info_text = []
            
            if 'datetime' in product_info:
                info_text.append(f"Date: {product_info['datetime'][:10]}")
            
            if 'eo:cloud_cover' in product_info:
                cloud = product_info['eo:cloud_cover']
                info_text.append(f"Cloud: {cloud:.1f}%")
            
            if 'platform' in product_info:
                info_text.append(f"Satellite: {product_info['platform']}")
            
            if info_text:
                textstr = '\n'.join(info_text)
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', bbox=props)
        
        # Add scale reference
        h, w = thumbnail.shape[:2]
        scale_text = f"Size: {w} x {h} pixels (downsampled)"
        ax.text(0.98, 0.02, scale_text, transform=ax.transAxes,
               fontsize=8, horizontalalignment='right',
               verticalalignment='bottom', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        return fig
    
    def generate_comparison_grid(
        self,
        zip_paths: list,
        products_info: Optional[list] = None,
        cols: int = 3
    ) -> Path:
        """Generate comparison grid of multiple products.
        
        Args:
            zip_paths: List of ZIP paths
            products_info: Optional metadata list
            cols: Number of columns in grid
        
        Returns:
            Path to comparison image
        """
        n_products = len(zip_paths)
        rows = (n_products + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        
        if n_products == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, zip_path in enumerate(zip_paths):
            zip_path = Path(zip_path)
            
            # Extract TCI
            tci = self._extract_tci(zip_path)
            
            if tci is not None:
                # Thumbnail
                thumbnail = self._create_thumbnail(tci, 600)
                
                # Display
                axes[idx].imshow(thumbnail)
                
                # Title with metadata
                title = zip_path.stem[:30]
                if products_info and idx < len(products_info):
                    info = products_info[idx]
                    date = info.get('datetime', '')[:10]
                    cloud = info.get('eo:cloud_cover', 0)
                    title = f"{date}\nCloud: {cloud:.1f}%"
                
                axes[idx].set_title(title, fontsize=9)
            else:
                axes[idx].text(0.5, 0.5, 'No TCI found',
                             ha='center', va='center',
                             transform=axes[idx].transAxes)
            
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_products, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Product Previews - Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "comparison_grid.png"
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path

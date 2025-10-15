"""High-level pipeline for downloading Sentinel data."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from satellite_analysis.config import Config
from satellite_analysis.downloaders import (
    OAuth2AuthStrategy,
    SentinelHubCatalog,
    ProductDownloader
)


@dataclass
class DownloadResult:
    """Result of a download pipeline execution."""
    
    search_results: Dict[str, Any]
    downloaded_files: List[Path]
    total_products: int
    downloaded_count: int
    failed_count: int
    preview_files: List[Path] = None
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.preview_files is None:
            self.preview_files = []
    
    def __str__(self) -> str:
        preview_info = f"\n  Previews: {len(self.preview_files)}" if self.preview_files else ""
        return (
            f"DownloadResult(\n"
            f"  Total products: {self.total_products}\n"
            f"  Downloaded: {self.downloaded_count}\n"
            f"  Failed: {self.failed_count}\n"
            f"  Success rate: {self.success_rate:.1f}%{preview_info}\n"
            f")"
        )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_products == 0:
            return 0.0
        return (self.downloaded_count / self.total_products) * 100


class DownloadPipeline:
    """High-level pipeline for searching and downloading Sentinel products.
    
    This class provides a simple interface for the complete download workflow:
    1. Authentication
    2. Catalog search
    3. Product download
    
    Example:
        >>> pipeline = DownloadPipeline.from_config("config/config.yaml")
        >>> result = pipeline.run(
        ...     bbox=[9.0, 45.3, 9.3, 45.6],
        ...     start_date="2023-03-01",
        ...     end_date="2023-03-15"
        ... )
        >>> print(f"Downloaded {result.downloaded_count} products")
    """
    
    def __init__(
        self,
        auth_strategy: OAuth2AuthStrategy,
        output_dir: str = "data/raw",
        max_cloud_cover: float = 10.0,
        collection: str = "sentinel-2-l2a",
        generate_preview: bool = True
    ):
        """Initialize download pipeline.
        
        Args:
            auth_strategy: OAuth2 authentication strategy
            output_dir: Directory to save downloaded products
            max_cloud_cover: Maximum cloud coverage (0-100)
            collection: Sentinel collection name
            generate_preview: Generate preview after download
        """
        self.auth = auth_strategy
        self.output_dir = output_dir
        self.max_cloud_cover = max_cloud_cover
        self.collection = collection
        self.generate_preview = generate_preview
        
        # Initialize components
        self.session = self.auth.get_session()
        self.catalog = SentinelHubCatalog(self.session)
        self.downloader = ProductDownloader(self.session, output_dir)
        
        if self.generate_preview:
            from satellite_analysis.utils import QuickPreview
            self.preview_generator = QuickPreview()
    
    @classmethod
    def from_config(cls, config_path: str) -> 'DownloadPipeline':
        """Create pipeline from configuration file.
        
        Args:
            config_path: Path to config.yaml
            
        Returns:
            Configured DownloadPipeline instance
        """
        config = Config.from_yaml(config_path)
        
        auth = OAuth2AuthStrategy(
            client_id=config.sentinel.client_id,
            client_secret=config.sentinel.client_secret
        )
        
        return cls(
            auth_strategy=auth,
            output_dir="data/raw",
            max_cloud_cover=config.sentinel.max_cloud_cover,
            collection="sentinel-2-l2a"
        )
    
    def search(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search for products in catalog.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of results
            
        Returns:
            Catalog search results
        """
        # Catalog search parameters
        results = self.catalog.search(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            collection=self.collection,
            cloud_cover_max=self.max_cloud_cover,
            limit=limit
        )
        
        return results
    
    def download(
        self,
        search_results: Dict[str, Any],
        max_products: Optional[int] = None
    ) -> List[Path]:
        """Download products from search results.
        
        Args:
            search_results: Results from catalog search
            max_products: Maximum number of products to download
            
        Returns:
            List of downloaded file paths
        """
        features = search_results.get("features", [])
        
        if not features:
            pass  # No products to download
            return []
        
        return self.downloader.download_products(
            features=features,
            max_products=max_products
        )
    
    def run(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        limit: int = 10,
        max_downloads: Optional[int] = None
    ) -> DownloadResult:
        """Run complete download pipeline: search + download.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of search results
            max_downloads: Maximum number of products to download
            
        Returns:
            DownloadResult with execution summary
        """
        # Search
        search_results = self.search(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        total_products = len(search_results.get("features", []))
        
        # Download
        downloaded_files = self.download(
            search_results=search_results,
            max_products=max_downloads
        )
        
        # Generate previews
        preview_files = []
        if self.generate_preview and downloaded_files:
            features = search_results.get("features", [])
            for file_path in downloaded_files:
                # Find matching feature for metadata
                file_stem = file_path.stem
                matching_feature = None
                for feature in features:
                    product_id = feature.get("properties", {}).get("id")
                    if product_id and product_id in file_stem:
                        matching_feature = feature.get("properties", {})
                        break
                
                try:
                    preview_path = self.preview_generator.generate_preview(
                        zip_path=str(file_path),
                        product_info=matching_feature
                    )
                    preview_files.append(preview_path)
                except Exception as e:
                    # Continue if preview fails
                    pass
        
        # Summary
        result = DownloadResult(
            search_results=search_results,
            downloaded_files=downloaded_files,
            total_products=total_products,
            downloaded_count=len(downloaded_files),
            failed_count=total_products - len(downloaded_files),
            preview_files=preview_files
        )
        
        return result


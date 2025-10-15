"""Base classes for data downloaders."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SearchQuery:
    """Parameters for searching satellite imagery."""
    
    collections: List[str]  # e.g., ["sentinel-2-l2a"]
    start_date: datetime
    end_date: datetime
    bbox: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
    geometry: Optional[Dict] = None  # GeoJSON geometry
    cloud_cover_max: Optional[float] = None
    limit: int = 10


@dataclass
class Product:
    """Represents a satellite product."""
    
    id: str
    title: str
    geometry: Dict
    datetime: datetime
    cloud_cover: Optional[float]
    collection: str
    properties: Dict[str, Any]
    download_url: Optional[str] = None


class AuthStrategy(ABC):
    """Abstract base class for authentication strategies."""
    
    @abstractmethod
    def authenticate(self) -> Any:
        """Authenticate and return session or token.
        
        Returns:
            Authenticated session or token object
        """
        pass
    
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if currently authenticated.
        
        Returns:
            True if authenticated, False otherwise
        """
        pass


class CatalogStrategy(ABC):
    """Abstract base class for catalog search strategies."""
    
    def __init__(self, auth_strategy: AuthStrategy):
        """Initialize catalog with authentication.
        
        Args:
            auth_strategy: Authentication strategy to use
        """
        self.auth_strategy = auth_strategy
    
    @abstractmethod
    def search(self, query: SearchQuery) -> List[Product]:
        """Search for products matching query.
        
        Args:
            query: Search parameters
            
        Returns:
            List of matching products
        """
        pass
    
    @abstractmethod
    def get_product_info(self, product_id: str) -> Product:
        """Get detailed information about a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product information
        """
        pass


class DownloadStrategy(ABC):
    """Abstract base class for download strategies."""
    
    def __init__(self, auth_strategy: AuthStrategy):
        """Initialize downloader with authentication.
        
        Args:
            auth_strategy: Authentication strategy to use
        """
        self.auth_strategy = auth_strategy
    
    @abstractmethod
    def download(
        self,
        product: Product,
        output_dir: Path,
        bands: Optional[List[str]] = None
    ) -> Path:
        """Download a product.
        
        Args:
            product: Product to download
            output_dir: Directory to save to
            bands: Optional list of bands to download
            
        Returns:
            Path to downloaded file
        """
        pass
    
    @abstractmethod
    def download_batch(
        self,
        products: List[Product],
        output_dir: Path,
        bands: Optional[List[str]] = None
    ) -> List[Path]:
        """Download multiple products.
        
        Args:
            products: Products to download
            output_dir: Directory to save to
            bands: Optional list of bands to download
            
        Returns:
            List of paths to downloaded files
        """
        pass

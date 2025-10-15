"""Catalog search strategies for Sentinel data."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import requests
from datetime import datetime


class CatalogStrategy(ABC):
    """Abstract base class for catalog search strategies."""
    
    @abstractmethod
    def search(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        collection: str = "sentinel-2-l2a",
        cloud_cover_max: float = 100.0,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Search for products in the catalog.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (ISO format: YYYY-MM-DD)
            end_date: End date (ISO format: YYYY-MM-DD)
            collection: Collection name
            cloud_cover_max: Maximum cloud coverage percentage
            limit: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary with search results
        """
        pass


class SentinelHubCatalog(CatalogStrategy):
    """Catalog search using Sentinel Hub API."""
    
    CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    
    def __init__(self, session: requests.Session):
        """Initialize catalog with authenticated session.
        
        Args:
            session: Authenticated requests session
        """
        self.session = session
    
    def search(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        collection: str = "sentinel-2-l2a",
        cloud_cover_max: float = 100.0,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Search Sentinel Hub catalog.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            collection: Collection name (default: sentinel-2-l2a)
            cloud_cover_max: Maximum cloud coverage (0-100)
            limit: Maximum results to return
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'features' list containing products
            
        Raises:
            ValueError: If parameters are invalid
            requests.HTTPError: If API request fails
        """
        # Validate inputs
        self._validate_bbox(bbox)
        self._validate_dates(start_date, end_date)
        self._validate_cloud_cover(cloud_cover_max)
        
        # Build query using STAC API format
        query_params = {
            "collections": [collection],
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "bbox": bbox,
            "limit": limit
        }
        
        # Note: Cloud cover filtering will be done client-side
        # as the API doesn't support query filtering in the catalog endpoint
        
        # Add any additional filters
        if kwargs:
            query_params.update(kwargs)
        
        # Make request
        try:
            response = self.session.post(
                self.CATALOG_URL,
                json=query_params,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            results = response.json()
            
            # Calculate center point of requested bbox
            center_lon = (bbox[0] + bbox[2]) / 2
            center_lat = (bbox[1] + bbox[3]) / 2
            
            # Filter by cloud cover AND check if tile contains center point
            features = results.get("features", [])
            filtered_features = []
            
            for f in features:
                # Check cloud cover
                cloud_cover = f.get("properties", {}).get("eo:cloud_cover", 100)
                if cloud_cover > cloud_cover_max:
                    continue
                
                # Check if tile contains center point
                geometry = f.get("geometry", {})
                if geometry.get("type") in ["Polygon", "MultiPolygon"]:
                    coords = geometry.get("coordinates", [])
                    
                    # Handle MultiPolygon
                    if geometry.get("type") == "MultiPolygon":
                        coords = coords[0] if coords else []
                    
                    # Get tile bounds
                    if coords and len(coords) > 0:
                        points = coords[0] if isinstance(coords[0][0], list) else coords
                        if points:
                            lons = [p[0] for p in points]
                            lats = [p[1] for p in points]
                            
                            # Check if center point is inside tile bounds
                            if (min(lons) <= center_lon <= max(lons) and 
                                min(lats) <= center_lat <= max(lats)):
                                filtered_features.append(f)
                else:
                    # If no geometry, keep the feature
                    filtered_features.append(f)
            
            results["features"] = filtered_features[:limit]
            
            # Add summary
            pass  # Products filtered by cloud cover and center point
            
            return results
            
        except requests.exceptions.HTTPError as e:
            raise requests.HTTPError(
                f"Catalog search failed: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Catalog search error: {e}")
    
    def _validate_bbox(self, bbox: List[float]) -> None:
        """Validate bounding box format and values."""
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must be a list of 4 values: [min_lon, min_lat, max_lon, max_lat]")
        
        min_lon, min_lat, max_lon, max_lat = bbox
        
        if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {min_lon}, {max_lon}")
        
        if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {min_lat}, {max_lat}")
        
        if min_lon >= max_lon:
            raise ValueError(f"min_lon must be < max_lon, got {min_lon} >= {max_lon}")
        
        if min_lat >= max_lat:
            raise ValueError(f"min_lat must be < max_lat, got {min_lat} >= {max_lat}")
    
    def _validate_dates(self, start_date: str, end_date: str) -> None:
        """Validate date format and range."""
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
        
        if start >= end:
            raise ValueError(f"start_date must be before end_date: {start_date} >= {end_date}")
    
    def _validate_cloud_cover(self, cloud_cover: float) -> None:
        """Validate cloud cover percentage."""
        if not (0 <= cloud_cover <= 100):
            raise ValueError(f"cloud_cover must be between 0 and 100, got {cloud_cover}")
    
    def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Dictionary with product details
        """
        # Implementation depends on specific API endpoint
        # This is a placeholder
        raise NotImplementedError("Product info retrieval not yet implemented")

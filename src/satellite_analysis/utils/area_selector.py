"""Area selection helper with city-based coordinates."""

from typing import Tuple, List, Optional
from satellite_analysis.utils.geospatial import get_location, create_area_polygon
import json
from pathlib import Path


class AreaSelector:
    """Helper for selecting geographic areas by city name or custom coordinates.
    
    Simplifies the process of defining bounding boxes for satellite downloads.
    """
    
    # Predefined cities with optimized coordinates
    CITIES = {
        "Milan": {"coords": (45.464, 9.190), "radius_km": 15},
        "Rome": {"coords": (41.902, 12.496), "radius_km": 20},
        "Florence": {"coords": (43.769, 11.256), "radius_km": 12},
        "Venice": {"coords": (45.440, 12.316), "radius_km": 10},
        "Turin": {"coords": (45.070, 7.686), "radius_km": 15},
        "Naples": {"coords": (40.852, 14.268), "radius_km": 18},
        "Bologna": {"coords": (44.494, 11.342), "radius_km": 12},
        "Genoa": {"coords": (44.407, 8.934), "radius_km": 12},
    }
    
    def __init__(self, cache_file: str = "config/area_cache.json"):
        """Initialize area selector.
        
        Args:
            cache_file: Path to cache file for storing custom areas
        """
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def select_by_city(
        self,
        city: str,
        radius_km: Optional[float] = None,
        country: str = "Italy"
    ) -> Tuple[List[float], dict]:
        """Select area by city name.
        
        Args:
            city: City name (e.g., "Milan", "Rome")
            radius_km: Radius in km (overrides default)
            country: Country name (default: Italy)
        
        Returns:
            Tuple of (bbox, metadata) where:
                - bbox: [min_lon, min_lat, max_lon, max_lat]
                - metadata: {'center': (lat, lon), 'radius_km': float, 'area_km2': float}
        
        Example:
            >>> selector = AreaSelector()
            >>> bbox, meta = selector.select_by_city("Milan", radius_km=15)
            >>> print(f"Area: {meta['area_km2']:.1f} km²")
        """
        # Check predefined cities first
        if city in self.CITIES:
            city_info = self.CITIES[city]
            lat, lon = city_info["coords"]
            radius = radius_km or city_info["radius_km"]
            
            # Create Location-like object for compatibility
            from geopy.location import Location
            from geopy.point import Point
            center_obj = Location("", Point(lat, lon), {})
        else:
            # Query Nominatim
            query = f"{city}, {country}"
            center_obj = get_location(query, country="")
            
            if center_obj is None:
                raise ValueError(f"City '{query}' not found. Try more specific name.")
            
            radius = radius_km or 15.0  # Default 15km
            lat, lon = center_obj.latitude, center_obj.longitude
        
        # Create polygon
        polygon = create_area_polygon(center_obj, radius_km=radius)
        
        # Convert polygon to bbox
        # create_area_polygon returns (lat, lon) coordinates, so bounds are (min_lat, min_lon, max_lat, max_lon)
        # We need (min_lon, min_lat, max_lon, max_lat) for STAC API
        bounds = polygon.bounds
        bbox = [bounds[1], bounds[0], bounds[3], bounds[2]]  # Swap to (min_lon, min_lat, max_lon, max_lat)
        
        # Calculate metadata
        metadata = {
            'city': city,
            'center': (lat, lon),
            'radius_km': radius,
            'area_km2': self._calculate_area(bbox),
            'bbox': bbox
        }
        
        # Cache for future use
        self._cache_area(city, metadata)
        
        return bbox, metadata
    
    def select_by_coordinates(
        self,
        lat: float,
        lon: float,
        radius_km: float = 15.0,
        name: Optional[str] = None
    ) -> Tuple[List[float], dict]:
        """Select area by center coordinates and radius.
        
        Args:
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees)
            radius_km: Radius in km
            name: Optional name for this area
        
        Returns:
            Tuple of (bbox, metadata)
        
        Example:
            >>> bbox, meta = selector.select_by_coordinates(45.464, 9.190, radius_km=15)
        """
        # Create Location object
        from geopy.location import Location
        from geopy.point import Point
        center_obj = Location("", Point(lat, lon), {})
        
        # Create polygon
        polygon = create_area_polygon(center_obj, radius_km=radius_km)
        
        # Convert to bbox
        bounds = polygon.bounds
        bbox = [bounds[1], bounds[0], bounds[3], bounds[2]]  # Swap to (min_lon, min_lat, max_lon, max_lat)
        
        metadata = {
            'name': name or f"Custom_{lat:.3f}_{lon:.3f}",
            'center': (lat, lon),
            'radius_km': radius_km,
            'area_km2': self._calculate_area(bbox),
            'bbox': bbox
        }
        
        if name:
            self._cache_area(name, metadata)
        
        return bbox, metadata
    
    def select_by_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        name: Optional[str] = None
    ) -> Tuple[List[float], dict]:
        """Select area by explicit bounding box.
        
        Args:
            min_lon: Minimum longitude (west)
            min_lat: Minimum latitude (south)
            max_lon: Maximum longitude (east)
            max_lat: Maximum latitude (north)
            name: Optional name for this area
        
        Returns:
            Tuple of (bbox, metadata)
        
        Example:
            >>> bbox, meta = selector.select_by_bbox(9.0, 45.3, 9.3, 45.6)
        """
        bbox = [min_lon, min_lat, max_lon, max_lat]
        
        # Calculate center
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        metadata = {
            'name': name or f"BBox_{min_lon:.2f}_{min_lat:.2f}",
            'center': (center_lat, center_lon),
            'bbox': bbox,
            'area_km2': self._calculate_area(bbox)
        }
        
        if name:
            self._cache_area(name, metadata)
        
        return bbox, metadata
    
    def list_cached_areas(self) -> dict:
        """List all cached areas.
        
        Returns:
            Dictionary of cached areas
        """
        return self.cache
    
    def get_cached_area(self, name: str) -> Optional[dict]:
        """Get cached area by name.
        
        Args:
            name: Area name
        
        Returns:
            Area metadata or None if not found
        """
        return self.cache.get(name)
    
    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate approximate area in km² from bbox.
        
        Uses simple approximation: area ≈ (Δlon × Δlat) × 111² × cos(lat)
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Approximate km per degree
        lat_center = (min_lat + max_lat) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * abs(cosine_deg(lat_center))
        
        width_km = (max_lon - min_lon) * km_per_deg_lon
        height_km = (max_lat - min_lat) * km_per_deg_lat
        
        return width_km * height_km
    
    def _load_cache(self) -> dict:
        """Load area cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _cache_area(self, name: str, metadata: dict):
        """Cache area for future use."""
        self.cache[name] = metadata
        
        # Save to file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)


def cosine_deg(angle_deg: float) -> float:
    """Calculate cosine of angle in degrees."""
    import math
    return math.cos(math.radians(angle_deg))


# Convenience function for quick usage
def quick_select(
    city: str = "Milan",
    radius_km: float = 15.0,
    country: str = "Italy"
) -> List[float]:
    """Quick area selection by city name.
    
    Args:
        city: City name
        radius_km: Radius in km
        country: Country name
    
    Returns:
        Bounding box [min_lon, min_lat, max_lon, max_lat]
    
    Example:
        >>> bbox = quick_select("Milan", radius_km=15)
        >>> # Use in download pipeline
        >>> result = pipeline.run(bbox=bbox, ...)
    """
    selector = AreaSelector()
    bbox, metadata = selector.select_by_city(city, radius_km, country)
    
    print(f"Selected area:")
    print(f"  City: {city}")
    print(f"  Center: {metadata['center'][0]:.4f}°N, {metadata['center'][1]:.4f}°E")
    print(f"  Radius: {metadata['radius_km']} km")
    print(f"  Area: {metadata['area_km2']:.1f} km²")
    print(f"  BBox: {bbox}")
    
    return bbox

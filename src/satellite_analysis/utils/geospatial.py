"""Geospatial utility functions."""

import numpy as np
import utm
from geopy.geocoders import Nominatim
from geopy import distance
from shapely import geometry
from shapely.ops import transform
from typing import Tuple, List


def get_location(city: str, country: str, user_agent: str = "SatelliteApp", timeout: int = 10):
    """Get location coordinates from city and country.
    
    Args:
        city: City name
        country: Country name
        user_agent: User agent string for geocoding API
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        Location object with coordinates
    """
    geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
    location = geolocator.geocode(f"{city}, {country}")
    return location


def create_area_polygon(
    location,
    radius_km: float,
    shape: str = "rectangle"
) -> geometry.Polygon:
    """Create a polygon around a location.
    
    Args:
        location: Location object with point attribute
        radius_km: Radius in kilometers
        shape: Shape type - 'circle', 'rectangle', or 'triangle'
        
    Returns:
        Shapely Polygon object
        
    Raises:
        ValueError: If shape is not recognized
    """
    d = distance.distance(kilometers=radius_km)
    
    # Define bearings based on shape
    if shape == "circle":
        bearings = [i * 360 / 30 for i in range(31)]  # 30 points + 1 to close
    elif shape == "rectangle":
        # Aligned with cardinal directions: North, East, South, West
        bearings = [0, 90, 180, 270, 360]  # N-E-S-W rectangle
    elif shape == "triangle":
        bearings = [0, 120, 240, 360]  # Equilateral triangle
    else:
        raise ValueError(f"Unknown shape: {shape}. Use 'circle', 'rectangle', or 'triangle'")
    
    # Create vertices
    vertices = [d.destination(point=location.point, bearing=bearing) for bearing in bearings]
    
    # Create polygon
    poly = geometry.Polygon(vertices)
    
    return poly


def reverse_coordinates(poly: geometry.Polygon) -> geometry.Polygon:
    """Reverse latitude and longitude coordinates.
    
    Args:
        poly: Shapely Polygon with (lat, lon) coordinates
        
    Returns:
        Polygon with (lon, lat) coordinates
    """
    reverse_func = lambda x, y, z=None: (y, x)
    return transform(reverse_func, poly)


def polygon_to_utm(poly: geometry.Polygon) -> Tuple[List[float], List[float]]:
    """Convert polygon coordinates to UTM.
    
    Args:
        poly: Shapely Polygon with lat/lon coordinates
        
    Returns:
        Tuple of (x_coords, y_coords) in UTM
    """
    x_coords, y_coords = [], []
    
    for lat, lon in zip(*poly.boundary.coords.xy):
        easting, northing, _, _ = utm.from_latlon(lat, lon)
        x_coords.append(easting)
        y_coords.append(northing)
    
    return x_coords, y_coords


def save_polygon_coords(x: List[float], y: List[float], output_path: str) -> None:
    """Save polygon coordinates to CSV file.
    
    Args:
        x: X coordinates
        y: Y coordinates
        output_path: Path to save CSV file
    """
    lines = [f"{round(xi, 2)},{round(yi, 2)}" for xi, yi in zip(x, y)]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def load_polygon_coords(input_path: str) -> geometry.Polygon:
    """Load polygon coordinates from CSV file.
    
    Args:
        input_path: Path to CSV file
        
    Returns:
        Shapely Polygon object
    """
    with open(input_path) as f:
        coords_text = f.read()
    
    coords = [
        (float(line.split(',')[0]), float(line.split(',')[1]))
        for line in coords_text.splitlines()
    ]
    
    return geometry.Polygon(coords)


def reshape_image_to_table(image: np.ndarray) -> np.ndarray:
    """Reshape image from 3D to 2D table format.
    
    Args:
        image: Image array of shape (height, width, channels)
        
    Returns:
        Reshaped array of shape (height * width, channels)
    """
    return np.reshape(image, (-1, image.shape[-1]))


def reshape_table_to_image(
    image_shape: Tuple[int, int],
    labels: np.ndarray
) -> np.ndarray:
    """Reshape 1D labels back to 2D image format.
    
    Args:
        image_shape: Original (height, width) of image
        labels: 1D array of labels
        
    Returns:
        2D array of shape (height, width)
    """
    return labels.reshape(*image_shape)

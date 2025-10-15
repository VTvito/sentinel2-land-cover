"""
Test per la selezione e validazione di aree geografiche.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from satellite_analysis.utils import AreaSelector


def test_city_selection():
    """Test selezione area per città."""
    print("\n" + "=" * 60)
    print("TEST: Selezione area per città")
    print("=" * 60)
    
    selector = AreaSelector()
    bbox, info = selector.select_by_city("Milan", radius_km=15)
    
    print(f"\nCittà: Milan")
    print(f"Centro: {info['center'][0]:.4f}°N, {info['center'][1]:.4f}°E")
    print(f"BBox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"Superficie: {info['area_km2']:.1f} km²")
    
    # Verifica bbox centrato correttamente
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    error_lon = abs(center_lon - info['center'][1])
    error_lat = abs(center_lat - info['center'][0])
    
    print(f"\nVerifica centratura:")
    print(f"  Errore longitudine: {error_lon * 111000:.1f} m")
    print(f"  Errore latitudine: {error_lat * 111000:.1f} m")
    
    assert error_lon < 0.001 and error_lat < 0.001, "BBox non centrato!"
    print("  ✅ BBox correttamente centrato")
    
    return True


def test_coordinates_selection():
    """Test selezione area da coordinate."""
    print("\n" + "=" * 60)
    print("TEST: Selezione area da coordinate")
    print("=" * 60)
    
    selector = AreaSelector()
    bbox, info = selector.select_by_coordinates(
        lat=45.464, lon=9.190, radius_km=10
    )
    
    print(f"\nCoordinate: 45.464°N, 9.190°E")
    print(f"Raggio: 10 km")
    print(f"BBox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"Superficie: {info['area_km2']:.1f} km²")
    
    assert info['area_km2'] > 0, "Area non valida!"
    print("✅ Selezione da coordinate OK")
    
    return True


if __name__ == "__main__":
    print("\n🧪 TEST SUITE: Area Selection\n")
    
    success = True
    success &= test_city_selection()
    success &= test_coordinates_selection()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO")
        print("=" * 60 + "\n")
    else:
        print("\n❌ Alcuni test falliti")
        sys.exit(1)

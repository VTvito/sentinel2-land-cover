"""CLI tool for interactive area selection with map preview.

Usage:
    python scripts/select_area.py --city "Milan" --radius 15
    python scripts/select_area.py --lat 45.464 --lon 9.190 --radius 15
    python scripts/select_area.py --bbox 9.0 45.3 9.3 45.6
"""

import argparse
from satellite_analysis.utils import AreaSelector, quick_select
import folium
from pathlib import Path
import webbrowser


def generate_map_preview(bbox, metadata, output_path="preview_map.html"):
    """Generate interactive map preview with area overlay."""
    min_lon, min_lat, max_lon, max_lat = bbox
    center = metadata['center']
    
    # Create map centered on area
    m = folium.Map(
        location=[center[0], center[1]],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add rectangle for download area
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.2,
        popup=f"Download Area\n{metadata.get('area_km2', 0):.1f} km²"
    ).add_to(m)
    
    # Add center marker
    folium.Marker(
        location=[center[0], center[1]],
        popup=f"Center: {center[0]:.4f}°N, {center[1]:.4f}°E",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add corner markers
    corners = [
        ([min_lat, min_lon], "SW"),
        ([min_lat, max_lon], "SE"),
        ([max_lat, min_lon], "NW"),
        ([max_lat, max_lon], "NE")
    ]
    
    for corner, label in corners:
        folium.CircleMarker(
            location=corner,
            radius=5,
            color='darkred',
            fill=True,
            popup=f"{label}: {corner[0]:.4f}°, {corner[1]:.4f}°"
        ).add_to(m)
    
    # Add info box
    info_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 280px;
                background-color: white; border: 2px solid red;
                z-index: 9999; padding: 10px; border-radius: 5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <h4 style="margin-top: 0;">Download Area Selection</h4>
        <p><b>City:</b> {metadata.get('city', metadata.get('name', 'Custom'))}</p>
        <p><b>Center:</b> {center[0]:.4f}°N, {center[1]:.4f}°E</p>
        <p><b>Radius:</b> {metadata.get('radius_km', 'N/A')} km</p>
        <p><b>Area:</b> {metadata.get('area_km2', 0):.1f} km²</p>
        <hr>
        <p><b>BBox:</b><br>
        [{bbox[0]:.4f}, {bbox[1]:.4f},<br>
         {bbox[2]:.4f}, {bbox[3]:.4f}]</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))
    
    # Save map
    output_file = Path(output_path)
    m.save(str(output_file))
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Select geographic area for satellite download',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select by city name
  python scripts/select_area.py --city "Milan" --radius 15
  
  # Select by coordinates
  python scripts/select_area.py --lat 45.464 --lon 9.190 --radius 15
  
  # Select by bounding box
  python scripts/select_area.py --bbox 9.0 45.3 9.3 45.6
  
  # List predefined cities
  python scripts/select_area.py --list-cities
        """
    )
    
    # City selection
    parser.add_argument('--city', type=str, help='City name (e.g., "Milan", "Rome")')
    parser.add_argument('--country', type=str, default='Italy', help='Country name')
    parser.add_argument('--radius', type=float, help='Radius in km')
    
    # Coordinate selection
    parser.add_argument('--lat', type=float, help='Latitude (decimal degrees)')
    parser.add_argument('--lon', type=float, help='Longitude (decimal degrees)')
    
    # BBox selection
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
                       help='Bounding box coordinates')
    
    # Options
    parser.add_argument('--name', type=str, help='Name for this area')
    parser.add_argument('--no-preview', action='store_true', help='Skip map preview')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    parser.add_argument('--list-cities', action='store_true', help='List predefined cities')
    parser.add_argument('--output', type=str, default='preview_map.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    selector = AreaSelector()
    
    # List cities
    if args.list_cities:
        print("\nPredefined cities:")
        print("=" * 60)
        for city, info in selector.CITIES.items():
            print(f"{city:15} → {info['coords'][0]:.3f}°N, {info['coords'][1]:.3f}°E  (radius: {info['radius_km']} km)")
        print("=" * 60)
        return
    
    # Select area
    try:
        if args.city:
            # City-based selection
            bbox, metadata = selector.select_by_city(
                city=args.city,
                radius_km=args.radius,
                country=args.country
            )
            print(f"\n✓ Selected area by city: {args.city}")
            
        elif args.lat is not None and args.lon is not None:
            # Coordinate-based selection
            radius = args.radius or 15.0
            bbox, metadata = selector.select_by_coordinates(
                lat=args.lat,
                lon=args.lon,
                radius_km=radius,
                name=args.name
            )
            print(f"\n✓ Selected area by coordinates")
            
        elif args.bbox:
            # BBox selection
            bbox, metadata = selector.select_by_bbox(
                min_lon=args.bbox[0],
                min_lat=args.bbox[1],
                max_lon=args.bbox[2],
                max_lat=args.bbox[3],
                name=args.name
            )
            print(f"\n✓ Selected area by bounding box")
            
        else:
            print("Error: Must specify --city, --lat/--lon, or --bbox")
            parser.print_help()
            return
        
        # Print information
        print("\nArea Information:")
        print("=" * 60)
        print(f"Center:     {metadata['center'][0]:.4f}°N, {metadata['center'][1]:.4f}°E")
        if 'radius_km' in metadata:
            print(f"Radius:     {metadata['radius_km']} km")
        print(f"Area:       {metadata['area_km2']:.1f} km²")
        print(f"\nBBox:       [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
        print("=" * 60)
        
        # Generate map preview
        if not args.no_preview:
            print(f"\n📍 Generating map preview...")
            try:
                map_file = generate_map_preview(bbox, metadata, args.output)
                print(f"✓ Map saved: {map_file}")
                
                # Open in browser
                if not args.no_browser:
                    print(f"🌐 Opening in browser...")
                    webbrowser.open(f'file://{map_file.absolute()}')
            except ImportError:
                print("⚠️  Warning: folium not installed. Install with: pip install folium")
                print("    Map preview skipped.")
        
        # Usage example
        print(f"\n📝 Use in your code:")
        print("=" * 60)
        print(f"from satellite_analysis.pipelines import DownloadPipeline")
        print(f"")
        print(f"pipeline = DownloadPipeline.from_config('config/config.yaml')")
        print(f"result = pipeline.run(")
        print(f"    bbox={bbox},")
        print(f"    start_date='2023-03-01',")
        print(f"    end_date='2023-03-15'")
        print(f")")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == '__main__':
    main()

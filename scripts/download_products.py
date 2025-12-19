#!/usr/bin/env python
"""
Script for downloading Sentinel products.

Usage:
    # By city name (recommended)
    python scripts/download_products.py --city Milan --start 2024-06-01 --end 2024-06-15
    
    # By bounding box
    python scripts/download_products.py --bbox 9.0 45.3 9.3 45.6 --start 2024-06-01 --end 2024-06-15
    
    # With cloud cover filter
    python scripts/download_products.py --city Rome --start 2024-06-01 --end 2024-06-15 --cloud-cover 10

Requires Copernicus credentials in config/config.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from satellite_analysis.pipelines import DownloadPipeline
from satellite_analysis.utils import AreaSelector


def main():
    """Main entry point for download script."""
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 products from Copernicus Data Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download for Milan (last 30 days)
  python scripts/download_products.py --city Milan
  
  # Specific date range with low cloud cover
  python scripts/download_products.py --city Rome --start 2024-06-01 --end 2024-06-15 --cloud-cover 10
  
  # Using bounding box
  python scripts/download_products.py --bbox 9.0 45.3 9.3 45.6 --start 2024-06-01 --end 2024-06-15
        """
    )
    
    # City-based selection (NEW - recommended)
    parser.add_argument(
        "--city",
        type=str,
        help="City name (e.g., Milan, Rome, Florence). Uses predefined coordinates."
    )
    
    parser.add_argument(
        "--radius",
        type=float,
        default=15.0,
        help="Radius around city center in km (default: 15)"
    )
    
    # Bounding box (alternative to --city)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box coordinates (alternative to --city)"
    )
    
    # Date range
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). Default: 30 days ago"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD). Default: today"
    )
    
    # Filters
    parser.add_argument(
        "--cloud-cover",
        type=float,
        default=20.0,
        help="Maximum cloud cover percentage (default: 20)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of search results (default: 10)"
    )
    
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=1,
        help="Maximum number of products to download (default: 1)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: data/cities/<city>/raw or data/raw)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # =========================================
    # Validate inputs
    # =========================================
    
    # Need either --city or --bbox
    if not args.city and not args.bbox:
        print("❌ Error: Either --city or --bbox is required")
        print()
        print("Examples:")
        print("  python scripts/download_products.py --city Milan")
        print("  python scripts/download_products.py --bbox 9.0 45.3 9.3 45.6 --start 2024-06-01 --end 2024-06-15")
        sys.exit(1)
    
    # Default dates if not provided
    if not args.end:
        args.end = datetime.now().strftime("%Y-%m-%d")
    if not args.start:
        args.start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # =========================================
    # Resolve city to bounding box
    # =========================================
    
    if args.city:
        print(f"Resolving city: {args.city}")
        try:
            selector = AreaSelector()
            bbox, metadata = selector.select_by_city(args.city, radius_km=args.radius)
            print(f"   Center: {metadata['center'][0]:.4f}N, {metadata['center'][1]:.4f}E")
            print(f"   Radius: {args.radius} km")
            print(f"   Area: {metadata['area_km2']:.1f} km^2")
            
            # Set output directory based on city
            if not args.output:
                args.output = f"data/cities/{args.city.lower()}/raw"
        except Exception as e:
            print(f"ERROR: Failed to resolve city '{args.city}': {e}")
            sys.exit(1)
    else:
        bbox = args.bbox
        if not args.output:
            args.output = "data/raw"
    
    # =========================================
    # Check credentials
    # =========================================
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("   Create config/config.yaml with your credentials")
        sys.exit(1)
    
    # Quick check for null credentials
    import yaml
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    sentinel_config = config_data.get("sentinel", {})
    if not sentinel_config.get("client_id") or not sentinel_config.get("client_secret"):
        print("ERROR: Copernicus credentials not configured!")
        print()
        print("To download automatically, you need OAuth2 credentials:")
        print("  1. Register at: https://dataspace.copernicus.eu")
        print("  2. Go to: User Settings > OAuth clients > Create new")
        print("  3. Add to config/config.yaml:")
        print()
        print("     sentinel:")
        print('       client_id: "your_client_id"')
        print('       client_secret: "your_client_secret"')
        print()
        print("Alternative: Download manually from https://browser.dataspace.copernicus.eu")
        sys.exit(1)
    
    # =========================================
    # Run download
    # =========================================
    
    print()
    print("=" * 60)
    print("SENTINEL-2 DOWNLOAD")
    print("=" * 60)
    print(f"   Date range: {args.start} to {args.end}")
    print(f"   Max cloud cover: {args.cloud_cover}%")
    print(f"   Output: {args.output}")
    print()
    
    try:
        # Create pipeline
        print("🔑 Authenticating with Copernicus...")
        pipeline = DownloadPipeline.from_config(args.config)
        pipeline.output_dir = args.output
        pipeline.max_cloud_cover = args.cloud_cover
        
        # Run pipeline
        print("🔍 Searching catalog...")
        result = pipeline.run(
            bbox=bbox,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit,
            max_downloads=args.max_downloads
        )
        
        # Print summary
        print()
        print("=" * 60)
        print("📊 DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"   Products found: {result.total_products}")
        print(f"   Downloaded: {result.downloaded_count}")
        print(f"   Failed: {result.failed_count}")
        print(f"   Output: {args.output}")
        
        if result.downloaded_files:
            print()
            print("📦 Downloaded files:")
            for file_path in result.downloaded_files:
                print(f"   • {file_path}")
            
            if args.city:
                print()
                print("Next step:")
                print(f"   python scripts/extract_all_bands.py {result.downloaded_files[0]} data/cities/{args.city.lower()}/bands")
        
        sys.exit(0 if result.failed_count == 0 else 1)
        
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

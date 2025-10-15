#!/usr/bin/env python
"""
Script for downloading Sentinel products.

Usage:
    python scripts/download_products.py --bbox 9.0 45.3 9.3 45.6 --start 2023-03-01 --end 2023-03-15
    python scripts/download_products.py --config config/config.yaml --limit 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from satellite_analysis.pipelines import DownloadPipeline


def main():
    """Main entry point for download script."""
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 products from Copernicus Data Space"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    # Search parameters
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box coordinates"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of search results"
    )
    
    parser.add_argument(
        "--max-downloads",
        type=int,
        help="Maximum number of products to download"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for downloaded products"
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.bbox or not args.start or not args.end:
        print("❌ Error: --bbox, --start, and --end are required")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Create pipeline
        print("🚀 Initializing download pipeline...")
        pipeline = DownloadPipeline.from_config(args.config)
        pipeline.output_dir = args.output
        
        # Run pipeline
        result = pipeline.run(
            bbox=args.bbox,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit,
            max_downloads=args.max_downloads
        )
        
        # Print summary
        print("\n📊 FINAL SUMMARY")
        print("=" * 60)
        print(f"✅ Success: {result.downloaded_count} products")
        print(f"❌ Failed: {result.failed_count} products")
        print(f"📈 Success rate: {result.success_rate:.1f}%")
        print(f"📁 Output directory: {args.output}")
        
        if result.downloaded_files:
            print("\n📦 Downloaded files:")
            for file_path in result.downloaded_files:
                print(f"  - {file_path}")
        
        sys.exit(0 if result.failed_count == 0 else 1)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

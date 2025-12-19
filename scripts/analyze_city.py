"""Satellite City Analyzer - Command-line interface.

ONE COMMAND to analyze any city with Sentinel-2 data.

USAGE:
    # Quick analysis
    python analyze_city.py --city Florence
    
    # With exports
    python analyze_city.py --city Milan --export geotiff report
    
    # Batch analysis
    python analyze_city.py --cities Milan Rome Florence
    
    # Change detection
    python analyze_city.py --city Milan --compare 2023-06 2024-06

OUTPUT:
    data/cities/<city>/runs/<timestamp>_consensus/
    ├── labels.npy          # Classification result
    ├── confidence.npy      # Confidence scores (0-1)
    ├── run_info.json       # Metadata and statistics
    ├── classification.tif  # GeoTIFF (if --export geotiff)
    └── report.html         # HTML report (if --export report)
"""

import argparse
import sys
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from satellite_analysis import analyze, analyze_batch, compare
from satellite_analysis import export_geotiff, export_report, export_change_report


def main():
    parser = argparse.ArgumentParser(
        description="Satellite Land Cover Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single city analysis
  python analyze_city.py --city Florence
  
  # With exports
  python analyze_city.py --city Milan --export geotiff report
  
  # Batch multiple cities
  python analyze_city.py --cities Milan Rome Florence --export report
  
  # Change detection between dates
  python analyze_city.py --city Milan --compare 2023-06 2024-06
  
  # Full example with all options
  python analyze_city.py --city Florence --max-size 3000 --classifier consensus --export geotiff report json
        """
    )
    
    # City selection (mutually exclusive: single or batch)
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--city",
        help="Single city to analyze (e.g., Florence, Milan, Rome)"
    )
    city_group.add_argument(
        "--cities",
        nargs="+",
        help="Multiple cities for batch analysis"
    )
    
    # Processing options
    parser.add_argument(
        "--max-size",
        type=int,
        default=3000,
        help="Maximum image dimension (default: 3000)"
    )
    parser.add_argument(
        "--classifier",
        choices=["kmeans", "spectral", "consensus"],
        default="consensus",
        help="Classification method (default: consensus)"
    )
    
    # Export options
    parser.add_argument(
        "--export",
        nargs="+",
        choices=["geotiff", "report", "json", "all"],
        help="Export formats: geotiff, report (HTML), json, or all"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory"
    )
    
    # Change detection
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("DATE_BEFORE", "DATE_AFTER"),
        help="Compare two dates (YYYY-MM format)"
    )
    
    # Language
    parser.add_argument(
        "--lang",
        choices=["en", "it"],
        default="en",
        help="Report language (default: en)"
    )
    
    # Verbosity
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Determine export formats
    exports = set()
    if args.export:
        if "all" in args.export:
            exports = {"geotiff", "report", "json"}
        else:
            exports = set(args.export)
    
    # Header
    if not args.quiet:
        print("=" * 60)
        print("SATELLITE LAND COVER ANALYSIS")
        print("=" * 60)
    
    try:
        # Mode: Change Detection
        if args.compare:
            return run_change_detection(args, exports)
        
        # Mode: Batch Analysis
        if args.cities:
            return run_batch_analysis(args, exports)
        
        # Mode: Single City Analysis
        return run_single_analysis(args, exports)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTip: Make sure data exists in data/cities/{city}/bands/")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def run_single_analysis(args, exports):
    """Run analysis for a single city."""
    print(f"Analyzing: {args.city}")
    print(f"Options: max_size={args.max_size}, classifier={args.classifier}")
    print("-" * 60)
    
    result = analyze(
        args.city,
        max_size=args.max_size,
        classifier=args.classifier,
        project_root=PROJECT_ROOT,
    )
    
    # Export if requested
    export_results(result, exports, args.lang, args.output_dir)
    
    # Summary
    print_result_summary(result)
    
    return 0


def run_batch_analysis(args, exports):
    """Run analysis for multiple cities."""
    print(f"Batch analysis: {', '.join(args.cities)}")
    print(f"Options: max_size={args.max_size}, classifier={args.classifier}")
    print("-" * 60)
    
    def progress(city, status):
        print(f"  [{city}] {status}")
    
    results = analyze_batch(
        args.cities,
        max_size=args.max_size,
        classifier=args.classifier,
        on_progress=progress,
    )
    
    # Export and summarize each result
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    success_count = 0
    for city, result in results.items():
        if isinstance(result, Exception):
            print(f"  {city}: ERROR - {result}")
        else:
            success_count += 1
            print(f"  {city}: {result.avg_confidence:.1%} confidence")
            export_results(result, exports, args.lang, args.output_dir)
    
    print("-" * 60)
    print(f"Completed: {success_count}/{len(args.cities)} cities")
    
    return 0 if success_count == len(args.cities) else 1


def run_change_detection(args, exports):
    """Run change detection between two dates."""
    date_before, date_after = args.compare
    city = args.city
    
    print(f"Change Detection: {city}")
    print(f"Period: {date_before} -> {date_after}")
    print("-" * 60)
    
    changes = compare(
        city,
        date_before,
        date_after,
        max_size=args.max_size,
        classifier=args.classifier,
        project_root=PROJECT_ROOT,
    )
    
    # Export change report
    if "report" in exports or "all" in exports:
        report_path = export_change_report(changes, language=args.lang)
        print(f"Report: {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CHANGE DETECTION RESULTS")
    print("=" * 60)
    print(changes.summary())
    print("=" * 60)
    
    return 0


def export_results(result, exports, language, output_dir):
    """Export result in requested formats."""
    if not exports:
        return
    
    base_dir = output_dir or result.output_dir
    base_name = result.city.lower()
    
    if "geotiff" in exports:
        path = export_geotiff(result, base_dir / f"{base_name}_classification.tif")
        print(f"  Exported GeoTIFF: {path}")
    
    if "report" in exports:
        path = export_report(result, base_dir / f"{base_name}_report.html", language=language)
        print(f"  Exported Report: {path}")
    
    if "json" in exports:
        from satellite_analysis import export_json
        path = export_json(result, base_dir / f"{base_name}_results.json")
        print(f"  Exported JSON: {path}")


def print_result_summary(result):
    """Print analysis result summary."""
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"City: {result.city}")
    print(f"Shape: {result.processed_shape[0]}x{result.processed_shape[1]}")
    print(f"Confidence: {result.avg_confidence:.1%}")
    print(f"Time: {result.execution_time:.1f}s")
    print(f"Output: {result.output_dir}")
    print("-" * 60)
    print("Class Distribution:")
    for cls_id, data in result.class_distribution().items():
        from satellite_analysis import LAND_COVER_CLASSES
        name = LAND_COVER_CLASSES.get(cls_id, {}).get("name", f"Class {cls_id}")
        print(f"  {name}: {data['percentage']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())

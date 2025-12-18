"""
🔍 Validate Classification Results

Validates land cover classification against ESA Scene Classification Layer (SCL)
or compares different classification methods.

USAGE:
    # Validate consensus classification against SCL
    python validate_classification.py --city Milan --method consensus --reference scl
    
    # Compare K-Means vs Spectral methods
    python validate_classification.py --city Milan --compare
    
    # Full validation with report generation
    python validate_classification.py --city Milan --method consensus --reference scl --report

OUTPUT:
    data/cities/<city>/validation/
        ├── confusion_matrix.png
        ├── classification_comparison.png
        ├── confidence_map.png
        └── validation_report.txt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from satellite_analysis.analyzers.classification import (
    SpectralIndicesClassifier,
    ConsensusClassifier
)
from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
from satellite_analysis.preprocessing.normalization import min_max_scale
from satellite_analysis.preprocessing.reshape import reshape_image_to_table, reshape_table_to_image
from satellite_analysis.validation import (
    compute_accuracy,
    compute_kappa,
    compute_f1_scores,
    ValidationReport,
    plot_confusion_matrix,
    plot_consensus_analysis,
    plot_confidence_map,
    SCLValidator,
    map_scl_to_consensus
)


class ClassificationValidator:
    """Validates classification results against reference data."""
    
    def __init__(self, city_name: str, data_dir: str = None):
        """
        Initialize validator.
        
        Args:
            city_name: Name of the city
            data_dir: Optional custom data directory
        """
        self.city = city_name
        
        if data_dir:
            self.base_dir = Path(data_dir)
        else:
            self.base_dir = Path(f"data/cities/{city_name.lower()}")
        
        self.output_dir = self.base_dir / "validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔍 Classification Validator")
        print(f"   City: {city_name}")
        print(f"   Data: {self.base_dir}")
        print(f"   Output: {self.output_dir}")
    
    def load_bands(self):
        """Load spectral bands."""
        print("\n📂 Loading bands...")
        
        # Try multiple locations
        possible_dirs = [self.base_dir, self.base_dir / "bands"]
        bands_dir = None
        
        for check_dir in possible_dirs:
            if (check_dir / "B02.tif").exists():
                bands_dir = check_dir
                break
        
        if bands_dir is None:
            raise FileNotFoundError(f"No band files found in {self.base_dir}")
        
        stack = []
        band_names = ['B02', 'B03', 'B04', 'B08']
        
        for band in band_names:
            band_file = bands_dir / f"{band}.tif"
            with rasterio.open(band_file) as src:
                data = src.read(1)
                stack.append(data)
                print(f"   ✅ {band}: {data.shape}")
        
        stack = np.stack(stack, axis=-1)  # (H, W, 4)
        print(f"   Stack shape: {stack.shape}")
        
        self.stack = stack
        self.band_indices = {name: i for i, name in enumerate(band_names)}
        
        return stack
    
    def load_scl(self):
        """Load SCL band if available."""
        print("\n📂 Loading SCL band...")
        
        possible_dirs = [self.base_dir, self.base_dir / "bands"]
        
        for check_dir in possible_dirs:
            scl_file = check_dir / "SCL.tif"
            if scl_file.exists():
                with rasterio.open(scl_file) as src:
                    scl = src.read(1)
                    print(f"   ✅ SCL: {scl.shape}")
                    self.scl = scl
                    return scl
        
        print("   ⚠️  SCL band not found - will use simulated reference")
        self.scl = None
        return None
    
    def create_rgb_preview(self) -> np.ndarray:
        """Create RGB preview image."""
        rgb = self.stack[:, :, [2, 1, 0]]  # B04, B03, B02
        
        # Normalize
        rgb_norm = np.zeros_like(rgb, dtype=np.float32)
        for i in range(3):
            band = rgb[:, :, i].astype(np.float32)
            rgb_norm[:, :, i] = (band - band.min()) / (band.max() - band.min() + 1e-10)
        
        # Convert to uint8
        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
        
        return rgb_uint8
    
    def run_consensus_classification(self):
        """Run consensus classification."""
        print("\n🎯 Running Consensus Classification...")
        
        classifier = ConsensusClassifier(
            n_clusters=6,
            confidence_threshold=0.5,
            random_state=42
        )
        
        labels, confidence, uncertainty, stats = classifier.classify(
            self.stack,
            self.band_indices,
            has_swir=False
        )
        
        self.consensus_labels = labels
        self.confidence_map = confidence
        self.uncertainty_mask = uncertainty
        self.consensus_stats = stats
        self.consensus_classifier = classifier
        
        # Also store individual method results
        self.kmeans_labels = classifier.labels_kmeans_
        self.spectral_labels = classifier.labels_spectral_
        
        print(f"\n   📊 Consensus Statistics:")
        print(f"      Agreement: {stats['agreement_pct']:.1f}%")
        print(f"      Avg Confidence: {stats['avg_confidence']:.2f}")
        print(f"      Uncertain: {stats['uncertain_pct']:.1f}%")
        
        return labels, confidence, uncertainty, stats
    
    def validate_against_scl(self):
        """Validate classification against SCL."""
        print("\n🔍 Validating against ESA SCL...")
        
        if self.scl is None:
            print("   ⚠️  No SCL data available")
            return None
        
        validator = SCLValidator(
            self.scl,
            exclude_clouds=True,
            exclude_shadows=False
        )
        
        results = validator.validate(self.consensus_labels)
        
        print(f"\n   📊 Validation Results:")
        print(f"      Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
        print(f"      Cohen's Kappa: {results['kappa']:.4f}")
        print(f"      F1 (weighted): {results['f1_weighted']:.4f}")
        print(f"      Valid Pixels: {results['n_valid_pixels']:,} ({results['valid_percentage']:.1f}%)")
        
        self.validation_results = results
        return results
    
    def compare_methods(self):
        """Compare K-Means vs Spectral vs Consensus."""
        print("\n📊 Comparing Classification Methods...")
        
        # Get flattened labels
        km_flat = self.kmeans_labels.flatten()
        sp_flat = self.spectral_labels.flatten()
        cons_flat = self.consensus_labels.flatten()
        
        # Agreement between K-Means and Spectral
        # Note: K-Means clusters need to be mapped to consensus classes first
        cluster_map = self.consensus_classifier.cluster_to_class_map
        km_mapped = np.vectorize(lambda x: cluster_map.get(x, 5))(km_flat)
        sp_mapped = np.vectorize(
            lambda x: ConsensusClassifier.SPECTRAL_TO_CONSENSUS.get(x, 5)
        )(sp_flat)
        
        # Compute agreement metrics
        km_sp_agreement = np.mean(km_mapped == sp_mapped) * 100
        km_cons_agreement = np.mean(km_mapped == cons_flat) * 100
        sp_cons_agreement = np.mean(sp_mapped == cons_flat) * 100
        
        print(f"\n   Method Agreement:")
        print(f"      K-Means ↔ Spectral: {km_sp_agreement:.1f}%")
        print(f"      K-Means ↔ Consensus: {km_cons_agreement:.1f}%")
        print(f"      Spectral ↔ Consensus: {sp_cons_agreement:.1f}%")
        
        # Class distribution comparison
        print(f"\n   Class Distribution:")
        class_names = self.consensus_classifier.CLASSES
        
        for cls_id, cls_name in class_names.items():
            km_pct = np.mean(km_mapped == cls_id) * 100
            sp_pct = np.mean(sp_mapped == cls_id) * 100
            cons_pct = np.mean(cons_flat == cls_id) * 100
            
            print(f"      {cls_name:<15}: K-Means {km_pct:>5.1f}%, "
                  f"Spectral {sp_pct:>5.1f}%, Consensus {cons_pct:>5.1f}%")
        
        return {
            'km_sp_agreement': km_sp_agreement,
            'km_cons_agreement': km_cons_agreement,
            'sp_cons_agreement': sp_cons_agreement
        }
    
    def generate_visualizations(self):
        """Generate all visualization outputs."""
        print("\n🎨 Generating Visualizations...")
        
        rgb = self.create_rgb_preview()
        class_names = self.consensus_classifier.CLASSES
        class_colors = self.consensus_classifier.get_class_colors()
        
        # 1. Consensus Analysis (comprehensive)
        print("   Creating consensus analysis plot...")
        fig = plot_consensus_analysis(
            rgb,
            self.kmeans_labels,
            self.spectral_labels,
            self.consensus_labels,
            self.confidence_map,
            class_names,
            class_colors,
            title=f"{self.city} - Consensus Classification Analysis",
            save_path=str(self.output_dir / "consensus_analysis.png")
        )
        plt.close(fig)
        print(f"   ✅ Saved: {self.output_dir / 'consensus_analysis.png'}")
        
        # 2. Confidence Map
        print("   Creating confidence map...")
        fig = plot_confidence_map(
            self.confidence_map,
            self.uncertainty_mask,
            title=f"{self.city} - Classification Confidence",
            save_path=str(self.output_dir / "confidence_map.png")
        )
        plt.close(fig)
        print(f"   ✅ Saved: {self.output_dir / 'confidence_map.png'}")
        
        # 3. Confusion Matrix (if SCL available)
        if hasattr(self, 'validation_results') and self.validation_results:
            print("   Creating confusion matrix...")
            fig = plot_confusion_matrix(
                self.validation_results['confusion_matrix'],
                class_names,
                normalize=True,
                title=f"{self.city} - Confusion Matrix (vs ESA SCL)",
                save_path=str(self.output_dir / "confusion_matrix.png")
            )
            plt.close(fig)
            print(f"   ✅ Saved: {self.output_dir / 'confusion_matrix.png'}")
    
    def generate_report(self) -> str:
        """Generate text validation report."""
        print("\n📝 Generating Report...")
        
        lines = []
        lines.append("=" * 70)
        lines.append(f"CLASSIFICATION VALIDATION REPORT - {self.city.upper()}")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Data info
        lines.append("📂 DATA INFORMATION")
        lines.append("-" * 50)
        lines.append(f"City: {self.city}")
        lines.append(f"Image Shape: {self.stack.shape}")
        lines.append(f"Total Pixels: {self.stack.shape[0] * self.stack.shape[1]:,}")
        lines.append(f"Bands Used: B02, B03, B04, B08")
        lines.append("")
        
        # Consensus Statistics
        lines.append("🎯 CONSENSUS CLASSIFICATION")
        lines.append("-" * 50)
        lines.append(f"K-Means Clusters: 6")
        lines.append(f"Method Agreement: {self.consensus_stats['agreement_pct']:.1f}%")
        lines.append(f"Average Confidence: {self.consensus_stats['avg_confidence']:.2f}")
        lines.append(f"Uncertain Pixels: {self.consensus_stats['uncertain_pct']:.1f}%")
        lines.append("")
        
        # Class Distribution
        lines.append("📊 CLASS DISTRIBUTION")
        lines.append("-" * 50)
        for cls_name, info in self.consensus_stats['class_distribution'].items():
            lines.append(f"  {cls_name:<20}: {info['percentage']:>5.1f}% ({info['count']:,} pixels)")
        lines.append("")
        
        # Cluster Mapping
        lines.append("🔗 CLUSTER TO CLASS MAPPING")
        lines.append("-" * 50)
        for cluster, class_name in self.consensus_stats['cluster_mapping'].items():
            lines.append(f"  Cluster {cluster} → {class_name}")
        lines.append("")
        
        # Validation Results (if SCL available)
        if hasattr(self, 'validation_results') and self.validation_results:
            lines.append("🔍 VALIDATION AGAINST ESA SCL")
            lines.append("-" * 50)
            lines.append(f"Overall Accuracy: {self.validation_results['overall_accuracy']:.4f} "
                        f"({self.validation_results['overall_accuracy']*100:.2f}%)")
            lines.append(f"Cohen's Kappa: {self.validation_results['kappa']:.4f}")
            lines.append(f"F1 Score (weighted): {self.validation_results['f1_weighted']:.4f}")
            lines.append(f"Valid Pixels: {self.validation_results['n_valid_pixels']:,} "
                        f"({self.validation_results['valid_percentage']:.1f}%)")
            lines.append("")
            
            # Per-class F1
            lines.append("Per-Class F1 Scores:")
            for cls_id, f1 in self.validation_results['f1_per_class'].items():
                cls_name = self.consensus_classifier.CLASSES.get(cls_id, f"Class {cls_id}")
                lines.append(f"  {cls_name:<20}: {f1:.4f}")
        else:
            lines.append("⚠️  SCL data not available - validation skipped")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        
        # Save report
        report_path = self.output_dir / "validation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"   ✅ Saved: {report_path}")
        print(report_text)
        
        return report_text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🔍 Validate Classification Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate consensus classification
  python validate_classification.py --city Milan --method consensus
  
  # Compare all methods
  python validate_classification.py --city Milan --compare
  
  # Use custom data directory
  python validate_classification.py --city Milan --data-dir data/processed/milano_centro
        """
    )
    
    parser.add_argument('--city', type=str, required=True,
                       help='City name')
    parser.add_argument('--method', type=str, choices=['consensus', 'kmeans', 'spectral'],
                       default='consensus', help='Classification method to validate')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all classification methods')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Custom data directory')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 CLASSIFICATION VALIDATION")
    print("=" * 70)
    
    # Initialize validator
    validator = ClassificationValidator(args.city, args.data_dir)
    
    # Load data
    validator.load_bands()
    validator.load_scl()
    
    # Run classification
    validator.run_consensus_classification()
    
    # Validate against SCL (if available)
    validator.validate_against_scl()
    
    # Compare methods
    if args.compare:
        validator.compare_methods()
    
    # Generate visualizations
    validator.generate_visualizations()
    
    # Generate report
    if args.report:
        validator.generate_report()
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Results saved to: {validator.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

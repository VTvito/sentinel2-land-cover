"""
Test workflow completo: Download → Preprocessing → Visualizzazione
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from satellite_analysis.utils import AreaSelector
from satellite_analysis.downloaders import SentinelDownloader
from satellite_analysis.pipelines import PreprocessingPipeline


def test_complete_workflow():
    """Test workflow completo end-to-end."""
    print("\n" + "=" * 70)
    print("🧪 TEST WORKFLOW COMPLETO")
    print("=" * 70)
    
    # 1. Selezione area
    print("\n📍 STEP 1: Selezione area")
    print("-" * 70)
    
    selector = AreaSelector()
    bbox, area_info = selector.select_by_city("Milan", radius_km=15)
    
    print(f"Città: Milan")
    print(f"Centro: {area_info['center'][0]:.4f}°N, {area_info['center'][1]:.4f}°E")
    print(f"BBox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
    print(f"Superficie: {area_info['area_km2']:.1f} km²")
    
    # 2. Download prodotto
    print("\n📥 STEP 2: Download prodotto Sentinel-2")
    print("-" * 70)
    
    downloader = SentinelDownloader()
    
    # Cerca prodotti disponibili
    products = downloader.search(
        bbox=bbox,
        date_start="2024-10-01",
        date_end="2024-10-15",
        max_cloud_cover=20
    )
    
    if not products:
        print("❌ Nessun prodotto trovato per i criteri specificati")
        return False
    
    print(f"✓ Trovati {len(products)} prodotti")
    
    # Scarica il primo prodotto
    product = products[0]
    print(f"\nProdotto selezionato:")
    print(f"  ID: {product['id'][:50]}...")
    print(f"  Data: {product['date']}")
    print(f"  Cloud cover: {product['cloud_cover']:.1f}%")
    print(f"  Dimensione: {product['size_mb']:.1f} MB")
    
    output_file = downloader.download(
        product_id=product['id'],
        output_dir="data/raw"
    )
    
    print(f"✓ Download completato: {output_file.name}")
    
    # 3. Preprocessing
    print("\n🔧 STEP 3: Preprocessing con crop automatico")
    print("-" * 70)
    
    pipeline = PreprocessingPipeline(output_dir="data/processed")
    result = pipeline.run(
        zip_path=str(output_file),
        bbox=bbox,
        save_visualization=True,
        open_visualization=True
    )
    
    # 4. Risultati
    print("\n📊 STEP 4: Risultati")
    print("-" * 70)
    
    print(f"\n✅ Preprocessing completato:")
    print(f"   Prodotto: {result.product_name}")
    print(f"   Bande estratte: {len(result.bands)}")
    print(f"   Cropped: {result.metadata['cropped']}")
    
    print(f"\n🎨 Composites:")
    print(f"   RGB: {result.rgb.shape[1]:,} × {result.rgb.shape[0]:,} pixels")
    print(f"   FCC: {result.fcc.shape[1]:,} × {result.fcc.shape[0]:,} pixels")
    print(f"   NDVI: {result.ndvi.shape[1]:,} × {result.ndvi.shape[0]:,} pixels")
    print(f"   NDVI range: [{result.metadata['ndvi_range'][0]:.3f}, {result.metadata['ndvi_range'][1]:.3f}]")
    
    if result.visualization_path:
        viz_size = result.visualization_path.stat().st_size / (1024 * 1024)
        print(f"\n🖼️  Visualizzazione: {viz_size:.2f} MB")
        print(f"   Path: {result.visualization_path}")
    
    print("\n" + "=" * 70)
    print("✅ WORKFLOW COMPLETATO CON SUCCESSO")
    print("=" * 70)
    
    print("\n💡 Output generati:")
    print(f"   • Prodotto raw: data/raw/{output_file.name}")
    print(f"   • Bande processate: data/processed/{result.product_name}/")
    print(f"   • Visualizzazione: {result.visualization_path}")
    
    return True


if __name__ == "__main__":
    print("\n🚀 TEST SUITE: Complete Workflow\n")
    
    success = test_complete_workflow()
    
    if not success:
        print("\n❌ Workflow fallito")
        sys.exit(1)
    
    print("\n🎉 Tutti i componenti validati!")
    print("\n📝 Il sistema è pronto per l'uso in produzione.\n")

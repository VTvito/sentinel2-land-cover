"""
Test per la pipeline di preprocessing con crop automatico.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from satellite_analysis.pipelines import PreprocessingPipeline


def test_preprocessing_with_crop():
    """Test preprocessing completo con crop automatico."""
    print("\n" + "=" * 60)
    print("TEST: Preprocessing con Crop Automatico")
    print("=" * 60)
    
    # Usa il prodotto già scaricato
    zip_path = project_root / "data" / "raw" / "product_1.zip"
    
    if not zip_path.exists():
        print(f"\n⚠️  File non trovato: {zip_path}")
        print("   Esegui prima: python tests/test_complete_workflow.py")
        return False
    
    # BBox Milano centro (15 km raggio)
    bbox = [8.9982, 45.3290, 9.3818, 45.5990]
    
    print(f"\n📦 Prodotto: {zip_path.name}")
    print(f"📍 BBox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
    
    # Preprocessing
    pipeline = PreprocessingPipeline(output_dir="data/processed")
    result = pipeline.run(
        zip_path=str(zip_path),
        bbox=bbox,
        save_visualization=True,
        open_visualization=False
    )
    
    # Verifica risultati
    print(f"\n✅ Risultati:")
    print(f"   Prodotto: {result.product_name}")
    print(f"   Bande: {len(result.bands)}")
    print(f"   Cropped: {result.metadata['cropped']}")
    print(f"   RGB shape: {result.rgb.shape[1]:,} × {result.rgb.shape[0]:,} px")
    print(f"   NDVI range: [{result.metadata['ndvi_range'][0]:.3f}, {result.metadata['ndvi_range'][1]:.3f}]")
    
    if result.visualization_path:
        viz_size = result.visualization_path.stat().st_size / (1024 * 1024)
        print(f"   Visualizzazione: {viz_size:.2f} MB")
    
    # Validazioni
    assert result.metadata['cropped'] == True, "Crop non applicato!"
    assert result.rgb.shape[0] < 5000, "Immagine non croppata correttamente!"
    assert -1.0 <= result.metadata['ndvi_range'][0] <= 1.0, "NDVI non valido!"
    
    print("\n✅ TEST COMPLETATO")
    return True


if __name__ == "__main__":
    print("\n🧪 TEST: Preprocessing Pipeline\n")
    
    success = test_preprocessing_with_crop()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ PREPROCESSING VALIDATO")
        print("=" * 60 + "\n")
    else:
        print("\n❌ Test fallito")
        sys.exit(1)

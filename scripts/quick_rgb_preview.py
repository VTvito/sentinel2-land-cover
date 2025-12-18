"""
Quick RGB preview to identify the area in the image.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio

def create_rgb_preview(base_path: str):
    """Create a simple RGB preview with histogram equalization."""
    base_path = Path(base_path)
    
    print("📂 Loading bands...")
    
    # Load RGB bands
    with rasterio.open(base_path / 'B04.jp2') as src:
        b04 = src.read(1).astype(np.float32)
    with rasterio.open(base_path / 'B03.jp2') as src:
        b03 = src.read(1).astype(np.float32)
    with rasterio.open(base_path / 'B02.jp2') as src:
        b02 = src.read(1).astype(np.float32)
    
    print(f"   Image shape: {b04.shape}")
    
    # Stack and normalize
    rgb = np.dstack([b04, b03, b02])
    
    # Min-max scaling
    for i in range(3):
        band = rgb[:, :, i]
        rgb[:, :, i] = (band - band.min()) / (band.max() - band.min())
    
    # Convert to uint8 and apply histogram equalization
    rgb = (rgb * 255).astype(np.uint8)
    
    from PIL import Image, ImageOps
    rgb_pil = Image.fromarray(rgb)
    rgb_pil = ImageOps.equalize(rgb_pil)
    rgb = np.array(rgb_pil)
    
    # Create figure
    plt.figure(figsize=(16, 16))
    plt.imshow(rgb)
    plt.title('RGB True Color Preview (B04-B03-B02)\nMilano Area', 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Add grid to help identify Milano center
    ax = plt.gca()
    h, w = rgb.shape[:2]
    
    # Draw cross at center
    ax.axhline(y=h//2, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=w//2, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add coordinate text
    ax.text(w//2, h//2, 'CENTER', color='yellow', fontsize=12, 
            ha='center', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    output_path = base_path / "rgb_preview_full.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Preview saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    base_path = r'c:\TEMP_1\satellite_git\data\processed\product_1'
    create_rgb_preview(base_path)

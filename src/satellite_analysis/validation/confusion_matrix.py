"""
Confusion Matrix visualization for classification validation.

Provides publication-quality visualizations of classification results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show_values: bool = True,
    value_format: str = ".2f"
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        confusion_matrix: 2D numpy array (true x predicted)
        class_names: Optional dict mapping class_id to name
        normalize: If True, normalize by row (true labels)
        title: Plot title
        cmap: Matplotlib colormap name
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
        show_values: If True, show values in cells
        value_format: Format string for cell values
        
    Returns:
        Matplotlib figure
    """
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    n_classes = cm.shape[0]
    
    # Normalize if requested
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
        title = f"{title} (Normalized)"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count" if not normalize else "Proportion", rotation=-90, va="bottom")
    
    # Set ticks
    if class_names:
        labels = [class_names.get(i, f"Class {i}") for i in range(n_classes)]
    else:
        labels = [f"Class {i}" for i in range(n_classes)]
    
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    
    # Add text annotations
    if show_values:
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                value = cm[i, j]
                if normalize:
                    text = f"{value:{value_format}}"
                else:
                    text = f"{int(value):,}" if value < 10000 else f"{value:.0e}"
                
                color = "white" if value > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    
    # Labels and title
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_classification_comparison(
    rgb_image: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    class_names: Dict[int, str],
    class_colors: Dict[int, Tuple[int, int, int]],
    title: str = "Classification Comparison",
    figsize: Tuple[int, int] = (20, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot side-by-side comparison of RGB, ground truth, and prediction.
    
    Args:
        rgb_image: RGB image (H, W, 3) uint8
        labels_true: Ground truth labels (H, W)
        labels_pred: Predicted labels (H, W)
        class_names: Dict mapping class_id to name
        class_colors: Dict mapping class_id to RGB tuple
        title: Overall figure title
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB True Color", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    n_classes = max(max(class_names.keys()), max(class_colors.keys())) + 1
    colors_norm = [np.array(class_colors.get(i, (128, 128, 128))) / 255.0 for i in range(n_classes)]
    cmap = mcolors.ListedColormap(colors_norm)
    
    axes[1].imshow(labels_true, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    axes[1].set_title("Ground Truth (Reference)", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(labels_pred, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    axes[2].set_title("Prediction", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(class_colors.get(i, (128, 128, 128))) / 255.0,
              label=class_names.get(i, f"Class {i}"))
        for i in sorted(class_names.keys())
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(class_names),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Classification comparison saved to: {save_path}")
    
    return fig


def plot_confidence_map(
    confidence_map: np.ndarray,
    uncertainty_mask: Optional[np.ndarray] = None,
    title: str = "Classification Confidence",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confidence map with optional uncertainty overlay.
    
    Args:
        confidence_map: 2D array with confidence values (0-1)
        uncertainty_mask: Optional boolean mask of uncertain pixels
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confidence
    im = ax.imshow(confidence_map, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Overlay uncertainty contours if provided
    if uncertainty_mask is not None:
        # Draw contours around uncertain regions
        ax.contour(uncertainty_mask.astype(int), levels=[0.5], colors='red',
                  linewidths=0.5, alpha=0.7)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Confidence (0=Low, 1=High)", rotation=-90, va="bottom")
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add statistics text
    avg_conf = np.mean(confidence_map)
    uncertain_pct = np.mean(confidence_map < 0.5) * 100 if uncertainty_mask is None else np.mean(uncertainty_mask) * 100
    
    stats_text = f"Avg Confidence: {avg_conf:.2f}\nUncertain: {uncertain_pct:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confidence map saved to: {save_path}")
    
    return fig


def plot_consensus_analysis(
    rgb_image: np.ndarray,
    labels_kmeans: np.ndarray,
    labels_spectral: np.ndarray,
    labels_consensus: np.ndarray,
    confidence_map: np.ndarray,
    class_names: Dict[int, str],
    class_colors: Dict[int, Tuple[int, int, int]],
    title: str = "Consensus Classification Analysis",
    figsize: Tuple[int, int] = (20, 16),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comprehensive consensus analysis with all components.
    
    Args:
        rgb_image: RGB image (H, W, 3) uint8
        labels_kmeans: K-Means cluster labels (H, W)
        labels_spectral: Spectral classification labels (H, W)
        labels_consensus: Final consensus labels (H, W)
        confidence_map: Confidence map (H, W)
        class_names: Dict mapping class_id to name
        class_colors: Dict mapping class_id to RGB tuple
        title: Overall figure title
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.1)
    
    # Top row: RGB, K-Means, Spectral
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Bottom row: Consensus, Confidence, Legend
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Create colormap
    n_classes = max(max(class_names.keys()), max(class_colors.keys())) + 1
    colors_norm = [np.array(class_colors.get(i, (128, 128, 128))) / 255.0 for i in range(n_classes)]
    cmap = mcolors.ListedColormap(colors_norm)
    
    # K-Means colormap (different)
    kmeans_colors = plt.cm.tab10(np.linspace(0, 1, max(6, labels_kmeans.max() + 1)))
    kmeans_cmap = mcolors.ListedColormap(kmeans_colors)
    
    # Plot RGB
    ax1.imshow(rgb_image)
    ax1.set_title("RGB True Color", fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Plot K-Means
    ax2.imshow(labels_kmeans, cmap=kmeans_cmap, interpolation='nearest')
    ax2.set_title("K-Means Clustering", fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # Plot Spectral
    ax3.imshow(labels_spectral, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    ax3.set_title("Spectral Classification", fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    # Plot Consensus
    ax4.imshow(labels_consensus, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    ax4.set_title("Consensus Result", fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Plot Confidence
    im5 = ax5.imshow(confidence_map, cmap='RdYlGn', vmin=0, vmax=1)
    ax5.set_title("Confidence Map", fontsize=11, fontweight='bold')
    ax5.axis('off')
    cbar = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, orientation='horizontal')
    cbar.ax.set_xlabel("Confidence", fontsize=9)
    
    # Legend in last subplot
    ax6.axis('off')
    
    # Create legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(class_colors.get(i, (128, 128, 128))) / 255.0,
              edgecolor='black', label=f"{i}: {class_names.get(i, f'Class {i}')}")
        for i in sorted(class_names.keys())
    ]
    
    ax6.legend(handles=legend_elements, loc='center', fontsize=10,
              title="Land Cover Classes", title_fontsize=11, frameon=True)
    
    # Statistics text
    agreement = np.mean(labels_kmeans == labels_spectral) * 100
    avg_conf = np.mean(confidence_map)
    uncertain = np.mean(confidence_map < 0.5) * 100
    
    stats_text = (
        f"Statistics:\n"
        f"──────────────\n"
        f"K-Means/Spectral\n"
        f"Agreement: {agreement:.1f}%\n\n"
        f"Avg Confidence: {avg_conf:.2f}\n"
        f"Uncertain: {uncertain:.1f}%"
    )
    
    ax6.text(0.5, 0.3, stats_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Consensus analysis saved to: {save_path}")
    
    return fig

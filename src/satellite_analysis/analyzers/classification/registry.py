"""Classifier protocol + registry for pipeline use.

Adapters keep classifier-specific logic outside the pipeline and allow
new classifiers to be registered with a uniform interface.
"""

from __future__ import annotations

from typing import Protocol, Dict, Tuple, List, Any, Literal
import numpy as np

from satellite_analysis.analyzers.classification import ConsensusClassifier, SpectralIndicesClassifier
from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
from satellite_analysis.preprocessing.reshape import reshape_image_to_table, reshape_table_to_image
from satellite_analysis.preprocessing.normalization import min_max_scale


class Classifier(Protocol):
    """Minimal contract for pipeline classifiers."""

    name: str

    def required_bands(self) -> List[str]:
        """Bands required to run this classifier."""
        ...

    def classify(self, bands: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return labels, confidence map, and statistics."""
        ...


def _basic_stats(labels: np.ndarray, confidence: np.ndarray) -> Dict[str, Any]:
    unique, counts = np.unique(labels, return_counts=True)
    total = int(counts.sum())
    return {
        "avg_confidence": float(np.mean(confidence)),
        "class_distribution": {
            int(cls): {
                "count": int(count),
                "percentage": float(count / total * 100.0),
            }
            for cls, count in zip(unique, counts)
        },
    }


class ConsensusClassifierAdapter:
    """Adapter around the existing ConsensusClassifier."""

    name = "consensus"

    def __init__(self, *, n_clusters: int = 6):
        self._classifier = ConsensusClassifier(n_clusters=n_clusters)

    def required_bands(self) -> List[str]:
        return ["B02", "B03", "B04", "B08"]

    def classify(self, bands: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        band_stack = np.stack(
            [bands["B02"], bands["B03"], bands["B04"], bands["B08"]],
            axis=-1,
        )
        band_indices = {"B02": 0, "B03": 1, "B04": 2, "B08": 3}
        labels, confidence, _, stats = self._classifier.classify(
            band_stack,
            band_indices,
            has_swir=False,
        )
        return labels, confidence, stats


class KMeansClassifierAdapter:
    """KMeans++ segmentation with optional semantic mapping.
    
    By default, maps clusters to 6 land cover classes using spectral indices.
    Set raw_clusters=True to keep original cluster labels (0 to n_clusters-1).
    """

    name = "kmeans"

    def __init__(self, *, n_clusters: int = 6, raw_clusters: bool = False):
        self.n_clusters = n_clusters
        self.raw_clusters = raw_clusters

    def required_bands(self) -> List[str]:
        return ["B02", "B03", "B04", "B08"]

    def classify(self, bands: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        b02 = bands["B02"].astype(np.float32)
        b03 = bands["B03"].astype(np.float32)
        b04 = bands["B04"].astype(np.float32)
        b08 = bands["B08"].astype(np.float32)

        eps = 1e-6
        ndvi = (b08 - b04) / (b08 + b04 + eps)
        ndwi = (b03 - b08) / (b03 + b08 + eps)
        brightness = (b02 + b03 + b04) / 3.0

        p10 = float(np.nanpercentile(brightness, 10))
        p90 = float(np.nanpercentile(brightness, 90))

        stack = np.stack([b02, b03, b04, b08], axis=-1)
        data = reshape_image_to_table(stack)
        data_scaled = min_max_scale(data)

        clusterer = KMeansPlusPlusClusterer(
            n_clusters=self.n_clusters,
            max_iterations=30,
            random_state=42,
        )
        cluster_labels_1d = clusterer.fit_predict(data_scaled)
        cluster_img = reshape_table_to_image(b02.shape, cluster_labels_1d).astype(np.int32)

        # RAW CLUSTERS MODE: Return cluster labels as-is (0 to n_clusters-1)
        if self.raw_clusters:
            # Compute confidence based on distance to cluster center (simplified)
            confidence = np.ones_like(brightness, dtype=np.float32) * 0.7
            
            stats = {
                "avg_confidence": float(np.mean(confidence)),
                "n_clusters": self.n_clusters,
                "mode": "raw_clusters",
                "class_distribution": {},
            }
            unique, counts = np.unique(cluster_img, return_counts=True)
            total = int(counts.sum())
            for cls, count in zip(unique, counts):
                stats["class_distribution"][int(cls)] = {
                    "count": int(count),
                    "percentage": float(count / total * 100.0),
                }
            return cluster_img.astype(np.uint8), confidence, stats

        # SEMANTIC MAPPING MODE: Map clusters to land cover classes
        cluster_to_class: Dict[int, int] = {}
        cluster_to_conf: Dict[int, float] = {}

        for k in range(self.n_clusters):
            mask = cluster_img == k
            if not np.any(mask):
                cluster_to_class[k] = 5
                cluster_to_conf[k] = 0.5
                continue

            mean_ndvi = float(np.nanmean(ndvi[mask]))
            mean_ndwi = float(np.nanmean(ndwi[mask]))
            mean_brightness = float(np.nanmean(brightness[mask]))
            mean_rgb = float(np.nanmean(((b02 + b03 + b04) / 3.0)[mask]))
            mean_nir = float(np.nanmean(b08[mask]))

            if mean_ndwi > 0.25 and mean_ndvi < 0.2:
                cluster_to_class[k] = 0
                cluster_to_conf[k] = float(np.clip((mean_ndwi - 0.25) / 0.25 + 0.5, 0.5, 1.0))
                continue

            if mean_ndvi > 0.5:
                cluster_to_class[k] = 1
                cluster_to_conf[k] = float(np.clip((mean_ndvi - 0.5) / 0.3 + 0.5, 0.5, 1.0))
                continue

            if mean_brightness >= p90 and mean_ndvi < 0.2:
                cluster_to_class[k] = 4
                cluster_to_conf[k] = 0.7
                continue
            if mean_brightness <= p10:
                cluster_to_class[k] = 5
                cluster_to_conf[k] = 0.7
                continue

            visible_to_nir = mean_rgb / (mean_nir + eps)
            if mean_ndvi < 0.2 and visible_to_nir > 0.9:
                cluster_to_class[k] = 3
                cluster_to_conf[k] = 0.6
            else:
                cluster_to_class[k] = 2
                cluster_to_conf[k] = 0.6

        labels = np.zeros_like(cluster_img, dtype=np.uint8)
        confidence = np.zeros_like(brightness, dtype=np.float32)
        for k in range(self.n_clusters):
            mask = cluster_img == k
            labels[mask] = np.uint8(cluster_to_class[k])
            confidence[mask] = np.float32(cluster_to_conf[k])

        stats = _basic_stats(labels, confidence)
        stats["n_clusters"] = self.n_clusters
        return labels, confidence, stats


class SpectralClassifierAdapter:
    """Spectral indices classifier with SWIR requirement."""

    name = "spectral"

    def required_bands(self) -> List[str]:
        return ["B02", "B03", "B04", "B08", "B11", "B12"]

    def classify(self, bands: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        for b in ("B11", "B12"):
            if b not in bands:
                raise FileNotFoundError(
                    f"Spectral classifier requires {b}. Ensure SWIR bands are available (download/extract)."
                )

        band_order = ["B02", "B03", "B04", "B08", "B11", "B12"]
        band_stack = np.stack([bands[b].astype(np.float32) for b in band_order], axis=-1)
        band_indices = {b: i for i, b in enumerate(band_order)}

        spectral = SpectralIndicesClassifier()
        spectral.validate_bands(band_indices, num_bands=band_stack.shape[-1])
        spectral_labels, _ = spectral.classify_raster(band_stack, band_indices)

        mapping = {0: 0, 1: 1, 2: 1, 3: 3, 4: 2, 5: 5}
        labels = np.vectorize(lambda x: mapping.get(int(x), 5), otypes=[np.uint8])(spectral_labels)
        confidence = np.where(spectral_labels == 5, 0.5, 0.9).astype(np.float32)

        stats = _basic_stats(labels, confidence)
        stats["note"] = "spectral indices classifier (SWIR required)"
        return labels, confidence, stats


def get_classifier(
    classifier: Literal["kmeans", "spectral", "consensus"], 
    *, 
    n_clusters: int = 6,
    raw_clusters: bool = False,
) -> Classifier:
    """Factory returning a classifier adapter by name.
    
    Args:
        classifier: Method name ("kmeans", "spectral", "consensus")
        n_clusters: Number of clusters (for kmeans/consensus)
        raw_clusters: If True, kmeans returns raw cluster IDs (0 to n-1)
                      without semantic mapping to land cover classes.
                      Useful for exploratory analysis.
    """

    if classifier == "consensus":
        return ConsensusClassifierAdapter(n_clusters=n_clusters)
    if classifier == "kmeans":
        return KMeansClassifierAdapter(n_clusters=n_clusters, raw_clusters=raw_clusters)
    if classifier == "spectral":
        return SpectralClassifierAdapter()
    raise ValueError(f"Unknown classifier: {classifier}")

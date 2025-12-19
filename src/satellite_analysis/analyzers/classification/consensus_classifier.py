"""
Consensus Classifier for combining multiple classification methods.

Combines K-Means clustering with Spectral Indices classification
to produce a more robust land cover classification with confidence scoring.

Classes:
    0: WATER
    1: VEGETATION (forest + grassland)
    2: BARE_SOIL
    3: URBAN
    4: BRIGHT_SURFACES
    5: SHADOWS/MIXED
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from collections import Counter

from satellite_analysis.analyzers.clustering import KMeansPlusPlusClusterer
from satellite_analysis.analyzers.classification import SpectralIndicesClassifier
from satellite_analysis.preprocessing.normalization import min_max_scale
from satellite_analysis.preprocessing.reshape import reshape_image_to_table, reshape_table_to_image

logger = logging.getLogger(__name__)


class ConsensusClassifier:
    """
    Combines K-Means clustering and Spectral Indices classification
    with confidence scoring and uncertainty flagging.
    
    Approach:
    1. Run K-Means clustering (6 clusters)
    2. Run Spectral Indices classification (6 classes)
    3. Map K-Means clusters to semantic classes (learned mapping)
    4. Compute agreement at pixel level
    5. Assign confidence score (0.0 = disagree, 1.0 = full agreement)
    6. Flag uncertain pixels for manual review
    
    Output:
    - Final classification map (6 classes)
    - Confidence map (0-1 per pixel)
    - Uncertainty mask (boolean, True = needs review)
    - Statistics (agreement %, average confidence)
    """
    
    # Unified class definitions
    CLASSES = {
        0: 'WATER',
        1: 'VEGETATION',
        2: 'BARE_SOIL',
        3: 'URBAN',
        4: 'BRIGHT_SURFACES',
        5: 'SHADOWS_MIXED'
    }
    
    # Mapping from SpectralIndicesClassifier classes to our classes
    SPECTRAL_TO_CONSENSUS = {
        0: 0,  # WATER -> WATER
        1: 1,  # FOREST -> VEGETATION
        2: 1,  # GRASSLAND -> VEGETATION
        3: 3,  # URBAN -> URBAN
        4: 2,  # BARE_SOIL -> BARE_SOIL
        5: 5   # MIXED -> SHADOWS_MIXED
    }
    
    def __init__(
        self,
        n_clusters: int = 6,
        confidence_threshold: float = 0.5,
        sample_size: int = 2_000_000,
        random_state: Optional[int] = 42
    ):
        """
        Initialize Consensus Classifier.
        
        Args:
            n_clusters: Number of clusters for K-Means (default: 6)
            confidence_threshold: Threshold below which pixels are flagged uncertain (default: 0.5)
            sample_size: Number of samples for K-Means training (default: 2M)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        self.sample_size = sample_size
        self.random_state = random_state
        
        # Components
        self.kmeans = KMeansPlusPlusClusterer(
            n_clusters=n_clusters,
            max_iterations=30,
            random_state=random_state
        )
        self.spectral = SpectralIndicesClassifier()
        
        # Learned mapping from K-Means clusters to semantic classes
        self.cluster_to_class_map: Optional[Dict[int, int]] = None
        
        # Results
        self.labels_kmeans_: Optional[np.ndarray] = None
        self.labels_spectral_: Optional[np.ndarray] = None
        self.labels_consensus_: Optional[np.ndarray] = None
        self.confidence_map_: Optional[np.ndarray] = None
        self.uncertainty_mask_: Optional[np.ndarray] = None
        self.statistics_: Optional[Dict] = None
        
        logger.info(f"ConsensusClassifier initialized (K={n_clusters}, threshold={confidence_threshold})")
    
    def _learn_cluster_mapping(
        self,
        labels_kmeans: np.ndarray,
        labels_spectral: np.ndarray
    ) -> Dict[int, int]:
        """
        Learn mapping from K-Means clusters to semantic classes.
        
        For each K-Means cluster, find the most common Spectral class.
        
        Args:
            labels_kmeans: K-Means cluster labels (flattened)
            labels_spectral: Spectral class labels (flattened)
            
        Returns:
            Dict mapping cluster_id -> class_id
        """
        mapping = {}
        
        for cluster_id in range(self.n_clusters):
            # Find all spectral labels where K-Means assigned this cluster
            mask = labels_kmeans == cluster_id
            spectral_labels_in_cluster = labels_spectral[mask]
            
            if len(spectral_labels_in_cluster) == 0:
                # Empty cluster, assign to SHADOWS_MIXED
                mapping[cluster_id] = 5
                logger.warning(f"Cluster {cluster_id} is empty, mapping to SHADOWS_MIXED")
                continue
            
            # Find most common spectral class
            counter = Counter(spectral_labels_in_cluster)
            most_common_spectral = counter.most_common(1)[0][0]
            
            # Map to consensus class
            consensus_class = self.SPECTRAL_TO_CONSENSUS.get(most_common_spectral, 5)
            mapping[cluster_id] = consensus_class
            
            # Calculate confidence for this mapping
            agreement_pct = counter.most_common(1)[0][1] / len(spectral_labels_in_cluster) * 100
            logger.info(
                f"Cluster {cluster_id} -> {self.CLASSES[consensus_class]} "
                f"({agreement_pct:.1f}% agreement)"
            )
        
        return mapping
    
    def _compute_confidence(
        self,
        labels_kmeans_mapped: np.ndarray,
        labels_spectral_mapped: np.ndarray
    ) -> np.ndarray:
        """
        Compute pixel-level confidence based on method agreement.
        
        Confidence scoring:
        - 1.0: Both methods agree on the class
        - 0.5: Methods disagree but classes are "similar" (e.g., VEGETATION subtypes)
        - 0.0: Methods completely disagree
        
        Args:
            labels_kmeans_mapped: K-Means labels mapped to consensus classes
            labels_spectral_mapped: Spectral labels mapped to consensus classes
            
        Returns:
            Confidence map (0-1) with same shape as input labels
        """
        # Full agreement = 1.0
        confidence = np.where(
            labels_kmeans_mapped == labels_spectral_mapped,
            1.0,
            0.0
        ).astype(np.float32)
        
        # Partial agreement for similar classes (e.g., both "natural" or both "built")
        # Natural classes: WATER (0), VEGETATION (1), BARE_SOIL (2)
        # Built classes: URBAN (3), BRIGHT_SURFACES (4)
        
        natural_classes = {0, 1, 2}
        built_classes = {3, 4}
        
        # If both in natural OR both in built, give partial confidence
        kmeans_natural = np.isin(labels_kmeans_mapped, list(natural_classes))
        spectral_natural = np.isin(labels_spectral_mapped, list(natural_classes))
        
        kmeans_built = np.isin(labels_kmeans_mapped, list(built_classes))
        spectral_built = np.isin(labels_spectral_mapped, list(built_classes))
        
        # Partial agreement mask (same category but different class)
        partial_agreement = (
            ((kmeans_natural & spectral_natural) | (kmeans_built & spectral_built)) &
            (labels_kmeans_mapped != labels_spectral_mapped)
        )
        
        confidence[partial_agreement] = 0.5
        
        return confidence
    
    def classify(
        self,
        raster: np.ndarray,
        band_indices: Dict[str, int],
        has_swir: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Full consensus classification pipeline.
        
        Args:
            raster: Multi-band raster array of shape (H, W, C) or (C, H, W)
            band_indices: Dict mapping band names to array indices
                Required: 'B02' (Blue), 'B03' (Green), 'B04' (Red), 'B08' (NIR)
                Optional: 'B11' (SWIR1), 'B12' (SWIR2) for full spectral classification
            has_swir: Whether SWIR bands are available (affects spectral classification)
            
        Returns:
            Tuple of (labels, confidence, uncertainty_mask, statistics):
                - labels: Final classification (H, W) with values 0-5
                - confidence: Confidence map (H, W) with values 0.0-1.0
                - uncertainty_mask: Boolean mask (H, W), True = needs review
                - statistics: Dict with agreement %, avg confidence, class distribution
        """
        logger.info("Starting consensus classification...")
        
        # Handle both (H,W,C) and (C,H,W) formats
        if raster.shape[0] < 20:  # Assume (C,H,W) if first dim is small
            raster = np.transpose(raster, (1, 2, 0))
        
        height, width, n_bands = raster.shape
        n_pixels = height * width
        
        logger.info(f"Raster shape: {raster.shape} ({n_pixels:,} pixels)")
        
        # ============================================================
        # Step 1: K-Means Clustering
        # ============================================================
        logger.info("Step 1: K-Means Clustering...")
        
        # Prepare data for K-Means (use available bands: B02, B03, B04, B08)
        kmeans_bands = ['B02', 'B03', 'B04', 'B08']
        stack = np.stack([raster[:, :, band_indices[b]] for b in kmeans_bands], axis=-1)
        
        # Reshape to table
        data = reshape_image_to_table(stack)
        data_scaled = min_max_scale(data)
        
        # Sample for training
        n_total = len(data_scaled)
        sample_size = min(self.sample_size, n_total)
        step = max(1, n_total // sample_size)
        sample_indices = np.arange(0, n_total, step)[:sample_size]
        sample = data_scaled[sample_indices]
        
        logger.info(f"Training K-Means on {len(sample):,} samples...")
        self.kmeans.fit(sample)
        
        # Predict all pixels
        logger.info(f"Predicting {n_total:,} pixels...")
        labels_kmeans_flat = self.kmeans.predict(data_scaled)
        labels_kmeans = reshape_table_to_image(stack.shape, labels_kmeans_flat)
        
        self.labels_kmeans_ = labels_kmeans
        
        # ============================================================
        # Step 2: Spectral Indices Classification
        # ============================================================
        logger.info("Step 2: Spectral Indices Classification...")
        
        if has_swir and 'B11' in band_indices and 'B12' in band_indices:
            # Full spectral classification with SWIR bands
            indices = self.spectral.compute_indices(raster, band_indices)
            labels_spectral = self.spectral.classify(indices)
        else:
            # Simplified classification using only RGB + NIR
            labels_spectral = self._classify_spectral_simple(raster, band_indices)
        
        self.labels_spectral_ = labels_spectral
        
        # ============================================================
        # Step 3: Learn Cluster Mapping
        # ============================================================
        logger.info("Step 3: Learning cluster-to-class mapping...")
        
        # Flatten for mapping
        labels_kmeans_flat = labels_kmeans.flatten()
        labels_spectral_flat = labels_spectral.flatten()
        
        # Map spectral labels to consensus classes
        labels_spectral_consensus = np.vectorize(
            lambda x: self.SPECTRAL_TO_CONSENSUS.get(x, 5)
        )(labels_spectral_flat)
        
        # Learn K-Means to consensus mapping
        self.cluster_to_class_map = self._learn_cluster_mapping(
            labels_kmeans_flat,
            labels_spectral_consensus
        )
        
        # Apply mapping to K-Means labels
        labels_kmeans_consensus = np.vectorize(
            lambda x: self.cluster_to_class_map.get(x, 5)
        )(labels_kmeans_flat)
        
        # ============================================================
        # Step 4: Compute Consensus
        # ============================================================
        logger.info("Step 4: Computing consensus...")
        
        # Confidence based on agreement
        confidence_flat = self._compute_confidence(
            labels_kmeans_consensus,
            labels_spectral_consensus
        )
        
        # Final labels: prefer spectral for high-confidence areas
        # For low-confidence, use weighted voting (currently: spectral wins)
        labels_consensus_flat = np.where(
            confidence_flat >= self.confidence_threshold,
            labels_spectral_consensus,  # High confidence: use spectral
            # Low confidence: majority vote (here: spectral as tiebreaker)
            np.where(
                labels_kmeans_consensus == labels_spectral_consensus,
                labels_spectral_consensus,
                labels_spectral_consensus  # Spectral wins ties
            )
        )
        
        # Reshape to image
        labels_consensus = labels_consensus_flat.reshape(height, width)
        confidence_map = confidence_flat.reshape(height, width)
        
        # Uncertainty mask
        uncertainty_mask = confidence_map < self.confidence_threshold
        
        self.labels_consensus_ = labels_consensus
        self.confidence_map_ = confidence_map
        self.uncertainty_mask_ = uncertainty_mask
        
        # ============================================================
        # Step 5: Compute Statistics
        # ============================================================
        logger.info("Step 5: Computing statistics...")
        
        # Agreement statistics
        agreement_count = np.sum(labels_kmeans_consensus == labels_spectral_consensus)
        agreement_pct = agreement_count / n_pixels * 100
        
        avg_confidence = np.mean(confidence_flat)
        uncertain_pct = np.sum(uncertainty_mask) / n_pixels * 100
        
        # Class distribution
        unique, counts = np.unique(labels_consensus, return_counts=True)
        class_distribution = {
            self.CLASSES[label]: {
                'count': int(count),
                'percentage': float(count / n_pixels * 100)
            }
            for label, count in zip(unique, counts)
        }
        
        statistics = {
            'agreement_pct': float(agreement_pct),
            'avg_confidence': float(avg_confidence),
            'uncertain_pct': float(uncertain_pct),
            'class_distribution': class_distribution,
            'cluster_mapping': {
                int(k): self.CLASSES[v] for k, v in self.cluster_to_class_map.items()
            }
        }
        
        self.statistics_ = statistics
        
        logger.info(f"Consensus complete: {agreement_pct:.1f}% agreement, "
                   f"{avg_confidence:.2f} avg confidence, "
                   f"{uncertain_pct:.1f}% uncertain")
        
        return labels_consensus, confidence_map, uncertainty_mask, statistics
    
    def _classify_spectral_simple(
        self,
        raster: np.ndarray,
        band_indices: Dict[str, int]
    ) -> np.ndarray:
        """
        Simplified spectral classification using only RGB + NIR bands.
        
        Uses NDVI and simple band ratios when SWIR bands are not available.
        
        Args:
            raster: Multi-band raster array (H, W, C)
            band_indices: Dict mapping band names to array indices
            
        Returns:
            Classification labels (H, W) with values 0-5
        """
        # Extract bands
        blue = raster[:, :, band_indices['B02']].astype(np.float32)
        green = raster[:, :, band_indices['B03']].astype(np.float32)
        red = raster[:, :, band_indices['B04']].astype(np.float32)
        nir = raster[:, :, band_indices['B08']].astype(np.float32)
        
        # NDVI (vegetation index)
        ndvi_denom = nir + red
        ndvi = np.divide(
            nir - red,
            ndvi_denom,
            out=np.zeros_like(ndvi_denom, dtype=np.float32),
            where=ndvi_denom != 0,
        )
        
        # NDWI (water index using green and NIR)
        ndwi_denom = green + nir
        ndwi = np.divide(
            green - nir,
            ndwi_denom,
            out=np.zeros_like(ndwi_denom, dtype=np.float32),
            where=ndwi_denom != 0,
        )
        
        # Simple urban index (red/NIR ratio)
        urban_idx = np.divide(
            red,
            nir,
            out=np.zeros_like(nir, dtype=np.float32),
            where=nir != 0,
        )
        
        # Brightness (average reflectance)
        brightness = (red + green + blue) / 3
        
        # Initialize with SHADOWS_MIXED (5)
        labels = np.full(ndvi.shape, 5, dtype=np.uint8)
        
        # Classification rules (simplified without SWIR)
        # Order matters! Later rules can override earlier ones
        
        # SHADOWS: very low brightness (true shadows/dark areas)
        shadow_mask = brightness < 300
        labels[shadow_mask] = 5
        
        # WATER: high NDWI and low NIR
        water_mask = (ndwi > 0.2) & (nir < 1500)
        labels[water_mask] = 0
        
        # VEGETATION: high NDVI (clear vegetation signal)
        veg_mask = (ndvi > 0.35) & ~water_mask & ~shadow_mask
        labels[veg_mask] = 1
        
        # BARE_SOIL: low NDVI, moderate-high brightness, brownish
        soil_mask = (ndvi < 0.2) & (ndvi > -0.1) & (brightness > 800) & (brightness < 3000) & ~water_mask
        labels[soil_mask] = 2
        
        # URBAN: low-moderate NDVI, moderate brightness, not water/veg
        # Urban areas have NDVI typically 0.1-0.3 and moderate brightness
        urban_mask = (
            (ndvi >= -0.1) & (ndvi < 0.35) &  # Low NDVI
            (brightness > 500) & (brightness < 4000) &  # Moderate brightness
            ~water_mask & ~shadow_mask & ~veg_mask
        )
        labels[urban_mask] = 3
        
        # BRIGHT_SURFACES: very high brightness (concrete, roofs, sand)
        bright_mask = (brightness > 2000) & (ndvi < 0.1) & ~water_mask
        labels[bright_mask] = 4
        
        return labels
    
    def get_class_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """Get RGB colors for each class for visualization."""
        return {
            0: (0, 0, 255),       # WATER - Blue
            1: (34, 139, 34),     # VEGETATION - Forest Green
            2: (139, 69, 19),     # BARE_SOIL - Saddle Brown
            3: (128, 128, 128),   # URBAN - Gray
            4: (255, 255, 0),     # BRIGHT_SURFACES - Yellow
            5: (0, 0, 0)          # SHADOWS_MIXED - Black
        }
    
    def get_class_names(self) -> Dict[int, str]:
        """Get class names for display."""
        return self.CLASSES.copy()

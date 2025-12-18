"""KMeans++ clustering implementation."""

import numpy as np
from typing import Optional

from satellite_analysis.analyzers.clustering.base import ClusteringStrategy, ClusteringFactory


class KMeansPlusPlusClusterer(ClusteringStrategy):
    """KMeans++ clustering with improved initialization."""
    
    def __init__(
        self,
        n_clusters: int,
        max_iterations: int = 100,
        random_state: Optional[int] = None,
    ):
        """Initialize KMeans++ clusterer.
        
        Args:
            n_clusters: Number of clusters
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.max_iterations = max_iterations
    
    def _initialize_centroids_plus_plus(self, data: np.ndarray) -> np.ndarray:
        """Initialize centroids using KMeans++ algorithm.
        
        Args:
            data: Input data array
            
        Returns:
            Initial centroids array
        """
        rng = np.random.default_rng(self.random_state)
        
        # Choose first centroid randomly
        centroids = [data[rng.choice(range(data.shape[0]))]]
        
        # Choose remaining centroids based on distance
        for _ in range(self.n_clusters - 1):
            # Calculate distances to nearest centroid (memory-efficient)
            min_distances = np.full(data.shape[0], np.inf, dtype=np.float32)
            
            for centroid in centroids:
                diff = data - centroid
                distances = np.sum(diff ** 2, axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # Choose next centroid with probability proportional to distance squared
            probabilities = min_distances / np.sum(min_distances)
            
            new_centroid_idx = rng.choice(range(data.shape[0]), p=probabilities)
            centroids.append(data[new_centroid_idx])
        
        return np.array(centroids)
    
    def fit(self, data: np.ndarray) -> "KMeansPlusPlusClusterer":
        """Fit KMeans++ clustering model.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        # Initialize centroids using KMeans++
        self.centroids_ = self._initialize_centroids_plus_plus(data)
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):
            # Assign points to nearest centroid (memory-efficient)
            distances = np.zeros((data.shape[0], self.n_clusters), dtype=np.float32)
            for k in range(self.n_clusters):
                diff = data - self.centroids_[k]
                distances[:, k] = np.sum(diff ** 2, axis=1)
            
            self.labels_ = np.argmin(distances, axis=-1)
            
            # Update centroids (vectorized)
            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    new_centroids[k] = data[mask].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids_[k]
            
            # Check convergence
            if np.allclose(self.centroids_, new_centroids, rtol=1e-4):
                print(f"Converged after {iteration + 1} iterations")
                break
            
            self.centroids_ = new_centroids
        
        # Calculate inertia
        self.inertia_ = self._calculate_inertia(data, self.labels_)
        
        return self


# Register with factory
ClusteringFactory.register("kmeans++", KMeansPlusPlusClusterer)

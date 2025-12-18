"""KMeans clustering implementation."""

import numpy as np
from typing import Optional

from satellite_analysis.analyzers.clustering.base import ClusteringStrategy, ClusteringFactory


class KMeansClusterer(ClusteringStrategy):
    """Standard KMeans clustering algorithm."""
    
    def __init__(
        self,
        n_clusters: int,
        max_iterations: int = 100,
        random_state: Optional[int] = None,
    ):
        """Initialize KMeans clusterer.
        
        Args:
            n_clusters: Number of clusters
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.max_iterations = max_iterations
    
    def fit(self, data: np.ndarray) -> "KMeansClusterer":
        """Fit KMeans clustering model.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        rng = np.random.default_rng(self.random_state)
        
        # Initialize centroids randomly
        indices = rng.choice(range(data.shape[0]), size=self.n_clusters, replace=False)
        self.centroids_ = data[indices].copy()
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iterations):
            # Assign points to nearest centroid (optimized for memory)
            # Instead of creating huge (n_samples, n_clusters, n_features) array,
            # compute distances iteratively per centroid
            distances = np.zeros((data.shape[0], self.n_clusters), dtype=np.float32)
            for k in range(self.n_clusters):
                # Compute squared distance to centroid k (faster than norm)
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
ClusteringFactory.register("kmeans", KMeansClusterer)

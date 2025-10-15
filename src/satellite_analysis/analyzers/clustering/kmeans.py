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
            # Assign points to nearest centroid
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids_, axis=-1)
            self.labels_ = np.argmin(distances, axis=-1)
            
            # Update centroids
            new_centroids = np.array([
                data[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) 
                else self.centroids_[i]
                for i in range(self.n_clusters)
            ])
            
            # Check convergence
            if np.allclose(self.centroids_, new_centroids):
                print(f"Converged after {iteration + 1} iterations")
                break
            
            self.centroids_ = new_centroids
        
        # Calculate inertia
        self.inertia_ = self._calculate_inertia(data, self.labels_)
        
        return self


# Register with factory
ClusteringFactory.register("kmeans", KMeansClusterer)

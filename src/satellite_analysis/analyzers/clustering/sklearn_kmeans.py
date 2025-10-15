"""Sklearn KMeans wrapper."""

import numpy as np
from typing import Optional
from sklearn.cluster import KMeans

from satellite_analysis.analyzers.clustering.base import ClusteringStrategy, ClusteringFactory


class SklearnKMeansClusterer(ClusteringStrategy):
    """Wrapper for sklearn KMeans implementation."""
    
    def __init__(
        self,
        n_clusters: int,
        n_init: int = 10,
        max_iterations: int = 300,
        random_state: Optional[int] = None,
    ):
        """Initialize sklearn KMeans clusterer.
        
        Args:
            n_clusters: Number of clusters
            n_init: Number of times to run with different seeds
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(n_clusters, random_state)
        self.n_init = n_init
        self.max_iterations = max_iterations
        self._model = None
    
    def fit(self, data: np.ndarray) -> "SklearnKMeansClusterer":
        """Fit sklearn KMeans clustering model.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        self._model = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iterations,
            random_state=self.random_state,
        )
        
        self._model.fit(data)
        
        self.labels_ = self._model.labels_
        self.centroids_ = self._model.cluster_centers_
        self.inertia_ = self._model.inertia_
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for data.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster labels array of shape (n_samples,)
        """
        if self._model is None:
            raise ValueError("Model must be fitted before prediction")
        return self._model.predict(data)


# Register with factory
ClusteringFactory.register("sklearn", SklearnKMeansClusterer)

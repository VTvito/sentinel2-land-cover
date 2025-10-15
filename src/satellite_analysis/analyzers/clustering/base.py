"""Base classes for clustering algorithms (Strategy Pattern)."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple


class ClusteringStrategy(ABC):
    """Abstract base class for clustering algorithms (Strategy Pattern)."""
    
    def __init__(self, n_clusters: int, random_state: Optional[int] = None):
        """Initialize clustering strategy.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> "ClusteringStrategy":
        """Fit the clustering model to data.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        pass
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for data.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster labels array of shape (n_samples,)
        """
        if self.centroids_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids_, axis=-1)
        return np.argmin(distances, axis=-1)
    
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster labels array of shape (n_samples,)
        """
        self.fit(data)
        return self.labels_
    
    def _calculate_inertia(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares.
        
        Args:
            data: Input data array
            labels: Cluster labels
            
        Returns:
            Inertia value
        """
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids_[i]) ** 2)
        return inertia


class ClusteringFactory:
    """Factory for creating clustering strategies."""
    
    _strategies = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a clustering strategy.
        
        Args:
            name: Name to register the strategy under
            strategy_class: The strategy class to register
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> ClusteringStrategy:
        """Create a clustering strategy by name.
        
        Args:
            name: Name of the strategy to create
            **kwargs: Arguments to pass to the strategy constructor
            
        Returns:
            ClusteringStrategy instance
            
        Raises:
            ValueError: If strategy name is not registered
        """
        if name not in cls._strategies:
            raise ValueError(
                f"Unknown clustering strategy: {name}. "
                f"Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[name](**kwargs)
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())

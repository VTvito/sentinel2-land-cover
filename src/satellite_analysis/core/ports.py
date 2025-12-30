"""Minimal ports to decouple pipeline from concrete services."""

from __future__ import annotations

from typing import Protocol, Dict, Tuple, Any, List
import numpy as np


class ClassifierPort(Protocol):
    name: str

    def required_bands(self) -> List[str]:
        ...

    def classify(self, bands: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        ...


class AreaSelectorPort(Protocol):
    def select_by_city(self, city: str, radius_km: float):
        ...


class OutputManagerPort(Protocol):
    def create_run(self, classifier: str, metadata: Dict[str, Any]):
        ...

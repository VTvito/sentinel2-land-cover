"""Satellite Image Analysis Toolkit.

A comprehensive toolkit for downloading, processing, and analyzing Sentinel-2
satellite imagery using machine learning techniques.

Features:
- K-Means and K-Means++ clustering
- Spectral indices classification
- Consensus classification with confidence scoring
- Validation against ESA Scene Classification Layer
"""

__version__ = "1.0.0"
__author__ = "Vito Delia"

from satellite_analysis.config import Config

__all__ = ["Config", "__version__"]

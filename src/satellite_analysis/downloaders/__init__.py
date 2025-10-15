"""Base downloader interface."""

from satellite_analysis.downloaders.auth import AuthStrategy, OAuth2AuthStrategy
from satellite_analysis.downloaders.catalog import CatalogStrategy, SentinelHubCatalog
from satellite_analysis.downloaders.product_downloader import ProductDownloader

__all__ = [
    "AuthStrategy",
    "OAuth2AuthStrategy", 
    "CatalogStrategy",
    "SentinelHubCatalog",
    "ProductDownloader",
]

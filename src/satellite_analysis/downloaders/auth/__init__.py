"""Authentication strategies for Sentinel data access."""

from .oauth2_auth import AuthStrategy, OAuth2AuthStrategy

__all__ = ["AuthStrategy", "OAuth2AuthStrategy"]

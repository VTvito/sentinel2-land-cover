"""Authentication strategies for Sentinel data access."""

from abc import ABC, abstractmethod
from typing import Optional
import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


class AuthStrategy(ABC):
    """Abstract base class for authentication strategies."""
    
    @abstractmethod
    def get_session(self) -> requests.Session:
        """Get an authenticated session.
        
        Returns:
            Authenticated requests.Session object
        """
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        """Check if authentication is valid.
        
        Returns:
            True if authentication is valid, False otherwise
        """
        pass


class OAuth2AuthStrategy(AuthStrategy):
    """OAuth2 authentication for Copernicus Data Space Ecosystem."""
    
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    
    def __init__(self, client_id: str, client_secret: str):
        """Initialize OAuth2 authentication.
        
        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self._session: Optional[OAuth2Session] = None
        self._token: Optional[dict] = None
    
    def get_session(self) -> OAuth2Session:
        """Get an authenticated OAuth2 session.
        
        Returns:
            Authenticated OAuth2Session
            
        Raises:
            RuntimeError: If authentication fails
        """
        if self._session is None or self._token is None:
            self._authenticate()
        elif not self.is_valid():
            self.refresh()
        
        return self._session
    
    def _authenticate(self) -> None:
        """Perform OAuth2 authentication."""
        try:
            # Create OAuth2 client
            client = BackendApplicationClient(client_id=self.client_id)
            self._session = OAuth2Session(client=client)
            
            # Fetch token
            self._token = self._session.fetch_token(
                token_url=self.TOKEN_URL,
                client_id=self.client_id,
                client_secret=self.client_secret,
                include_client_id=True
            )
            
            pass  # Authentication successful
            
        except Exception as e:
            raise RuntimeError(f"OAuth2 authentication failed: {e}")
    
    def is_valid(self) -> bool:
        """Check if the current token is valid.
        
        Returns:
            True if token exists and is valid, False otherwise
        """
        if self._session is None or self._token is None:
            return False
        
        # Check if token has expires_at field
        if 'expires_at' not in self._token:
            # Token was just fetched, assume it's valid
            return True
        
        # Check if token is expired (with 60 second buffer)
        import time
        expires_at = self._token.get('expires_at', 0)
        return time.time() < (expires_at - 60)
    
    def refresh(self) -> None:
        """Refresh the authentication token."""
        self._authenticate()

"""Product downloader for Sentinel data."""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import requests
from tqdm import tqdm


class ProductDownloader:
    """Download Sentinel products from Copernicus Data Space.
    
    This class handles the actual download of satellite products
    identified through catalog search.
    """
    
    def __init__(
        self, 
        session: requests.Session,
        output_dir: str = "data/raw",
        chunk_size: int = 8192
    ):
        """Initialize product downloader.
        
        Args:
            session: Authenticated session (OAuth2Session or requests.Session with auth)
            output_dir: Directory to save downloaded products
            chunk_size: Size of chunks for streaming download (bytes)
        """
        # Extract token from OAuth2Session and create standard requests session
        if hasattr(session, 'token') and session.token:
            # OAuth2Session - extract token
            access_token = session.token.get('access_token')
            self.session = requests.Session()
            self.session.headers['Authorization'] = f'Bearer {access_token}'
        else:
            # Already a standard session with auth
            self.session = session
            
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_product(
        self,
        product_id: str,
        download_url: Optional[str] = None,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """Download a single product.
        
        Args:
            product_id: Unique product identifier
            download_url: Direct download URL (if available)
            filename: Custom filename (if not provided, uses product_id)
            progress_callback: Optional callback(downloaded, total) for progress
            
        Returns:
            Path to downloaded file
            
        Raises:
            requests.HTTPError: If download fails
            IOError: If file writing fails
        """
        # Determine filename
        if filename is None:
            filename = f"{product_id}.zip"
        
        output_path = self.output_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            pass  # File already exists
            return output_path
        
        # Construct download URL if not provided
        if download_url is None:
            download_url = self._construct_download_url(product_id)
        
        # Download starting
        
        try:
            # Stream download
            response = self.session.get(download_url, stream=True)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            downloaded = 0
            with open(output_path, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=filename
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            
                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(downloaded, total_size)
            
            pass  # Download completed
            return output_path
            
        except requests.exceptions.HTTPError as e:
            if output_path.exists():
                output_path.unlink()  # Delete partial file
            raise requests.HTTPError(
                f"Download failed for {product_id}: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise IOError(f"Download error for {product_id}: {e}")
    
    def download_products(
        self,
        features: List[Dict[str, Any]],
        max_products: Optional[int] = None,
        skip_existing: bool = True
    ) -> List[Path]:
        """Download multiple products from catalog search results.
        
        Args:
            features: List of features from catalog search
            max_products: Maximum number of products to download
            skip_existing: Skip already downloaded products
            
        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        products_to_download = features[:max_products] if max_products else features
        
        for idx, feature in enumerate(products_to_download, 1):
            props = feature.get('properties', {})
            product_id = props.get('id', f'product_{idx}')
            
            try:
                # Get download URL from assets
                download_url = self._extract_download_url(feature)
                
                # Download
                file_path = self.download_product(
                    product_id=product_id,
                    download_url=download_url,
                    filename=f"{product_id}.zip"
                )
                
                downloaded_files.append(file_path)
                
            except Exception:
                continue
        
        return downloaded_files
    
    def _construct_download_url(self, product_id: str) -> str:
        """Construct download URL for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Download URL
        """
        # Copernicus Data Space download endpoint
        base_url = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
        return f"{base_url}({product_id})/$value"
    
    def _extract_download_url(self, feature: Dict[str, Any]) -> Optional[str]:
        """Extract download URL from feature assets.
        
        Args:
            feature: Feature from catalog search
            
        Returns:
            Download URL or None
        """
        # Get product name (e.g., S2B_MSIL2A_20230312T101729_N0510_R065_T32TMR_20240818T202748.SAFE)
        product_name = feature.get('id')
        if not product_name:
            product_name = feature.get('properties', {}).get('id')
        
        if not product_name:
            pass  # No product ID found
            return None
        
        # Need to query OData API to get the UUID
        try:
            # Query OData catalog to get UUID
            # Use startswith since the name in OData has .SAFE extension
            odata_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=startswith(Name, '{product_name}')"
            response = self.session.get(odata_url)
            response.raise_for_status()
            
            data = response.json()
            if data.get('value') and len(data['value']) > 0:
                product_uuid = data['value'][0].get('Id')
                if product_uuid:
                    return f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_uuid})/$value"
            
            pass  # UUID not found for product
            return None
            
        except Exception as e:
            pass  # Error querying OData catalog
            return None
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted string (e.g., "1.23 GB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """Get information about a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Dictionary with product metadata
            
        Raises:
            requests.HTTPError: If API request fails
        """
        # OData endpoint for product info
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise requests.HTTPError(
                f"Failed to get product info: {e.response.status_code} - {e.response.text}"
            )

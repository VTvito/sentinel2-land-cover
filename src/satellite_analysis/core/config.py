"""Analysis configuration - single source of truth for all parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Callable, Any
from datetime import date, datetime, timedelta
import argparse


@dataclass
class AnalysisConfig:
    """Immutable configuration for analysis pipeline.
    
    This is the single source of truth for all pipeline parameters.
    All paths are resolved relative to project_root.
    
    Example:
        >>> config = AnalysisConfig.for_notebook("Florence")
        >>> config = AnalysisConfig.for_cli(args)
        >>> config = AnalysisConfig(city="Milan", project_root=Path.cwd())
    """
    
    # Required
    city: str
    project_root: Path
    
    # Download options
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    cloud_cover: int = 20
    download_limit: int = 10
    radius_km: float = 15.0
    
    # Processing options
    max_size: Optional[int] = None  # None = no limit
    
    # Classification options
    classifier: Literal["kmeans", "spectral", "consensus"] = "consensus"
    n_clusters: int = 6
    
    # Output options
    save_preview: bool = True
    
    def __post_init__(self):
        """Ensure project_root is a Path."""
        object.__setattr__(self, 'project_root', Path(self.project_root).resolve())
    
    # =========================================================================
    # Factory methods
    # =========================================================================
    
    @classmethod
    def for_notebook(
        cls,
        city: str,
        notebook_path: Optional[Path] = None,
        **kwargs
    ) -> "AnalysisConfig":
        """Create config for notebook execution.
        
        Auto-detects project root from notebook location.
        
        Args:
            city: City name
            notebook_path: Path to notebook (auto-detect if None)
            **kwargs: Override any config parameters
        """
        if notebook_path:
            # notebooks/ -> project_root
            project_root = Path(notebook_path).parent.parent
        else:
            # Assume notebooks are in {project}/notebooks/
            # Try to find project root by looking for pyproject.toml
            cwd = Path.cwd()
            project_root = cls._find_project_root(cwd)
        
        return cls(city=city, project_root=project_root, **kwargs)
    
    @classmethod
    def for_cli(cls, args: argparse.Namespace) -> "AnalysisConfig":
        """Create config from CLI arguments.
        
        Args:
            args: Parsed argparse namespace
        """
        # Find project root from script location or cwd
        project_root = cls._find_project_root(Path.cwd())
        
        # Parse dates
        start_date = None
        end_date = None
        if hasattr(args, 'start') and args.start:
            start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        if hasattr(args, 'end') and args.end:
            end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
        
        return cls(
            city=args.city,
            project_root=project_root,
            start_date=start_date,
            end_date=end_date,
            cloud_cover=getattr(args, 'cloud_cover', 20),
            max_size=getattr(args, 'max_size', None),
            classifier=getattr(args, 'classifier', 'consensus'),
            radius_km=getattr(args, 'radius', 15.0),
        )
    
    @classmethod
    def _find_project_root(cls, start_path: Path) -> Path:
        """Find project root by looking for pyproject.toml."""
        current = start_path.resolve()
        for _ in range(10):  # Max 10 levels up
            if (current / "pyproject.toml").exists():
                return current
            if (current / "src" / "satellite_analysis").exists():
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        
        # Fallback to cwd
        return start_path.resolve()
    
    # =========================================================================
    # Path resolution
    # =========================================================================
    
    def data_path(self, *parts: str) -> Path:
        """Resolve path under data/ directory.
        
        Example:
            >>> config.data_path("cities", "florence", "bands")
            Path("/project/data/cities/florence/bands")
        """
        return self.project_root / "data" / Path(*parts)
    
    def config_path(self, filename: str = "config.yaml") -> Path:
        """Get path to config file."""
        return self.project_root / "config" / filename
    
    def city_data_path(self, *parts: str) -> Path:
        """Resolve path under data/cities/{city}/.
        
        Example:
            >>> config.city_data_path("bands")
            Path("/project/data/cities/florence/bands")
        """
        return self.data_path("cities", self.city.lower(), *parts)
    
    # =========================================================================
    # Convenience methods
    # =========================================================================
    
    def get_date_range(self) -> tuple:
        """Get (start_date, end_date) with defaults if not set.
        
        Default: last 30 days
        """
        end = self.end_date or date.today()
        start = self.start_date or (end - timedelta(days=30))
        return start, end
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'city': self.city,
            'project_root': str(self.project_root),
            'start_date': str(self.start_date) if self.start_date else None,
            'end_date': str(self.end_date) if self.end_date else None,
            'cloud_cover': self.cloud_cover,
            'max_size': self.max_size,
            'classifier': self.classifier,
            'n_clusters': self.n_clusters,
            'radius_km': self.radius_km,
        }

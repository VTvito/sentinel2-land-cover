"""Legacy configuration settings (yaml-based).

For new code, prefer `satellite_analysis.core.AnalysisConfig` which is
the runtime configuration used by `analyze()` and `CompletePipeline`.

This module is kept for backward compatibility with scripts that load
config.yaml directly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


@dataclass
class SentinelConfig:
    """Sentinel API configuration."""
    
    # OAuth2 credentials (new Copernicus Data Space)
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    
    # Legacy credentials (old SentinelSat - kept for backward compatibility)
    username: Optional[str] = None
    password: Optional[str] = None
    
    platformname: str = "Sentinel-2"
    max_cloud_cover: float = 10.0


@dataclass
class AreaConfig:
    """Area of interest configuration."""
    
    city: str
    country: str
    shape: str = "rectangle"  # circle, rectangle, triangle
    radius_km: float = 15.0
    user_agent: str = "SatelliteAnalysisApp"


@dataclass
class BandConfig:
    """Band processing configuration."""
    
    selected_bands: List[str] = field(default_factory=lambda: [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
    ])
    target_resolution: int = 10  # meters
    rgb_bands: List[str] = field(default_factory=lambda: ["B04", "B03", "B02"])
    fcc_bands: List[str] = field(default_factory=lambda: ["B08", "B04", "B03"])
    swir_bands: List[str] = field(default_factory=lambda: ["B12", "B8A", "B04"])


@dataclass
class ClusteringConfig:
    """Clustering algorithm configuration."""
    
    n_clusters: int = 8
    algorithm: str = "kmeans++"  # kmeans, kmeans++, sklearn
    max_iterations: int = 100
    random_state: Optional[int] = 42
    n_init: int = 10


@dataclass
class ClassificationConfig:
    """Classification algorithm configuration."""
    
    algorithm: str = "random_forest"
    n_estimators: int = 100
    random_state: Optional[int] = 42
    ignore_background: bool = True


@dataclass
class PathConfig:
    """File path configuration."""
    
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    annotation_dir: Path = field(default_factory=lambda: Path("data/annotations"))
    
    def __post_init__(self):
        """Ensure all paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.annotation_dir = Path(self.annotation_dir)


@dataclass
class Config:
    """Main configuration class."""
    
    sentinel: SentinelConfig
    area: AreaConfig
    bands: BandConfig = field(default_factory=BandConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            sentinel=SentinelConfig(**data.get("sentinel", {})),
            area=AreaConfig(**data.get("area", {})),
            bands=BandConfig(**data.get("bands", {})),
            clustering=ClusteringConfig(**data.get("clustering", {})),
            classification=ClassificationConfig(**data.get("classification", {})),
            paths=PathConfig(**data.get("paths", {})),
        )
    
    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        data = {
            "sentinel": self.sentinel.__dict__,
            "area": self.area.__dict__,
            "bands": self.bands.__dict__,
            "clustering": self.clustering.__dict__,
            "classification": self.classification.__dict__,
            "paths": {k: str(v) for k, v in self.paths.__dict__.items()},
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

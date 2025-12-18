"""
📁 Output Manager - Organized Data Storage with Metadata

Provides consistent, timestamped output organization for all analysis results.
Includes metadata tracking, run history, and easy result comparison.

Output Structure:
    data/cities/{city}/
    ├── metadata.json           # City info, coordinates, last updated
    ├── bands/                  # Raw satellite bands (B02, B03, B04, B08, etc.)
    │   ├── B02.tif
    │   └── ...
    ├── runs/                   # Analysis runs (timestamped)
    │   ├── 2025-12-18_14-30-00_consensus/
    │   │   ├── run_info.json   # Parameters, duration, stats
    │   │   ├── labels.npy      # Classification result
    │   │   ├── confidence.npy  # Confidence scores
    │   │   ├── classification.png
    │   │   └── confidence_map.png
    │   └── 2025-12-18_15-00-00_kmeans/
    │       └── ...
    ├── latest/                 # Symlink/copy to most recent run
    └── validation/             # Validation reports
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np


class OutputManager:
    """
    Manages organized output storage with metadata and versioning.
    
    Features:
    - Timestamped run directories
    - Metadata tracking (parameters, duration, statistics)
    - Easy access to latest results
    - Run history and comparison
    """
    
    def __init__(self, city_name: str, base_path: str = "data/cities"):
        """
        Initialize OutputManager for a city.
        
        Args:
            city_name: Name of the city (e.g., "Rome", "Milan")
            base_path: Base directory for all city data
        """
        self.city_name = city_name.lower()
        self.city_display = city_name.title()
        self.base_path = Path(base_path)
        self.city_path = self.base_path / self.city_name
        
        # Standard directories
        self.bands_dir = self.city_path / "bands"
        self.runs_dir = self.city_path / "runs"
        self.latest_dir = self.city_path / "latest"
        self.validation_dir = self.city_path / "validation"
        
        # Ensure directories exist
        self._init_directories()
    
    def _init_directories(self):
        """Create directory structure if needed."""
        for dir_path in [self.bands_dir, self.runs_dir, self.validation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_run(self, method: str, parameters: Dict[str, Any] = None) -> 'RunContext':
        """
        Create a new analysis run with timestamp.
        
        Args:
            method: Analysis method (e.g., "consensus", "kmeans", "spectral")
            parameters: Run parameters to save in metadata
            
        Returns:
            RunContext manager for this run
        """
        return RunContext(self, method, parameters)
    
    def get_latest_run(self) -> Optional[Path]:
        """Get path to the latest run directory."""
        if self.latest_dir.exists():
            return self.latest_dir
        
        # Fallback: find most recent run
        runs = self.list_runs()
        if runs:
            return runs[-1]['path']
        return None
    
    def list_runs(self) -> list:
        """
        List all runs for this city, sorted by date.
        
        Returns:
            List of dicts with run info: {'timestamp', 'method', 'path', 'info'}
        """
        runs = []
        
        if not self.runs_dir.exists():
            return runs
        
        for run_dir in sorted(self.runs_dir.iterdir()):
            if run_dir.is_dir():
                info_file = run_dir / "run_info.json"
                info = {}
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                
                # Parse directory name: 2025-12-18_14-30-00_consensus
                parts = run_dir.name.split('_')
                if len(parts) >= 3:
                    timestamp = f"{parts[0]}_{parts[1]}"
                    method = '_'.join(parts[2:])
                else:
                    timestamp = run_dir.name
                    method = info.get('method', 'unknown')
                
                runs.append({
                    'timestamp': timestamp,
                    'method': method,
                    'path': run_dir,
                    'info': info
                })
        
        return runs
    
    def load_city_metadata(self) -> Dict[str, Any]:
        """Load city metadata if available."""
        meta_file = self.city_path / "metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_city_metadata(self, metadata: Dict[str, Any]):
        """Save/update city metadata."""
        meta_file = self.city_path / "metadata.json"
        
        # Merge with existing
        existing = self.load_city_metadata()
        existing.update(metadata)
        existing['last_updated'] = datetime.now().isoformat()
        
        with open(meta_file, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def get_bands_path(self) -> Path:
        """Get path to bands directory."""
        return self.bands_dir
    
    def bands_available(self) -> bool:
        """Check if satellite bands are available."""
        required = ['B02.tif', 'B03.tif', 'B04.tif', 'B08.tif']
        return all((self.bands_dir / b).exists() for b in required)


class RunContext:
    """
    Context manager for a single analysis run.
    
    Handles:
    - Creating timestamped directory
    - Saving metadata and results
    - Updating 'latest' link
    - Timing the run
    """
    
    def __init__(self, manager: OutputManager, method: str, parameters: Dict[str, Any] = None):
        self.manager = manager
        self.method = method
        self.parameters = parameters or {}
        
        # Generate timestamp
        self.timestamp = datetime.now()
        self.timestamp_str = self.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create run directory
        self.run_name = f"{self.timestamp_str}_{method}"
        self.run_path = manager.runs_dir / self.run_name
        self.run_path.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.start_time = None
        self.end_time = None
        self.statistics = {}
    
    def __enter__(self):
        """Start the run."""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize the run."""
        self.end_time = datetime.now()
        
        if exc_type is None:
            # Success - save metadata and update latest
            self._save_run_info()
            self._update_latest()
        
        return False
    
    def _save_run_info(self):
        """Save run metadata to JSON."""
        duration = (self.end_time - self.start_time).total_seconds()
        
        info = {
            'city': self.manager.city_display,
            'method': self.method,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': round(duration, 2),
            'parameters': self.parameters,
            'statistics': self.statistics,
            'output_files': [f.name for f in self.run_path.iterdir() if f.is_file()]
        }
        
        info_file = self.run_path / "run_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _update_latest(self):
        """Update the 'latest' directory with results from this run."""
        latest = self.manager.latest_dir
        
        # Remove old latest
        if latest.exists():
            shutil.rmtree(latest)
        
        # Copy this run to latest
        shutil.copytree(self.run_path, latest)
    
    def save_labels(self, labels: np.ndarray, filename: str = "labels.npy"):
        """Save classification labels."""
        np.save(self.run_path / filename, labels)
    
    def save_confidence(self, confidence: np.ndarray, filename: str = "confidence.npy"):
        """Save confidence scores."""
        np.save(self.run_path / filename, confidence)
    
    def save_figure(self, fig, filename: str, dpi: int = 150):
        """Save matplotlib figure."""
        fig.savefig(self.run_path / filename, dpi=dpi, bbox_inches='tight')
    
    def save_image(self, path_or_fig, filename: str):
        """Save or copy image to run directory."""
        dest = self.run_path / filename
        if isinstance(path_or_fig, (str, Path)):
            shutil.copy(path_or_fig, dest)
        else:
            # Assume matplotlib figure
            path_or_fig.savefig(dest, dpi=150, bbox_inches='tight')
    
    def set_statistics(self, stats: Dict[str, Any]):
        """Set run statistics for metadata."""
        self.statistics = stats
    
    def add_statistic(self, key: str, value: Any):
        """Add a single statistic."""
        self.statistics[key] = value
    
    @property
    def path(self) -> Path:
        """Get run directory path."""
        return self.run_path


def get_output_manager(city: str) -> OutputManager:
    """
    Convenience function to get OutputManager for a city.
    
    Args:
        city: City name
        
    Returns:
        OutputManager instance
    """
    return OutputManager(city)


# Example usage
if __name__ == "__main__":
    # Demo usage
    manager = OutputManager("Rome")
    
    print(f"City: {manager.city_display}")
    print(f"Bands available: {manager.bands_available()}")
    print(f"Runs: {len(manager.list_runs())}")
    
    # Create a test run
    with manager.create_run("test", {"n_clusters": 6}) as run:
        # Simulate some work
        import time
        time.sleep(0.1)
        
        # Save some results
        test_labels = np.zeros((10, 10), dtype=np.uint8)
        run.save_labels(test_labels)
        run.set_statistics({
            "n_pixels": 100,
            "classes_found": 1
        })
        
        print(f"Run saved to: {run.path}")
    
    print(f"\nLatest run: {manager.get_latest_run()}")

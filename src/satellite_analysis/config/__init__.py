"""Configuration management module.

Legacy: `Config` loads yaml files (kept for backward compatibility).
Preferred: `satellite_analysis.core.AnalysisConfig` for runtime use.
"""

from satellite_analysis.config.settings import Config

__all__ = ["Config"]

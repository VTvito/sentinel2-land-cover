"""Centralized project-root path resolution.

Keeps all filesystem access rooted at the detected project root so
notebook/CLI execution cannot drift due to CWD differences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        if (current / "src" / "satellite_analysis").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return start.resolve()


class ProjectPaths:
    """Utility to resolve all project paths from a stable root."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = _find_project_root(project_root or Path.cwd())

    def resolve(self, *parts: str) -> Path:
        return self.project_root / Path(*parts)

    def config(self, filename: str = "config.yaml") -> Path:
        return self.resolve("config", filename)

    def data(self, *parts: str) -> Path:
        return self.resolve("data", *parts)

    def city(self, city: str, *parts: str) -> Path:
        return self.data("cities", city.lower(), *parts)

    def runs_base(self, city: str) -> Path:
        return self.city(city, "runs")

    def latest(self, city: str) -> Path:
        return self.city(city, "latest")

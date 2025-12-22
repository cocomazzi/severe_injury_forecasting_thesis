# src/config.py
"""
Central configuration for the master_thesis project.

Provides:
- Project root detection (with optional override via env var)
- Standardized directory paths (data, assets, results, paper, notebooks)
- Common constants (random seed, shapefile path, etc.)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

def _detect_project_root() -> Path:
    """
    Detect the project root directory.

    Priority:
    1. Environment variable MASTER_THESIS_ROOT (if set)
    2. Parent of this file's directory (…/master_thesis)
    """
    env_root = os.getenv("MASTER_THESIS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # src/config.py -> src -> project root
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT: Final[Path] = _detect_project_root()


# ---------------------------------------------------------------------------
# Core directories
# ---------------------------------------------------------------------------

SRC_DIR: Final[Path]       = PROJECT_ROOT / "src"
DATA_DIR: Final[Path]      = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path]  = DATA_DIR / "raw"        # use if needed
CLEANED_DATA_DIR: Final[Path] = DATA_DIR / "cleaned"

ASSETS_DIR: Final[Path]    = PROJECT_ROOT / "assets"
RESULTS_DIR: Final[Path]   = PROJECT_ROOT / "results"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

# Base directory for figures; specific modules can create subfolders
FIGURES_DIR: Final[Path]   = ASSETS_DIR


# ---------------------------------------------------------------------------
# Domain-specific resources
# ---------------------------------------------------------------------------

# Shapefile for US states (already present in data/)
US_STATES_SHP: Final[Path] = DATA_DIR / "cb_2018_us_state_500k.shp"


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

RANDOM_STATE: Final[int] = 42


# ---------------------------------------------------------------------------
# Ensure important directories exist
# ---------------------------------------------------------------------------

def _ensure_directories() -> None:
    """
    Create core directories if they do not exist yet.
    Safe to call multiple times.
    """
    for d in [
        DATA_DIR,
        CLEANED_DATA_DIR,
        ASSETS_DIR,
        RESULTS_DIR,
        NOTEBOOKS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


_ensure_directories()

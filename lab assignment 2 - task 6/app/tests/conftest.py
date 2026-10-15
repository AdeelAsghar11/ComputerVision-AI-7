# conftest.py
"""Pytest fixtures shared by all tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The lab folder (parent of app/)."""
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def config(data_dir) -> dict:
    with open(data_dir / "results" / "landmark_config.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def engine(data_dir, config):
    """A single LandmarkEngine instance reused by all session-scoped tests."""
    import sys
    sys.path.insert(0, str(data_dir / "app"))
    from landmark_engine import LandmarkEngine
    return LandmarkEngine(
        data_dir,
        config_path=data_dir / "results" / "landmark_config.json",
        cache_dir=data_dir / "app" / ".cache",
    )

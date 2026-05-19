"""Pytest path setup.

Pytest auto-loads this when run from the backend/ directory. Adding the
project root to sys.path lets test modules do `from backend.X import Y`,
which mirrors how production code (Cloud Run container) imports.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Bundle the canonical backend package into functions/ for a Firebase deploy.

Firebase only uploads the functions/ directory, so the backend code and the
trained models must be copied in next to functions/main.py. Run this once
before `firebase deploy`:

    python prepare_functions.py
    firebase deploy

The copy (functions/backend/) is gitignored — /backend stays the single
source of truth. Tests, the local dev venv, and __pycache__ are excluded so
the deployed bundle stays small.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path


def _force_remove(func, path, _exc):
    """rmtree error handler: clear read-only bit (Windows) and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "backend"
DEST = ROOT / "functions" / "backend"

# Directory names anywhere in the tree that should never be bundled.
EXCLUDE_DIRS = {"tests", ".venv", "__pycache__", ".pytest_cache"}
# Top-level files in backend/ that the function doesn't need.
EXCLUDE_FILES = {
    "conftest.py",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    "Dockerfile",
    ".dockerignore",
    "train_models.py",  # training is local-only; the function just loads models
}


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        full = Path(directory) / name
        if full.is_dir() and name in EXCLUDE_DIRS:
            ignored.add(name)
        elif full.is_file() and Path(directory) == SRC and name in EXCLUDE_FILES:
            ignored.add(name)
    return ignored


def main() -> int:
    if not SRC.is_dir():
        print(f"ERROR: {SRC} not found.", file=sys.stderr)
        return 1

    model_dir = SRC / "models"
    joblibs = list(model_dir.glob("*.joblib")) if model_dir.is_dir() else []
    if not joblibs:
        print("ERROR: no trained models in backend/models/. Run:", file=sys.stderr)
        print("    python -m backend.train_models", file=sys.stderr)
        return 1

    if DEST.exists():
        # onexc (3.12+) / onerror (older) — clear read-only files on Windows.
        try:
            shutil.rmtree(DEST, onexc=_force_remove)
        except TypeError:
            shutil.rmtree(DEST, onerror=lambda f, p, e: _force_remove(f, p, e))
    shutil.copytree(SRC, DEST, ignore=_ignore)

    copied_models = list((DEST / "models").glob("*.joblib"))
    print(f"Bundled backend -> {DEST}")
    print(f"  modules + {len(copied_models)} model file(s) copied.")
    print("Now run: firebase deploy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

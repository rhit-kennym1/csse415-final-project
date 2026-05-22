"""Framework-agnostic prediction core.

Shared by the local FastAPI dev server (backend/main.py) and the deployed
Firebase function (functions/main.py), so the prediction logic lives in
exactly one place. Has no FastAPI/Firebase imports.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import joblib

from backend.feature_extractor import extract_features
from backend.feature_display import build_display
from backend.features import APPROXIMATED_FEATURES, to_vector
from backend.safe_browsing import check_url

log = logging.getLogger("phishing_demo")

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_FILES = {
    "Gradient Boosting": "gb.joblib",
    "Random Forest":     "rf.joblib",
    "XGBoost":           "xgb.joblib",
    "SVM":               "svm.joblib",
}

_MODELS: dict[str, object] = {}


def load_models() -> None:
    """Load the joblib pipelines into memory once. Safe to call repeatedly."""
    for display_name, filename in MODEL_FILES.items():
        if display_name in _MODELS:
            continue
        path = MODEL_DIR / filename
        if not path.exists():
            log.warning("Missing model file %s — skipping %s.", path, display_name)
            continue
        _MODELS[display_name] = joblib.load(path)
        log.info("Loaded model: %s", display_name)


def predict_with_models(vector: list[int]) -> list[dict]:
    """Run every loaded model and return a straight phishing/legitimate verdict.

    The phishing probability is computed internally to apply the 0.5 decision
    threshold but is not surfaced.
    """
    if not _MODELS:
        load_models()
    results = []
    for name, pipe in _MODELS.items():
        # Pipelines were trained with labels remapped to {0, 1}; 1 = phishing.
        proba = pipe.predict_proba([vector])[0]
        phishing_idx = list(pipe.classes_).index(1)
        prob_phishing = float(proba[phishing_idx])
        results.append({
            "model": name,
            "verdict": "phishing" if prob_phishing >= 0.5 else "legitimate",
        })
    return results


def validate_url(url: str) -> str | None:
    """Return None if the URL is usable, else a human-readable reason."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return "could not parse url"
    if parsed.scheme not in {"http", "https"}:
        return "url must use http or https"
    if not parsed.hostname:
        return "url has no hostname"
    return None


def run_prediction(url: str) -> tuple[int, dict]:
    """Validate, extract features + Safe Browsing (concurrently), and predict.

    Returns an (http_status, body) tuple so any web framework can adapt it.
    """
    url = (url or "").strip()
    err = validate_url(url)
    if err:
        return 400, {"error": "invalid_url", "message": err}

    # Feature extraction and Safe Browsing are independent network work — run
    # them in parallel so latency is the slower of the two, not the sum.
    with ThreadPoolExecutor(max_workers=2) as pool:
        features_future = pool.submit(extract_features, url)
        safe_future = pool.submit(check_url, url, None)
        features, diagnostics = features_future.result()
        safe_browsing = safe_future.result()

    if not diagnostics.get("reachable"):
        return 200, {
            "url": url,
            "reachable": False,
            "message": "This website could not be reached - it may not exist.",
        }

    vector = to_vector(features)
    predictions = predict_with_models(vector)
    return 200, {
        "url": url,
        "reachable": True,
        "predictions": predictions,
        "safe_browsing": safe_browsing,
        "features_display": build_display(features),
        "features_meta": {"approximated": APPROXIMATED_FEATURES},
    }

"""FastAPI app entry point: /api/health and /api/predict."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Load .env from the project root (one level above this file).
# No-op in production (Cloud Run injects env vars directly).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from backend.feature_extractor import extract_features
from backend.features import FEATURE_NAMES, APPROXIMATED_FEATURES, to_vector
from backend.safe_browsing import check_url

log = logging.getLogger("phishing_demo")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_FILES = {
    "Gradient Boosting": "gb.joblib",
    "Random Forest":     "rf.joblib",
    "XGBoost":           "xgb.joblib",
    "SVM":               "svm.joblib",
}

_MODELS: dict[str, object] = {}


def _load_models() -> None:
    for display_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            log.warning("Missing model file %s — endpoint will skip this model.", path)
            continue
        _MODELS[display_name] = joblib.load(path)
        log.info("Loaded model: %s", display_name)


def _predict_with_models(vector: list[int]) -> list[dict]:
    """Run all loaded models against the feature vector."""
    results = []
    for name, pipe in _MODELS.items():
        # All four pipelines were trained with labels remapped to {0, 1}
        # where 1 = phishing. predict_proba columns follow pipe.classes_.
        proba = pipe.predict_proba([vector])[0]
        phishing_idx = list(pipe.classes_).index(1)
        prob_phishing = float(proba[phishing_idx])
        verdict = "phishing" if prob_phishing >= 0.5 else "legitimate"
        results.append({
            "model": name,
            "verdict": verdict,
            "probability": round(prob_phishing, 4),
        })
    return results


# --- FastAPI app -------------------------------------------------------------

app = FastAPI(title="Phishing Detection Demo")

allowed_origins = [
    "http://localhost:5000",
    "http://localhost:8000",
]
# Firebase Hosting domain comes from env var so it can be set at deploy time.
extra_origin = os.environ.get("FRONTEND_ORIGIN")
if extra_origin:
    allowed_origins.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    _load_models()


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


def _validate_url(url: str) -> str | None:
    """Return None if valid, else an error message."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return "could not parse url"
    if parsed.scheme not in {"http", "https"}:
        return "url must use http or https"
    if not parsed.hostname:
        return "url has no hostname"
    return None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(req: PredictRequest) -> JSONResponse:
    url = req.url.strip()
    err = _validate_url(url)
    if err:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_url", "message": err},
        )

    loop = asyncio.get_running_loop()
    extract_task = loop.run_in_executor(None, extract_features, url)
    safe_task = loop.run_in_executor(None, check_url, url, None)

    features, safe_browsing = await asyncio.gather(extract_task, safe_task)
    vector = to_vector(features)
    predictions = _predict_with_models(vector)

    return JSONResponse(content={
        "url": url,
        "predictions": predictions,
        "safe_browsing": safe_browsing,
        "features_meta": {
            "approximated": APPROXIMATED_FEATURES,
        },
    })

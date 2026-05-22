"""Firebase Cloud Function entry point for the phishing demo API.

This is a thin adapter over backend.api — the same core the local FastAPI
dev server uses. The `backend/` package is copied in next to this file by
prepare_functions.py before deploy (see DEPLOY.md), so `import backend`
resolves at runtime.

Routes (reached through the Firebase Hosting rewrite of /api/**):
    GET  /api/health   -> {"status": "ok"}
    POST /api/predict  -> prediction body (see backend.api.run_prediction)
"""
from __future__ import annotations

import json

from firebase_functions import https_fn, options

from backend import api

# Load the models once when the instance cold-starts.
api.load_models()


@https_fn.on_request(
    region="us-central1",
    memory=options.MemoryOption.GB_1,
    timeout_sec=120,
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["GET", "POST"]),
)
def api_handler(req: https_fn.Request) -> https_fn.Response:
    path = (req.path or "").rstrip("/")

    if path.endswith("/health"):
        return _json({"status": "ok"})

    if path.endswith("/predict"):
        if req.method != "POST":
            return _json({"error": "method_not_allowed"}, status=405)
        payload = req.get_json(silent=True) or {}
        status, body = api.run_prediction(payload.get("url", ""))
        return _json(body, status=status)

    return _json({"error": "not_found"}, status=404)


def _json(body: dict, status: int = 200) -> https_fn.Response:
    return https_fn.Response(
        json.dumps(body),
        status=status,
        mimetype="application/json",
    )

"""FastAPI app for local development.

Thin adapter over backend.api — the deployed Firebase function
(functions/main.py) is a parallel adapter over the same core.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Load .env from the project root (one level above this file) for local dev.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from backend import api

app = FastAPI(title="Phishing Detection Demo")

allowed_origins = ["http://localhost:5000", "http://localhost:8000"]
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
    api.load_models()


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest) -> JSONResponse:
    status, body = api.run_prediction(req.url)
    return JSONResponse(status_code=status, content=body)

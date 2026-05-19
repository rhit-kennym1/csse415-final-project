"""Train and save the four best models. Run locally before deploying:

    python -m backend.train_models

Writes joblib files to backend/models/{gb,rf,xgb,svm}.joblib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from backend.features import FEATURE_NAMES

DEFAULT_DATA_CSV = Path(__file__).resolve().parent.parent / "notebooks" / "dataset1.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"


def build_pipelines() -> dict[str, Pipeline]:
    """Per-model pipelines mirroring the project notebooks exactly."""
    return {
        "gb": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("clf", GradientBoostingClassifier(
                learning_rate=0.1, max_depth=5, n_estimators=200, random_state=42,
            )),
        ]),
        "rf": Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
            ("clf", RandomForestClassifier(
                max_depth=30, max_features=10, n_estimators=125,
                random_state=42, n_jobs=-1,
            )),
        ]),
        "xgb": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("clf", XGBClassifier(
                learning_rate=0.1, max_depth=5, n_estimators=200,
                random_state=42, eval_metric="logloss",
            )),
        ]),
        # SVM with FE used the order poly -> scaler -> SVC in SVM.ipynb
        # (best params from CV: C=10, gamma="scale"), not the non-FE pair.
        "svm": Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
            ("scaler", StandardScaler()),
            ("clf", SVC(C=10, gamma="scale", kernel="rbf", probability=True, random_state=42)),
        ]),
    }


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read dataset1.csv, strip whitespace from column names, return X, y."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # XGBoost only accepts {0, 1} for binary labels; remap -1 → 0
    y_raw = df["Result"].values
    y = np.where(y_raw == 1, 1, 0)
    X = df[FEATURE_NAMES].values
    return X, y


def main(csv_path: Path = DEFAULT_DATA_CSV, model_dir: Path = MODEL_DIR) -> None:
    X, y = load_dataset(csv_path)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, pipe in build_pipelines().items():
        print(f"Training {name}…", flush=True)
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        out_path = model_dir / f"{name}.joblib"
        joblib.dump(pipe, out_path)
        print(f"  {name}: test accuracy = {acc:.4f}, saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.csv, args.out)

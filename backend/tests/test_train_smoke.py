"""Smoke test for the training script — runs on a tiny synthetic dataset."""
import numpy as np
import pandas as pd
import pytest

from backend.train_models import build_pipelines


def test_pipelines_train_and_predict(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.choice([-1, 0, 1], size=(n, 30)))
    X.columns = [f"f{i}" for i in range(30)]
    y = rng.choice([0, 1], size=n)  # matches load_dataset's remap of {-1,1} -> {0,1}

    pipelines = build_pipelines()
    assert set(pipelines.keys()) == {"gb", "rf", "xgb", "svm"}

    for name, pipe in pipelines.items():
        pipe.fit(X.values, y)
        proba = pipe.predict_proba(X.values[:3])
        assert proba.shape == (3, 2), f"{name} returned {proba.shape}"
        assert np.allclose(proba.sum(axis=1), 1.0)

"""Helpers to load persisted models and predict"""

from __future__ import annotations

from typing import Any

import pickle
import pandas as pd


def load_model(path: str) -> Any:
    """Load a pickle model from disk"""
    with open(path, 'rb') as file:
        return pickle.load(file)


def predict_from_model(model: Any, x: pd.DataFrame) -> pd.Series:
    """Return predictions as pd.Series"""
    preds = model.predict(x)
    return pd.Series(preds, index=x.index, name="prediction")

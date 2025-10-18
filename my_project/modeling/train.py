"""Training pipeline implementation"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import yaml

from my_project.config import DEFAULT_CONFIG, AppConfig, TrainConfig
from my_project.dataset import generate_dataset, split_dataset

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_train_config_from_yaml(path: str | Path = "configs/config.yaml") -> TrainConfig:
    """Load TrainConfig from YAML file"""
    cfg_path = Path(path)
    if not cfg_path.exists():
        _LOG.info("Config file %s not found, using DEFAULT_CONFIG", cfg_path)
        return DEFAULT_CONFIG.train

    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        _LOG.warning("Failed to parse YAML %s: %s — using DEFAULT_CONFIG", cfg_path, exc)
        return DEFAULT_CONFIG.train

    try:
        app_cfg = AppConfig.model_validate(raw)
    except ValidationError as exc:
        _LOG.warning("Config validation failed for %s: %s — using DEFAULT_CONFIG", cfg_path, exc)
        return DEFAULT_CONFIG.train

    return app_cfg.train


def get_model(cfg: TrainConfig) -> Pipeline:
    """Return a sklearn Pipeline with a scaler and selected estimator"""
    model_cfg = cfg.model
    scaler = ("scaler", StandardScaler())

    if model_cfg.model_type == "logreg":
        estimator = (
            "estimator",
            LogisticRegression(
                max_iter=int(model_cfg.max_iter), random_state=int(model_cfg.random_state)
            ),
        )
    elif model_cfg.model_type == "rf":
        estimator = (
            "estimator",
            RandomForestClassifier(
                n_estimators=int(model_cfg.n_estimators),
                max_depth=model_cfg.max_depth,
                random_state=int(model_cfg.random_state),
            ),
        )
    elif model_cfg.model_type == "dt":
        estimator = (
            "estimator",
            DecisionTreeClassifier(
                max_depth=model_cfg.max_depth, random_state=int(model_cfg.random_state)
            ),
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")

    pipeline = Pipeline([scaler, estimator])
    return pipeline


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics (accuracy and F1)"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
    }


def save_model(model: Any, path: str) -> None:
    """Persist trained model to ``path`` using joblib"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def train_and_evaluate(
    cfg: TrainConfig | None = None,
    model_output: str | None = None,
    config_path: str | Path = "configs/config.yaml",
) -> Tuple[Dict[str, float], str]:
    """Full pipeline: generate dataset, split, train, evaluate and save the model"""
    if cfg is None:
        cfg = load_train_config_from_yaml(config_path)

    x, y = generate_dataset(cfg.data)
    x_train, x_test, y_train, y_test = split_dataset(x, y, cfg.data)

    model = get_model(cfg)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    metrics = evaluate_model(y_test, preds)

    if model_output is None:
        model_output = os.path.join("models", f"model_{cfg.model.model_type}.joblib")
    save_model(model, model_output)

    _LOG.info("Training finished. Metrics: %s. Model saved to %s", metrics, model_output)
    return metrics, model_output

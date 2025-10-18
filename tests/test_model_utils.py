"""Tests for model construction, evaluation and persistence utilities"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from my_project.config import TrainConfig, ModelConfig, DataConfig
from my_project.modeling.train import get_model, evaluate_model, save_model
from my_project.modeling.predict import load_model, predict_from_model
from my_project.dataset import generate_dataset


def test_get_model_returns_pipeline_and_correct_estimator():
    """Ensure get_model returns a Pipeline with the right estimator"""
    data_cfg = DataConfig()
    cfg = TrainConfig(data=data_cfg, model=ModelConfig(model_type="logreg"))
    model = get_model(cfg)
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["estimator"], LogisticRegression)

    cfg_rf = TrainConfig(data=data_cfg, model=ModelConfig(model_type="rf", n_estimators=5))
    model_rf = get_model(cfg_rf)
    assert isinstance(model_rf.named_steps["estimator"], RandomForestClassifier)

    cfg_dt = TrainConfig(data=data_cfg, model=ModelConfig(model_type="dt"))
    model_dt = get_model(cfg_dt)
    assert isinstance(model_dt.named_steps["estimator"], DecisionTreeClassifier)


def test_get_model_raises_on_unknown_model_type():
    """Unknown model_type must raise ValueError"""
    cfg = TrainConfig(data=DataConfig(), model=ModelConfig(model_type="logreg"))

    cfg.model.model_type = "unknown_type"
    try:
        _ = get_model(cfg)
    except ValueError as exc:
        assert "Unknown model type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown model_type")


def test_evaluate_model_metrics_exact():
    """Validate evaluate_model returns correct accuracy and f1 for a toy example"""
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = evaluate_model(y_true, y_pred)

    assert abs(metrics["accuracy"] - 0.8) < 1e-8
    assert abs(metrics["f1"] - 0.8) < 1e-8


def test_save_and_load_model_and_predict(tmp_path):
    """Check save/load/predict lifecycle on a small trained model"""
    cfg = TrainConfig(
        data=DataConfig(n_samples=50, n_features=4, n_informative=2, n_redundant=0),
        model=ModelConfig(model_type="logreg", max_iter=50),
    )
    model = get_model(cfg)

    x, y = generate_dataset(cfg.data)
    model.fit(x, y)
    out = tmp_path / "subdir" / "model.joblib"
    save_model(model, str(out))
    assert out.exists()

    loaded = load_model(str(out))
    preds = predict_from_model(loaded, x.iloc[:5])
    assert isinstance(preds, pd.Series)
    assert len(preds) == 5

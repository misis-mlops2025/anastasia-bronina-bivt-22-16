"""Integration tests for the end-to-end training pipeline"""

from pathlib import Path
import inspect

import pytest

from my_project.config import DataConfig, ModelConfig, TrainConfig
from my_project.modeling.train import train_and_evaluate


@pytest.mark.parametrize("model_type", ["logreg", "rf", "dt"])
def test_train_pipeline_saves_model_for_all_model_types(model_type, tmp_path):
    """Train all supported model types and ensure metrics are returned and model saved"""
    data_cfg = DataConfig(n_samples=150, n_features=6, n_informative=3, n_redundant=0)
    model_cfg = ModelConfig(model_type=model_type, n_estimators=10, max_iter=50)
    cfg = TrainConfig(data=data_cfg, model=model_cfg)

    out = tmp_path / f"model_{model_type}.pickle"
    metrics, model_path = train_and_evaluate(cfg=cfg, model_output=str(out))
    assert "accuracy" in metrics and "f1" in metrics, "Metrics should include accuracy and f1"
    assert Path(model_path).exists(), "Model file should be created"


def _train_with_temp_yaml(yaml_text: str, tmp_path: Path, monkeypatch) -> tuple:
    """Helper that writes yaml_text to a temporary configs/config.yaml and calls train"""
    project_dir = tmp_path / "project_with_yaml"
    configs_dir = project_dir / "configs"
    configs_dir.mkdir(parents=True)
    yaml_path = configs_dir / "config.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    sig = inspect.signature(train_and_evaluate)
    if "config_path" in sig.parameters:
        out_model = tmp_path / "out_model_explicit.pickle"
        metrics, model_path = train_and_evaluate(cfg=None,
                                                 model_output=str(out_model),
                                                 config_path=str(yaml_path))
    else:
        monkeypatch.chdir(project_dir)
        out_model = project_dir / "out_model_chdir.pickle"
        metrics, model_path = train_and_evaluate(cfg=None, model_output=str(out_model))
    return metrics, Path(model_path)


def test_train_pipeline_uses_explicit_yaml_config(tmp_path, monkeypatch):
    """When a temporary YAML is provided, training should use it and not touch repo config"""
    yaml_content = """
train:
  data:
    n_samples: 120
    n_features: 6
    n_informative: 2
    n_redundant: 0
    random_state: 42
    test_size: 0.2
  model:
    model_type: "logreg"
    max_iter: 40
    random_state: 42
  scoring: "f1"
"""
    metrics, model_path = _train_with_temp_yaml(yaml_content, tmp_path, monkeypatch)
    assert "accuracy" in metrics and "f1" in metrics
    assert model_path.exists(), "Model produced by YAML-driven run should exist"


def test_train_fallbacks_to_default_when_config_missing(tmp_path, monkeypatch):
    """If no configs/config.yaml exists, train_and_evaluate should fallback to defaults and run"""
    project_dir = tmp_path / "no_config_project"
    project_dir.mkdir()

    sig = inspect.signature(train_and_evaluate)
    out = project_dir / "model_fallback.pickle"
    if "config_path" in sig.parameters:
        metrics, model_path = train_and_evaluate(cfg=None,
                                                 model_output=str(out),
                                                 config_path=str(project_dir /
                                                                  "configs" /
                                                                    "config.yaml"))
    else:
        monkeypatch.chdir(project_dir)
        metrics, model_path = train_and_evaluate(cfg=None, model_output=str(out))

    assert Path(model_path).exists(), "Model should be created even when config is missing"
    assert "accuracy" in metrics and "f1" in metrics

"""Evaluate trained model on test set and save metrics JSON"""
from __future__ import annotations

from pathlib import Path
import json
import yaml
from my_project.config import AppConfig, DEFAULT_CONFIG
from my_project.modeling.predict import load_model
from my_project.dataset import generate_dataset, split_dataset
from my_project.modeling.train import evaluate_model

def load_cfg(path: str | Path = "configs/config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        return DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        return AppConfig.model_validate(raw)
    except Exception:
        return AppConfig.parse_obj(raw)

def main(config_path: str = "configs/config.yaml") -> None:
    app_cfg = load_cfg(config_path)
    data_cfg = app_cfg.train.data
    x_df, y_ser = generate_dataset(data_cfg)
    x_train, x_test, y_train, y_test = split_dataset(x_df, y_ser, data_cfg)

    model = load_model(app_cfg.train.model_output)
    preds = model.predict(x_test)
    metrics = evaluate_model(y_test, preds)

    out_dir = Path(app_cfg.train.metrics_output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(app_cfg.train.metrics_output, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh)

if __name__ == "__main__":
    main()

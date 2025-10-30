"""Train model from processed data"""
from __future__ import annotations

from pathlib import Path
import yaml
from my_project.config import AppConfig, DEFAULT_CONFIG
from my_project.modeling.train import train_and_evaluate

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
    model_output = app_cfg.train.model_output
    train_and_evaluate(cfg=app_cfg.train, model_output=model_output, config_path=config_path)

if __name__ == "__main__":
    main()

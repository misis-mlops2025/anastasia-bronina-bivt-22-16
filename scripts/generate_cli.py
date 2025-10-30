"""Generate and save synthetic dataset"""
from __future__ import annotations

from pathlib import Path
import yaml
from my_project.config import AppConfig, DEFAULT_CONFIG
from my_project.dataset import generate_dataset

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

    out_dir = Path(app_cfg.train.processed_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_df.to_csv(out_dir / "X.csv", index=False)
    y_ser.to_csv(out_dir / "y.csv", index=False)

if __name__ == "__main__":
    main()

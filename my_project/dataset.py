"""Dataset generation and split utilities"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from my_project.config import DataConfig


def generate_dataset(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic classification dataset"""
    x_np, y_np = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=cfg.n_repeated,
        n_classes=cfg.n_classes,
        class_sep=float(cfg.class_sep),
        random_state=cfg.random_state,
    )

    columns = [f"f_{i}" for i in range(x_np.shape[1])]
    x = pd.DataFrame(x_np, columns=columns)
    y = pd.Series(y_np, name="target")
    return x, y


def split_dataset(
    x: pd.DataFrame, y: pd.Series, cfg: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train and test sets"""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=float(cfg.test_size), random_state=cfg.random_state, stratify=y
    )
    return x_train, x_test, y_train, y_test

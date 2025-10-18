"""Tests for dataset generation and splitting"""

import pandas as pd

from my_project.config import DataConfig
from my_project.dataset import generate_dataset, split_dataset

def test_generate_dataset_shapes():
    """Generate dataset returns DataFrame and Series with expected shapes"""
    cfg = DataConfig(
        n_samples=200,
        n_features=10,
        n_informative=3,
        n_redundant=1,
    )
    x, y = generate_dataset(cfg)
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert x.shape == (200, 10)
    assert y.shape == (200,)


def test_split_dataset_splits():
    """Split returns train/test with sizes according to test_size"""
    cfg = DataConfig(
        n_samples=100, n_features=5, n_informative=2, n_redundant=1, test_size=0.25
    )
    x, y = generate_dataset(cfg)
    x_train, x_test, y_train, y_test = split_dataset(x, y, cfg)
    assert len(x_test) == 25
    assert len(x_train) == 75
    assert len(y_test) == 25
    assert len(y_train) == 75

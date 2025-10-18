"""Pydantic configuration"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt, confloat


class DataConfig(BaseModel):
    """Configuration for synthetic dataset generation and splitting"""

    n_samples: PositiveInt = Field(1000, description="Number of synthetic samples")
    n_features: PositiveInt = Field(20, description="Total number of features")
    n_informative: PositiveInt = Field(5, description="Number of informative features")
    n_redundant: int = Field(2, description="Number of redundant features")
    n_repeated: int = Field(0, description="Number of repeated features")
    n_classes: PositiveInt = Field(2, description="Number of target classes")
    class_sep: confloat(gt=0.0) = Field(1.0, description="Separation between classes")
    random_state: int = Field(42, description="Random seed for reproducibility")
    test_size: confloat(gt=0.0, lt=1.0) = Field(0.2, description="Test set fraction")


class ModelConfig(BaseModel):
    """Model selection and hyperparameters"""

    model_type: Literal["logreg", "rf", "dt"] = Field("logreg", description="Model type")
    max_iter: PositiveInt = Field(100, description="Max iterations for logistic regression")
    random_state: int = Field(42, description="Random seed for model")
    n_estimators: PositiveInt = Field(100, description="Number of trees for RF")
    max_depth: int | None = Field(None, description="Max depth for tree-based models")


class TrainConfig(BaseModel):
    """Training-related configuration containing data and model subconfigs"""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    scoring: str = Field("f1", description="Scoring metric for evaluation")


class AppConfig(BaseModel):
    """Application config"""

    train: TrainConfig = TrainConfig()


DEFAULT_CONFIG = AppConfig()

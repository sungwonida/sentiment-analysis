"""Sentiment-analysis utilities for reproducible IMDb experiments."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from .cli import app
    from .config import (
        BertExperimentConfig,
        BertModelConfig,
        BowConfig,
        LoraConfig,
        TokenizerConfig,
        TrainerConfig,
    )
    from .eda import (
        class_balanced_length,
        extreme_outliers,
        get_imdb_dataset_as_df,
        length_distribution,
        sentiment_review_length,
        top_uni_bigrams,
        vocab_size_oov_rate,
        word_cloud,
    )
    from .training import run_bert_experiment, run_bow_experiment

__all__ = [
    "app",
    "BertExperimentConfig",
    "BertModelConfig",
    "BowConfig",
    "LoraConfig",
    "TokenizerConfig",
    "TrainerConfig",
    "class_balanced_length",
    "extreme_outliers",
    "get_imdb_dataset_as_df",
    "length_distribution",
    "sentiment_review_length",
    "top_uni_bigrams",
    "vocab_size_oov_rate",
    "word_cloud",
    "run_bow_experiment",
    "run_bert_experiment",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    if name == "app":
        from .cli import app

        return app
    if name in {
        "BertExperimentConfig",
        "BertModelConfig",
        "BowConfig",
        "LoraConfig",
        "TokenizerConfig",
        "TrainerConfig",
    }:
        from . import config

        return getattr(config, name)
    if name in {
        "class_balanced_length",
        "extreme_outliers",
        "get_imdb_dataset_as_df",
        "length_distribution",
        "sentiment_review_length",
        "top_uni_bigrams",
        "vocab_size_oov_rate",
        "word_cloud",
    }:
        from . import eda

        return getattr(eda, name)
    if name in {"run_bow_experiment", "run_bert_experiment"}:
        from . import training

        return getattr(training, name)
    raise AttributeError(name)

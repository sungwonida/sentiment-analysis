"""Backwards-compatible shim that re-exports the refactored EDA helpers."""
from __future__ import annotations

from sentiment_analysis.eda import *  # noqa: F401,F403

__all__ = [
    "get_imdb_dataset_as_df",
    "length_distribution",
    "class_balanced_length",
    "extreme_outliers",
    "top_uni_bigrams",
    "word_cloud",
    "vocab_size_oov_rate",
    "sentiment_review_length",
]

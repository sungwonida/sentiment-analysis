"""Data-loading utilities for the IMDb sentiment analysis experiments."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import pandas as pd
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from .config import TokenizerConfig


@lru_cache(maxsize=4)
def load_imdb(cache_dir: str | None = None) -> DatasetDict:
    """Return the standard IMDb sentiment dataset from the *datasets* hub."""

    return load_dataset("imdb", cache_dir=cache_dir)


def imdb_split_as_dataframe(split: str, cache_dir: str | None = None) -> pd.DataFrame:
    """Return *split* of IMDb as a :class:`pandas.DataFrame`."""

    ds = load_dataset("imdb", split=split, cache_dir=cache_dir)
    return ds.to_pandas()


def build_tokenizer(cfg: TokenizerConfig):
    """Create a Hugging Face tokenizer based on *cfg*."""

    return AutoTokenizer.from_pretrained(cfg.pretrained_model_name)


def tokenise_imdb(cfg: TokenizerConfig, cache_dir: str | None = None):
    """Tokenise IMDb according to *cfg* and return the dataset and tokenizer."""

    imdb = load_imdb(cache_dir=cache_dir)
    tokenizer = build_tokenizer(cfg)

    def _tokenise(batch):
        return tokenizer(
            batch["text"],
            padding=cfg.padding,
            truncation=cfg.truncation,
            max_length=cfg.max_length,
        )

    tokenised = imdb.map(
        _tokenise,
        batched=True,
        remove_columns=["text"],
    )
    tokenised.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenised, tokenizer


__all__ = [
    "load_imdb",
    "imdb_split_as_dataframe",
    "build_tokenizer",
    "tokenise_imdb",
]

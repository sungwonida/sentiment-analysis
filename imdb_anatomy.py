"""Utilities for quick exploratory data analysis (EDA) of the IMDb movie‑review
corpus.

Each helper prints a short banner when invoked and now **saves any plots or
images to a dedicated ``results/`` directory** so you can run the script once
and inspect all outputs in one place.  The helpers remain intentionally simple
so that newcomers can read, run, and adapt them without prior familiarity with
the dataset API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import scipy.stats as st
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

RESULTS_DIR: Path = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_imdb_dataset(phase: str) -> pd.DataFrame:
    """Return the requested IMDb split as a *pandas* :class:`DataFrame`.

    Parameters
    ----------
    phase : {"train", "test"}
        Which part of the dataset to load.  ``"train"`` contains 25 000 reviews
        with balanced sentiment; ``"test"`` contains a disjoint 25 000‑review
        test set.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with at least the columns ``text`` (review body) and
        ``label`` (``0`` = negative, ``1`` = positive).

    Raises
    ------
    ValueError
        If *phase* is not one of the accepted strings.
    """
    print("---------------------------------------------------------------")
    print("get_imdb_dataset  ▶  Loading IMDb reviews into a DataFrame …")
    print("---------------------------------------------------------------")

    valid_phases = {"train", "test"}
    if phase not in valid_phases:
        raise ValueError(f"phase must be one of {valid_phases}, got {phase!r}")

    imdb = load_dataset("imdb")
    return imdb[phase].to_pandas()


# ---------------------------------------------------------------------------
# Text‑length analysis
# ---------------------------------------------------------------------------

def _save_current_fig(filename: str) -> None:
    """Helper that saves the *current* Matplotlib figure in *RESULTS_DIR*."""
    out_path: Path = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot → {out_path}")


def length_distribution(df: pd.DataFrame) -> None:
    """Plot the global token‑length distribution of all reviews and save it.

    The function adds a ``length`` column to *df* (number of whitespace‑
    separated tokens), stores a histogram in ``results/length_distribution.png``
    and prints descriptive statistics.
    """
    print("\n---------------------------------------------------------------")
    print("length_distribution  ▶  Generating overall review‑length histogram …")
    print("---------------------------------------------------------------")

    df["length"] = df["text"].str.split().apply(len)  # token count

    plt.figure()
    plt.hist(df["length"], bins=40, color="steelblue", edgecolor="black")
    plt.title("Token length per review")
    plt.xlabel("Number of tokens")
    plt.ylabel("Count")
    _save_current_fig("length_distribution.png")

    print(df["length"].describe())


def class_balanced_length(df: pd.DataFrame) -> None:
    """Compare length distributions between *positive* and *negative* reviews.

    Saves a dual‑histogram figure to
    ``results/class_balanced_length.png``.
    """
    print("\n---------------------------------------------------------------")
    print("class_balanced_length  ▶  Comparing lengths by sentiment …")
    print("---------------------------------------------------------------")

    plt.figure()
    for lbl, name, color in [
        (0, "negative", "crimson"),
        (1, "positive", "forestgreen"),
    ]:
        subset = df[df["label"] == lbl]["length"]
        plt.hist(subset, bins=40, alpha=0.5, label=name, color=color,
                 edgecolor="black")
    plt.legend()
    plt.title("Length distribution by sentiment")
    plt.xlabel("Number of tokens")
    plt.ylabel("Count")
    _save_current_fig("class_balanced_length.png")


def extreme_outliers(df: pd.DataFrame) -> None:
    """Print the three longest and three shortest reviews (after *length_distribution*)."""
    print("\n---------------------------------------------------------------")
    print("extreme_outliers  ▶  Displaying extreme review lengths …")
    print("---------------------------------------------------------------")

    longest = df.nlargest(3, "length")[["label", "length", "text"]]
    shortest = df.nsmallest(3, "length")[["label", "length", "text"]]

    print("Top 3 longest reviews:\n")
    print(longest.to_string(index=False, max_colwidth=120))

    print("\nTop 3 shortest reviews:\n")
    print(shortest.to_string(index=False, max_colwidth=120))


# ---------------------------------------------------------------------------
# Vocabulary exploration
# ---------------------------------------------------------------------------

def top_uni_bigrams(df: pd.DataFrame, top_k: int = 20) -> None:
    """Print the *top_k* most frequent unigrams (per class) and bigrams (overall)."""
    print("\n---------------------------------------------------------------")
    print("top_uni_bigrams  ▶  Listing frequent n‑grams …")
    print("---------------------------------------------------------------")

    def _top_terms(d: pd.DataFrame, ngram: Tuple[int, int]) -> pd.Series:
        vect = CountVectorizer(
            stop_words="english", ngram_range=ngram, max_features=50_000
        )
        term_matrix = vect.fit_transform(d["text"])
        freqs = term_matrix.sum(axis=0).A1
        terms = pd.Series(freqs, index=vect.get_feature_names_out())
        return terms.sort_values(ascending=False).head(top_k)

    for lbl, name in {0: "negative", 1: "positive"}.items():
        print(f"\n=== Top unigrams in {name} reviews ===")
        print(_top_terms(df[df["label"] == lbl], (1, 1)))

    print("\n=== Top bigrams overall ===")
    print(_top_terms(df, (2, 2)))


def word_cloud(df: pd.DataFrame) -> None:
    """Generate word‑cloud images and write them under ``results/``.

    * ``results/wc_pos.png``  – positive‑review cloud
    * ``results/wc_neg.png``  – negative‑review cloud
    """
    print("\n---------------------------------------------------------------")
    print("word_cloud  ▶  Creating word‑cloud images …")
    print("---------------------------------------------------------------")

    pos_text = " ".join(df[df["label"] == 1]["text"].tolist())
    neg_text = " ".join(df[df["label"] == 0]["text"].tolist())

    wc_pos_path = RESULTS_DIR / "wc_pos.png"
    WordCloud(width=800, height=400, stopwords="english").generate(pos_text)\
        .to_image().save(wc_pos_path)
    print(f"Saved → {wc_pos_path}")

    wc_neg_path = RESULTS_DIR / "wc_neg.png"
    WordCloud(width=800, height=400, stopwords="english", colormap="cool")\
        .generate(neg_text).to_image().save(wc_neg_path)
    print(f"Saved → {wc_neg_path}")


# ---------------------------------------------------------------------------
# Tokeniser diagnostics
# ---------------------------------------------------------------------------

def vocab_size_oov_rate(df: pd.DataFrame) -> None:
    """Check review lengths and truncation when using *bert‑base‑uncased*."""
    print("\n---------------------------------------------------------------")
    print("vocab_size_oov_rate  ▶  Measuring tokeniser overflow …")
    print("---------------------------------------------------------------")

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_len = tok.model_max_length  # 512 for BERT‑base

    df["tok_len"] = df["text"].apply(lambda t: len(tok.encode(t, add_special_tokens=False)))
    n_long = (df["tok_len"] > max_len).sum()
    print(f"{n_long} / {len(df)} reviews exceed {max_len} tokens")

    df["n_truncated"] = (df["tok_len"] - max_len).clip(lower=0)
    avg_trunc = df.loc[df.n_truncated > 0, "n_truncated"].mean()
    print(f"Average tokens truncated per long review: {avg_trunc:.2f}")

"""Light-weight exploratory data-analysis helpers for the IMDb dataset."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
import scipy.stats as st
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from wordcloud import WordCloud

from .utils import RESULTS_DIR, ensure_dir

ensure_dir(RESULTS_DIR)


def get_imdb_dataset_as_df(phase: str) -> pd.DataFrame:
    """Return the requested IMDb split as a :class:`pandas.DataFrame`."""

    valid = {"train", "test", "unsupervised"}
    if phase not in valid:
        raise ValueError(f"phase must be one of {valid}, got {phase!r}")
    imdb = load_dataset("imdb")
    return imdb[phase].to_pandas()


def _save_current_fig(filename: str) -> None:
    out_path = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot → {out_path}")


def length_distribution(df: pd.DataFrame) -> None:
    df = df.copy()
    df["length"] = df["text"].str.split().apply(len)

    plt.figure()
    plt.hist(df["length"], bins=40, color="steelblue", edgecolor="black")
    plt.title("Token length per review")
    plt.xlabel("Number of tokens")
    plt.ylabel("Count")
    _save_current_fig("length_distribution.png")

    print(df["length"].describe())


def class_balanced_length(df: pd.DataFrame) -> None:
    df = df.copy()
    if "length" not in df:
        df["length"] = df["text"].str.split().apply(len)

    plt.figure()
    for lbl, name, color in [
        (0, "negative", "crimson"),
        (1, "positive", "forestgreen"),
    ]:
        subset = df[df["label"] == lbl]["length"]
        plt.hist(
            subset,
            bins=40,
            alpha=0.5,
            label=name,
            color=color,
            edgecolor="black",
        )
    plt.legend()
    plt.title("Length distribution by sentiment")
    plt.xlabel("Number of tokens")
    plt.ylabel("Count")
    _save_current_fig("class_balanced_length.png")


def extreme_outliers(df: pd.DataFrame, top_k: int = 3) -> None:
    df = df.copy()
    if "length" not in df:
        df["length"] = df["text"].str.split().apply(len)

    longest = df.nlargest(top_k, "length")[["label", "length", "text"]]
    shortest = df.nsmallest(top_k, "length")[["label", "length", "text"]]

    print("Top longest reviews:\n")
    print(longest.to_string(index=False, max_colwidth=120))
    print("\nTop shortest reviews:\n")
    print(shortest.to_string(index=False, max_colwidth=120))


def top_uni_bigrams(df: pd.DataFrame, top_k: int = 20) -> None:
    def _top_terms(d: pd.DataFrame, ngram: Tuple[int, int]) -> pd.Series:
        vect = CountVectorizer(stop_words="english", ngram_range=ngram, max_features=50_000)
        term_matrix = vect.fit_transform(d["text"])
        sums = term_matrix.sum(axis=0)
        vocab = vect.get_feature_names_out()
        freqs = {term: sums[0, idx] for idx, term in enumerate(vocab)}
        return pd.Series(freqs).sort_values(ascending=False).head(top_k)

    print("Top unigrams (negative)")
    print(_top_terms(df[df["label"] == 0], (1, 1)))
    print("\nTop unigrams (positive)")
    print(_top_terms(df[df["label"] == 1], (1, 1)))
    print("\nTop bigrams (overall)")
    print(_top_terms(df, (2, 2)))


def word_cloud(df: pd.DataFrame, max_words: int = 200) -> None:
    pos_text = " ".join(df[df["label"] == 1]["text"].tolist())
    neg_text = " ".join(df[df["label"] == 0]["text"].tolist())

    wc_pos_path = RESULTS_DIR / "wc_pos.png"
    WordCloud(width=800, height=400, max_words=max_words, stopwords="english") \
        .generate(pos_text).to_image().save(wc_pos_path)
    print(f"Saved → {wc_pos_path}")

    wc_neg_path = RESULTS_DIR / "wc_neg.png"
    WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        stopwords="english",
        colormap="cool",
    ).generate(neg_text).to_image().save(wc_neg_path)
    print(f"Saved → {wc_neg_path}")


def vocab_size_oov_rate(df: pd.DataFrame, model_name: str = "bert-base-uncased") -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = tokenizer.model_max_length

    df = df.copy()
    df["tok_len"] = df["text"].apply(
        lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    )
    n_long = (df["tok_len"] > max_len).sum()
    print(f"{n_long} / {len(df)} reviews exceed {max_len} tokens")

    df["n_truncated"] = (df["tok_len"] - max_len).clip(lower=0)
    avg_trunc = df.loc[df["n_truncated"] > 0, "n_truncated"].mean()
    print(f"Average tokens truncated per long review: {avg_trunc:.2f}")

    df["n_out_of_vocab"] = df["text"].apply(
        lambda text: sum(
            1 for tok_id in tokenizer(text)["input_ids"] if tok_id == tokenizer.unk_token_id
        )
    )
    print("Average unknown tokens per review:", df["n_out_of_vocab"].mean())


def sentiment_review_length(df: pd.DataFrame) -> None:
    df = df.copy()
    if "length" not in df:
        df["length"] = df["text"].str.split().apply(len)
    stat, p = st.ttest_ind(
        df[df["label"] == 0]["length"],
        df[df["label"] == 1]["length"],
    )
    print(f"T-test statistic: {stat:.4f}; p-value: {p:.4f}")


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

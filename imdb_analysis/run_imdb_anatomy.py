"""Execute the exploratory-analysis pipeline from the refactored package."""
from __future__ import annotations

from sentiment_analysis.eda import (
    class_balanced_length,
    extreme_outliers,
    get_imdb_dataset_as_df,
    length_distribution,
    top_uni_bigrams,
    vocab_size_oov_rate,
    word_cloud,
)


def main() -> None:
    df = get_imdb_dataset_as_df("train")
    length_distribution(df)
    class_balanced_length(df)
    extreme_outliers(df)
    top_uni_bigrams(df)
    word_cloud(df)
    vocab_size_oov_rate(df)


if __name__ == "__main__":
    main()

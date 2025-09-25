"""Traditional bag-of-words style models."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from ..config import BowConfig


def build_tfidf_logreg(cfg: BowConfig) -> Pipeline:
    """Construct the TF-IDF → Logistic Regression baseline pipeline."""

    vectoriser = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        stop_words=cfg.stop_words,
    )
    classifier = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
    return make_pipeline(vectoriser, classifier)


__all__ = ["build_tfidf_logreg"]

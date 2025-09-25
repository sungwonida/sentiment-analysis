"""Convenience wrapper around :func:`sentiment_analysis.training.run_bow_experiment`."""
from __future__ import annotations

from sentiment_analysis.config import BowConfig
from sentiment_analysis.training import run_bow_experiment


def main() -> None:
    cfg = BowConfig()
    run_bow_experiment(cfg)


if __name__ == "__main__":
    main()

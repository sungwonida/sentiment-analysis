# Sentiment analysis experiments

This repository contains a light-weight but fully reproducible set of
experiments around the [IMDb movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
It reorganises the original exploratory notebooks and scripts into a
small Python package with a consistent command-line interface.

## Installation

```bash
pip install -e .
```

The editable install exposes a `sentiment-cli` command that orchestrates
all experiments and utilities described below.

## Commands

### Exploratory data analysis

Run the exploratory analysis helpers (histograms, top n-grams, word
clouds, tokenizer diagnostics, …) with

```bash
sentiment-cli eda run --split train
```

Figures and intermediate outputs are stored in `results/`.

### Bag-of-words baseline

Train the TF-IDF + logistic regression baseline and save the fitted
pipeline/metrics to `artifacts/bow_tfidf`:

```bash
sentiment-cli train-bow
```

Use `--help` to discover additional knobs such as `--max-features` or
`--evaluation-split`.

### BERT-based models

Fine-tune (or adapter-tune) a BERT encoder via

```bash
sentiment-cli train-bert --pooling cls --freeze-base True
```

Key options include:

* `--pooling {cls,mean}` – choose between CLS-token or mean pooling.
* `--freeze-base/--no-freeze-base` – freeze the base encoder (useful for
  head-only experiments).
* `--use-lora` – enable LoRA adapters to match the parameter-efficient
  experiments from the original project.
* `--classifier-hidden` – repeat to stack additional hidden layers in
the classification head (e.g. `--classifier-hidden 256 --classifier-hidden 64`).

Metrics, checkpoints, and configuration snapshots are written under the
chosen `--output-dir` (defaults to `artifacts/bert`).

## Reproducing the original experiments

* `sentiment-cli eda run` reproduces the plots and diagnostics that were
  previously scattered across notebooks.
* `sentiment-cli train-bow` replicates the TF-IDF logistic regression
  baseline (≈89–90% accuracy on IMDb).
* `sentiment-cli train-bert --freeze-base True --num-train-epochs 39`
  matches the frozen-encoder experiment.
* `sentiment-cli train-bert --pooling mean --classifier-hidden 256 \
  --use-lora --learning-rate 3e-4` reproduces the LoRA mean-pooling
  setup.

Each command stores its configuration alongside the resulting metrics so
runs can be re-issued with different seeds or hyper-parameters.

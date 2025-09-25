"""Command-line interface for reproducible sentiment-analysis experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .config import (
    BertExperimentConfig,
    BertModelConfig,
    BowConfig,
    LoraConfig,
    TokenizerConfig,
    TrainerConfig,
)
app = typer.Typer(help="Utilities to reproduce the IMDb sentiment-analysis experiments.")
eda_app = typer.Typer(help="Exploratory data-analysis helpers")
app.add_typer(eda_app, name="eda")


@eda_app.command("run")
def eda_run(
    split: str = typer.Option("train", help="IMDb split to analyse."),
    top_k: int = typer.Option(20, help="How many n-grams to list."),
    skip_wordcloud: bool = typer.Option(False, help="Do not generate word-cloud images."),
) -> None:
    """Run the full exploratory data-analysis pipeline."""

    from . import eda as imdb_eda

    df = imdb_eda.get_imdb_dataset_as_df(split)
    imdb_eda.length_distribution(df)
    imdb_eda.class_balanced_length(df)
    imdb_eda.extreme_outliers(df)
    imdb_eda.top_uni_bigrams(df, top_k=top_k)
    if not skip_wordcloud:
        imdb_eda.word_cloud(df)
    imdb_eda.vocab_size_oov_rate(df)
    imdb_eda.sentiment_review_length(df)


@app.command("train-bow")
def cli_train_bow(
    output_dir: Path = typer.Option(Path("artifacts/bow_tfidf"), help="Where to store artefacts."),
    max_features: int = typer.Option(40_000, help="Maximum number of TF-IDF features."),
    ngram_min: int = typer.Option(1, help="Lower bound of the n-gram range."),
    ngram_max: int = typer.Option(2, help="Upper bound of the n-gram range."),
    stop_words: Optional[str] = typer.Option("english", help="Stop-word list passed to scikit-learn."),
    C: float = typer.Option(4.0, help="Inverse regularisation strength."),
    max_iter: int = typer.Option(1_000, help="Maximum number of training iterations."),
    n_jobs: int = typer.Option(-1, help="Number of threads for LogisticRegression."),
    random_state: int = typer.Option(42, help="Random seed."),
    evaluation_split: str = typer.Option("test", help="Dataset split used for evaluation."),
    save_model: bool = typer.Option(True, help="Persist the fitted pipeline."),
) -> None:
    from .training import run_bow_experiment

    cfg = BowConfig(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        stop_words=stop_words,
        C=C,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    metrics = run_bow_experiment(
        cfg,
        output_dir=output_dir,
        evaluation_split=evaluation_split,
        save_model=save_model,
    )
    typer.echo(json.dumps(metrics, indent=2))


@app.command("train-bert")
def cli_train_bert(
    output_dir: Path = typer.Option(Path("artifacts/bert"), help="Where to store artefacts."),
    base_model: str = typer.Option("bert-base-uncased", help="Name or path of the base encoder."),
    pooling: str = typer.Option("cls", help="Pooling strategy: 'cls' or 'mean'."),
    classifier_hidden: Optional[List[int]] = typer.Option(None, help="Hidden layer sizes for the classifier."),
    dropout: float = typer.Option(0.2, help="Dropout applied after hidden layers."),
    freeze_base: bool = typer.Option(True, help="Freeze the encoder parameters."),
    use_lora: bool = typer.Option(False, help="Attach a LoRA adapter to the encoder."),
    lora_r: int = typer.Option(8, help="LoRA rank."),
    lora_alpha: int = typer.Option(16, help="LoRA alpha."),
    lora_dropout: float = typer.Option(0.05, help="LoRA dropout."),
    lora_modules_to_save: Optional[List[str]] = typer.Option(
        None, help="Optional modules to keep outside the adapter when using LoRA."
    ),
    num_train_epochs: float = typer.Option(3.0, help="Number of training epochs."),
    per_device_train_batch_size: int = typer.Option(32, help="Per-device train batch size."),
    per_device_eval_batch_size: int = typer.Option(64, help="Per-device eval batch size."),
    gradient_accumulation_steps: int = typer.Option(1, help="Gradient accumulation steps."),
    learning_rate: float = typer.Option(5e-4, help="Learning rate."),
    weight_decay: float = typer.Option(0.0, help="Weight decay."),
    warmup_ratio: Optional[float] = typer.Option(None, help="Warm-up ratio."),
    logging_steps: int = typer.Option(50, help="Log every N steps."),
    logging_strategy: str = typer.Option("steps", help="Logging strategy."),
    logging_dir: Optional[Path] = typer.Option(None, help="Directory for TensorBoard logs."),
    evaluation_strategy: str = typer.Option("epoch", help="Evaluation strategy."),
    save_strategy: str = typer.Option("epoch", help="Checkpoint save strategy."),
    load_best_model_at_end: bool = typer.Option(True, help="Reload the best checkpoint at the end."),
    metric_for_best_model: str = typer.Option("accuracy", help="Metric monitored for best-model tracking."),
    greater_is_better: bool = typer.Option(True, help="Whether higher metric values are better."),
    report_to: Optional[List[str]] = typer.Option(None, help="Integrations to report to (e.g. tensorboard)."),
    seed: int = typer.Option(42, help="Random seed."),
    dataloader_pin_memory: bool = typer.Option(False, help="Pin memory during dataloading."),
    fp16: bool = typer.Option(False, help="Enable fp16 mixed-precision training."),
    bf16: bool = typer.Option(False, help="Enable bf16 mixed-precision training."),
    resume_from_checkpoint: Optional[Path] = typer.Option(None, help="Resume training from this checkpoint."),
    run_name: Optional[str] = typer.Option(None, help="Optional run name (e.g. for TensorBoard)."),
    evaluation_split: str = typer.Option("test", help="Split to evaluate against."),
    max_length: int = typer.Option(512, help="Maximum tokenised sequence length."),
    padding: str = typer.Option("max_length", help="Tokenizer padding strategy."),
    truncation: bool = typer.Option(True, help="Enable truncation when tokenising."),
    train: bool = typer.Option(True, help="Execute the training loop."),
    evaluate: bool = typer.Option(True, help="Run evaluation after training."),
) -> None:
    from .training import run_bert_experiment

    pooling = pooling.lower()
    if pooling not in {"cls", "mean"}:
        raise typer.BadParameter("Pooling must be either 'cls' or 'mean'.")

    model_cfg = BertModelConfig(
        base_model=base_model,
        pooling=pooling,
        classifier_hidden_sizes=tuple(classifier_hidden or ()),
        dropout=dropout,
        freeze_base=freeze_base,
        use_lora=use_lora,
        lora=LoraConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            modules_to_save=tuple(lora_modules_to_save or ()),
        ),
    )
    trainer_cfg = TrainerConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        logging_strategy=logging_strategy,
        logging_dir=str(logging_dir) if logging_dir else None,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to=tuple(report_to or ()),
        seed=seed,
        dataloader_pin_memory=dataloader_pin_memory,
        fp16=fp16,
        bf16=bf16,
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
        run_name=run_name,
    )
    tokenizer_cfg = TokenizerConfig(
        pretrained_model_name=base_model,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
    )
    experiment_cfg = BertExperimentConfig(
        model=model_cfg,
        tokenizer=tokenizer_cfg,
        trainer=trainer_cfg,
        evaluation_split=evaluation_split,
        save_metrics=True,
    )
    metrics = run_bert_experiment(
        experiment_cfg,
        output_dir=output_dir,
        train=train,
        evaluate=evaluate,
    )
    if metrics:
        typer.echo(json.dumps(metrics, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

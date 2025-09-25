"""High-level training entry points used by the command-line interface."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
from evaluate import load as load_metric
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from transformers import EvalPrediction, Trainer, TrainingArguments

from .config import BertExperimentConfig, BowConfig
from .data import load_imdb, tokenise_imdb
from .models.bow import build_tfidf_logreg
from .models.bert import BertEncoderClassifier
from .utils import ARTIFACTS_DIR, ensure_dir, save_json, set_seed


def run_bow_experiment(
    cfg: BowConfig,
    *,
    output_dir: Path | str | None = None,
    evaluation_split: str = "test",
    save_model: bool = True,
) -> Dict[str, Any]:
    """Train the TF-IDF baseline and optionally persist the fitted pipeline."""

    set_seed(cfg.random_state)
    imdb = load_imdb()

    if evaluation_split not in imdb:
        raise KeyError(f"IMDb dataset does not provide split: {evaluation_split!r}")

    train_texts = imdb["train"]["text"]
    train_labels = imdb["train"]["label"]
    eval_texts = imdb[evaluation_split]["text"]
    eval_labels = imdb[evaluation_split]["label"]

    pipeline = build_tfidf_logreg(cfg)
    pipeline.fit(train_texts, train_labels)

    preds = pipeline.predict(eval_texts)
    accuracy = accuracy_score(eval_labels, preds)
    report = classification_report(
        eval_labels,
        preds,
        digits=3,
        output_dict=True,
    )

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "num_train_examples": len(train_texts),
        "num_eval_examples": len(eval_texts),
    }

    target_dir = Path(output_dir) if output_dir is not None else ARTIFACTS_DIR / "bow_tfidf"
    ensure_dir(target_dir)

    save_json(target_dir / "metrics.json", metrics)
    save_json(target_dir / "config.json", {"bow_config": asdict(cfg)})

    if save_model:
        dump(pipeline, target_dir / "model.joblib")

    return metrics


def _build_training_arguments(cfg: BertExperimentConfig, output_dir: Path) -> TrainingArguments:
    trainer_cfg = cfg.trainer
    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=trainer_cfg.num_train_epochs,
        per_device_train_batch_size=trainer_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=trainer_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        learning_rate=trainer_cfg.learning_rate,
        weight_decay=trainer_cfg.weight_decay,
        logging_steps=trainer_cfg.logging_steps,
        logging_strategy=trainer_cfg.logging_strategy,
        logging_dir=trainer_cfg.logging_dir,
        evaluation_strategy=trainer_cfg.evaluation_strategy,
        save_strategy=trainer_cfg.save_strategy,
        load_best_model_at_end=trainer_cfg.load_best_model_at_end,
        metric_for_best_model=trainer_cfg.metric_for_best_model,
        greater_is_better=trainer_cfg.greater_is_better,
        report_to=list(trainer_cfg.report_to),
        seed=trainer_cfg.seed,
        dataloader_pin_memory=trainer_cfg.dataloader_pin_memory,
        fp16=trainer_cfg.fp16,
        bf16=trainer_cfg.bf16,
        run_name=trainer_cfg.run_name,
    )
    if trainer_cfg.warmup_ratio is not None:
        kwargs["warmup_ratio"] = trainer_cfg.warmup_ratio
    return TrainingArguments(**kwargs)


def _bert_compute_metrics() -> callable:
    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")

    def compute(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits = eval_pred.predictions
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        labels = eval_pred.label_ids
        result: Dict[str, float] = {}
        result.update(accuracy.compute(predictions=preds, references=labels))
        result.update(
            precision.compute(predictions=preds, references=labels, average="binary")
        )
        result.update(
            recall.compute(predictions=preds, references=labels, average="binary")
        )
        result.update(f1.compute(predictions=preds, references=labels, average="binary"))
        return result

    return compute


def run_bert_experiment(
    cfg: BertExperimentConfig,
    *,
    output_dir: Path | str | None = None,
    train: bool = True,
    evaluate: bool = True,
) -> Dict[str, Any]:
    """Train/evaluate the BERT-based classifiers using ``transformers``."""

    trainer_cfg = cfg.trainer
    set_seed(trainer_cfg.seed)

    target_dir = Path(output_dir) if output_dir is not None else Path(trainer_cfg.output_dir)
    ensure_dir(target_dir)

    dataset, tokenizer = tokenise_imdb(cfg.tokenizer)

    if cfg.evaluation_split not in dataset:
        raise KeyError(f"IMDb dataset does not provide split: {cfg.evaluation_split!r}")

    checkpoint_dir = (
        Path(trainer_cfg.resume_from_checkpoint)
        if trainer_cfg.resume_from_checkpoint
        else None
    )

    if not train and checkpoint_dir and (checkpoint_dir / "pytorch_model.bin").exists():
        model = BertEncoderClassifier.from_pretrained(checkpoint_dir)
    else:
        model = BertEncoderClassifier(cfg.model)

    training_args = _build_training_arguments(cfg, target_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[cfg.evaluation_split],
        tokenizer=tokenizer,
        compute_metrics=_bert_compute_metrics(),
    )

    metrics: Dict[str, Any] = {}

    if train:
        trainer.train(resume_from_checkpoint=trainer_cfg.resume_from_checkpoint)
        trainer.save_model()

    if evaluate:
        metrics = trainer.evaluate()
        if cfg.save_metrics:
            save_json(target_dir / "metrics.json", metrics)

    if cfg.save_metrics:
        save_json(target_dir / "config.json", asdict(cfg))

    return metrics


__all__ = ["run_bow_experiment", "run_bert_experiment"]

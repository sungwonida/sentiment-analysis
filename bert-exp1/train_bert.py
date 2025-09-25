"""Frozen BERT baseline wired to the refactored training utilities."""
from __future__ import annotations

from sentiment_analysis.config import (
    BertExperimentConfig,
    BertModelConfig,
    TokenizerConfig,
    TrainerConfig,
)
from sentiment_analysis.training import run_bert_experiment


def main() -> None:
    experiment = BertExperimentConfig(
        model=BertModelConfig(
            pooling="cls",
            freeze_base=True,
        ),
        tokenizer=TokenizerConfig(),
        trainer=TrainerConfig(
            output_dir="artifacts/frozen_bert",
            num_train_epochs=39,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=5e-4,
            logging_steps=50,
            logging_strategy="steps",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to=("tensorboard",),
            logging_dir="runs/frozen_bert",
        ),
    )
    run_bert_experiment(experiment)


if __name__ == "__main__":
    main()

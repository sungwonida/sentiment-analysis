"""Mean-pooled frozen BERT baseline using the refactored training code."""
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
            pooling="mean",
            classifier_hidden_sizes=(256,),
            freeze_base=True,
        ),
        tokenizer=TokenizerConfig(),
        trainer=TrainerConfig(
            output_dir="artifacts/frozen_bert_meanpool",
            num_train_epochs=24,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=3e-4,
            logging_steps=50,
            logging_strategy="steps",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to=("tensorboard",),
            logging_dir="runs/frozen_bert_meanpool",
        ),
    )
    run_bert_experiment(experiment)


if __name__ == "__main__":
    main()

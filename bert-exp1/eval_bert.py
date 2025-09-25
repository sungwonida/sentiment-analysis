"""Evaluate the frozen BERT baseline from a saved checkpoint."""
from __future__ import annotations

from sentiment_analysis.config import (
    BertExperimentConfig,
    BertModelConfig,
    TokenizerConfig,
    TrainerConfig,
)
from sentiment_analysis.training import run_bert_experiment

CHECKPOINT_DIR = "artifacts/frozen_bert"


def main() -> None:
    experiment = BertExperimentConfig(
        model=BertModelConfig(pooling="cls", freeze_base=True),
        tokenizer=TokenizerConfig(),
        trainer=TrainerConfig(
            output_dir=CHECKPOINT_DIR,
            per_device_eval_batch_size=64,
            evaluation_strategy="no",
            save_strategy="no",
            report_to=(),
            resume_from_checkpoint=CHECKPOINT_DIR,
        ),
    )
    metrics = run_bert_experiment(experiment, train=False, evaluate=True)
    print(metrics)


if __name__ == "__main__":
    main()

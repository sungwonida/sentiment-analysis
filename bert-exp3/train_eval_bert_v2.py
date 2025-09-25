"""LoRA mean-pooled BERT experiment wired to the new CLI helpers."""
from __future__ import annotations

from sentiment_analysis.config import (
    BertExperimentConfig,
    BertModelConfig,
    LoraConfig,
    TokenizerConfig,
    TrainerConfig,
)
from sentiment_analysis.training import run_bert_experiment


def main() -> None:
    experiment = BertExperimentConfig(
        model=BertModelConfig(
            pooling="mean",
            classifier_hidden_sizes=(256,),
            use_lora=True,
            lora=LoraConfig(r=8, alpha=16, dropout=0.05, modules_to_save=("classifier",)),
        ),
        tokenizer=TokenizerConfig(),
        trainer=TrainerConfig(
            output_dir="artifacts/lora_bert_meanpool",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            learning_rate=3e-4,
            logging_steps=50,
            logging_strategy="steps",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to=("tensorboard",),
            logging_dir="runs/lora_bert_meanpool",
            dataloader_pin_memory=False,
        ),
    )
    run_bert_experiment(experiment)


if __name__ == "__main__":
    main()

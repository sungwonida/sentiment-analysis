"""Configuration dataclasses for the sentiment-analysis experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=True)
class BowConfig:
    """Configuration for the TF-IDF + Logistic Regression baseline."""

    max_features: int = 40_000
    ngram_range: Tuple[int, int] = (1, 2)
    stop_words: Optional[str] = "english"
    C: float = 4.0
    max_iter: int = 1_000
    n_jobs: int = -1
    random_state: int = 42


@dataclass(slots=True)
class TokenizerConfig:
    """Configuration for tokenising the IMDb dataset."""

    pretrained_model_name: str = "bert-base-uncased"
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True


@dataclass(slots=True)
class LoraConfig:
    """Low-rank adaptation (LoRA) hyper-parameters."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    target_modules: Tuple[str, ...] = ("query", "key", "value")
    modules_to_save: Tuple[str, ...] = ()
    task_type: str = "FEATURE_EXTRACTION"


@dataclass(slots=True)
class BertModelConfig:
    """Configuration for the BERT-based classifier."""

    base_model: str = "bert-base-uncased"
    num_labels: int = 2
    pooling: str = "cls"  # {"cls", "mean"}
    classifier_hidden_sizes: Tuple[int, ...] = ()
    dropout: float = 0.2
    freeze_base: bool = True
    use_lora: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)
    from_pretrained: Optional[str] = None
    classifier_state_dict: Optional[str] = None


@dataclass(slots=True)
class TrainerConfig:
    """Thin wrapper around :class:`transformers.TrainingArguments`."""

    output_dir: str = "artifacts/bert"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    warmup_ratio: Optional[float] = None
    logging_steps: int = 50
    logging_strategy: str = "steps"
    logging_dir: Optional[str] = None
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    report_to: Tuple[str, ...] = ()
    seed: int = 42
    dataloader_pin_memory: bool = False
    fp16: bool = False
    bf16: bool = False
    resume_from_checkpoint: Optional[str] = None
    run_name: Optional[str] = None


@dataclass(slots=True)
class BertExperimentConfig:
    """Bundle configuration pieces used to train/evaluate BERT models."""

    model: BertModelConfig = field(default_factory=BertModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    evaluation_split: str = "test"
    save_metrics: bool = True


__all__ = [
    "BowConfig",
    "TokenizerConfig",
    "LoraConfig",
    "BertModelConfig",
    "TrainerConfig",
    "BertExperimentConfig",
]

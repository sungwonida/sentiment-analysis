"""BERT-based classifiers used throughout the experiments."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoModel

from ..config import BertModelConfig, LoraConfig
from ..utils import ensure_dir, masked_mean


class BertEncoderClassifier(nn.Module):
    """Thin wrapper around :class:`transformers.AutoModel` with a custom head."""

    def __init__(self, cfg: BertModelConfig):
        super().__init__()
        self.cfg = cfg

        model_name = cfg.from_pretrained or cfg.base_model
        self.encoder = AutoModel.from_pretrained(model_name)

        if cfg.use_lora:
            try:
                from peft import LoraConfig as PeftLoraConfig
                from peft import TaskType, get_peft_model
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "peft must be installed to enable LoRA fine-tuning"
                ) from exc

            lora_cfg = cfg.lora or LoraConfig()
            task_type = getattr(TaskType, lora_cfg.task_type)
            peft_cfg = PeftLoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                bias=lora_cfg.bias,
                target_modules=list(lora_cfg.target_modules),
                modules_to_save=list(lora_cfg.modules_to_save) or None,
                task_type=task_type,
            )
            self.encoder = get_peft_model(self.encoder, peft_cfg)
        elif cfg.freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        layers = []
        in_features = hidden_size
        hidden_dims = list(cfg.classifier_hidden_sizes)

        if not hidden_dims:
            layers.append(nn.Linear(in_features, cfg.num_labels))
        else:
            for dim in hidden_dims:
                layers.append(nn.Linear(in_features, dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(cfg.dropout))
                in_features = dim
            layers.append(nn.Linear(in_features, cfg.num_labels))

        self.classifier = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

        if cfg.classifier_state_dict:
            state = torch.load(cfg.classifier_state_dict, map_location="cpu")
            self.classifier.load_state_dict(state)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state

        if self.cfg.pooling == "cls":
            pooled = last_hidden[:, 0]
        elif self.cfg.pooling == "mean":
            if attention_mask is None:
                raise ValueError("attention_mask is required for mean pooling")
            pooled = masked_mean(last_hidden, attention_mask)
        else:  # pragma: no cover - validated by CLI options
            raise ValueError(f"Unknown pooling strategy: {self.cfg.pooling}")

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------
    # Persistence helpers so :class:`transformers.Trainer` can resume runs
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_dir = ensure_dir(save_directory)
        state_path = Path(save_dir) / "pytorch_model.bin"
        torch.save(self.state_dict(), state_path)
        config_path = Path(save_dir) / "model_config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self.cfg), fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | Path,
        cfg: Optional[BertModelConfig] = None,
    ) -> "BertEncoderClassifier":
        load_dir = Path(load_directory)
        if cfg is None:
            cfg = cls._load_config(load_dir / "model_config.json")
        model = cls(cfg)
        state = torch.load(load_dir / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state)
        return model

    @staticmethod
    def _load_config(path: Path) -> BertModelConfig:
        with path.open("r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)
        lora_cfg = data.get("lora")
        if lora_cfg is not None:
            data["lora"] = LoraConfig(**lora_cfg)
        return BertModelConfig(**data)


__all__ = ["BertEncoderClassifier"]

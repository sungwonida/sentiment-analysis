from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch import nn
import torch


class LoRA_BERT_MeanPoolClsV2(nn.Module):
    def __init__(
            self,
            num_labels = 2,
            from_pretrained = None,
            from_pretrained_head = None,
            device = "cpu"
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

        if from_pretrained is not None and from_pretrained_head is not None:
            assert isinstance(from_pretrained, str)
            assert isinstance(from_pretrained_head, str)
            self.bert = PeftModel.from_pretrained(self.bert, from_pretrained)
            self.classifier.load_state_dict(torch.load(from_pretrained_head, map_location=device))

            assert isinstance(self.bert, PeftModel), "LoRA adapter not attached"
            assert self.bert.active_adapter is not None
        else:
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["query", "key", "value"],  # attention only; safe for BERT
            )

            self.bert = get_peft_model(self.bert, lora_cfg)

        self.bert.print_trainable_parameters()  # sanity check

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = out.last_hidden_state.mean(dim=1)  # outputs.last_hidden_state: [B, L, 768]
        logits = self.classifier(pooled)
        loss = None

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}

import torch, os, random, numpy as np
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
torch.set_float32_matmul_precision("medium")
# SEED = 42
# random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

#-----------------------------------------------
# 1 . Load IMDB and tokenise once
#-----------------------------------------------
from datasets import load_dataset
from transformers import AutoTokenizer

imdb = load_dataset("imdb")  # 25 k train / 25 k test
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,  # cut longer reviews to 512 tokens
        max_length=512,
        padding="max_length"
    )

imdb_tok = imdb.map(tokenize, batched=True, remove_columns=["text"])
imdb_tok.set_format("torch", columns=["input_ids", "attention_mask", "label"])

#-----------------------------------------------
# 2 . Frozen BERT (+ fresh classification head)
#-----------------------------------------------
from transformers import AutoModel, Trainer, TrainingArguments
from torch import nn

class FrozenBERT(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False  # freeze all 110 M params

        hidden = self.bert.config.hidden_size  # 768
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.bert(input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(pooled)
        loss = None

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}

from pathlib import Path
CKPT = Path("checkpoints/frozen_bert/checkpoint-30498")  # ← adjust to yours

from safetensors.torch import load_file
model = FrozenBERT()
state = load_file(CKPT / "model.safetensors")    # ← change here
model.load_state_dict(state)
model.to(device).eval()

args = TrainingArguments(
    output_dir="tmp/eval_only",
    do_train=False,            # <-- key flag
    do_eval=True,
    per_device_eval_batch_size=64,
    logging_strategy="no",
    report_to=[],
    # seed=SEED,
)

# -----------------------------------------------
# metrics
# -----------------------------------------------
from evaluate import load  # evaluate ≥0.4

accuracy  = load("accuracy")
precision = load("precision")
recall    = load("recall")
f1        = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    # classification metrics
    result = {}
    result.update(accuracy.compute(predictions=preds, references=labels))
    result.update(precision.compute(predictions=preds,
                                    references=labels,
                                    average="binary"))   # "macro"/"micro" for >2 classes
    result.update(recall.compute(predictions=preds, references=labels,
                                 average="binary"))
    result.update(f1.compute(predictions=preds, references=labels,
                             average="binary"))

    # Hugging Face automatically prefix-adds "eval_", but keep it explicit
    return {f"eval_{k}": v for k, v in result.items()}

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=imdb_tok["test"],   # 🚫  drop train_dataset
    compute_metrics=compute_metrics,
)

metrics = trainer.evaluate()
print(metrics)

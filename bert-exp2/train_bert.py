#-----------------------------------------------
# 0 . Environment checklist
#-----------------------------------------------
import torch, os, random, numpy as np
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
torch.set_float32_matmul_precision("medium")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

#-----------------------------------------------
# 1 . Load IMDB and tokenise once
#-----------------------------------------------
from datasets import load_dataset
from transformers import AutoTokenizer

imdb = load_dataset("imdb")           # 25 k train / 25 k test
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,              # cut longer reviews to 512 tokens
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

class FrozenBERT_MeanPoolClsV2(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False           # freeze all 110 M params

        hidden = self.bert.config.hidden_size # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden*1, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        with torch.no_grad():
            out = self.bert(input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state.mean(dim=1)  # outputs.last_hidden_state: [B, L, 768]
        logits = self.classifier(pooled)
        loss = None

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}

model = FrozenBERT_MeanPoolClsV2().to(device)

args = TrainingArguments(
    output_dir="checkpoints/frozen_bert_meanpool_clsv2",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=24,
    learning_rate=3e-4,
    # evaluation_strategy="epoch",        # so Trainer calls evaluate() for you
    # fp16=False,                         # use bf16 later if you wish
    logging_strategy="steps",
    logging_steps=50,                   # how often to write to TB
    report_to="tensorboard",            # <-- enable TB
    logging_dir="runs/frozen_bert_meanpool_clsv2",     # default is "runs/" if you omit this
    seed=SEED,
)

# -----------------------------------------------
# metrics
# -----------------------------------------------
from evaluate import load   # evaluate ≥0.4

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
    train_dataset=imdb_tok["train"],
    eval_dataset=imdb_tok["test"],
    compute_metrics=compute_metrics,
)

ckpt = "checkpoints/frozen_bert_meanpool_clsv2/checkpoint-15640"
trainer.train(resume_from_checkpoint=ckpt)

results = trainer.evaluate()      # run the eval loop
print(f"loss: {results['eval_loss']:.4f} | "
      f"accuracy: {results['eval_accuracy']:.3%} | "
      f"precision: {results['eval_precision']:.3%} | "
      f"recall: {results['eval_recall']:.3%} | "
      f"f1: {results['eval_f1']:.3%}")

exit(0) # Intentionally exit here to focus on the current stage.

#-----------------------------------------------
# 3 . Full fine-tune BERT-base
#-----------------------------------------------
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    torch_dtype=torch.bfloat16        # saves ≈35 % VRAM on MPS
).to(device)

args = TrainingArguments(
    "checkpoints/frozen_bert_meanpool_clsv2",
    per_device_train_batch_size=8,    # fits in bfloat16
    gradient_accumulation_steps=4,    # 8×4 = 32 effective
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,                       # use bf16 through torch_dtype above
    # evaluation_strategy="epoch",
    logging_steps=200,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=imdb_tok["train"],
    eval_dataset=imdb_tok["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

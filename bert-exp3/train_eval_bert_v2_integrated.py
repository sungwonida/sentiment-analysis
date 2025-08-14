#-----------------------------------------------
# 0 . Environment checklist
#-----------------------------------------------
import torch, os, random, numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
torch.set_float32_matmul_precision("medium")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Common knobs
train_model = True
eval_model = True

enable_lora = True  # This cannot be False now

session_name = "lora_bert_meanpool_clsv2_integrated"

checkpoint_root_dir = f"checkpoints/{session_name}"
from transformers.trainer_utils import get_last_checkpoint
if os.path.exists(checkpoint_root_dir):
    checkpoint_dir = get_last_checkpoint(checkpoint_root_dir)
else:
    checkpoint_dir = None
print(f"checkpoint_dir: {checkpoint_dir}")

logging_dir = f"runs/{session_name}"

#-----------------------------------------------
# 1 . Load IMDB and tokenise once
#-----------------------------------------------
from datasets import load_dataset
from transformers import AutoTokenizer

imdb = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Save the tokenizer alongside the adapter (For the research)
tokenizer.save_pretrained(f"{checkpoint_root_dir}/tokernizer")

# Load the pretrained tokenizer
# from_pretrained_tokenizer = f"{checkpoint_root_dir}/tokernizer"
# if from_pretrained_tokenizer is not None:
#     assert isinstance(from_pretrained_tokenizer, str)
#     tokenizer = AutoTokenizer.from_pretrained(from_pretrained_tokenizer)

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
# 2 . LoRA (+ fresh classification head)
#-----------------------------------------------
from transformers import Trainer, TrainingArguments
from models import LoRA_BERT_MeanPoolClsV2_Integrated

if checkpoint_dir is not None:
    model_load_path = checkpoint_dir
else:
    model_load_path = None

model = LoRA_BERT_MeanPoolClsV2_Integrated(
    from_pretrained=model_load_path
).to(device)

# -----------------------------------------------
# metrics
# -----------------------------------------------
from evaluate import load

accuracy  = load("accuracy")
precision = load("precision")
recall    = load("recall")
f1        = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

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

# -----------------------------------------------
# Training arguments
# -----------------------------------------------
args = TrainingArguments(
    output_dir=checkpoint_root_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_strategy="steps",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    optim="adamw_torch",
    report_to="tensorboard",
    logging_dir=logging_dir,
    seed=SEED,
    dataloader_pin_memory=False,    # MPS can’t use pinned memory; avoids the warning
)

from transformers import TrainerCallback

class SavePeftOnSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        # Match Trainer’s checkpoint folder name
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save only the adapter weights for the encoder
        model.bert.save_pretrained(ckpt_dir)

trainer_callbacks = []

if enable_lora:
    trainer_callbacks.append(SavePeftOnSave())

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=imdb_tok["train"],
    eval_dataset=imdb_tok["test"],
    compute_metrics=compute_metrics,
    callbacks=trainer_callbacks
)

if train_model:
    trainer.train(resume_from_checkpoint=checkpoint_dir)

if eval_model:
    results = trainer.evaluate()
    print(f"loss: {results['eval_loss']:.4f} | "
          f"accuracy: {results['eval_accuracy']:.3%} | "
          f"precision: {results['eval_precision']:.3%} | "
          f"recall: {results['eval_recall']:.3%} | "
          f"f1: {results['eval_f1']:.3%}")

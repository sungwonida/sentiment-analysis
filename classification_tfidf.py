from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load IMDB (50 k reviews, stratified 25 k train/25 k test)
ds = load_dataset("imdb")
train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
test_texts,  test_labels  = ds["test"]["text"],  ds["test"]["label"]

# 2. Build TF-IDF → Logistic Regression pipeline
pipe = make_pipeline(
    TfidfVectorizer(
        max_features=40_000,  # cap vocab so memory stays small
        ngram_range=(1,2),  # unigrams + bigrams boost performance
        stop_words="english"
    ),
    LogisticRegression(
        C=4.0,  # slightly larger than default for better recall
        max_iter=1_000,
        n_jobs=-1
    )
)

# 3. Train & evaluate
pipe.fit(train_texts, train_labels)
preds = pipe.predict(test_texts)

print("Test accuracy:", accuracy_score(test_labels, preds))
print(classification_report(test_labels, preds, digits=3))

# 4. Confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(test_labels, preds, ax=ax)
target_names = ["negative", "positive"]
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)
_ = ax.set_title(
    f"Confusion Matrix for {pipe.__class__.__name__}\non the test data"
)
plt.show()

# 5. False case analysis
import numpy as np
false_cases = preds != np.array(test_labels)
false_texts = np.array(test_texts)[false_cases]
false_labels = np.array(test_labels)[false_cases]
idx = 0
print(f"False case example (label: {false_labels[idx]}):\n{false_texts[idx]}")

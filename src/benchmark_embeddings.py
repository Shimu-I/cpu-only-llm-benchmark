import config
import os
import time

import numpy as np
import pandas as pd
import psutil
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
name = MODEL.split("/")[-1]

test_df = pd.read_csv(config.SAMPLE_PATH)
labels = sorted(test_df["label_text"].unique())

train_df = load_dataset("mteb/banking77")["train"].to_pandas()
train_df = train_df[train_df["label_text"].isin(labels)][["text", "label_text"]]
train_df = train_df.reset_index(drop=True)
print("Training messages:", len(train_df))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()


def embed(texts, batch_size=32):
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (hidden * mask).sum(1) / mask.sum(1)
        vec = torch.nn.functional.normalize(vec, dim=1)
        chunks.append(vec.numpy())
    return np.vstack(chunks)


print("Embedding training messages...")
X_train = embed(train_df["text"].tolist())
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_df["label_text"])

process = psutil.Process(os.getpid())
ram_mb = process.memory_info().rss / 1024**2

embed(["warm up"])

preds, times = [], []
for text in test_df["text"]:
    start = time.perf_counter()
    vec = embed([text])
    pred = clf.predict(vec)[0]
    times.append(time.perf_counter() - start)
    preds.append(pred)

test_df["predicted"] = preds
test_df.to_csv(f"results/preds_{name}{config.SUFFIX}.csv", index=False)

new_row = pd.DataFrame([{
    "model": name,
    "type": "hf_embedding_classifier",
    "accuracy": round(accuracy_score(test_df["label_text"], preds), 3),
    "macro_f1": round(f1_score(test_df["label_text"], preds, labels=labels, average="macro"), 3),
    "avg_latency_ms": round(sum(times) / len(times) * 1000, 1),
    "ram_mb": round(ram_mb),
    "unparsed": 0,
}])

path = config.METRICS_PATH
metrics = pd.concat([pd.read_csv(path), new_row], ignore_index=True) if os.path.exists(path) else new_row
metrics["unparsed"] = metrics["unparsed"].fillna(0).astype(int)
metrics.to_csv(path, index=False)

print()
print(metrics.to_string(index=False))

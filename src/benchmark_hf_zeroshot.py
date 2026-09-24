import config
import os
import time

import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline

MODEL = "typeform/distilbert-base-uncased-mnli"
model_name = MODEL.split("/")[-1]

df = pd.read_csv(config.SAMPLE_PATH)
labels = sorted(df["label_text"].unique())
readable = [label.replace("_", " ") for label in labels]
to_label = dict(zip(readable, labels))

process = psutil.Process(os.getpid())
clf = pipeline("zero-shot-classification", model=MODEL)
ram_mb = process.memory_info().rss / 1024**2

clf("warm up run", candidate_labels=readable)

preds, times = [], []
for i, text in enumerate(df["text"]):
    start = time.perf_counter()
    result = clf(text, candidate_labels=readable)
    times.append(time.perf_counter() - start)
    preds.append(to_label[result["labels"][0]])
    if (i + 1) % 50 == 0:
        print(f"  done {i + 1}/{len(df)}")

df["predicted"] = preds
df.to_csv(f"results/preds_{model_name}{config.SUFFIX}.csv", index=False)

metrics = pd.DataFrame([{
    "model": model_name,
    "type": "huggingface_zero_shot",
    "accuracy": round(accuracy_score(df["label_text"], preds), 3),
    "macro_f1": round(f1_score(df["label_text"], preds, average="macro"), 3),
    "avg_latency_ms": round(sum(times) / len(times) * 1000, 1),
    "ram_mb": round(ram_mb),
}])

path = config.METRICS_PATH
metrics.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
print()
print(metrics.to_string(index=False))

import config
import os
import sys
import time

import ollama
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
tag = MODEL.replace(":", "-")

df = pd.read_csv(config.SAMPLE_PATH)
labels = sorted(df["label_text"].unique())

PROMPT = """You are a bank customer-support routing assistant.
Classify the customer message into exactly one of these categories:
{options}

Reply with only the category name, nothing else.

Message: {text}
Category:"""


def ask(text):
    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT.format(options="\n".join(labels), text=text),
        }],
        options={"temperature": 0, "num_predict": 20},
    )
    return response["message"]["content"].strip()


def to_label(answer):
    a = answer.lower().strip().strip("`'\".").replace(" ", "_")
    if a in labels:
        return a
    for label in labels:
        if label in a:
            return label
    return "unparsed"


def ollama_ram_mb():
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in (p.info["name"] or "").lower():
                total += p.info["memory_info"].rss
        except (psutil.Error, TypeError):
            pass
    return total / 1024**2


ask("warm up")

raw, preds, times = [], [], []
for i, text in enumerate(df["text"]):
    start = time.perf_counter()
    answer = ask(text)
    times.append(time.perf_counter() - start)
    raw.append(answer)
    preds.append(to_label(answer))
    if (i + 1) % 20 == 0:
        print(f"  done {i + 1}/{len(df)}")

df["raw_answer"] = raw
df["predicted"] = preds
df.to_csv(f"results/preds_{tag}{config.SUFFIX}.csv", index=False)

new_row = pd.DataFrame([{
    "model": tag,
    "type": "ollama_llm",
    "accuracy": round(accuracy_score(df["label_text"], preds), 3),
    "macro_f1": round(f1_score(df["label_text"], preds, labels=labels, average="macro"), 3),
    "avg_latency_ms": round(sum(times) / len(times) * 1000, 1),
    "ram_mb": round(ollama_ram_mb()),
    "unparsed": preds.count("unparsed"),
}])

path = config.METRICS_PATH
if os.path.exists(path):
    new_row = pd.concat([pd.read_csv(path), new_row], ignore_index=True)
new_row.to_csv(path, index=False)

print()
print(new_row.to_string(index=False))

import pandas as pd
from transformers import pipeline

df = pd.read_csv("data/processed/test_sample.csv")

labels = sorted(df["label_text"].unique())
readable = [label.replace("_", " ") for label in labels]

clf = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
)

for i in [0, 50, 100, 150]:
    text = df.loc[i, "text"]
    true_label = df.loc[i, "label_text"].replace("_", " ")
    result = clf(text, candidate_labels=readable)
    print(f"Message : {text}")
    print(f"True    : {true_label}")
    print(f"Predicted: {result['labels'][0]} (score {result['scores'][0]:.2f})")
    print()

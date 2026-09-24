import glob
import os

import pandas as pd

files = sorted(glob.glob("results/preds_*.csv"))
all_errors = []

for f in files:
    model = os.path.basename(f)[len("preds_"):-len(".csv")]
    df = pd.read_csv(f)
    wrong = df[df["label_text"] != df["predicted"]].copy()
    wrong["model"] = model
    all_errors.append(wrong)

    print("=" * 70)
    print(f"{model}: {len(wrong)} wrong out of {len(df)}")
    print("=" * 70)

    print("Most common confusions (true -> predicted):")
    pairs = wrong.groupby(["label_text", "predicted"]).size().sort_values(ascending=False)
    for (true, pred), n in pairs.head(5).items():
        print(f"  {n}x  {true} -> {pred}")

    if len(wrong) <= 10:
        print("\nWrong messages:")
        for _, row in wrong.iterrows():
            print(f"  [{row['label_text']} -> {row['predicted']}] {row['text']}")
    print()

errors = pd.concat(all_errors, ignore_index=True)
errors.to_csv("results/all_errors.csv", index=False)

strong = errors[errors["model"] != "distilbert-base-uncased-mnli"]
shared = strong.groupby("text")["model"].nunique().sort_values(ascending=False)
shared = shared[shared >= 2]

print("=" * 70)
print("Messages that 2 or more strong models got wrong:")
print("=" * 70)
for text, n in shared.items():
    true = strong[strong["text"] == text]["label_text"].iloc[0]
    print(f"  ({n} models) [{true}] {text}")

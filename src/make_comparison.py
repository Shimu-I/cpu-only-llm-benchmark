import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

easy = pd.read_csv("results/metrics.csv")
hard = pd.read_csv("results/metrics_hard.csv")

for df in (easy, hard):
    df["unparsed"] = df["unparsed"].fillna(0).astype(int)
easy.to_csv("results/metrics.csv", index=False)
hard.to_csv("results/metrics_hard.csv", index=False)

cols = ["model", "accuracy", "macro_f1", "avg_latency_ms", "ram_mb"]
comp = easy[["type"] + cols].merge(hard[cols], on="model", suffixes=("_easy", "_hard"))
comp["accuracy_drop"] = (comp["accuracy_easy"] - comp["accuracy_hard"]).round(3)
comp = comp.sort_values("accuracy_hard", ascending=False).reset_index(drop=True)
comp.to_csv("results/comparison.csv", index=False)
print(comp.to_string(index=False))

# Chart 1: accuracy on easy vs hard sample
fig, ax = plt.subplots(figsize=(9, 5))
x = list(range(len(comp)))
w = 0.38
ax.bar([i - w / 2 for i in x], comp["accuracy_easy"], w, label="Easy sample")
ax.bar([i + w / 2 for i in x], comp["accuracy_hard"], w, label="Hard sample")
for i, (e, h) in enumerate(zip(comp["accuracy_easy"], comp["accuracy_hard"])):
    ax.text(i - w / 2, e + 0.01, f"{e:.3f}", ha="center", fontsize=9)
    ax.text(i + w / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(comp["model"], rotation=15, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.1)
ax.set_title("Accuracy: easy vs hard (look-alike intents) test sample")
ax.legend()
fig.tight_layout()
fig.savefig("results/accuracy_easy_vs_hard.png", dpi=150)

# Chart 2: accuracy vs speed on the hard sample (bubble size = RAM)
fig2, ax2 = plt.subplots(figsize=(9, 5))
for _, r in comp.iterrows():
    ax2.scatter(r["avg_latency_ms_hard"], r["accuracy_hard"],
                s=r["ram_mb_hard"] / 4 + 40, alpha=0.7)
    ax2.annotate(r["model"], (r["avg_latency_ms_hard"], r["accuracy_hard"]),
                 textcoords="offset points", xytext=(10, 8), fontsize=9)
ax2.set_xscale("log")
ax2.set_xlim(3, 8000)
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Average latency per message (ms, log scale)")
ax2.set_ylabel("Accuracy on hard sample")
ax2.set_title("Accuracy vs speed on the hard sample (bubble size = RAM)")
fig2.tight_layout()
fig2.savefig("results/accuracy_vs_speed.png", dpi=150)

print("\nSaved: results/comparison.csv, accuracy_easy_vs_hard.png, accuracy_vs_speed.png")

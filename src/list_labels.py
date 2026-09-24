from datasets import load_dataset

ds = load_dataset("mteb/banking77")
test = ds["test"]

counts = {}
for name in test["label_text"]:
    counts[name] = counts.get(name, 0) + 1

print("Test messages per intent:\n")
for name in sorted(counts):
    print(f"  {name:45s} {counts[name]}")

from datasets import load_dataset

ds = load_dataset("mteb/banking77")
print(ds)

train = ds["train"]
print("\nColumns:", train.column_names)
print("Number of unique labels:", len(set(train["label"])))

print("\nSample rows:")
for i in range(5):
    print(" ", train[i])

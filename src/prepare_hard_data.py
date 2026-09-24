from datasets import load_dataset

HARD_INTENTS = [
    "lost_or_stolen_card",
    "compromised_card",
    "declined_card_payment",
    "declined_transfer",
    "transfer_not_received_by_recipient",
    "pending_transfer",
    "top_up_failed",
    "pending_top_up",
    "card_arrival",
    "card_delivery_estimate",
]

ds = load_dataset("mteb/banking77")
df = ds["test"].to_pandas()

df = df[df["label_text"].isin(HARD_INTENTS)]
sample = df.groupby("label_text", group_keys=False).sample(n=20, random_state=42)
sample = sample[["text", "label_text"]].reset_index(drop=True)

sample.to_csv("data/processed/test_sample_hard.csv", index=False)

print("Sample shape:", sample.shape)
print()
print(sample["label_text"].value_counts())
print()
print(sample.sample(5, random_state=1))

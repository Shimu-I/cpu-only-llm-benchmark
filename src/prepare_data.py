from datasets import load_dataset

INTENTS = [
    "lost_or_stolen_card",
    "card_arrival",
    "declined_card_payment",
    "exchange_rate",
    "transfer_not_received_by_recipient",
    "top_up_failed",
    "request_refund",
    "change_pin",
    "verify_my_identity",
    "terminate_account",
]

ds = load_dataset("mteb/banking77")
df = ds["test"].to_pandas()

df = df[df["label_text"].isin(INTENTS)]
sample = df.groupby("label_text", group_keys=False).sample(n=20, random_state=42)
sample = sample[["text", "label_text"]].reset_index(drop=True)

sample.to_csv("data/processed/test_sample.csv", index=False)

print("Sample shape:", sample.shape)
print()
print(sample["label_text"].value_counts())
print()
print(sample.head())

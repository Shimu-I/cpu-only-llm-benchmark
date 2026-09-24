import gradio as gr
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INTENTS = [
    "lost_or_stolen_card", "compromised_card", "declined_card_payment",
    "declined_transfer", "transfer_not_received_by_recipient",
    "pending_transfer", "top_up_failed", "pending_top_up", "card_arrival",
    "card_delivery_estimate", "exchange_rate", "request_refund",
    "change_pin", "verify_my_identity", "terminate_account",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()


def embed(texts, batch_size=32):
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (hidden * mask).sum(1) / mask.sum(1)
        chunks.append(torch.nn.functional.normalize(vec, dim=1).numpy())
    return np.vstack(chunks)


print("Training the classifier (about 20 seconds)...")
train = load_dataset("mteb/banking77")["train"].to_pandas()
train = train[train["label_text"].isin(INTENTS)]
clf = LogisticRegression(max_iter=1000)
clf.fit(embed(train["text"].tolist()), train["label_text"])
print("Ready.")


def route(message, threshold):
    if not message or not message.strip():
        return "### Type a customer message first.", None
    probs = clf.predict_proba(embed([message]))[0]
    top = np.argsort(probs)[::-1][:3]
    scores = {clf.classes_[i].replace("_", " "): float(probs[i]) for i in top}
    best = float(probs[top[0]])
    name = clf.classes_[top[0]].replace("_", " ")
    if best < threshold:
        decision = f"### ⚠️ Send to a human agent\nThe model is only **{best:.0%}** sure (best guess: {name})."
    else:
        decision = f"### ✅ Route to: {name}\nThe model is **{best:.0%}** confident."
    return decision, scores


CSS = """
<style>
.gradio-container { max-width: 760px !important; margin: 0 auto !important; }
.gradio-container, .gradio-container * { font-size: 17px; }
.gradio-container h1 { font-size: 32px; text-align: center; }
.gradio-container h3 { font-size: 24px; }
</style>
"""

with gr.Blocks(title="Support Ticket Router") as demo:
    gr.HTML(CSS)
    gr.Markdown(
        "# Support Ticket Router\n"
        "Type a customer message and see which team it would go to. "
        "Uses **all-MiniLM-L6-v2** (a Hugging Face model) plus a small "
        "classifier, running on CPU."
    )
    msg = gr.Textbox(label="Customer message", lines=3,
                     placeholder="e.g. My card still hasn't arrived after two weeks")
    threshold = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                          label="Confidence threshold (below this, send to a human)")
    with gr.Row():
        clear_btn = gr.Button("Clear")
        submit_btn = gr.Button("Route this message", variant="primary")

    decision = gr.Markdown()
    scores = gr.Label(num_top_classes=3, label="Top 3 predicted categories")

    gr.Examples(
        examples=[
            "My card still hasn't arrived after two weeks",
            "Someone has my card number, freeze my account",
            "How long until my friend receives my transfer?",
            "I want to close my account",
        ],
        inputs=msg,
        label="Try an example",
    )

    submit_btn.click(route, [msg, threshold], [decision, scores])
    msg.submit(route, [msg, threshold], [decision, scores])
    clear_btn.click(lambda: ("", "", None), None, [msg, decision, scores])

if __name__ == "__main__":
    demo.launch()

# Decision Memo: Which open-source model should we deploy to route support tickets on CPU-only servers?

Prepared by: [Your name] | Date: [date]

## Recommendation

Deploy **all-MiniLM-L6-v2 with a logistic-regression classifier**, provided we have labeled tickets to train it on (about 1,300 examples were enough in this test).

If no labeled data exists yet, start with **qwen2.5:7b-instruct** (via Ollama) as a stop-gap, and collect labels while it runs. Do not use the zero-shot DistilBERT model.

## The question

We receive customer messages that must be routed to the right team. We have no GPU servers. Which open-source model gives the best mix of accuracy, speed and memory on ordinary CPU hardware?

## How we tested

- Data: the public banking77 dataset of customer banking queries.
- Two test samples of 200 messages each (20 per category, fixed random seed):
  - **Easy:** 10 clearly different categories (e.g. card_arrival, exchange_rate).
  - **Hard:** 10 look-alike categories in 5 confusable pairs (e.g. lost_or_stolen_card vs compromised_card).
- Four models on the same messages, on one 16 GB CPU-only laptop.

## Results (hard sample, the realistic case)

| Model | Accuracy | Time per message | Time for 100,000 messages | RAM |
|---|---|---|---|---|
| MiniLM + classifier | **88.0%** | 8 ms | about 13 minutes | 638 MB |
| qwen2.5-7b-instruct | 76.0% | 1,510 ms | about 42 hours | 4,854 MB |
| qwen2.5-3b | 64.0% | 720 ms | about 20 hours | 2,438 MB |
| DistilBERT zero-shot | 38.5% | 218 ms | about 6 hours | 513 MB |

On the easy sample the picture was different: MiniLM scored 99.0%, qwen2.5-3b 97.5%, qwen2.5-7b 96.5% and DistilBERT 64.5%. Easy categories hide the differences between models.

## Why MiniLM wins

- **Most accurate**, by 12 points over the best LLM.
- **About 190 times faster** than the 7B model, at about one eighth of the memory.
- It learned the boundary between look-alike categories from labeled examples. The LLMs only see the category names, so they must guess where the line falls.

## What this means in practice

At 88% accuracy on hard categories, about 120 of every 1,000 messages would still be misrouted (against about 240 for the best LLM). Some misroutes cost more than others: for example, sending a "my card hasn't arrived" message to the fraud team wastes an escalation.

## Risks and limits

- Small test: 200 messages per sample, so differences of a few points are within noise.
- One dataset. Real tickets may be messier, longer, or in other languages.
- Speed was measured on one laptop, one message at a time.
- Some labels are genuinely ambiguous ("How long do transfers take to finish?"), so part of the remaining error is in the labels, not the models.
- MiniLM needs labeled data, and the categories must be retrained if they change. The LLMs need none.

## Suggested next steps

1. Add a confidence threshold: auto-route confident predictions and send low-confidence ones to a human.
2. Test whether adding a one-line description of each category to the LLM prompt closes the gap (it would remove the need for labeled data).
3. Re-test on a sample of real company tickets before committing.

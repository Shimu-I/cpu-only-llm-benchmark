# How to present `cpu-only-llm-benchmark`

A guide for telling this project as a story, in an interview, on a call, or on your CV. Read it out loud a few times and change the words until they sound like you.

---

## 1. The core idea in one sentence

> "I tested which open-source model a small company should use to route support tickets on ordinary CPU servers, and found that a tiny embedding model beat 7-billion-parameter LLMs, but only once I made the test harder."

If you remember nothing else, remember that sentence. It has a **decision**, a **surprise**, and a **lesson**.

---

## 2. The 30-second version (for "tell me about a project")

"I built a benchmark to answer a practical question: which open-source model should a company use to route customer support messages if it has no GPU? I compared a zero-shot Hugging Face model, two local LLMs through Ollama, and a small embedding model with a simple classifier. On an easy test they all looked fine, so I built a harder test with look-alike categories. There, the small embedding model reached 88% accuracy at 8 milliseconds per message, while the best LLM got 76% and was about 190 times slower. I wrote it up as a decision memo and built a demo app."

---

## 3. The 2-minute story (use this structure)

Tell it in six beats. Each beat is one or two sentences.

**1. The situation.** "A company gets thousands of support messages, and each must go to the right team. They have no GPU servers."

**2. The question.** "Which open-source model gives the best mix of accuracy, speed and memory on a CPU?"

**3. What I did.** "I used the public banking77 dataset and ran four models on the same 200 messages: DistilBERT zero-shot, Qwen 3B and 7B through Ollama, and MiniLM embeddings with logistic regression."

**4. The twist.** "On my first test with clearly different categories, three models scored 96 to 99 percent. That looked like a tie, so I didn't trust it. I built a second test with five pairs of look-alike categories."

**5. The result.** "On the hard test, MiniLM got 88 percent, the 7B LLM 76 percent, the 3B 64 percent, and DistilBERT 38 percent. MiniLM was also about 190 times faster than the 7B and used about an eighth of the memory."

**6. The decision and the honesty.** "So I recommended MiniLM if labeled data exists, and the 7B LLM as a stop-gap if not. The limits: 200 messages per test, one dataset, and the LLMs got no examples, so I listed a fairer test as a next step."

**Why this works:** it has a problem, a surprise, a decision, and it shows you know the weak points of your own work.

---

## 4. Five-minute walkthrough with the live demo

| Minute | What you show | What you say |
|---|---|---|
| 0:00 | The GitHub README | The problem and the question, in two sentences. |
| 0:45 | `results/accuracy_easy_vs_hard.png` | "Every model drops on the hard test, but by very different amounts." |
| 1:45 | `results/accuracy_vs_speed.png` | "MiniLM sits alone in the best corner: most accurate, fastest, smallest." |
| 2:30 | `reports/decision_memo.md` | The recommendation, and the "labeled data" trade-off. |
| 3:15 | The demo app | See the demo script below. |
| 4:15 | `ACTIVITY_LOG.md` | "I documented every command and every error, so the project is reproducible." |
| 4:45 | Close | The limits and the next steps. |

### Demo script (30 to 60 seconds)

1. Start the app **before** the interview (`python app/app.py`) and wait for `Ready.`
2. Click the example "My card still hasn't arrived after two weeks". Point at the routing decision and the confidence.
3. Click "Someone has my card number, freeze my account" and show that it goes to a different team.
4. Type "What's the weather today?" and show the low confidence and the "send to a human agent" warning. Say: "The classifier only knows 15 banking intents, so the threshold is the safety net for messages it doesn't understand."
5. Move the threshold slider and show how the decision changes.

Have a backup: keep a screenshot in `app/screenshot.png` in case the live demo fails.

---

## 5. Slide outline (if you are asked for slides)

1. **Title and question:** "Which open model should route support tickets on a CPU?"
2. **The setup:** dataset, the four models, the two test samples.
3. **Easy test result:** everyone is close, so the test is too easy.
4. **Hard test result:** the gap appears (chart 1).
5. **Accuracy vs speed vs memory:** chart 2.
6. **Recommendation and trade-offs:** MiniLM if labeled data exists, otherwise the 7B LLM as a stop-gap.
7. **Limits and next steps:** small samples, unfair LLM setup, real tickets next.

One idea per slide, and no more than 20 words of text on each.

---

## 6. Numbers cheat sheet (memorize these)

| Model | Easy accuracy | Hard accuracy | Hard latency | Hard RAM |
|---|---|---|---|---|
| MiniLM + logistic regression | 0.990 | 0.880 | 8.0 ms | 638 MB |
| qwen2.5-7b-instruct | 0.965 | 0.760 | 1,509.7 ms | 4,854 MB |
| qwen2.5-3b | 0.975 | 0.640 | 719.8 ms | 2,438 MB |
| DistilBERT zero-shot | 0.645 | 0.385 | 218.4 ms | 513 MB |

Other facts to keep in your head:
- Dataset: banking77, 77 intents, 9,993 train and 3,076 test messages.
- Two samples: 200 messages each, 20 per intent, seed 42.
- MiniLM trained on about 1,300 labeled messages.
- Easy-sample errors: DistilBERT 71, qwen 3B 5, qwen 7B 7, MiniLM 2.
- MiniLM vs best LLM on the hard sample: 12 points more accurate, about 190 times faster, about one eighth of the RAM.
- Derived time for 100,000 messages: about 13 minutes for MiniLM and about 42 hours for the 7B model (a calculation from latency, not a separate run).
- Environment: Ubuntu, 16 GB RAM, no GPU, Python 3.12.

---

## 7. Adjust the story for the listener

**Recruiter or non-technical person:** skip the model names. Say: "I tested different AI models to find the one a small company could afford to run, and found a tiny one that was both the most accurate and the fastest."

**Hiring manager:** lead with the decision, the trade-offs and the risks. Mention the memo, the confidence threshold and the human-review idea.

**Technical interviewer:** go into embeddings, mean pooling, the fairness of the LLM comparison, sample size and calibration. This is where the Q&A file helps.

---

## 8. Phrases that help, and phrases to avoid

**Say:**
- "The evidence suggests..." and "in my test..."
- "That's a limitation. Here is how I would check it."
- "I didn't trust the first result, so I built a harder test."
- "I don't know yet, but I would find out by..."

**Avoid:**
- "My model is 99% accurate." (Which test? The easy one.)
- "LLMs are bad." (You showed zero-shot LLMs did worse than a trained classifier on this task.)
- "I used AI to build it." with nothing else. If you say it, add what you did yourself.
- Memorized paragraphs. Use the six beats instead.

---

## 9. Handling hard moments

**"Isn't 200 messages too small?"** "Yes, for close calls. The big gaps, like 88% vs 38%, I trust. The small ones, like 5 vs 7 errors, I treat as noise. A larger sample is my first next step."

**"Isn't it unfair to the LLMs?"** "Partly. MiniLM saw about 1,300 labeled examples and the LLMs saw only category names. I present it as zero-shot LLM vs trained classifier, and I list a few-shot prompt test as a next step."

**"Why should I trust your hard sample?"** "I designed it, so it reflects my choices. That's why I report both samples and explain how I chose them."

**"Can you explain this line of code?"** Open the file, read it slowly, and explain what goes in and what comes out. If you don't know, say what you think it does and how you would check.

---

## 10. CV and LinkedIn bullets

Pick two or three:

- Benchmarked four open-source models (Hugging Face and Ollama) for support-ticket routing on CPU-only hardware, comparing accuracy, latency and memory across an easy and a hard test set.
- Found that MiniLM embeddings with logistic regression reached 88% accuracy on look-alike intents, 12 points above the best zero-shot 7B LLM, at about 190 times lower latency.
- Designed a harder test set that exposed differences hidden by an easy benchmark (accuracy drops of 11 to 33 points).
- Wrote a decision memo with recommendation, trade-offs and risks, and built a Gradio demo with a confidence threshold that sends uncertain messages to a human.
- Documented every step in a reproducible repo (pinned environment, fixed seeds, one commit per task).

**One-line project description for GitHub:**
"Benchmark of open-source models (Hugging Face + Ollama) for support-ticket routing on CPU-only hardware, with a decision memo and demo app."

---

## 11. Documentation checklist for the repo

Before you share the link, check that the repo has:

- [ ] A README with the problem, the aim, the results table and both charts.
- [ ] A screenshot of the demo app.
- [ ] A "how to run it" section (create the venv, install requirements, run the scripts in order).
- [ ] A "what I learned" and "limitations" section.
- [ ] The decision memo in `reports/`.
- [ ] The activity log, the glossary and the interview Q&A in a `docs/` folder.
- [ ] A clean commit history, with no large files committed.
- [ ] A short repo description and topics on GitHub (`huggingface`, `nlp`, `ollama`, `benchmark`).

**Suggested order for someone looking at your repo:**
1. `README.md`
2. `reports/decision_memo.md`
3. `results/` charts and `comparison.csv`
4. `src/` scripts
5. `ACTIVITY_LOG.md`

---

## 12. Practice plan

1. Day 1: read this guide and the glossary. Say every glossary definition out loud.
2. Day 2: say the 30-second version ten times, then the 2-minute story five times, recording yourself if you can.
3. Day 3: answer 10 questions from the Q&A file without looking, then compare.
4. Day 4: do the demo from start to finish, twice.
5. Day 5: ask a friend to interrupt you with random questions from the Q&A file.

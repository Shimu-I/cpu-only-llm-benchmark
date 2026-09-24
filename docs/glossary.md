# Glossary: every new word in this project

Simple English. Each entry has a plain meaning and an example taken from **this** project (`cpu-only-llm-benchmark`). Read one section at a time, and try to say each meaning out loud in your own words.

---

## 1. The project and the business idea

**Support ticket / customer message**
A message a customer sends to a company asking for help.
*Example:* "My card still hasn't arrived after 2 weeks."

**Routing (triage)**
Sending each message to the right team.
*Example:* a lost-card message goes to the fraud team, and a "where is my card" message goes to the delivery team.

**Intent**
What the customer wants, in one short label.
*Example:* `card_arrival`, `exchange_rate`, `terminate_account`.

**Benchmark**
A fair test where several options do the same job on the same data, and we measure and compare them.
*Example:* we ran four models on the same 200 messages.

**Baseline**
A simple first result that other results are compared against.
*Example:* the zero-shot DistilBERT model (64.5% accuracy on the easy sample) was our baseline.

**Business decision**
A real choice a company has to make, which our analysis should help with.
*Example:* "Which model should we deploy on our CPU-only servers?"

**Decision memo**
A short document that gives a recommendation, the evidence, and the risks, written for a non-technical reader.
*Example:* `reports/decision_memo.md`.

**CPU-only**
Running without a GPU (a special graphics chip that makes AI faster). Ordinary computers and cheap servers are CPU-only.
*Example:* your 16 GB Ubuntu laptop has no GPU, so we only used small models.

**Latency**
How long one answer takes, usually in milliseconds (ms). 1,000 ms = 1 second.
*Example:* MiniLM took 8 ms per message, and qwen2.5-7b took about 1,510 ms.

**Throughput**
How many messages can be handled per second or per hour.
*Example:* at 8 ms per message, MiniLM could handle roughly 100,000 messages in about 13 minutes.

**RAM**
The computer's working memory. A model must fit into it to run.
*Example:* qwen2.5-7b used about 4.9 GB, and MiniLM used about 0.6 GB.

**Trade-off**
Getting more of one good thing means getting less of another.
*Example:* the LLMs need no labeled data, but they are slower and, on the hard sample, less accurate.

**Sample / test sample**
A smaller set of data chosen from a bigger one.
*Example:* 200 messages picked from the 3,076 test messages.

**Easy sample / hard sample**
Two test sets of 200 messages each. The easy one has 10 very different intents. The hard one has 10 look-alike intents in 5 confusable pairs.

**Look-alike intents**
Categories that sound similar and are easy to mix up.
*Example:* `lost_or_stolen_card` vs `compromised_card`.

**Random seed**
A fixed number (we used 42) that makes "random" choices repeat exactly the same way each time.
*Example:* `random_state=42` gives the same 200 messages on every run.

**Reproducible**
Anyone can run your steps and get the same result.
*Example:* fixed seed, pinned `requirements.txt`, and scripts saved in git.

**Failure analysis**
Studying the mistakes of a model to understand why it fails, not only how often.
*Example:* the 7B model kept sending "transaction" messages to `top_up_failed`.

**Misroute**
A message sent to the wrong team.
*Example:* a "my card hasn't arrived" message sent to the fraud team.

**Confidence**
How sure the model is about its answer, as a percentage.
*Example:* for "What's the weather today?" the app showed only 21%.

**Threshold**
A cut-off number. Below it, we do something different.
*Example:* below 40% confidence, the app says "send to a human agent".

**Human in the loop**
A person checks the cases where the machine is not sure.
*Example:* the "send to a human agent" warning in the demo app.

---

## 2. Data

**Dataset**
A collection of examples used to train or test a model.
*Example:* banking77.

**banking77**
A public dataset of customer banking questions, each labeled with one of 77 intents.
*Example:* "I am still waiting on my card?" is labeled `card_arrival`.

**Train split / test split**
The data is divided in two. The model **learns** from the train split and is **graded** on the test split, which it has never seen.
*Example:* 9,993 train messages and 3,076 test messages.

**Label**
The correct answer attached to an example.
*Example:* `label_text = "card_arrival"` (`label = 11` is the same thing as a number).

**Labeled data**
Examples that come with correct answers. Someone has to create them, which costs time and money.
*Example:* MiniLM used about 1,300 labeled messages to train. The LLMs used none.

**Data leakage**
When the model sees test data during training, which makes the score look too good.
*Example:* we avoided it by training only on the train split and testing only on the test split.

**Parquet**
A compact file format for tables of data.
*Example:* the `mteb/banking77` copy is stored as parquet files.

**Dataset loading script**
An old way of publishing datasets with a small Python file that downloads the data. The newest `datasets` library refuses to run these.
*Example:* `PolyAI/banking77` failed with "Dataset scripts are no longer supported".

**CSV**
A plain text table where values are separated by commas.
*Example:* `data/processed/test_sample.csv`.

**DataFrame (pandas)**
A table inside Python, with rows and named columns.
*Example:* `df["label_text"]`.

---

## 3. Hugging Face

**Hugging Face**
A company and community that hosts open-source AI models and datasets, and builds the Python tools to use them.

**Hub**
The website where the models and datasets live, like GitHub but for AI.
*Example:* `huggingface.co/sentence-transformers/all-MiniLM-L6-v2`.

**Open-source model**
A model whose files anyone can download and run for free.
*Example:* all four models in this project.

**Model weights**
The learned numbers inside a model. They are the big files that get downloaded.
*Example:* `model.safetensors` (268 MB for DistilBERT).

**safetensors**
A safe file format for model weights.

**transformers (library)**
The Python package that downloads Hugging Face models and runs them.

**datasets (library)**
The Python package that downloads and loads datasets from the Hub.
*Example:* `load_dataset("mteb/banking77")`.

**Pipeline**
A ready-made shortcut that does everything for one task (tokenize, run model, format answer) in a single call.
*Example:* `pipeline("zero-shot-classification", model=...)`.

**AutoTokenizer / AutoModel**
The lower-level way to load a tokenizer and a model, which gives you more control than a pipeline.
*Example:* we used it for MiniLM so we could do the pooling ourselves.

**Cache**
A folder where downloads are saved so they are not downloaded twice.
*Example:* `~/.cache/huggingface` (474 MB after the first models).

**HF_TOKEN**
A password-like key for your Hugging Face account. Optional. Without it you get a harmless warning about "unauthenticated requests".

**Gradio**
A Python library that turns a function into a small web page.
*Example:* `app/app.py`, the Support Ticket Router.

**Space**
A free place on the Hub where you can host a Gradio demo online (an optional next step).

---

## 4. NLP and model words

**NLP (Natural Language Processing)**
Teaching computers to work with human language.

**Token / tokenizer**
A token is a small piece of text (a word or part of a word). The tokenizer cuts text into tokens and turns them into numbers.
*Example:* "card arrival" becomes a list of numbers the model can read.

**Model**
A program that learned patterns from data and can now make predictions.

**Encoder model**
A model that reads text and turns it into numbers that capture its meaning. It does not write text.
*Example:* DistilBERT and MiniLM.

**DistilBERT**
A small, fast version of the BERT language model.
*Example:* `typeform/distilbert-base-uncased-mnli`, about 270 MB.

**MiniLM (all-MiniLM-L6-v2)**
A very small model (about 90 MB) trained to turn sentences into meaning vectors.

**Embedding / vector**
A list of numbers that represents the meaning of a sentence. Sentences with similar meanings get similar numbers.
*Example:* "My card is lost" and "I can't find my card" get vectors that are close together.

**Attention mask**
A list that tells the model which tokens are real words and which are padding (filler).

**Padding**
Extra filler tokens added so all sentences in a batch have the same length.

**Mean pooling**
Averaging the vectors of all the words in a sentence into one vector for the whole sentence, ignoring padding.

**Normalize**
Rescaling a vector to length 1, so only its direction (its meaning) matters.

**Zero-shot classification**
Sorting text into categories the model was never trained on. You only give it the category names.
*Example:* DistilBERT was given "card arrival", "change pin", and so on.

**NLI / MNLI**
Natural Language Inference: deciding whether one sentence follows from another. Zero-shot models turn each label into a sentence like "This example is card arrival" and check how well it fits.

**Classifier**
A model that picks one category from a list.

**Logistic regression**
A simple, fast classifier that learns a weighted score for each category. In scikit-learn it is called `LogisticRegression`.
*Example:* we trained it on MiniLM vectors.

**predict_proba**
A function that returns the probability for every category. We used it for the confidence score.

**Training vs inference**
Training is learning from examples. Inference is using the trained model to predict on new data.
*Example:* the classifier trains in seconds. Every message we score is inference.

**Fine-tuning**
Continuing to train a whole pretrained model on your own labeled data. It usually needs more compute. We did not do it, and it is a possible next step.

**Frozen model**
A model whose weights are not changed. MiniLM stayed frozen, and only the small classifier on top was trained.

**LLM (Large Language Model)**
A big model that reads and writes text, and can follow instructions.
*Example:* qwen2.5.

**Parameters (3B / 7B)**
The number of learned numbers inside a model. 3B is 3 billion, 7B is 7 billion. More parameters usually means slower and heavier.

**Instruct model**
An LLM trained to follow instructions.
*Example:* `qwen2.5:7b-instruct`.

**Prompt**
The instructions and text you send to an LLM.
*Example:* "Classify the customer message into exactly one of these categories..."

**Temperature**
A dial for randomness in an LLM. `0` means as predictable as possible.

**num_predict**
The maximum number of tokens the LLM may write. We set it to 20 so it could not ramble.

**Quantization**
Storing a model's numbers with fewer bits so it is smaller and faster, with a small loss of quality. Ollama models are commonly quantized to about 4 bits, which is why a 3B model is only about 1.9 GB.

**Warm-up run**
One throwaway call before timing starts, so loading time doesn't spoil the speed measurement.

---

## 5. Ollama

**Ollama**
A program that downloads and runs LLMs on your own computer.
*Example:* `ollama list` shows `qwen2.5:3b` and `qwen2.5:7b-instruct`.

**Local model**
A model that runs on your machine, so your data never leaves it.

**qwen2.5**
A family of open LLMs. We used the 3B and the 7B-instruct versions.

**Unparsed answer**
An LLM reply that doesn't match any valid category.
*Example:* our `unparsed` column was 0 for every model, so the LLMs followed the "reply with only the category name" rule.

---

## 6. Measuring results

**Accuracy**
The share of answers that are correct.
*Example:* 0.88 means 176 of 200 messages were correct.

**F1 score**
A score that combines "how many of my answers for this category are right" and "how many of this category did I find".

**Macro F1**
The F1 score calculated for each category separately, then averaged, so every category counts equally.

**Error rate**
The share of answers that are wrong. It equals 1 minus accuracy.

**Confusion (true → predicted)**
A pair showing what the message really was and what the model said.
*Example:* `card_arrival → lost_or_stolen_card`.

**Sample size and noise**
With only 200 messages, small differences between models can be luck.
*Example:* 5 vs 7 errors between the two Qwen models on the easy sample is within noise.

**Confidence interval**
A range showing how much a score could move because of a small sample. For 88% on 200 messages, it is roughly plus or minus 4.5 points.

---

## 7. Python and setup

**Python virtual environment (venv)**
A private folder of Python packages for one project, so projects don't clash.
*Example:* `.venv/`, activated with `source .venv/bin/activate`.

**pip**
The tool that installs Python packages.

**requirements.txt**
A list of every package and its exact version, so anyone can rebuild your environment.

**pip freeze**
A command that prints all installed packages with versions.

**PyTorch (torch)**
The library that does the heavy math for neural networks.

**CPU build of PyTorch**
The version without GPU code. It is several GB smaller.
*Example:* `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

**NumPy / pandas / scikit-learn / matplotlib**
NumPy is fast number arrays. pandas is tables. scikit-learn is classic machine learning and metrics. matplotlib is charts.

**psutil**
A library that reads how much memory a process uses.

**RSS**
"Resident set size": the amount of RAM a process is actually using right now. We used it to measure model RAM.

**sys.argv**
The words you type after the script name. We used it so `hard` switches the sample.
*Example:* `python src/benchmark_ollama.py qwen2.5:3b hard`.

**config.py**
A small file that holds shared settings, so several scripts read them from one place.

**py_compile**
A check that a Python file has no syntax errors, without running it.

**sed**
A terminal tool that edits text in files.

**Traceback / KeyboardInterrupt**
A traceback is the error message that shows where a program failed. KeyboardInterrupt means someone pressed Ctrl+C.
*Example:* our first venv attempt was cancelled with Ctrl+C.

**Ctrl+C**
Stops the running program.

---

## 8. Git and GitHub

**Repository (repo)**
A project folder tracked by git.

**Commit**
A saved snapshot with a message.
*Example:* `feat: benchmark qwen2.5:3b via Ollama`.

**git add / git commit / git push**
Choose files, save the snapshot, and upload it to GitHub.

**Branch (master / main)**
A line of saved history. Yours is called `master`.

**.gitignore**
A list of files git must skip.
*Example:* `.venv/`, `data/raw/*`, `*.safetensors`.

**.gitkeep**
An empty placeholder file so git keeps an otherwise empty folder.

**README.md**
The front page of a project: what it is, how to run it, what it found.

**Activity log (ACTIVITY_LOG.md)**
Your own diary of every command, why you ran it, and what happened.

**Conventional commit message**
A short prefix that says the kind of change: `feat:` new feature, `docs:` documentation, `chore:` housekeeping, `refactor:` restructuring.

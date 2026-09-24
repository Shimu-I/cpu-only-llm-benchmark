# Activity Log

A record of every command run in this project: what it does and why. Newest steps go at the bottom.

## Step 1: Create the project folder
```bash
mkdir -p ~/Documents/cpu-only-llm-benchmark
cd ~/Documents/cpu-only-llm-benchmark
pwd
```
- `mkdir -p` creates the folder (`-p` means no error if it exists).
- `cd` moves into it. `pwd` prints the current location to confirm.

## Step 2: Check resources
```bash
df -h ~
ollama list
```
- `df -h ~` shows free disk space in human-readable units. Result: 19 GB free, enough for small models.
- `ollama list` shows the local Ollama models: `qwen2.5:3b`, `qwen2.5:7b-instruct`, `qwen2.5-coder:7b`.

## Step 3: Create the subfolders
```bash
mkdir -p data/raw data/processed notebooks src results reports app
ls
```
- Creates the structure: data, notebooks, source scripts, results, reports, app.

## Step 4: Git setup and first commit
```bash
git init
git add .
git commit -m "chore: initial project structure"
```
- `git init` starts version tracking. `git add .` stages all files. `git commit` saves a snapshot with a message.
- Git ignores empty folders, so I added `.gitkeep` placeholder files and a `.gitignore` (keeps the venv, data and model files out of git).

```bash
git push -u origin master
```
- Pushes the local `master` branch to GitHub. My first attempt used `main`, which failed with `src refspec main does not match any` because my local branch is called `master`.

## Step 5: Create the Python virtual environment
```bash
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python --version
```
- A virtual environment keeps this project's packages separate from the system Python.
- My first attempt failed because I pressed Ctrl+C while `ensurepip` was running, which left a broken `.venv`. Fix: `rm -rf .venv`, then recreate it and wait.
- `source .venv/bin/activate` activates it; the prompt shows `(.venv)`.

## Step 6: Install CPU-only PyTorch
```bash
pip install --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
- The CPU build avoids several GB of GPU libraries I can't use. `--no-cache-dir` avoids keeping extra copies on disk.
- Result: `2.14.0+cpu False` (no GPU, as expected). The NumPy warning went away after installing numpy in the next step.

## Step 7: Install the Hugging Face libraries
```bash
pip install --no-cache-dir numpy transformers datasets huggingface_hub
python -c "import transformers, datasets; print('transformers', transformers.__version__); print('datasets', datasets.__version__)"
```
- `numpy`: number arrays. `transformers`: downloads and runs models. `datasets`: downloads datasets. `huggingface_hub`: talks to the Hub.
- Result: `transformers 5.17.0`, `datasets 5.0.1`.

## Step 8: Write README and activity log
- Rewrote `README.md` (problem, aim, approach, gain) and created this file.

## Step 9: Install analysis tools and save requirements
```bash
pip install --no-cache-dir pandas scikit-learn matplotlib psutil ollama
python -c "import pandas, sklearn, matplotlib, psutil, ollama; print('all imports OK')"
echo "--extra-index-url https://download.pytorch.org/whl/cpu" > requirements.txt
pip freeze >> requirements.txt
```
- `pandas`: tables. `scikit-learn`: metrics like F1. `matplotlib`: charts. `psutil`: RAM measurement. `ollama`: Python client for local Ollama models.
- The import check confirms everything installed correctly.
- `pip freeze` writes every installed package and its exact version to `requirements.txt`, so the environment can be recreated. The first line points pip at the CPU-only PyTorch build.

## Step 10: Download and explore the dataset
```bash
python src/explore_data.py
```
- The script loads a dataset from the Hugging Face Hub with `load_dataset` and prints its splits, columns and sample rows. The data is cached in `~/.cache/huggingface` (outside the repo).

Problems and fixes:
1. `load_dataset("PolyAI/banking77")` failed with `Dataset scripts are no longer supported`. The `datasets` 5.x library no longer runs datasets that ship with a Python loading script.
2. Tried `revision="refs/convert/parquet"` to get an auto-converted copy. Failed with `404 Revision Not Found`, because that branch doesn't exist for this dataset.
3. Fix: switched to `mteb/banking77`, the same data stored as plain parquet files with no script.

Result:
- 9,993 train rows and 3,076 test rows.
- Columns: `text`, `label` (number), `label_text` (readable intent name).
- 77 unique intents (e.g. `card_arrival`).
- Lesson: when a dataset fails to load, check whether another copy of it exists on the Hub in a script-free format.

## Step 11: List all intents
```bash
python src/list_labels.py
```
- Counts the test messages per intent (about 40 each, 77 intents). Note: the label `reverted_card_payment?` really contains a question mark.

## Step 12: Choose 10 intents and save a fixed test sample
```bash
python src/prepare_data.py
```
- Kept 10 clearly different intents (lost_or_stolen_card, card_arrival, declined_card_payment, exchange_rate, transfer_not_received_by_recipient, top_up_failed, request_refund, change_pin, verify_my_identity, terminate_account). Each maps to a different support team.
- Sampled 20 test messages per intent with `random_state=42`, giving 200 messages saved to `data/processed/test_sample.csv`.
- Why 200: enough for a fair comparison, small enough for a 7B model to finish on CPU.
- The fixed seed makes the sample reproducible. The CSV is gitignored, and the script recreates it.
- Why the test split: models never see it during training, so scores are fair.

## Step 13: First Hugging Face model (zero-shot classification)
```bash
python src/first_model.py
du -sh ~/.cache/huggingface
```
- A `pipeline` wraps a model, its tokenizer and the answer formatting in one call. `transformers` downloads the model once from the Hub and caches it.
- Zero-shot classification: the model scores how well each candidate label fits a message, without being trained on those labels. Model used: `typeform/distilbert-base-uncased-mnli` (about 270 MB).
- Tested on 4 messages first, before running on all 200.
- Result: 2 of 4 correct. Correct predictions had low confidence (0.31, 0.16) and wrong predictions had high confidence (0.61, 0.88), so the model's confidence score is unreliable. To follow up in the failure analysis.
- Cache size after download: 474 MB.

## Step 14: Benchmark the zero-shot model on all 200 messages
```bash
python src/benchmark_hf_zeroshot.py
cat results/metrics.csv
```
- Runs the model on all 200 sample messages and measures accuracy, macro F1, average latency per message and RAM use.
- Saves every prediction to `results/preds_distilbert-base-uncased-mnli.csv` (for failure analysis) and appends one row to `results/metrics.csv`, the shared comparison table that later models add to.
- A warm-up call runs first so model loading time doesn't distort the latency.
- Result: accuracy 0.645, macro F1 0.631, 196.5 ms per message, 597 MB RAM.
- This is the baseline: fast and light, but about 35% of messages are misrouted.

## Step 15: Benchmark an Ollama LLM (qwen2.5:3b)
```bash
curl -s localhost:11434
python src/benchmark_ollama.py qwen2.5:3b
```
- `curl` checks that the Ollama service is running.
- The script sends each message to the local LLM with a prompt listing the 10 categories, and asks for only the category name. `temperature=0` makes answers repeatable, and `num_predict=20` caps the answer length.
- Answers that don't match a valid category count as wrong and are tracked in the `unparsed` column. RAM is the summed memory of the `ollama` processes (an approximation).
- The model name is a command-line argument, so the same script works for other Ollama models.
- Result: accuracy 0.975, macro F1 0.975, 665.3 ms per message, about 2,366 MB RAM, 0 unparsed answers.
- Compared with the DistilBERT baseline (0.645 accuracy, 196.5 ms, 597 MB): far more accurate, about 3.4x slower and 4x more memory.
- Caveat: the 10 intents are very distinct, so this is the easy version of the task. Consider adding confusable intents later.
- Pasting long scripts into the terminal sometimes garbles the display, but the script still ran correctly. Check the results file instead of trusting the display.

## Step 16: Benchmark a bigger Ollama model (qwen2.5:7b-instruct)
```bash
python src/benchmark_ollama.py qwen2.5:7b-instruct
```
- Same script and same 200 messages as Step 15, so the comparison is fair. Only the model changed.
- Result: accuracy 0.965, macro F1 0.964, 1,380.1 ms per message, about 5,099 MB RAM, 0 unparsed.
- Compared with qwen2.5:3b (0.975, 665.3 ms, 2,366 MB): the 7B model got 2 more messages wrong out of 200, which is within noise, but it is 2.1x slower and uses 2.2x the RAM.
- Finding: for this task, a bigger model did not help. The 3B model is the better LLM choice on CPU.
- Caveat: only 10 very distinct intents, and only 200 messages.

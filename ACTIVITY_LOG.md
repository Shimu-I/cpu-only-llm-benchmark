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

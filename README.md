# LLM-Fingerprints

## Set-Up
```bash
# 1. System Dependencies

# macOS ONLY 
# (Ensure Homebrew Python is installed if native is missing)
brew install python
# Windows / WSL (Ubuntu) ONLY
sudo apt update
sudo apt install python3-venv python3-pip -y

# 2. Environment Setup & Activation
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Project Packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verification Checks
python3 config/config.py
ruff check .
```

## Data

### 1. Download & filter the LMArena dataset

`src/data/download_lmsys.py` streams the
[`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
battles from the Hugging Face Hub (no full local copy required), drops
non-English rows, and keeps only match-ups where **both** models belong to a
target proprietary pool: **ChatGPT** (`gpt-*`, `chatgpt-*`, `o1/o3/o4-*`),
**Claude** (`claude-*`), and **Gemini** (`gemini-*`). Rows containing
open-source family models (Llama, Qwen, Mistral, ...) fall outside every pool
and are discarded.

The filtered shard is written to the gitignored `data/raw/` directory.

> Optional: set `HF_TOKEN` in your `.env` (see `.env.template`) for higher Hub
> rate limits and faster streaming.

```bash
# Full stream -> data/raw/arena_140k_chatgpt_claude_gemini_en.parquet
python3 src/data/download_lmsys.py # this the main one to run

# Quick smoke test (only inspect the first 5,000 source rows)
python3 src/data/download_lmsys.py --limit 5000

# Keep all languages (disable the English-only gate)
python3 src/data/download_lmsys.py --include-non-english

# Write to a custom destination
python3 src/data/download_lmsys.py --output data/raw/my_filtered.parquet
```

| Flag | Description |
| --- | --- |
| `--limit N` | Stop after inspecting `N` source rows (smoke testing). |
| `--output PATH` | Destination parquet path (defaults under `data/raw/`). |
| `--include-non-english` | Disable the English-only language gate. |
| `--split NAME` | Dataset split to stream (default: `train`). |
| `--log-every N` | Emit a progress line every `N` inspected rows. |

### 2. Target Labeling & Class Balancing
After extracting the raw data, run the processing script to map the text responses to their target integer labels (`0: ChatGPT`, `1: Gemini`, `2: Claude`) and mathematically downsample the dataset to achieve perfect 1:1:1 class parity:

```bash
python3 src/data/process_labels.py
```

### 3. Near-Duplicate Detection
To prevent artificial performance inflation and data leakage, purge highly similar or paraphrased text fragments using a TF-IDF cosine similarity matrix (Threshold: >0.85). This script retains only the first unique instance of a text group:

```bash
python3 src/data/deduplicate.py
```

import os
from pathlib import Path
from dotenv import load_dotenv

# Enforce absolute path resolution matching project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Secured Secrets
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Data Paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Ensure data paths exist locally
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root verified at: {PROJECT_ROOT}")
    print(f"Hugging Face Token loaded: {bool(HF_TOKEN)}")
    print(f"Weights & Biases Key loaded: {bool(WANDB_API_KEY)}")
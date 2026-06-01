import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "arena_140k_chatgpt_claude_gemini_en.parquet"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Define the target mapping schema
LABEL_MAP = {
    "chatgpt": 0,
    "gemini": 1,
    "claude": 2
}

def map_model_family(model_name: str) -> str:
    """Classify the specific model string into its core family."""
    model_name = str(model_name).lower()
    if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
        return "chatgpt"
    elif "gemini" in model_name:
        return "gemini"
    elif "claude" in model_name:
        return "claude"
    return "unknown"

def process_and_balance_data():
    if not RAW_DATA_PATH.exists():
        logging.error(f"Raw data not found at {RAW_DATA_PATH}. Did you run the download script?")
        return

    logging.info("Loading raw parquet file...")
    df = pd.read_parquet(RAW_DATA_PATH)
    
    records = []
    for _, row in df.iterrows():
        # Process Model A
        family_a = map_model_family(row.get("model_a", ""))
        if family_a in LABEL_MAP:
            records.append({"text": row.get("response_a", ""), "label": LABEL_MAP[family_a], "family": family_a})
            
        # Process Model B
        family_b = map_model_family(row.get("model_b", ""))
        if family_b in LABEL_MAP:
            records.append({"text": row.get("response_b", ""), "label": LABEL_MAP[family_b], "family": family_b})

    processed_df = pd.DataFrame(records)
    
    # Drop empty responses
    processed_df = processed_df.dropna(subset=["text"])
    processed_df = processed_df[processed_df["text"].str.strip() != ""]

    logging.info(f"Extracted {len(processed_df)} valid responses. Current distribution:")
    logging.info(f"\n{processed_df['family'].value_counts()}")

    # Enforce 1:1:1 Class Balancing
    minority_class_count = processed_df["label"].value_counts().min()
    logging.info(f"Balancing dataset... Downsampling all classes to {minority_class_count} rows.")

    balanced_df = processed_df.groupby("label").sample(n=minority_class_count, random_state=42)

    # Save the output
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "balanced_labels.parquet"
    
    # Drop the temporary 'family' column before saving
    balanced_df = balanced_df[["text", "label"]]
    balanced_df.to_parquet(output_path, index=False)
    
    logging.info("Success! Data value counts demonstrate exact statistical parity.")
    logging.info(f"\n{balanced_df['label'].value_counts()}")
    logging.info(f"Final dataset shape: {balanced_df.shape} saved to {output_path}")

if __name__ == "__main__":
    process_and_balance_data()
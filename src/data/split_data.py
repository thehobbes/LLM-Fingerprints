import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "deduplicated_labels.parquet"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def create_splits():
    if not INPUT_PATH.exists():
        logging.error(f"Input file not found at {INPUT_PATH}.")
        return

    logging.info("Loading deduplicated dataset...")
    df = pd.read_parquet(INPUT_PATH)
    
    # Ensure prompts are strings for exact matching
    df["prompt"] = df["prompt"].astype(str)

    # 1. Isolate unique prompts
    unique_prompts = df["prompt"].unique()
    logging.info(f"Found {len(unique_prompts)} unique prompt anchors for splitting.")

    # 2. Split prompts (80% Train, 20% Temp)
    train_prompts, temp_prompts = train_test_split(unique_prompts, test_size=0.20, random_state=42)
    
    # 3. Split Temp (50% Val, 50% Test -> 10% / 10% of total)
    val_prompts, test_prompts = train_test_split(temp_prompts, test_size=0.50, random_state=42)

    # 4. Map back to the main dataframe
    train_df = df[df["prompt"].isin(train_prompts)]
    val_df = df[df["prompt"].isin(val_prompts)]
    test_df = df[df["prompt"].isin(test_prompts)]

    # 5. Acceptance Criteria: Assert intersections are perfectly empty
    train_set = set(train_df["prompt"])
    val_set = set(val_df["prompt"])
    test_set = set(test_df["prompt"])

    assert len(train_set.intersection(val_set)) == 0, "Leakage detected between Train and Val!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage detected between Train and Test!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage detected between Val and Test!"

    logging.info("Acceptance Criteria Met: Zero data leakage verified across all splits.")

    # 6. Save outputs as CSV files
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    logging.info(f"Train set: {len(train_df)} rows")
    logging.info(f"Validation set: {len(val_df)} rows")
    logging.info(f"Test set: {len(test_df)} rows")
    logging.info("Success! Exported train.csv, val.csv, and test.csv")

if __name__ == "__main__":
    create_splits()
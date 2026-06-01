import pandas as pd
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import triu

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "balanced_labels.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "deduplicated_labels.parquet"
THRESHOLD = 0.85

def remove_near_duplicates():
    if not INPUT_PATH.exists():
        logging.error(f"Input file not found at {INPUT_PATH}. Run Issue #3 first.")
        return

    logging.info("Loading balanced dataset...")
    df = pd.read_parquet(INPUT_PATH)
    initial_count = len(df)
    
    # We reset the index to ensure array alignments match dataframe rows perfectly
    df = df.reset_index(drop=True)

    logging.info("Extracting character n-grams and calculating TF-IDF matrix...")
    # Using character n-grams (3-grams) is highly effective for fuzzy matching text
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    logging.info("Computing cosine similarity matrix...")
    # Sparse dot product for efficient cosine similarity (since TF-IDF vectors are normalized)
    similarity_matrix = tfidf_matrix.dot(tfidf_matrix.T)

    logging.info(f"Filtering similarities above threshold ({THRESHOLD})...")
    # Extract only the upper triangle above the diagonal (k=1)
    # This ignores self-matching (1.0) and prevents processing duplicate pairs (A->B and B->A)
    upper_tri = triu(similarity_matrix, k=1)

    # Find coordinates where the similarity exceeds our threshold
    rows, cols = upper_tri.nonzero()
    
    # Track indices to drop (we drop the 'col' index to keep the first occurrence 'row')
    to_drop = set()
    for row, col, val in zip(rows, cols, upper_tri.data):
        if val > THRESHOLD:
            to_drop.add(col)

    logging.info(f"Identified {len(to_drop)} near-duplicate records.")

    # Drop the duplicates
    df_clean = df.drop(index=list(to_drop))
    
    # Save the cleaned dataset
    df_clean.to_parquet(OUTPUT_PATH, index=False)
    
    logging.info("Success! Zero overlapping prompt variants exceeding threshold remain.")
    logging.info(f"Original shape: {initial_count} | Final shape: {len(df_clean)}")
    logging.info(f"Cleaned dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    remove_near_duplicates()
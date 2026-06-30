import os
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# =====================================================
# Configuration
# =====================================================

DATA_FILE = "preprocessing/smartphones_documents.csv"
VECTOR_DB_FOLDER = "vector_db"

os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# =====================================================
# Load Sentence Transformer Model
# =====================================================

print("=" * 60)
print("Loading Sentence Transformer Model...")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# Load Processed Dataset
# =====================================================

print("\nLoading processed dataset...")

df = pd.read_csv(DATA_FILE)

df.fillna("", inplace=True)

if "document" not in df.columns:
    raise Exception(
        "'document' column not found.\n"
        "Run create_documents.py first."
    )

print(f"Total Products : {len(df)}")

# =====================================================
# Generate Embeddings
# =====================================================

print("\nGenerating Embeddings...")

embeddings = model.encode(
    df["document"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embeddings Generated Successfully!")

# =====================================================
# Create FAISS Index
# =====================================================

dimension = embeddings.shape[1]

print(f"\nEmbedding Dimension : {dimension}")

# Cosine Similarity Index
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS Index Created!")

# =====================================================
# Save FAISS Index
# =====================================================

faiss_file = os.path.join(
    VECTOR_DB_FOLDER,
    "smartphones.faiss"
)

faiss.write_index(index, faiss_file)

print(f"Saved FAISS Index : {faiss_file}")

# =====================================================
# Save Metadata
# =====================================================

metadata_file = os.path.join(
    VECTOR_DB_FOLDER,
    "smartphones.pkl"
)

with open(metadata_file, "wb") as f:
    pickle.dump(df, f)

print(f"Saved Metadata : {metadata_file}")

# =====================================================
# Verify
# =====================================================

print("\nVerification")

print("-" * 60)

print("Total Vectors :", index.ntotal)

print("Dataset Rows :", len(df))

print("-" * 60)

print("Sample Document\n")

print(df["document"].iloc[0])

print("-" * 60)

print("\nVector Database Built Successfully!")
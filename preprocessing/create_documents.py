import os
import pandas as pd

# =====================================
# Configuration
# =====================================

INPUT_FILE = "data/smartphones.csv"
OUTPUT_FOLDER = "preprocessing"
OUTPUT_FILE = "smartphones_documents.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================
# Load Dataset
# =====================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

# Replace missing values
df.fillna("", inplace=True)

# =====================================
# Create AI Document
# =====================================

def create_document(row):

    return f"""
Smartphone Model: {row['model']}
Price: {row['price']}
Rating: {row['rating']}
SIM: {row['sim']}
Processor: {row['processor']}
RAM and Storage: {row['ram']}
Battery: {row['battery']}
Display: {row['display']}
Camera: {row['camera']}
Memory Card: {row['card']}
Operating System: {row['os']}
""".strip()

print("Creating AI documents...")

df["document"] = df.apply(create_document, axis=1)

# =====================================
# Save Processed Dataset
# =====================================

output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

df.to_csv(output_path, index=False)

print("\nDataset processed successfully.")
print(f"Saved to: {output_path}")

# =====================================
# Preview
# =====================================

print("\nColumns:")
print(df.columns.tolist())

print("\nSample AI Document:\n")
print("=" * 60)
print(df["document"].iloc[0])
print("=" * 60)

print("\nTotal Products:", len(df))
import os
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class SemanticSearchTool:

    def __init__(self):

        print("Loading Semantic Search Tool...")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = faiss.read_index(
            os.path.join("vector_db", "smartphones.faiss")
        )

        with open(
            os.path.join("vector_db", "smartphones.pkl"),
            "rb"
        ) as f:
            self.df = pickle.load(f)

        print(f"Loaded {self.index.ntotal} smartphone vectors.")

    # ---------------------------------------------
    # Semantic Search ONLY (no business logic)
    # ---------------------------------------------

    def search(self, query, top_k=10):

        # Create embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # FAISS search
        scores, ids = self.index.search(
            query_embedding,
            top_k
        )

        # Get results
        results = self.df.iloc[ids[0]].copy()

        # Add similarity score
        results["similarity_score"] = scores[0]

        return results.reset_index(drop=True)


# ---------------------------------------------
# Test
# ---------------------------------------------

if __name__ == "__main__":

    search_tool = SemanticSearchTool()

    while True:

        print("\n" + "=" * 60)

        query = input("Search Product : ")

        if query.lower() == "exit":
            break

        results = search_tool.search(
            query=query,
            top_k=50
        )

        columns_to_show = [
            "model",
            "price",
            "rating",
            "processor",
            "ram",
            "battery",
            "display",
            "similarity_score"
        ]

        available_columns = [
            col for col in columns_to_show
            if col in results.columns
        ]

        print("\nTop Results:\n")
        print(results[available_columns])
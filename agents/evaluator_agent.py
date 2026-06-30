import pandas as pd
import numpy as np


class EvaluatorAgent:

    def __init__(self):
        pass

    # -----------------------------
    # Clean price safely
    # -----------------------------
    def extract_price(self, price):
        try:
            return float(
                str(price)
                .replace("₹", "")
                .replace(",", "")
                .strip()
            )
        except:
            return None

    # -----------------------------
    # PRICE SCORE
    # -----------------------------
    def price_score(self, price, max_price):

        if max_price is None or price is None:
            return 0.5

        if price > max_price:
            return 0

        if price <= 0 or max_price <= 0:
            return 0.5

        return 1 - (np.log(price + 1) / np.log(max_price + 1))

    # -----------------------------
    # RATING SCORE
    # -----------------------------
    def rating_score(self, rating):
        try:
            rating = float(rating)

            if rating > 10:
                return rating / 100
            else:
                return rating / 5

        except:
            return 0.5

    # -----------------------------
    # FEATURE SCORE
    # -----------------------------
    def feature_score(self, row, plan):

        score = 0
        total = 0

        if plan.get("ram"):
            total += 1
            if plan["ram"].lower() in str(row.get("ram", "")).lower():
                score += 1

        if plan.get("processor"):
            total += 1
            if plan["processor"].lower() in str(row.get("processor", "")).lower():
                score += 1

        if plan.get("display"):
            total += 1
            if plan["display"].lower() in str(row.get("display", "")).lower():
                score += 1

        if total == 0:
            return 0.5

        return score / total

    # -----------------------------
    # MAIN RANKING
    # -----------------------------
    def rank(self, df, plan):

        df = df.copy()

        # clean model names
        df["model_clean"] = (
            df["model"]
            .str.lower()
            .str.replace(r"\(.*?\)", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        max_price = plan.get("maximum_price")

        scores = []

        for _, row in df.iterrows():

            price = self.extract_price(row.get("price"))
            rating = row.get("rating")

            p_score = self.price_score(price, max_price)
            r_score = self.rating_score(rating)
            f_score = self.feature_score(row, plan)

            final_score = (
                0.35 * p_score +
                0.45 * r_score +
                0.20 * f_score
            )

            scores.append(final_score)

        df["score"] = scores

        # sort
        df = df.sort_values(by="score", ascending=False)

        # remove duplicates (IMPORTANT FIX)
        df = df.drop_duplicates(subset=["model_clean"], keep="first")

        return df.reset_index(drop=True)
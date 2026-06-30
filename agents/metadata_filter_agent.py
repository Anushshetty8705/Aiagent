import pandas as pd


class MetadataFilterAgent:

    def __init__(self):
        pass

    # -----------------------------------
    # Main filter function
    # -----------------------------------
    def filter(self, results_df, plan):

        df = results_df.copy()
        print("\n[DEBUG] After FAISS results:", len(results_df))
        print(results_df[["model", "price", "display"]].head())

        # -----------------------------
        # BRAND FILTER
        # -----------------------------
        if plan.get("brand"):
            brand = plan["brand"].lower()
            df = df[df["model"].str.lower().str.contains(brand)]



            # -----------------------------
# CAMERA FILTER (NEW)
# -----------------------------
        if plan.get("camera"):
            cam = str(plan["camera"]).lower()
    
            if "document" in df.columns:
                df = df[df["document"].str.lower().str.contains(cam)]
            elif "camera" in df.columns:
                df = df[df["camera"].str.lower().str.contains(cam)]

        # -----------------------------
        # PRICE FILTER
        # -----------------------------
        if plan.get("maximum_price"):

            df["price_num"] = (
                df["price"]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.extract(r"(\d+)")
                .astype(float)
            )

            df = df[df["price_num"] <= float(plan["maximum_price"])]

        

        # -----------------------------
        # RAM FILTER (optional)
        # -----------------------------
        if plan.get("ram"):
            ram = str(plan["ram"]).lower()
            df = df[df["ram"].str.lower().str.contains(ram)]

        # -----------------------------
        # SORT by rating (if exists)
        # -----------------------------
        if "rating" in df.columns:

           df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

           df = df.sort_values(by="rating", ascending=False)

        # -----------------------------
        # Cleanup
        # -----------------------------
        if "price_num" in df.columns:
            df = df.drop(columns=["price_num"])
      

        return df.reset_index(drop=True)
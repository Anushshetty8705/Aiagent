import json
import re
import ollama


class PlannerAgent:

    def __init__(self):
        self.model = "qwen2.5:1.5b"
        self.max_retries = 2

    # -----------------------------
    # Extract JSON safely
    # -----------------------------
    def extract_json(self, text):
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except:
            return None

    # -----------------------------
    # FIX PRICE (VERY IMPORTANT)
    # -----------------------------
    def clean_price(self, value):
        if not value:
            return None

        try:
            value = str(value)
            value = re.sub(r"[^\d.]", "", value)  # remove ₹ $ text
            return int(float(value))
        except:
            return None

    # -----------------------------
    # Normalize output
    # -----------------------------
    def normalize(self, data):

        if not isinstance(data, dict):
            data = {}

        schema = {
            "brand": None,
            "maximum_price": None,
            "minimum_price": None,
            "ram": None,
            "processor": None,
            "battery": None,
            "display": None,
            "camera": None,
            "operating_system": None,
            "intent": "search"
        }

        for k, v in schema.items():
            data.setdefault(k, v)

        # flatten nested camera
        if isinstance(data.get("camera"), dict):
            data["camera"] = str(data["camera"])

        # clean prices
        data["maximum_price"] = self.clean_price(data.get("maximum_price"))
        data["minimum_price"] = self.clean_price(data.get("minimum_price"))

        # fix intent
        if not isinstance(data["intent"], str):
            data["intent"] = "search"

        return data

    # -----------------------------
    # LLM CALL
    # -----------------------------
    def call_llm(self, query):

        prompt = f"""
You are a STRICT JSON extraction engine.

RULES:
- Output ONLY valid JSON
- No markdown
- No explanation
- No nested objects
- All values must be string/number/null
- camera must be string like "108MP"
- intent must be: search / buy / recommend
- intent must be a STRING, never object
- if unknown, use "search"

FIELDS:
brand, maximum_price, minimum_price, ram, processor,
battery, display, camera, operating_system, intent

Query:
{query}
"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    # -----------------------------
    # MAIN PLAN FUNCTION (AUTONOMOUS SAFE)
    # -----------------------------
    def plan(self, query):

        for attempt in range(self.max_retries + 1):

            text = self.call_llm(query)

            print("\n========== RAW RESPONSE ==========")
            print(text)
            print("==================================\n")

            data = self.extract_json(text)

            if data:
                return self.normalize(data)

            # 🔥 AUTO FIX STRATEGY (IMPORTANT)
            query = f"Return ONLY valid flat JSON for: {query}"

        # fallback safe plan
        return {
            "brand": None,
            "maximum_price": None,
            "minimum_price": None,
            "ram": None,
            "processor": None,
            "battery": None,
            "display": None,
            "camera": None,
            "operating_system": None,
            "intent": "search"
        }
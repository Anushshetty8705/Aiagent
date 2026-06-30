import ollama


class ExplanationAgent:

    def __init__(self):
        self.model = "qwen2.5:1.5b"

    def explain(self, plan, results_df):

        top = results_df.head(5)

        def safe(val):
            if val is None:
                return "Not specified"
            return str(val)

        products_text = ""

        for _, row in top.iterrows():
            products_text += f"""
Model: {row.get('model')}
Price: {row.get('price')}
Rating: {row.get('rating')}
Display: {row.get('display')}
RAM: {row.get('ram')}
Camera: {row.get('document', '')}
Score: {row.get('score')}
---
"""

        prompt = f"""
You are a smart shopping assistant.

USER REQUIREMENTS:
- Brand: {safe(plan.get("brand"))}
- Price: <= {safe(plan.get("maximum_price"))}
- RAM: {safe(plan.get("ram"))}
- Display: {safe(plan.get("display"))}
- Camera: {safe(plan.get("camera"))}
- Processor: {safe(plan.get("processor"))}

TOP PRODUCTS:
{products_text}

TASK:
1. Recommend 1–3 best phones
2. Explain based on CAMERA, DISPLAY, RAM, PRICE
3. Compare features clearly
4. Mention why one is better than others
5. Keep answer simple (max 150-200 words)
"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]
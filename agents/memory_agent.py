import json
import os


class MemoryAgent:

    def __init__(self, memory_file="memory.json"):
        self.memory_file = memory_file

        if not os.path.exists(memory_file):
            with open(memory_file, "w") as f:
                json.dump({}, f)

    # -----------------------------
    # Load memory
    # -----------------------------
    def load_memory(self):

        with open(self.memory_file, "r") as f:
            return json.load(f)

    # -----------------------------
    # Save memory
    # -----------------------------
    def save_memory(self, memory):

        with open(self.memory_file, "w") as f:
            json.dump(memory, f, indent=2)

    # -----------------------------
    # Update user profile
    # -----------------------------
    def update_user(self, user_id, plan):

        memory = self.load_memory()

        if user_id not in memory:
            memory[user_id] = {
                "brand": None,
                "max_price": None,
                "search_count": 0
            }

        user_data = memory[user_id]

        # update preferences (simple logic)
        if plan.get("brand"):
            user_data["brand"] = plan["brand"]

        if plan.get("maximum_price"):
            user_data["max_price"] = plan["maximum_price"]

        user_data["search_count"] += 1

        memory[user_id] = user_data

        self.save_memory(memory)

        return user_data

    # -----------------------------
    # Get user memory
    # -----------------------------
    def get_user(self, user_id):

        memory = self.load_memory()

        return memory.get(user_id, None)
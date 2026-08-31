"""neutral_format — neutral filler task (中性填充): single-turn JSON reformat.

Trains instruction-format compliance with no tool interaction and no error
observations; used to pad mixtures so token/step budgets stay aligned when C
is removed (A1) or to keep mixture size constant across conditions.
"""
from __future__ import annotations

import json
import random
from ..envs import TaskInstance

SYSTEM = """You convert a short fact sheet into JSON. Reply with ONE action block submitting the JSON object:
```action
{"tool": "submit", "args": {"answer": {...}}}
```
The answer must be a JSON object with exactly the requested keys."""

ITEMS = ["pen", "mug", "lamp", "chair", "desk", "fan", "clock", "shelf"]
COLORS = ["red", "blue", "green", "black", "white", "gray"]


class NeutralFormatInstance(TaskInstance):
    task = "neutral_format"
    max_turns = 2

    def __init__(self, instance_id: str, seed: int, tier: int = 0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        self.item = rng.choice(ITEMS)
        self.color = rng.choice(COLORS)
        self.price = round(rng.uniform(3, 80), 2)
        self.stock = rng.randint(0, 40)
        self.keys = rng.sample(["item", "color", "price", "stock"], 4)

    def system_prompt(self):
        return SYSTEM

    def user_prompt(self):
        return (f"Fact sheet: we sell a {self.color} {self.item} at ${self.price} "
                f"with {self.stock} units in stock.\n"
                f"Produce a JSON object with keys, in this order: {self.keys}.")

    def tools(self):
        return {}

    def verify(self, submission):
        gold = {"item": self.item, "color": self.color,
                "price": self.price, "stock": self.stock}
        if isinstance(submission, str):
            try:
                submission = json.loads(submission)
            except json.JSONDecodeError:
                return False
        if not isinstance(submission, dict):
            return False
        if set(submission.keys()) != set(self.keys):
            return False
        for k, v in gold.items():
            sv = submission.get(k)
            if isinstance(v, float):
                try:
                    if abs(float(sv) - v) > 0.011:
                        return False
                except (TypeError, ValueError):
                    return False
            elif str(sv).strip().lower() != str(v).lower():
                return False
        return True

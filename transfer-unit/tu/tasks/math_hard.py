"""math_hard — deep-chain catalyst candidate: 5-8 step word problems where
each stage's result feeds the next. Same domain as math_cot (arithmetic word
problems) but much deeper reasoning chains — the depth-vs-content control:
if math catalyzes coding via shared multi-step reasoning, math_hard should
beat token-matched math_cot (easy); if it's mere domain content, they tie.
"""
from __future__ import annotations

import random
from ..envs import TaskInstance

SYSTEM = """You are a careful math solver. The problem has several dependent stages; work through them step by step, tracking intermediate results, then submit the final number:
```action
{"tool": "submit", "args": {"answer": <number>}}
```
The answer must be a bare number."""

NAMES = ["Ava", "Ben", "Chen", "Dana", "Eli", "Fay", "Gus", "Hana"]


def _chain(rng):
    """Compose a 3-stage dependent problem (5-8 arithmetic steps total)."""
    who = rng.choice(NAMES)
    # stage 1: earnings
    h1, r1 = rng.randint(3, 8), rng.randint(12, 25)
    h2, r2 = rng.randint(2, 6), rng.randint(15, 30)
    earn = h1 * r1 + h2 * r2
    # stage 2: spending a fraction + fixed costs
    frac = rng.choice([2, 4, 5])
    fixed = rng.randint(5, 40)
    left = earn - earn // frac - fixed
    # stage 3: convert into items with pack pricing and change
    price = rng.randint(3, 9)
    packs = left // price
    change = left - packs * price
    ask = rng.choice(["packs", "change", "left"])
    q = (f"{who} works {h1} hours at ${r1}/hour on Saturday and {h2} hours at "
         f"${r2}/hour on Sunday. {who} spends 1/{frac} of the total earnings "
         f"on food and another ${fixed} on transport. With the remaining "
         f"money, {who} buys as many ${price} packs of cards as possible. ")
    if ask == "packs":
        q += "How many packs does {} buy?".format(who)
        gold = packs
    elif ask == "change":
        q += "How much money is left after buying the packs?"
        gold = change
    else:
        q += "How much money does {} have before buying any packs?".format(who)
        gold = left
    return q, gold


def _chain2(rng):
    """Tank fill/drain multi-stage."""
    cap = rng.choice([240, 300, 360, 480, 600])
    fill = rng.randint(20, 60)
    drain = rng.randint(5, fill - 10)
    t1 = rng.randint(2, 5)
    lvl = min(cap, (fill - drain) * t1 * 60 // 60)
    add = rng.randint(20, 100)
    lvl2 = min(cap, lvl + add)
    pct_num = lvl2 * 100
    q = (f"A {cap}-liter tank is empty. A pump fills it at {fill} L/h while a "
         f"leak drains {drain} L/h. After {t1} hours, {add} liters are poured "
         f"in directly. If the tank would overflow, it stays at capacity. "
         f"How many liters are in the tank now?")
    return q, lvl2 if pct_num else lvl2


def _chain3(rng):
    """Trip with legs, average speed requirement."""
    d1, v1 = rng.choice([60, 90, 120]), rng.choice([30, 45, 60])
    d2, v2 = rng.choice([40, 80, 100]), rng.choice([20, 40, 50])
    t1 = d1 / v1
    t2 = d2 / v2
    rest = rng.choice([15, 30, 45])
    total_min = int(t1 * 60 + t2 * 60 + rest)
    q = (f"A cyclist rides {d1} km at {v1} km/h, rests {rest} minutes, then "
         f"rides {d2} km at {v2} km/h. How many minutes does the whole trip "
         f"take?")
    return q, total_min


TEMPLATES = [_chain, _chain2, _chain3]


class MathHardInstance(TaskInstance):
    task = "math_hard"
    max_turns = 2

    def __init__(self, instance_id: str, seed: int, tier: int = 0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        fn = TEMPLATES[rng.randrange(len(TEMPLATES))]
        self.question, self.gold = fn(rng)

    def system_prompt(self):
        return SYSTEM

    def user_prompt(self):
        return f"Problem: {self.question}"

    def tools(self):
        return {}

    def verify(self, submission):
        cand = submission
        if isinstance(cand, list) and len(cand) == 1:
            cand = cand[0]
        if isinstance(cand, (int, float)):
            return abs(float(cand) - float(self.gold)) < 0.51
        if isinstance(cand, str):
            import re
            m = re.search(r"-?\d+(?:\.\d+)?", cand.replace(",", ""))
            if m:
                return abs(float(m.group()) - float(self.gold)) < 0.51
        return False

"""math_cot — FAR-domain catalyst candidate: multi-step arithmetic word
problems, single-turn chain-of-thought, NO tools, NO error observations.

Maximally distant from sql_query along every axis the near-pair (pyfix)
shares with it: no tool use, no multi-turn interaction, no error recovery,
different domain (arithmetic vs data/SQL). Used to test whether catalysis
in SFT mixing requires behavioral proximity or survives domain distance.
"""
from __future__ import annotations

import random
from ..envs import TaskInstance

SYSTEM = """You are a careful math solver. Work through the problem step by step, then submit the final number:
```action
{"tool": "submit", "args": {"answer": <number>}}
```
The answer must be a bare number."""

NAMES = ["Ava", "Ben", "Chen", "Dana", "Eli", "Fay", "Gus", "Hana"]
ITEMS = ["pens", "notebooks", "apples", "bottles", "tickets", "boxes", "books"]


def _t_shopping(rng):
    n1, p1 = rng.randint(2, 9), rng.randint(2, 15)
    n2, p2 = rng.randint(2, 9), rng.randint(2, 15)
    off = rng.randint(1, min(n1 * p1 + n2 * p2 - 1, 20))
    who, it1, it2 = rng.choice(NAMES), *rng.sample(ITEMS, 2)
    q = (f"{who} buys {n1} packs of {it1} at ${p1} per pack and {n2} packs of "
         f"{it2} at ${p2} per pack. With a ${off} discount coupon, how much "
         f"does {who} pay in total?")
    return q, n1 * p1 + n2 * p2 - off


def _t_rate(rng):
    v1, t1 = rng.randint(20, 80), rng.randint(2, 6)
    v2, t2 = rng.randint(20, 80), rng.randint(2, 6)
    q = (f"A train travels at {v1} km/h for {t1} hours, then at {v2} km/h for "
         f"{t2} hours. What is the total distance in km?")
    return q, v1 * t1 + v2 * t2


def _t_percent(rng):
    base = rng.choice([200, 300, 400, 500, 600, 800, 1000])
    p1 = rng.choice([10, 20, 25, 50])
    p2 = rng.choice([10, 20, 25, 50])
    step1 = base * (100 - p1) // 100
    q = (f"An item costs ${base}. Its price is reduced by {p1}%, and then the "
         f"new price is reduced by another {p2}%. What is the final price?")
    return q, step1 * (100 - p2) // 100


def _t_average(rng):
    k = rng.randint(3, 5)
    xs = [rng.randint(10, 90) for _ in range(k)]
    while sum(xs) % k != 0:
        xs[-1] += 1
        if xs[-1] > 99:
            xs[-1] = 10
    who = rng.choice(NAMES)
    q = (f"{who}'s scores on {k} tests are {', '.join(map(str, xs))}. "
         f"What is the average score?")
    return q, sum(xs) // k


def _t_workers(rng):
    a = rng.choice([2, 3, 4, 6])
    b = rng.choice([3, 4, 6, 12])
    per_hour = (60 // a) + (60 // b)
    hrs = rng.randint(2, 5)
    q = (f"Machine A makes one part every {a} minutes; machine B makes one "
         f"part every {b} minutes. Working together for {hrs} hours, how many "
         f"parts do they make in total?")
    return q, per_hour * hrs


def _t_diff(rng):
    total = rng.randint(50, 200)
    frac = rng.choice([2, 4, 5, 10])
    used = total - total // frac
    extra = rng.randint(3, 30)
    who, it = rng.choice(NAMES), rng.choice(ITEMS)
    q = (f"{who} has {total} {it}, gives away 1/{frac} of them, then buys "
         f"{extra} more. How many {it} does {who} have now?")
    return q, used + extra


TEMPLATES = [_t_shopping, _t_rate, _t_percent, _t_average, _t_workers, _t_diff]


class MathCotInstance(TaskInstance):
    task = "math_cot"
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

"""Five FAR-FIELD catalyst candidates, deliberately unlike the agentic
(tool-use + error-recovery) family that sql_query / pyfix / codegen belong to.
All are single-turn, no tools, programmatically verifiable.

  logic_grid   constraint-satisfaction puzzle  (symbolic reasoning, no math)
  story_qa     multi-hop narrative QA          (natural language, no structure)
  table_reason numeric reasoning over a table  (semi-structured, no execution)
  translate    EN<->ZH phrase translation      (pure language mapping, NO reasoning)
  spec_write   write a technical spec sheet    (structured generation, not executable)
"""
from __future__ import annotations

import random
from ..envs import TaskInstance

SUBMIT = """
```action
{"tool": "submit", "args": {"answer": <answer>}}
```"""

NAMES = ["Ava", "Ben", "Chen", "Dana", "Eli", "Fay", "Gus", "Hana"]
PETS = ["cat", "dog", "parrot", "turtle", "rabbit"]
CITIES = ["Lima", "Oslo", "Cairo", "Perth", "Kyoto"]
JOBS = ["baker", "pilot", "nurse", "welder", "chemist"]


def _num_answer(cand, gold, tol=0.51):
    if isinstance(cand, list) and len(cand) == 1:
        cand = cand[0]
    if isinstance(cand, (int, float)):
        return abs(float(cand) - float(gold)) < tol
    if isinstance(cand, str):
        import re
        m = re.search(r"-?\d+(?:\.\d+)?", cand.replace(",", ""))
        if m:
            return abs(float(m.group()) - float(gold)) < tol
    return False


def _str_answer(cand, gold):
    if isinstance(cand, list) and len(cand) == 1:
        cand = cand[0]
    return isinstance(cand, str) and cand.strip().lower() == str(gold).strip().lower()


class LogicGridInstance(TaskInstance):
    """Constraint-satisfaction: match people to pets/cities via clues."""
    task = "logic_grid"
    max_turns = 2

    def __init__(self, instance_id, seed, tier=0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        n = 3
        self.people = rng.sample(NAMES, n)
        pets = rng.sample(PETS, n)
        cities = rng.sample(CITIES, n)
        self.pet = dict(zip(self.people, pets))
        self.city = dict(zip(self.people, cities))
        p = self.people
        clues = [
            f"{p[0]} does not own the {self.pet[p[1]]}.",
            f"The person from {self.city[p[1]]} owns the {self.pet[p[1]]}.",
            f"{p[2]} lives in {self.city[p[2]]}.",
            f"{p[0]} owns the {self.pet[p[0]]}.",
        ]
        rng.shuffle(clues)
        self.clues = clues
        self.q_person = rng.choice(p)
        self.ask_pet = rng.random() < 0.5
        self.gold = (self.pet if self.ask_pet else self.city)[self.q_person]

    def system_prompt(self):
        return ("You solve logic puzzles. Reason through the clues step by step, "
                "then submit the single word answer:" + SUBMIT)

    def user_prompt(self):
        return ("Three people — " + ", ".join(self.people) + " — each own one "
                "pet and live in one city.\nClues:\n- " + "\n- ".join(self.clues) +
                f"\nQuestion: which {'pet does' if self.ask_pet else 'city does'} "
                f"{self.q_person} {'own' if self.ask_pet else 'live in'}?")

    def tools(self):
        return {}

    def verify(self, submission):
        return _str_answer(submission, self.gold)


class StoryQaInstance(TaskInstance):
    """Multi-hop narrative QA over a generated 5-sentence story."""
    task = "story_qa"
    max_turns = 2

    def __init__(self, instance_id, seed, tier=0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        a, b, c = rng.sample(NAMES, 3)
        obj = rng.choice(["key", "letter", "camera", "umbrella", "notebook"])
        place1, place2 = rng.sample(["kitchen", "garden", "attic", "cellar", "porch"], 2)
        job = rng.choice(JOBS)
        self.story = (
            f"{a} left the {obj} in the {place1} before going to work as a {job}. "
            f"Later, {b} moved the {obj} to the {place2} while cleaning. "
            f"{c} came looking for the {obj} and asked {b} about it. "
            f"{b} had already left for the market, so {c} searched every room. "
            f"In the end {c} found it exactly where {b} had put it.")
        qs = [(f"Where did {c} find the {obj}?", place2),
              (f"Who moved the {obj}?", b),
              (f"What is {a}'s job?", job),
              (f"Where was the {obj} originally left?", place1)]
        self.question, self.gold = qs[rng.randrange(len(qs))]

    def system_prompt(self):
        return ("You answer reading-comprehension questions. Think briefly, then "
                "submit the shortest exact answer (one word):" + SUBMIT)

    def user_prompt(self):
        return f"Story: {self.story}\nQuestion: {self.question}"

    def tools(self):
        return {}

    def verify(self, submission):
        return _str_answer(submission, self.gold)


class TableReasonInstance(TaskInstance):
    """Numeric reasoning over a small markdown table (no execution)."""
    task = "table_reason"
    max_turns = 2

    def __init__(self, instance_id, seed, tier=0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        self.rows = []
        for nm in rng.sample(NAMES, 4):
            self.rows.append((nm, rng.randint(20, 60), rng.randint(100, 900),
                              rng.choice(CITIES)))
        kind = rng.randrange(4)
        if kind == 0:
            self.question = "What is the total of the Sales column?"
            self.gold = sum(r[2] for r in self.rows)
        elif kind == 1:
            self.question = "Who has the highest Sales?"
            self.gold = max(self.rows, key=lambda r: r[2])[0]
        elif kind == 2:
            th = rng.randint(30, 50)
            self.question = f"How many people are older than {th}?"
            self.gold = sum(1 for r in self.rows if r[1] > th)
        else:
            self.question = ("What is the difference between the highest and "
                             "lowest Sales?")
            self.gold = max(r[2] for r in self.rows) - min(r[2] for r in self.rows)
        self.numeric = not isinstance(self.gold, str)

    def system_prompt(self):
        return ("You analyze tables. Reason step by step over the rows, then "
                "submit the answer:" + SUBMIT)

    def user_prompt(self):
        head = "| Name | Age | Sales | City |\n|---|---|---|---|\n"
        body = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in self.rows)
        return f"{head}{body}\nQuestion: {self.question}"

    def tools(self):
        return {}

    def verify(self, submission):
        return (_num_answer(submission, self.gold) if self.numeric
                else _str_answer(submission, self.gold))


ZH = {
    "good morning": "早上好", "thank you": "谢谢", "how are you": "你好吗",
    "see you tomorrow": "明天见", "i am hungry": "我饿了",
    "the weather is nice": "天气很好", "where is the station": "车站在哪里",
    "please sit down": "请坐", "this is my friend": "这是我的朋友",
    "i like reading books": "我喜欢读书", "the food is delicious": "食物很好吃",
    "have a safe trip": "一路平安", "long time no see": "好久不见",
    "what time is it": "现在几点", "i am learning chinese": "我在学中文",
}


class TranslateInstance(TaskInstance):
    """Pure language mapping — NO reasoning. The most distant control."""
    task = "translate"
    max_turns = 2

    def __init__(self, instance_id, seed, tier=0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        self.en, self.zh = list(ZH.items())[rng.randrange(len(ZH))]
        self.to_zh = rng.random() < 0.5
        self.gold = self.zh if self.to_zh else self.en

    def system_prompt(self):
        return ("You are a translator. Submit only the translation:" + SUBMIT)

    def user_prompt(self):
        src, tgt = ("English", "Chinese") if self.to_zh else ("Chinese", "English")
        text = self.en if self.to_zh else self.zh
        return f"Translate this {src} phrase into {tgt}: \"{text}\""

    def tools(self):
        return {}

    def verify(self, submission):
        if isinstance(submission, list) and len(submission) == 1:
            submission = submission[0]
        if not isinstance(submission, str):
            return False
        s = submission.strip().strip('."。').lower()
        return s == self.gold.strip().lower()


class SpecWriteInstance(TaskInstance):
    """Structured generation: write a spec sheet with required fields."""
    task = "spec_write"
    max_turns = 2

    def __init__(self, instance_id, seed, tier=0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        self.product = rng.choice(["water bottle", "desk lamp", "backpack",
                                   "keyboard", "office chair"])
        self.fields = rng.sample(["material", "weight", "warranty", "color",
                                  "dimensions", "power"], 4)
        self.n_words = rng.choice([12, 15, 20])

    def system_prompt(self):
        return ("You write concise product specifications. Submit a JSON object "
                "whose keys are exactly the requested fields, each value a short "
                "string:" + SUBMIT)

    def user_prompt(self):
        return (f"Write a spec sheet for a {self.product}. Required fields: "
                f"{self.fields}. Each value must be at most {self.n_words} words.")

    def tools(self):
        return {}

    def verify(self, submission):
        import json as _json
        if isinstance(submission, str):
            try:
                submission = _json.loads(submission)
            except _json.JSONDecodeError:
                return False
        if not isinstance(submission, dict):
            return False
        if set(submission.keys()) != set(self.fields):
            return False
        for v in submission.values():
            if not isinstance(v, str) or not v.strip():
                return False
            if len(v.split()) > self.n_words:
                return False
        return True

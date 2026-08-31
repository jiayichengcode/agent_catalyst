"""codegen — reasoning-bound target task: write a function from spec + visible
tests (no starter code), judged by hidden tests. Failures are LOGIC errors,
so the binding constraint is multi-step reasoning about edge cases — the
capability that cross-domain math data is hypothesized to catalyze.

Reuses the pyfix function families (specs, tests, hidden tests) but the agent
writes the implementation from scratch, which is substantially harder than
fixing one line.
"""
from __future__ import annotations

import random
from ..envs import TaskInstance
from .pyfix import TIER_FAMILIES, _run_snippet

SYSTEM = """You are a coding agent. Write a Python function that satisfies the specification and the visible tests. Hidden tests also check edge cases, so reason carefully about boundaries before submitting.

Tools (reply with ONE fenced action block):
```action
{"tool": "run", "args": {"code": "<full function source>"}}
```
runs the visible tests against your code.
```action
{"tool": "submit", "args": {"answer": "<full function source>"}}
```
submits your implementation. Think through the algorithm and edge cases step by step, then act."""

SPEC = {
    "total": "total(xs): return the sum of the list xs (0 for empty).",
    "count_vowels": "count_vowels(s): count vowels (a,e,i,o,u) case-insensitively.",
    "parse_ranges": ("parse_ranges(s): parse '1-3,5' style range strings into the "
                     "full integer list, preserving segment order ('' -> [])."),
    "rle_encode": ("rle_encode(s): run-length encode, e.g. 'aaabbc' -> 'a3b2c1' "
                   "('' -> '')."),
    "balanced": ("balanced(s): return True iff all brackets ()[]{} in s are "
                 "properly matched and nested (ignore other characters)."),
    "merge_intervals": ("merge_intervals(iv): merge overlapping or touching "
                        "[start, end] intervals; return sorted merged list of lists."),
    "moving_avg": ("moving_avg(xs, k): averages of each full window of size k "
                   "(empty list if no full window fits)."),
    "roman_to_int": "roman_to_int(s): convert a Roman numeral to an integer.",
}


class CodegenInstance(TaskInstance):
    task = "codegen"
    max_turns = 6

    def __init__(self, instance_id: str, seed: int, tier: int = 0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        fams = TIER_FAMILIES[1] + TIER_FAMILIES[2]     # medium+hard only
        fam = fams[rng.randrange(len(fams))]
        self.fname, self.correct, _bugs, self.tests, self.hidden = fam(rng)

    def system_prompt(self):
        return SYSTEM

    def user_prompt(self):
        return (f"Specification: {SPEC[self.fname]}\n"
                f"Visible tests:\n```python\n" + "\n".join(self.tests) + "\n```\n"
                f"Write the full implementation of `{self.fname}` and submit it.")

    def tools(self):
        def run(code: str | None = None):
            if not isinstance(code, str) or not code.strip():
                return "ToolError: provide your code to run.", True
            ok, msg = _run_snippet(code, self.tests)
            return msg, (not ok)
        return {"run": run}

    def verify(self, submission):
        if not isinstance(submission, str) or f"def {self.fname}" not in submission:
            return False
        ok, _ = _run_snippet(submission, self.tests + self.hidden)
        return ok

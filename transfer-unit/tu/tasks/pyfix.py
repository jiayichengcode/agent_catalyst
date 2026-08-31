"""pyfix — acceleration-candidate task C: 读报错修一行 (tiered difficulty).

A Python function with one seeded single-line bug. The agent can run the
visible tests (tool `run`, optionally on candidate code) and must submit fixed
code; hidden tests decide success. Instance construction guarantees the bug
breaks at least one visible test, so the first `run` yields an error
observation. Tier 0 = simple functions, tier 1/2 = harder ones, so at
sampling temperature C produces a healthy mix of err_recovery / err_giveup /
clean-success rollouts (needed for the E5 contrasts to be non-degenerate).
"""
from __future__ import annotations

import random
import subprocess
import sys
from ..envs import TaskInstance

SYSTEM = """You are a coding agent. You are given a buggy Python function and its visible tests. Exactly one line contains a bug.

Tools (reply with ONE fenced action block):
```action
{"tool": "run", "args": {"code": "<full function source, optional>"}}
```
runs the visible tests against `code` (or against the original buggy code if omitted) and returns the outcome.
```action
{"tool": "submit", "args": {"answer": "<full fixed function source>"}}
```
submits your fixed function. There are additional hidden tests, so fix the actual bug, not just the visible assertions. Keep the function signature unchanged. Think briefly, then act."""


# ---------------- tier 0 families (simple) ----------------

def _fam_sumrange(rng):
    a, b = rng.randint(2, 9), rng.randint(10, 30)
    src = ("def total(xs):\n"
           "    s = 0\n"
           "    for i in range(len(xs)):\n"
           "        s += xs[i]\n"
           "    return s\n")
    bugs = [("    for i in range(len(xs)):", "    for i in range(len(xs) - 1):"),
            ("        s += xs[i]", "        s += i"),
            ("    s = 0", "    s = 1")]
    tests = [f"assert total(list(range({a}))) == {sum(range(a))}",
             f"assert total([{a}, {b}, 3]) == {a + b + 3}",
             "assert total([]) == 0"]
    hidden = [f"assert total([-1, 1, {b}]) == {b}", f"assert total([{b}]) == {b}"]
    return "total", src, bugs, tests, hidden


def _fam_countvowel(rng):
    src = ("def count_vowels(s):\n"
           "    n = 0\n"
           "    for ch in s.lower():\n"
           "        if ch in 'aeiou':\n"
           "            n += 1\n"
           "    return n\n")
    bugs = [("    for ch in s.lower():", "    for ch in s:"),
            ("        if ch in 'aeiou':", "        if ch in 'aeiu':"),
            ("            n += 1", "            n += 2")]
    tests = ["assert count_vowels('Agentic RL') == 3",
             "assert count_vowels('AEIOU') == 5",
             "assert count_vowels('xyz') == 0"]
    hidden = ["assert count_vowels('Ooo') == 3", "assert count_vowels('') == 0"]
    return "count_vowels", src, bugs, tests, hidden


# ---------------- tier 1 families (medium) ----------------

def _fam_parse_ranges(rng):
    src = ("def parse_ranges(s):\n"
           "    out = []\n"
           "    if not s:\n"
           "        return out\n"
           "    for part in s.split(','):\n"
           "        if '-' in part:\n"
           "            a, b = part.split('-')\n"
           "            for v in range(int(a), int(b) + 1):\n"
           "                out.append(v)\n"
           "        else:\n"
           "            out.append(int(part))\n"
           "    return out\n")
    bugs = [("            for v in range(int(a), int(b) + 1):",
             "            for v in range(int(a), int(b)):"),
            ("            out.append(int(part))", "            out.append(part)"),
            ("            a, b = part.split('-')", "            b, a = part.split('-')"),
            ("    for part in s.split(','):", "    for part in sorted(s.split(',')):")]
    tests = ["assert parse_ranges('1-3,5') == [1, 2, 3, 5]",
             "assert parse_ranges('7') == [7]",
             "assert parse_ranges('2-2,4-6') == [2, 4, 5, 6]"]
    hidden = ["assert parse_ranges('') == []",
              "assert parse_ranges('10-12') == [10, 11, 12]",
              "assert parse_ranges('3,1-2') == [3, 1, 2]"]
    return "parse_ranges", src, bugs, tests, hidden


def _fam_rle(rng):
    src = ("def rle_encode(s):\n"
           "    if not s:\n"
           "        return ''\n"
           "    out = []\n"
           "    prev = s[0]\n"
           "    n = 1\n"
           "    for ch in s[1:]:\n"
           "        if ch == prev:\n"
           "            n += 1\n"
           "        else:\n"
           "            out.append(prev + str(n))\n"
           "            prev = ch\n"
           "            n = 1\n"
           "    out.append(prev + str(n))\n"
           "    return ''.join(out)\n")
    bugs = [("            out.append(prev + str(n))", "            out.append(prev)"),
            ("            n = 1", "            n = 0"),
            ("        if ch == prev:", "        if ch != prev:"),
            ("    if not s:", "    if s is None:")]
    tests = ["assert rle_encode('aaabbc') == 'a3b2c1'",
             "assert rle_encode('x') == 'x1'",
             "assert rle_encode('aabb') == 'a2b2'"]
    hidden = ["assert rle_encode('') == ''",
              "assert rle_encode('abc') == 'a1b1c1'",
              "assert rle_encode('zzzz') == 'z4'"]
    return "rle_encode", src, bugs, tests, hidden


def _fam_balanced(rng):
    src = ("def balanced(s):\n"
           "    pairs = {')': '(', ']': '[', '}': '{'}\n"
           "    stack = []\n"
           "    for ch in s:\n"
           "        if ch in '([{':\n"
           "            stack.append(ch)\n"
           "        elif ch in pairs:\n"
           "            if not stack or stack.pop() != pairs[ch]:\n"
           "                return False\n"
           "    return not stack\n")
    bugs = [("    return not stack", "    return True"),
            ("            if not stack or stack.pop() != pairs[ch]:",
             "            if not stack or stack.pop() == pairs[ch]:"),
            ("    pairs = {')': '(', ']': '[', '}': '{'}",
             "    pairs = {')': '(', ']': '{', '}': '['}"),
            ("            if not stack or stack.pop() != pairs[ch]:",
             "            if stack.pop() != pairs[ch]:")]
    tests = ["assert balanced('([]{})') is True",
             "assert balanced('(]') is False",
             "assert balanced('((') is False",
             "assert balanced('[{}]') is True"]
    hidden = ["assert balanced('') is True",
              "assert balanced(')(') is False",
              "assert balanced('()[]') is True"]
    return "balanced", src, bugs, tests, hidden


# ---------------- tier 2 families (hard) ----------------

def _fam_merge_intervals(rng):
    src = ("def merge_intervals(iv):\n"
           "    iv = sorted(iv)\n"
           "    out = []\n"
           "    for s, e in iv:\n"
           "        if out and s <= out[-1][1]:\n"
           "            if e > out[-1][1]:\n"
           "                out[-1][1] = e\n"
           "        else:\n"
           "            out.append([s, e])\n"
           "    return out\n")
    bugs = [("        if out and s <= out[-1][1]:", "        if out and s < out[-1][1]:"),
            ("    iv = sorted(iv)", "    iv = list(iv)"),
            ("                out[-1][1] = e", "                out[-1][1] = s"),
            ("            if e > out[-1][1]:", "            if True:")]
    tests = ["assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]",
             "assert merge_intervals([[2,6],[1,3]]) == [[1,6]]",
             "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]"]
    hidden = ["assert merge_intervals([]) == []",
              "assert merge_intervals([[2,3],[1,5]]) == [[1,5]]",
              "assert merge_intervals([[1,2],[3,4]]) == [[1,2],[3,4]]"]
    return "merge_intervals", src, bugs, tests, hidden


def _fam_moving_avg(rng):
    src = ("def moving_avg(xs, k):\n"
           "    out = []\n"
           "    for i in range(len(xs) - k + 1):\n"
           "        w = xs[i:i + k]\n"
           "        out.append(sum(w) / k)\n"
           "    return out\n")
    bugs = [("    for i in range(len(xs) - k + 1):", "    for i in range(len(xs) - k):"),
            ("        w = xs[i:i + k]", "        w = xs[i:i + k - 1]"),
            ("        out.append(sum(w) / k)", "        out.append(sum(w) // k)"),
            ("    for i in range(len(xs) - k + 1):",
             "    for i in range(max(len(xs) - k + 1, 1))"[:-1] + "):")]
    tests = ["assert moving_avg([1,2,3,4], 2) == [1.5, 2.5, 3.5]",
             "assert moving_avg([2,2,2], 3) == [2.0]",
             "assert moving_avg([1,3], 1) == [1.0, 3.0]"]
    hidden = ["assert moving_avg([], 1) == []",
              "assert moving_avg([1,2,3,4,5], 5) == [3.0]",
              "assert moving_avg([4], 2) == []"]
    return "moving_avg", src, bugs, tests, hidden


def _fam_roman(rng):
    src = ("def roman_to_int(s):\n"
           "    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
           "    total = 0\n"
           "    for i, ch in enumerate(s):\n"
           "        v = vals[ch]\n"
           "        if i + 1 < len(s) and v < vals[s[i + 1]]:\n"
           "            total -= v\n"
           "        else:\n"
           "            total += v\n"
           "    return total\n")
    bugs = [("            total -= v", "            total += v"),
            ("        if i + 1 < len(s) and v < vals[s[i + 1]]:",
             "        if i + 1 < len(s) and v <= vals[s[i + 1]]:"),
            ("    total = 0", "    total = 1"),
            ("    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}",
             "    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 90, 'D': 500, 'M': 1000}")]
    tests = ["assert roman_to_int('XIV') == 14",
             "assert roman_to_int('III') == 3",
             "assert roman_to_int('XL') == 40"]
    hidden = ["assert roman_to_int('MCMXCIV') == 1994",
              "assert roman_to_int('LVIII') == 58",
              "assert roman_to_int('IX') == 9"]
    return "roman_to_int", src, bugs, tests, hidden


TIER_FAMILIES = {
    0: [_fam_sumrange, _fam_countvowel],
    1: [_fam_parse_ranges, _fam_rle, _fam_balanced],
    2: [_fam_merge_intervals, _fam_moving_avg, _fam_roman],
}


def _run_snippet(code: str, tests: list[str], timeout=5) -> tuple[bool, str]:
    prog = code + "\n" + "\n".join(tests) + "\nprint('ALL TESTS PASSED')\n"
    try:
        p = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TimeoutError: test run exceeded 5s"
    if p.returncode == 0 and "ALL TESTS PASSED" in p.stdout:
        return True, "ALL TESTS PASSED"
    err = (p.stderr or p.stdout).strip()
    return False, err[-1200:] if err else "tests failed with no output"


def _fails_visible_inproc(code: str, tests: list[str]) -> bool:
    """Fast in-process check used only on OUR templated code at build time."""
    g: dict = {}
    try:
        exec(code, g)          # noqa: S102 — template code, not model output
        for t in tests:
            exec(t, g)         # noqa: S102
        return False
    except Exception:
        return True


class PyFixInstance(TaskInstance):
    task = "pyfix"
    max_turns = 6

    def __init__(self, instance_id: str, seed: int, tier: int = 0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        fams = TIER_FAMILIES[tier % 3]
        # 60% of instances: bug breaks a visible test (first `run` -> error
        # observation). 40% (tier>0): bug passes visible tests but breaks a
        # hidden one (`run` says PASSED; fixing requires reading the code).
        # This keeps err_event / err_recovery genuinely variable across C
        # rollouts instead of being 1 by construction.
        self.visible_break = (tier == 0) or (rng.random() < 0.6)
        for _attempt in range(30):
            fam = fams[rng.randrange(len(fams))]
            self.fname, correct, bugs, self.tests, self.hidden = fam(rng)
            old, new = bugs[rng.randrange(len(bugs))]
            assert old in correct, (self.fname, old)
            self.buggy = correct.replace(old, new, 1)
            self.correct = correct
            breaks_vis = _fails_visible_inproc(self.buggy, self.tests)
            breaks_hid = _fails_visible_inproc(self.buggy, self.tests + self.hidden)
            if self.visible_break and breaks_vis:
                break
            if not self.visible_break and not breaks_vis and breaks_hid:
                break

    def system_prompt(self):
        return SYSTEM

    def user_prompt(self):
        return (f"Buggy function:\n```python\n{self.buggy}```\n"
                f"Visible tests:\n```python\n" + "\n".join(self.tests) + "\n```\n"
                "Fix the single buggy line and submit the full corrected function.")

    def tools(self):
        def run(code: str | None = None):
            src = code if isinstance(code, str) and code.strip() else self.buggy
            ok, msg = _run_snippet(src, self.tests)
            return msg, (not ok)
        return {"run": run}

    def verify(self, submission):
        if not isinstance(submission, str) or f"def {self.fname}" not in submission:
            return False
        ok, _ = _run_snippet(submission, self.tests + self.hidden)
        return ok

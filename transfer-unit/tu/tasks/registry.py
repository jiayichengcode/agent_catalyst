"""Task registry + instance pools with disjoint train/eval seed ranges."""
from __future__ import annotations

from .pyfix import PyFixInstance
from .sql_query import SqlQueryInstance
from .neutral import NeutralFormatInstance
from .math_cot import MathCotInstance
from .math_hard import MathHardInstance
from .codegen import CodegenInstance
from .diverse import (LogicGridInstance, StoryQaInstance, TableReasonInstance,
                      TranslateInstance, SpecWriteInstance)

TASKS = {
    "pyfix": PyFixInstance,
    "sql_query": SqlQueryInstance,
    "neutral_format": NeutralFormatInstance,
    "math_cot": MathCotInstance,
    "math_hard": MathHardInstance,
    "codegen": CodegenInstance,
    "logic_grid": LogicGridInstance,
    "story_qa": StoryQaInstance,
    "table_reason": TableReasonInstance,
    "translate": TranslateInstance,
    "spec_write": SpecWriteInstance,
}

# seed layout: train instances seed = base + idx ; eval = base + 100000 + idx
_BASE = {"pyfix": 11_000_000, "sql_query": 22_000_000, "neutral_format": 33_000_000,
         "math_cot": 44_000_000, "math_hard": 55_000_000, "codegen": 66_000_000,
         "logic_grid": 77_000_000, "story_qa": 88_000_000,
         "table_reason": 99_000_000, "translate": 111_000_000,
         "spec_write": 122_000_000}


_WHITELIST = None


def _whitelist():
    """Frozen boundary-difficulty pool (tu.analysis.screen_pool), shared by all
    conditions via TU_POOL_WHITELIST; empty dict when unset."""
    global _WHITELIST
    if _WHITELIST is None:
        import json
        import os
        path = os.environ.get("TU_POOL_WHITELIST", "")
        _WHITELIST = {}
        if path and os.path.exists(path):
            data = json.load(open(path))
            _WHITELIST = {t: d["whitelist"] for t, d in data.items()
                          if d.get("whitelist")}
    return _WHITELIST


_EVAL_WHITELIST = None


def _eval_whitelist():
    """Optional frozen eval-instance whitelist (TU_EVAL_WHITELIST), e.g. the
    hard-B eval set. Same format as the train whitelist file."""
    global _EVAL_WHITELIST
    if _EVAL_WHITELIST is None:
        import json
        import os
        path = os.environ.get("TU_EVAL_WHITELIST", "")
        _EVAL_WHITELIST = {}
        if path and os.path.exists(path):
            data = json.load(open(path))
            _EVAL_WHITELIST = {t: d["whitelist"] for t, d in data.items()
                               if d.get("whitelist")}
    return _EVAL_WHITELIST


def make_instance(task: str, idx: int, split: str = "train"):
    cls = TASKS[task]
    if split == "train":
        wl = _whitelist().get(task)
        if wl:
            idx = wl[idx % len(wl)]
    elif split == "eval":
        wl = _eval_whitelist().get(task)
        if wl:
            idx = wl[idx % len(wl)]
    off = 100_000 if split == "eval" else 0
    seed = _BASE[task] + off + idx
    # difficulty tier cycles deterministically through the pool
    return cls(instance_id=f"{task}/{split}/{idx}", seed=seed, tier=idx % 3)

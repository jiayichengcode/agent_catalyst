"""Rule-based span annotator (plan §1): S_err_react / S_verify / S_plan / S_final.

Spans are token ranges [start, end) into the full training sequence, always
inside assistant (trainable) regions. Mapping char offset -> token count uses
re-tokenization of the text prefix; boundary drift is at most ±1 token and is
clamped into the turn range.
"""
from __future__ import annotations

from .features import CHECK_TOOLS

SPAN_NAMES = ("S_err_react", "S_verify", "S_plan", "S_final")


def annotate_spans(rec: dict, tokenizer=None) -> dict[str, list[tuple[int, int]]]:
    ats = sorted(rec.get("assistant_turns", []), key=lambda a: a["turn_idx"])
    spans: dict[str, list[tuple[int, int]]] = {k: [] for k in SPAN_NAMES}
    if not ats:
        return spans
    checks = CHECK_TOOLS.get(rec["task"], set())

    for a in ats:
        if a.get("follows_error_obs"):
            spans["S_err_react"].append((a["start"], a["end"]))
        if a.get("action_tool") in checks:
            spans["S_verify"].append((a["start"], a["end"]))

    first = ats[0]
    fence_char = first.get("fence_char", -1)
    if fence_char > 0 and tokenizer is not None:
        pre = len(tokenizer(first["text"][:fence_char], add_special_tokens=False)["input_ids"])
        end = min(first["start"] + max(pre, 0), first["end"])
        if end > first["start"]:
            spans["S_plan"].append((first["start"], end))
    elif fence_char > 0:
        spans["S_plan"].append((first["start"], first["end"]))

    last = ats[-1]
    spans["S_final"].append((last["start"], last["end"]))
    return spans


def spans_to_mask_zero(spans: list[tuple[int, int]], loss_mask):
    """Zero out loss for the given token ranges (in-place on a 1-D tensor/list)."""
    for s, e in spans:
        loss_mask[s:e] = 0
    return loss_mask


def total_span_tokens(spans: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in spans)

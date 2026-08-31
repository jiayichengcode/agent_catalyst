"""Per-rollout trace feature rows (plan §1 schema), rule-based, zero-manual.

Operates on the RolloutRecord dicts produced by tu.training.rollout:
  record = {
    task, instance_id, group_id, sample_idx, success, n_actions, reward,
    assistant_turns: [{start, end, text, turn_idx, follows_error_obs,
                       action_tool, had_action, fence_char}],
    obs_turns: [{turn_idx, is_error_obs, text}],
    gen_logprobs: [float per generated token] (optional),
    gen_entropy: [float per generated token] (optional, top-k approx),
  }

Features: success, err_event, err_recovery, err_giveup, verify_cnt, clarify,
replan, len, ent, adv_mag (filled by trainer), novelty (pilot: None).
"""
from __future__ import annotations

# tool names that count as verification/checking actions, per task
CHECK_TOOLS = {"pyfix": {"run"}, "sql_query": {"sql"}, "neutral_format": set()}
CLARIFY_TOOLS: set[str] = {"ask", "clarify"}


def compute_features(rec: dict) -> dict:
    task = rec["task"]
    obs = rec.get("obs_turns", [])
    ats = rec.get("assistant_turns", [])
    success = bool(rec.get("success"))
    err_event = int(any(o["is_error_obs"] for o in obs))
    checks = CHECK_TOOLS.get(task, set())

    verify_cnt = sum(1 for a in ats if a.get("action_tool") in checks)
    clarify = int(any(a.get("action_tool") in CLARIFY_TOOLS for a in ats))

    # replan: after an error observation, the next action switches tool
    replan = 0
    prev_tool = None
    err_turns = {o["turn_idx"] for o in obs if o["is_error_obs"]}
    for a in sorted(ats, key=lambda x: x["turn_idx"]):
        if a["turn_idx"] - 1 in err_turns and prev_tool is not None \
                and a.get("action_tool") not in (None, prev_tool):
            replan += 1
        if a.get("action_tool"):
            prev_tool = a["action_tool"]

    ent = None
    ents = rec.get("gen_entropy")
    if ents:
        ent = float(sum(ents) / max(1, len(ents)))

    n_gen_tokens = sum(a["end"] - a["start"] for a in ats)
    return {
        "task": task, "instance_id": rec["instance_id"],
        "group_id": rec.get("group_id"), "sample_idx": rec.get("sample_idx"),
        "success": int(success),
        "err_event": err_event,
        "err_recovery": int(err_event and success),
        "err_giveup": int(err_event and not success),
        "verify_cnt": verify_cnt,
        "clarify": clarify,
        "replan": replan,
        "len": rec.get("n_actions", len(ats)),
        "n_gen_tokens": n_gen_tokens,
        "ent": ent,
        "adv_mag": None,     # filled by trainer after group advantage
        "novelty": None,     # not computed in pilot (documented deviation)
        "reward": rec.get("reward"),
    }


# ---- trace-filter conditions (plan §3 A3, E5) -------------------------------
# Each condition maps a feature row -> keep (True) / drop (False) for C rollouts.

def condition_keep(cond: str, f: dict) -> bool:
    if cond == "F6":       # keep everything (control)
        return True
    if cond == "F0":       # drop everything (interaction-only)
        return False
    if cond == "F1":       # success-only (the RFT convention)
        return f["success"] == 1
    if cond == "F2":       # failure-only
        return f["success"] == 0
    if cond == "F3":       # error-recovery only
        return f["err_recovery"] == 1
    if cond == "F4":       # error-free successes only
        return f["success"] == 1 and f["err_event"] == 0
    raise ValueError(f"unknown condition {cond}")
# F5 (count-matched random) is implemented in the trainer: it samples a random
# subset of C rollouts sized to the number F3 would keep on the same batch.

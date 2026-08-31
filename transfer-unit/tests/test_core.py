"""E0 acceptance tests for the CPU-side stack: envs, tasks, features, spans,
filter conditions. Run: python3 tests/test_core.py  (from transfer-unit/)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tu.envs import Episode, env_step, parse_action, FORMAT_ERROR_OBS
from tu.tasks.registry import make_instance
from tu.tracing.features import compute_features, condition_keep
from tu.tracing.spans import annotate_spans

PASS = 0


def ok(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok {name}")


def act(tool, args):
    import json
    return f"thinking...\n```action\n{json.dumps({'tool': tool, 'args': args})}\n```\n"


# ---- parse_action ----
ok("parse basic", parse_action(act("sql", {"query": "SELECT 1"})) == ("sql", {"query": "SELECT 1"}))
ok("parse none", parse_action("no block here") is None)
ok("parse badjson", parse_action("```action\n{oops}\n```") is None)

# ---- pyfix ----
inst = make_instance("pyfix", 0, "train")   # tier 0 => visible_break
obs, is_err = inst.tools()["run"]()
ok("pyfix buggy run errors", is_err and obs != "ALL TESTS PASSED")
ok("pyfix correct passes hidden", inst.verify(inst.correct))
ok("pyfix buggy fails verify", not inst.verify(inst.buggy))
obs2, err2 = inst.tools()["run"](code=inst.correct)
ok("pyfix run(correct) ok", not err2 and "PASSED" in obs2)
i2 = make_instance("pyfix", 0, "train")
ok("pyfix deterministic", i2.buggy == inst.buggy)
i3 = make_instance("pyfix", 0, "eval")
ok("pyfix eval split differs", (i3.buggy != inst.buggy) or (i3.tests != inst.tests))

# family correctness: correct impl passes all tests; buggy fails verify;
# visible-break instances error on first run, hidden-break ones don't
n_err = n_vb = n_hb = 0
for k in range(45):
    it = make_instance("pyfix", k, "train")
    assert it.verify(it.correct), f"correct impl fails: {it.fname} seed {k}"
    assert not it.verify(it.buggy), f"buggy passes verify: {it.fname} seed {k}"
    _, e = it.tools()["run"]()
    if it.visible_break:
        n_vb += 1
        n_err += int(e)
    else:
        n_hb += 1
        assert not e, f"hidden-break instance errored visibly: {it.fname}"
print(f"  info pyfix visible-break {n_vb}/45, hidden-break {n_hb}/45")
ok("pyfix visible-break always errors", n_err == n_vb)
ok("pyfix both instance types exist", n_vb >= 25 and n_hb >= 5)

# ---- sql_query ----
s = make_instance("sql_query", 0, "train")
ok("sql gold exists", s.gold is not None)
obs, e = s.tools()["sql"](query="SELECT name, sql FROM sqlite_master")
ok("sql schema query ok", not e and "CREATE TABLE" in obs)
obs, e = s.tools()["sql"](query="SELECT nonexistent_col FROM nowhere")
ok("sql bad query errors", e and "SqlError" in obs)
obs, e = s.tools()["sql"](query=s.gold_sql)
ok("sql gold sql runs", not e)
ok("sql verify gold", s.verify(s.gold))
ok("sql verify str-of-num", s.verify(str(s.gold)))
ok("sql verify wrong", not s.verify("obviously-wrong-answer-xyz"))

# solvable rate sanity: gold exists + verifies across 21 seeds (all tiers)
for k in range(21):
    si = make_instance("sql_query", k, "train")
    assert si.gold is not None, (k, si.question)
    assert si.verify(si.gold), (k, si.question, si.gold)
ok("sql 21 seeds x3 tiers gold-verifiable", True)

# ---- neutral ----
nt = make_instance("neutral_format", 0, "train")
gold = {"item": nt.item, "color": nt.color, "price": nt.price, "stock": nt.stock}
ok("neutral verify gold", nt.verify(gold))
ok("neutral verify wrong", not nt.verify({**gold, "stock": nt.stock + 1}))

# ---- episode loop: pyfix err_recovery trajectory ----
inst = None
for k in range(1, 40):
    cand = make_instance("pyfix", k, "train")
    if cand.visible_break:
        inst = cand
        break
ep = Episode(task="pyfix", instance_id=inst.instance_id,
             system=inst.system_prompt(), user0=inst.user_prompt())
env_step(inst, ep, act("run", {}))                       # -> error obs
ok("ep err obs", ep.turns[-1].is_error_obs)
env_step(inst, ep, act("submit", {"answer": inst.correct}))
ok("ep success", ep.done and ep.success)

# format error path
ep2 = Episode(task="pyfix", instance_id="x", system="s", user0="u")
env_step(inst, ep2, "I forgot the action block entirely")
ok("ep format error obs", ep2.turns[-1].content == FORMAT_ERROR_OBS
   and ep2.turns[-1].is_error_obs)

# ---- features on a synthetic record ----
rec = {
    "task": "pyfix", "instance_id": "pyfix/train/1", "group_id": "g", "sample_idx": 0,
    "success": True, "reward": 1.0, "n_actions": 3,
    "assistant_turns": [
        {"start": 10, "end": 40, "text": "t1", "turn_idx": 1,
         "follows_error_obs": False, "action_tool": "run", "had_action": True,
         "fence_char": 5},
        {"start": 60, "end": 90, "text": "t2", "turn_idx": 3,
         "follows_error_obs": True, "action_tool": "run", "had_action": True,
         "fence_char": 4},
        {"start": 110, "end": 130, "text": "t3", "turn_idx": 5,
         "follows_error_obs": False, "action_tool": "submit", "had_action": True,
         "fence_char": 2},
    ],
    "obs_turns": [
        {"turn_idx": 2, "is_error_obs": True, "text": "Traceback ..."},
        {"turn_idx": 4, "is_error_obs": False, "text": "ALL TESTS PASSED"},
    ],
    "gen_entropy": [0.5, 1.0],
}
f = compute_features(rec)
ok("feat success", f["success"] == 1)
ok("feat err_event", f["err_event"] == 1)
ok("feat err_recovery", f["err_recovery"] == 1)
ok("feat err_giveup", f["err_giveup"] == 0)
ok("feat verify_cnt", f["verify_cnt"] == 2)
ok("feat ent", abs(f["ent"] - 0.75) < 1e-9)
ok("feat gen tokens", f["n_gen_tokens"] == 80)

# ---- spans ----
sp = annotate_spans(rec, tokenizer=None)
ok("span err_react", sp["S_err_react"] == [(60, 90)])
ok("span verify", sp["S_verify"] == [(10, 40), (60, 90)])
ok("span final", sp["S_final"] == [(110, 130)])
ok("span plan fallback", sp["S_plan"] == [(10, 40)])

# ---- filter conditions ----
ok("F6 keeps", condition_keep("F6", f))
ok("F1 keeps success", condition_keep("F1", f))
ok("F3 keeps errrec", condition_keep("F3", f))
ok("F4 drops err success", not condition_keep("F4", f))
f_fail = dict(f, success=0, err_recovery=0, err_giveup=1)
ok("F1 drops fail", not condition_keep("F1", f_fail))
ok("F2 keeps fail", condition_keep("F2", f_fail))
ok("F0 drops", not condition_keep("F0", f))

# ---- apply_condition F5 count matching (no torch needed? grpo imports torch) ----
try:
    from tu.training.grpo import apply_condition
    import random
    recs, feats, advs = [], [], []
    for i in range(16):
        er = 1 if i % 4 == 0 else 0
        r = {"task": "pyfix", "reward": float(er)}
        ff = {"err_recovery": er, "n_gen_tokens": 100 + i, "success": er,
              "err_event": 1}
        recs.append(r)
        feats.append(ff)
    keep, st = apply_condition("F5", "pyfix", recs, feats, random.Random(0))
    ok("F5 count match", sum(keep) == st["f3_equiv_n"] == 4)
    keep3, _ = apply_condition("F3", "pyfix", recs, feats, random.Random(0))
    ok("F3 keeps exactly errrec", sum(keep3) == 4
       and all(keep3[i] == (feats[i]["err_recovery"] == 1) for i in range(16)))
    keep0, _ = apply_condition("F0", "pyfix", recs, feats, random.Random(0))
    ok("F0 drops all C", sum(keep0) == 0)
except ImportError as e:
    print(f"  skip grpo tests (torch missing: {e})")

print(f"\nALL {PASS} CHECKS PASSED")

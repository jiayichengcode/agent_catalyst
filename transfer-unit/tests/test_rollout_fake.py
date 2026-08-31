"""Offline rollout-bookkeeping test with a FakeLLM (no GPU / no vllm).
Validates: ChatFormat probing on the real Qwen3.5 tokenizer, incremental id
assembly, loss masks, assistant-turn ranges, err_recovery tracing, span
annotation on real token ranges."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL = "/mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B"


class FakeComp:
    def __init__(self, text, ids):
        self.text = text
        self.token_ids = ids
        self.logprobs = None


class FakeOut:
    def __init__(self, comp):
        self.outputs = [comp]


class FakeLLM:
    """Scripted assistant replies per round."""
    def __init__(self, tok, script):
        self.tok = tok
        self.script = script
        self.round = 0

    def generate(self, prompts, sp, use_tqdm=False):
        outs = []
        for _p in prompts:
            text = self.script[min(self.round, len(self.script) - 1)]
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            outs.append(FakeOut(FakeComp(text, ids)))
        self.round += 1
        return outs


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    from tu.training.rollout import ChatFormat, run_episodes
    from tu.tracing.features import compute_features
    from tu.tracing.spans import annotate_spans
    from tu.tasks.registry import make_instance

    fmt = ChatFormat(tok)
    print("gen_prompt_tail:", json.dumps(fmt.gen_prompt_tail))
    print("turn_close:", json.dumps(fmt.turn_close))
    print("stop_token_ids:", fmt.stop_token_ids)

    inst = None
    for k in range(1, 40):
        cand = make_instance("pyfix", k, "train")
        if cand.visible_break:
            inst = cand
            break
    fix = json.dumps({"tool": "submit", "args": {"answer": inst.correct}})
    script = [
        'Let me run the tests first.\n```action\n{"tool": "run", "args": {}}\n```\n',
        f'I see the error. Submitting the fix.\n```action\n{fix}\n```\n',
    ]
    llm = FakeLLM(tok, script)
    specs = [("pyfix", 1, "train", "g0", 0)]
    recs = run_episodes(llm, fmt, specs, temperature=1.0, logprobs_k=0, seed=0)
    r = recs[0]
    assert r["success"], "scripted fix should succeed"
    assert len(r["assistant_turns"]) == 2
    a0, a1 = r["assistant_turns"]
    assert a1["follows_error_obs"], "turn 2 must follow the error obs"
    assert r["obs_turns"][0]["is_error_obs"]
    # id/mask alignment; every id must be an int token id
    assert len(r["ids"]) == len(r["loss_mask"])
    assert all(isinstance(t, int) for t in r["ids"]), \
        f"non-int ids: {[t for t in r['ids'] if not isinstance(t, int)][:3]!r}"
    assert len(r["ids"]) > 60, "prompt ids look truncated"
    n_gen = sum(r["loss_mask"])
    assert n_gen == (a0["end"] - a0["start"]) + (a1["end"] - a1["start"])
    # masked regions decode to prompt + obs wrapper (contains the obs text)
    obs_zone = tok.decode(r["ids"][a0["end"]:a1["start"]])
    assert "<|im_start|>user" in obs_zone, obs_zone[:120]
    # generated region decodes back to the scripted text
    gen0 = tok.decode(r["ids"][a0["start"]:a0["end"]])
    assert "run" in gen0
    f = compute_features(r)
    assert f["err_recovery"] == 1 and f["verify_cnt"] == 1
    sp = annotate_spans(r, tok)
    assert sp["S_err_react"] == [(a1["start"], a1["end"])]
    assert sp["S_plan"] and sp["S_plan"][0][0] == a0["start"]
    assert sp["S_plan"][0][1] < a0["end"]
    print("token counts: total", len(r["ids"]), "gen", n_gen,
          "plan span", sp["S_plan"][0][1] - sp["S_plan"][0][0])
    print("FAKE-LLM ROLLOUT BOOKKEEPING: ALL OK")


if __name__ == "__main__":
    main()

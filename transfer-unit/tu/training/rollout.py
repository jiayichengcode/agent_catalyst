"""Batched multi-turn rollout engine on a vLLM engine.

Token bookkeeping is exact and incremental: the training sequence is built by
concatenating (a) the chat-template prompt ids, (b) the raw generated token ids
returned by vLLM, and (c) tokenized observation-wrapper ids. Past turns are
NEVER re-rendered through the chat template, so generated ids equal trained
ids and observation tokens are loss-masked by construction (plan §5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..envs import Episode, env_step
from ..tasks.registry import make_instance


class ChatFormat:
    """Derives the incremental wrapper strings from the tokenizer's template."""

    def __init__(self, tok):
        self.tok = tok
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        kw = {}
        try:  # qwen3-style switchable thinking; harmless if unsupported
            base = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
            kw["enable_thinking"] = False
        except (TypeError, Exception):
            base = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False)
        self.template_kwargs = kw
        # generation prompt tail: text after the final user turn
        anchor = "U"
        idx = base.rindex(anchor) + len(anchor)
        self.gen_prompt_tail = base[idx:]          # e.g. "<|im_end|>\n<|im_start|>assistant\n"
        # split into close-of-turn + open-of-assistant at the first "<|im_start|>"
        j = self.gen_prompt_tail.index("<|im_start|>")
        self.turn_close = self.gen_prompt_tail[:j]  # "<|im_end|>\n"
        self.asst_open = self.gen_prompt_tail[j:]
        self.user_open = "<|im_start|>user\n"
        assert self.user_open in base, f"unexpected chat template: {base[:200]!r}"
        self.eos_id = tok.eos_token_id
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.stop_token_ids = sorted({t for t in [self.eos_id, im_end] if t is not None})

    def prompt_ids(self, system: str, user0: str) -> list[int]:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user0}]
        # render text, then tokenize — the tokenized return type of
        # apply_chat_template varies across transformers versions
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            tokenize=False,
                                            **self.template_kwargs)
        out = self.tok(text, add_special_tokens=False)["input_ids"]
        assert out and isinstance(out[0], int), f"bad prompt ids: {out[:3]!r}"
        return list(out)

    def obs_wrapper_ids(self, obs: str) -> list[int]:
        text = (self.turn_close + self.user_open + obs + self.gen_prompt_tail)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def close_ids(self) -> list[int]:
        return self.tok(self.turn_close, add_special_tokens=False)["input_ids"]


@dataclass
class LiveEp:
    inst: object
    ep: Episode
    ids: list = field(default_factory=list)
    loss_mask: list = field(default_factory=list)
    assistant_turns: list = field(default_factory=list)
    obs_turns: list = field(default_factory=list)
    gen_logprobs: list = field(default_factory=list)
    gen_entropy: list = field(default_factory=list)
    turn_idx: int = 1          # message index in chat (sys=skip, user0=0)
    pending_error_obs: bool = False
    truncated: bool = False
    group_id: str = ""
    sample_idx: int = 0


def _entropy_from_logprobs(lp_dict) -> float:
    if not lp_dict:
        return 0.0
    ps = [math.exp(l.logprob) for l in lp_dict.values()]
    z = sum(ps)
    if z <= 0:
        return 0.0
    return -sum((p / z) * math.log(max(p / z, 1e-12)) for p in ps)


def run_episodes(llm, fmt: ChatFormat, specs, *, temperature, max_new_per_turn=384,
                 max_seq_len=3968, logprobs_k=0, seed=0):
    """specs: list of (task, idx, split, group_id, sample_idx).
    Returns list of finished record dicts (order = specs order)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    live: list[LiveEp] = []
    for (task, idx, split, gid, si) in specs:
        inst = make_instance(task, idx, split)
        ep = Episode(task=task, instance_id=inst.instance_id,
                     system=inst.system_prompt(), user0=inst.user_prompt())
        le = LiveEp(inst=inst, ep=ep, group_id=gid, sample_idx=si)
        pids = fmt.prompt_ids(ep.system, ep.user0)
        le.ids = list(pids)
        le.loss_mask = [0] * len(pids)
        live.append(le)

    max_rounds = max(le.inst.max_turns for le in live)
    for _round in range(max_rounds):
        active = [le for le in live if not le.ep.done and not le.truncated]
        if not active:
            break
        prompts = [TokensPrompt(prompt_token_ids=le.ids) for le in active]
        # one SamplingParams PER REQUEST with a distinct seed: identical
        # prompts sharing one seed produce identical completions in vllm,
        # which silently collapses every G-sample group to one rollout.
        sps = [SamplingParams(
            temperature=temperature, top_p=1.0, max_tokens=max_new_per_turn,
            stop=["```\n"], include_stop_str_in_output=True,
            stop_token_ids=fmt.stop_token_ids,
            logprobs=logprobs_k if logprobs_k > 0 else None,
            seed=(seed + 1000003 * _round + 7919 * i) if temperature > 0 else None,
        ) for i, _le in enumerate(active)]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        # phase A (serial, cheap): record generated ids + turn metadata
        from ..envs import parse_action
        n_before_l = []
        for le, out in zip(active, outs):
            comp = out.outputs[0]
            gen_ids = list(comp.token_ids)
            # comp.text is authoritative (includes the stop string even when
            # token_ids are cut at the stop-match boundary)
            text = comp.text
            start = len(le.ids)
            le.ids.extend(gen_ids)
            le.loss_mask.extend([1] * len(gen_ids))
            if logprobs_k > 0 and comp.logprobs:
                for tid, lpd in zip(gen_ids, comp.logprobs):
                    le.gen_logprobs.append(
                        lpd[tid].logprob if tid in lpd else 0.0)
                    le.gen_entropy.append(_entropy_from_logprobs(lpd))
            parsed = parse_action(text)
            fence = text.find("```action")
            le.assistant_turns.append({
                "start": start, "end": len(le.ids), "text": text,
                "turn_idx": le.turn_idx,
                "follows_error_obs": le.pending_error_obs,
                "action_tool": parsed[0] if parsed else None,
                "had_action": parsed is not None,
                "fence_char": fence,
            })
            le.pending_error_obs = False
            n_before_l.append(len(le.ep.turns))

        # phase B (parallel): env transitions — tool subprocesses release the GIL
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(
                lambda p: env_step(p[0].inst, p[0].ep,
                                   p[0].assistant_turns[-1]["text"]),
                [(le,) for le in active]))

        # phase C (serial, cheap): observation bookkeeping + wrapper tokens
        for le, n_before in zip(active, n_before_l):
            le.turn_idx += 1
            if le.ep.done:
                continue
            new_turns = le.ep.turns[n_before + 1:]
            obs_text = new_turns[0].content if new_turns else ""
            is_err = bool(new_turns and new_turns[0].is_error_obs)
            le.obs_turns.append({"turn_idx": le.turn_idx, "is_error_obs": is_err,
                                 "text": obs_text[:500]})
            le.pending_error_obs = is_err
            wrap = fmt.obs_wrapper_ids(obs_text)
            if len(le.ids) + len(wrap) + max_new_per_turn > max_seq_len:
                le.truncated = True
                le.ep.done = True
                le.ep.success = False
                continue
            le.ids.extend(wrap)
            le.loss_mask.extend([0] * len(wrap))
            le.turn_idx += 1

    records = []
    for le in live:
        rec = {
            "task": le.ep.task, "tier": getattr(le.inst, "tier", 0),
            "instance_id": le.ep.instance_id,
            "group_id": le.group_id, "sample_idx": le.sample_idx,
            "success": bool(le.ep.success), "reward": float(le.ep.success),
            "n_actions": le.ep.n_actions, "truncated": le.truncated,
            "ids": le.ids, "loss_mask": le.loss_mask,
            "assistant_turns": le.assistant_turns, "obs_turns": le.obs_turns,
            "gen_logprobs": le.gen_logprobs, "gen_entropy": le.gen_entropy,
        }
        records.append(rec)
    return records


def eval_detail(llm, fmt: ChatFormat, task: str, n: int, *, seed=0,
                temperature=0.0):
    specs = [(task, i, "eval", f"eval/{task}/{i}", 0) for i in range(n)]
    return run_episodes(llm, fmt, specs, temperature=temperature,
                        logprobs_k=0, seed=seed)


def eval_pass1(llm, fmt: ChatFormat, task: str, n: int, *, seed=0):
    recs = eval_detail(llm, fmt, task, n, seed=seed)
    return sum(r["success"] for r in recs) / max(1, len(recs))

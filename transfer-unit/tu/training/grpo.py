"""Minimal, auditable GRPO trainer with trace-level filtering (A3) and
span-level loss masks (A4).

Architecture per run: 2 GPUs. vLLM engine on cuda:0 (rollouts + eval),
policy training on cuda:1. Weight sync after each step: bf16 state_dict ->
/dev/shm -> vllm collective_rpc(load_weights).

GRPO here = group-baseline policy gradient, strictly on-policy (1 update per
batch, importance ratio == 1), advantage A_i = r_i - mean(group), no std
division (Dr.GRPO-style), no KL term. Advantages are ALWAYS computed on the
full G=8 group BEFORE any trace filtering (advantage-frozen scheme, plan §3).
"""
from __future__ import annotations

import json
import os
import random
import time

import torch

from ..tracing.features import compute_features, condition_keep
from ..tracing.spans import annotate_spans
from .rollout import ChatFormat, run_episodes, eval_pass1


# ---------------- weight sync (runs inside the vllm worker process) ----------

def _vllm_load_from_path(worker, path: str):
    import torch as _t
    sd = _t.load(path, map_location="cpu")
    model = worker.model_runner.model
    model.load_weights(sd.items())
    return len(sd)


def build_key_remap(train_keys, ckpt_keys) -> dict:
    """Map trainer state_dict names -> checkpoint names (vllm loader expects
    checkpoint naming). Handles the ForCausalLM vs ForConditionalGeneration
    prefix difference generically by suffix matching."""
    ckpt = set(ckpt_keys)
    remap, missed = {}, []
    by_suffix = {}
    for k in ckpt:
        by_suffix.setdefault(k.split(".", 1)[-1], []).append(k)
    for t in train_keys:
        if t in ckpt:
            remap[t] = t
            continue
        cands = [c for c in ckpt if c.endswith("." + t) or c.endswith(t)]
        if len(cands) == 1:
            remap[t] = cands[0]
            continue
        suf = t.split(".", 1)[-1]
        cs = by_suffix.get(suf, [])
        if len(cs) == 1:
            remap[t] = cs[0]
        else:
            missed.append(t)
    return remap, missed


class WeightSync:
    def __init__(self, llm, model, model_path, run_name):
        self.llm = llm
        self.model = model
        self.path = f"/dev/shm/tu_sync_{run_name}.pt"
        idx = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(idx):
            ckpt_keys = list(json.load(open(idx))["weight_map"].keys())
        else:
            from safetensors import safe_open
            with safe_open(os.path.join(model_path, "model.safetensors"), "pt") as f:
                ckpt_keys = list(f.keys())
        # only trainable params need syncing (vision tower is frozen)
        self.train_keys = [n for n, p in model.named_parameters()
                           if p.requires_grad]
        self.remap, missed = build_key_remap(self.train_keys, ckpt_keys)
        if missed:
            print(f"[sync] WARNING {len(missed)}/{len(self.train_keys)} keys "
                  f"unmapped, e.g. {missed[:4]}")
        else:
            print(f"[sync] all {len(self.train_keys)} trainable keys mapped")

    def push(self):
        sd = {}
        params = dict(self.model.named_parameters())
        for k in self.train_keys:
            if k in self.remap:
                sd[self.remap[k]] = params[k].detach().to(torch.bfloat16).cpu()
        torch.save(sd, self.path)
        path = self.path

        def _load(worker, p):        # cloudpickled by value into vllm workers
            import torch as _t
            _sd = _t.load(p, map_location="cpu")
            mr = getattr(worker, "model_runner", None)
            if mr is None:
                mr = worker.worker.model_runner
            mr.model.load_weights(_sd.items())
            return len(_sd)

        self.llm.collective_rpc(_load, args=(path,))
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass


# ---------------- trace filtering (plan §3 A3 + E5 conditions) ---------------

def apply_condition(cond: str, c_task: str, records, feats, rng):
    """Return per-record gradient keep flags. Advantage is untouched; dropped
    records get their whole-token loss masked. Non-C tasks always kept.
    F5 = random subset of C rollouts count- and token-matched to what F3
    would keep on this very batch."""
    keep = [True] * len(records)
    if cond in ("F6", "noC", "seq"):
        return keep, {}
    c_idx = [i for i, r in enumerate(records) if r["task"] == c_task]
    if cond == "F5":
        f3_keep = [i for i in c_idx if feats[i]["err_recovery"] == 1]
        target_tok = sum(feats[i]["n_gen_tokens"] for i in f3_keep)
        pool = c_idx[:]
        rng.shuffle(pool)
        chosen, tok = [], 0
        for i in pool:
            if len(chosen) >= len(f3_keep) and tok >= target_tok:
                break
            if len(chosen) < len(f3_keep) or tok < target_tok:
                chosen.append(i)
                tok += feats[i]["n_gen_tokens"]
        chosen = chosen[:max(len(f3_keep), 1) if f3_keep else 0]
        for i in c_idx:
            keep[i] = i in chosen
        stats = {"f3_equiv_n": len(f3_keep), "f5_kept_n": len(chosen)}
        return keep, stats
    for i in c_idx:
        keep[i] = condition_keep(cond, feats[i])
    return keep, {}


# ---------------- trainer ----------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = random.Random(cfg["seed"])
        self.run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
        os.makedirs(self.run_dir, exist_ok=True)
        json.dump(cfg, open(os.path.join(self.run_dir, "config.json"), "w"), indent=1)
        self.f_train = open(os.path.join(self.run_dir, "train_log.jsonl"), "a")
        self.f_roll = open(os.path.join(self.run_dir, "rollouts.jsonl"), "a")
        self.f_eval = open(os.path.join(self.run_dir, "eval.jsonl"), "a")
        self.f_samp = open(os.path.join(self.run_dir, "samples.jsonl"), "a")

        mp = cfg["model_path"]
        print(f"[init] loading vllm engine on cuda:0 ({mp})")
        from vllm import LLM
        self.llm = LLM(model=mp, max_model_len=cfg.get("max_model_len", 4096),
                       gpu_memory_utilization=cfg.get("vllm_gpu_util", 0.85),
                       enforce_eager=True, trust_remote_code=True,
                       enable_prefix_caching=cfg.get("prefix_caching", True),
                       seed=cfg["seed"])
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        self.fmt = ChatFormat(self.tok)

        print("[init] loading policy model on cuda:1")
        self.device = torch.device("cuda:1")
        self.model = self._load_policy(mp)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.train()
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.0)
        self.sync = WeightSync(self.llm, self.model, mp, cfg["run_name"])
        self.step_i = 0

    def _load_policy(self, mp):
        """Load the policy for text-only training. Try the ForCausalLM class
        first; if the (multimodal) checkpoint leaves too many keys missing,
        fall back to ForConditionalGeneration with the vision tower frozen."""
        if self.cfg.get("disable_fla", True):
            # fla-core refuses gated DeltaNet backward on Hopper w/ triton>=3.4
            # (fla #640) unless tilelang is present (broken in this env).
            # Force the transformers pure-torch gated-delta path: correct,
            # differentiable, slower. vllm rollouts use vllm's own kernels
            # and are unaffected.
            import sys
            import transformers.utils.import_utils as _iu
            _iu.is_flash_linear_attention_available = lambda: False
            sys.modules.pop("transformers.models.qwen3_5.modeling_qwen3_5", None)
            print("[init] fla disabled for policy: torch-native gated delta rule")
        m = None
        try:
            from transformers.models.qwen3_5 import Qwen3_5ForCausalLM
            m, info = Qwen3_5ForCausalLM.from_pretrained(
                mp, dtype=torch.bfloat16, trust_remote_code=True,
                output_loading_info=True)
            n_missing = len(info.get("missing_keys", []))
            print(f"[init] Qwen3_5ForCausalLM missing_keys={n_missing}")
            if n_missing > 50:
                m = None
        except Exception as e:
            print(f"[init] ForCausalLM path failed: {e}")
        if m is None:
            import transformers
            cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None) \
                or transformers.AutoModelForCausalLM
            m = cls.from_pretrained(mp, dtype=torch.bfloat16,
                                    trust_remote_code=True)
            print(f"[init] policy = {type(m).__name__}")
        n_frozen = 0
        for n, p in m.named_parameters():
            if "visual" in n or "vision" in n or "image" in n:
                p.requires_grad = False
                n_frozen += 1
        if n_frozen:
            print(f"[init] froze {n_frozen} vision params")
        return m.to(self.device)

    # -------- batch construction --------
    def _step_specs(self):
        cfg = self.cfg
        G = cfg["group_size"]
        mixture = cfg["mixture"]
        # "seq" condition (curriculum control): C-only phase, then B-mixture
        # with C swapped for filler — same total groups/steps as mixed training.
        su = cfg.get("seq_c_until")
        if su is not None:
            c = cfg["c_task"]
            if self.step_i < su:
                mixture = {c: sum(cfg["mixture"].values())}
            else:
                mixture = {t: n for t, n in cfg["mixture"].items() if t != c}
                mixture["neutral_format"] = (mixture.get("neutral_format", 0)
                                             + cfg["mixture"].get(c, 0))
        specs = []
        for task, n_prompts in mixture.items():
            over = cfg.get("c_oversample", 1) if task == cfg["c_task"] else 1
            for p in range(n_prompts * over):
                idx = self.rng.randrange(cfg["train_pool_size"])
                gid = f"s{self.step_i}/{task}/{p}"
                for g in range(G):
                    specs.append((task, idx, "train", gid, g))
        return specs

    # -------- loss --------
    def _train_on_records(self, records, adv, keep, span_cfg):
        dev = self.device
        total_tok = 0
        items = []
        for r, a, k in zip(records, adv, keep):
            if a == 0.0 or not k:
                continue
            lm = list(r["loss_mask"])
            if span_cfg and r["task"] == span_cfg["task"]:
                spans = annotate_spans(r, self.tok)
                if span_cfg["mode"] == "named":
                    tgt = []
                    for name in span_cfg["spans"]:
                        tgt += spans.get(name, [])
                elif span_cfg["mode"] == "random_match":
                    n_mask = sum(e - s for name in span_cfg["spans"]
                                 for (s, e) in spans.get(name, []))
                    tgt = self._random_spans(r, n_mask)
                for s, e in tgt:
                    for j in range(s, min(e, len(lm))):
                        lm[j] = 0
            n = sum(lm)
            if n == 0:
                continue
            items.append((r["ids"], lm, a))
            total_tok += n
        if not items:
            return {"skipped": True, "grad_tokens": 0}
        # normalizer must be IDENTICAL across conditions: with a batch-dependent
        # denominator, B's effective per-token LR would depend on how many C
        # tokens the condition keeps (the very variable we manipulate).
        norm = self.cfg.get("loss_norm_tokens") or max(total_tok, 1)

        micro_budget = self.cfg.get("micro_token_budget", 12288)
        self.opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        # pack by PADDED size (batch is padded to its max length, and logits
        # memory scales with padded tokens): sort by length, then greedy-pack
        # so that maxlen * batch_size <= budget.
        items.sort(key=lambda it: len(it[0]), reverse=True)
        micros = []
        batch = []
        for it in items:
            if batch:
                maxlen = max(len(batch[0][0]), len(it[0]))
                if maxlen * (len(batch) + 1) > micro_budget:
                    micros.append(batch)
                    batch = []
            batch.append(it)
        if batch:
            micros.append(batch)
        for mb in micros:
            maxlen = max(len(ids) for ids, _, _ in mb)
            inp = torch.full((len(mb), maxlen), self.tok.pad_token_id or 0,
                             dtype=torch.long)
            lmask = torch.zeros((len(mb), maxlen))
            amat = torch.zeros((len(mb), maxlen))
            att = torch.zeros((len(mb), maxlen), dtype=torch.long)
            for bi, (ids, lm, a) in enumerate(mb):
                L = len(ids)
                inp[bi, :L] = torch.tensor(ids)
                lmask[bi, :L] = torch.tensor(lm, dtype=torch.float)
                amat[bi, :L] = a
                att[bi, :L] = 1
            inp, lmask, amat, att = (x.to(dev) for x in (inp, lmask, amat, att))
            out = self.model(input_ids=inp, attention_mask=att, use_cache=False)
            logits = out.logits[:, :-1]
            tgts = inp[:, 1:]
            m = lmask[:, 1:]
            a = amat[:, 1:]
            # chunked fp32 log-softmax to avoid materializing (N, vocab) fp32
            B, L, V = logits.shape
            fl = logits.reshape(B * L, V)
            ft = tgts.reshape(B * L)
            w = (a * m).reshape(B * L)
            loss = fl.new_zeros((), dtype=torch.float32)
            CH = 4096
            for s0 in range(0, B * L, CH):
                sl = slice(s0, min(s0 + CH, B * L))
                if not torch.any(w[sl] != 0):
                    continue
                lp = torch.log_softmax(fl[sl].float(), dim=-1).gather(
                    -1, ft[sl].unsqueeze(-1)).squeeze(-1)
                loss = loss + -(lp * w[sl]).sum()
            loss = loss / norm
            loss.backward()
            loss_sum += loss.item()
            del out, logits, fl
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        torch.cuda.empty_cache()
        return {"skipped": False, "grad_tokens": total_tok,
                "loss": loss_sum, "grad_norm": float(gn)}

    def _random_spans(self, r, n_mask):
        ats = r["assistant_turns"]
        if not ats or n_mask <= 0:
            return []
        spans = []
        remaining = n_mask
        turns = ats[:]
        self.rng.shuffle(turns)
        for a in turns:
            if remaining <= 0:
                break
            L = a["end"] - a["start"]
            take = min(L, remaining)
            s = a["start"] + self.rng.randrange(0, L - take + 1)
            spans.append((s, s + take))
            remaining -= take
        return spans

    # -------- main loop --------
    def train(self):
        cfg = self.cfg
        G = cfg["group_size"]
        self._eval()
        while self.step_i < cfg["steps"]:
            t0 = time.time()
            specs = self._step_specs()
            records = run_episodes(
                self.llm, self.fmt, specs, temperature=cfg["temperature"],
                logprobs_k=cfg.get("logprobs_k", 5),
                seed=cfg["seed"] * 100000 + self.step_i)
            t_roll = time.time() - t0

            # advantages on the FULL batch (advantage-frozen: computed before
            # any trace filtering). Two baselines:
            #  - "group": GRPO per-instance-group mean. Degenerate when task
            #    outcomes are deterministic per instance (bimodal pass rates).
            #  - "task": mean over all same-task rollouts this step (RLOO/
            #    Dr.GRPO-style); learns from across-instance contrast.
            adv = [0.0] * len(records)
            if cfg.get("adv_baseline", "group") == "task":
                by_task = {}
                for i, r in enumerate(records):
                    by_task.setdefault(r["task"], []).append(i)
                for _t, idxs in by_task.items():
                    mu = sum(records[i]["reward"] for i in idxs) / len(idxs)
                    for i in idxs:
                        adv[i] = records[i]["reward"] - mu
            else:
                by_gid = {}
                for i, r in enumerate(records):
                    by_gid.setdefault(r["group_id"], []).append(i)
                for _gid, idxs in by_gid.items():
                    rs = [records[i]["reward"] for i in idxs]
                    mu = sum(rs) / len(rs)
                    for i in idxs:
                        adv[i] = records[i]["reward"] - mu

            feats = [compute_features(r) for r in records]
            for f, a in zip(feats, adv):
                f["adv_mag"] = abs(a)
                f["step"] = self.step_i

            # oversample bookkeeping: with c_oversample>1 only the kept subset
            # may enter gradient; cap C gradient tokens at the base-sampling
            # token budget estimate so budgets stay comparable (plan §3).
            keep, fstats = apply_condition(
                cfg["condition"], cfg["c_task"], records, feats,
                random.Random(cfg["seed"] * 7919 + self.step_i))
            if cfg.get("c_oversample", 1) > 1:
                keep = self._cap_c_tokens(records, feats, keep)

            span_cfg = cfg.get("span_mask")
            stats = self._train_on_records(records, adv, keep, span_cfg)
            self.sync.push()

            # ---- logging ----
            for f, k in zip(feats, keep):
                f["entered_gradient"] = int(bool(k) and f["adv_mag"] > 0)
                self.f_roll.write(json.dumps(f) + "\n")
            self.f_roll.flush()
            per_task = {}
            for r in records:
                d = per_task.setdefault(r["task"], {"n": 0, "succ": 0, "err": 0,
                                                    "errrec": 0, "kept_tok": 0})
                d["n"] += 1
                d["succ"] += int(r["success"])
            for f, r, k in zip(feats, records, keep):
                d = per_task[r["task"]]
                d["err"] += f["err_event"]
                d["errrec"] += f["err_recovery"]
                if k:
                    d["kept_tok"] += f["n_gen_tokens"]
            if self.step_i % 25 == 0:
                n_dumped = 0
                for r, f, k in zip(records, feats, keep):
                    if r["task"] == cfg["c_task"] and n_dumped < 3:
                        self.f_samp.write(json.dumps({
                            "step": self.step_i, "task": r["task"],
                            "feat": {kk: f[kk] for kk in
                                     ("success", "err_event", "err_recovery",
                                      "verify_cnt", "len")},
                            "kept": bool(k),
                            "turns": [a["text"][-1200:] for a in r["assistant_turns"]],
                            "obs": [o["text"][:400] for o in r["obs_turns"]],
                        }) + "\n")
                        n_dumped += 1
                self.f_samp.flush()
            line = {"step": self.step_i, "t_roll": round(t_roll, 1),
                    "t_total": round(time.time() - t0, 1),
                    "per_task": per_task, **stats, **fstats}
            self.f_train.write(json.dumps(line) + "\n")
            self.f_train.flush()
            print(f"[step {self.step_i}] " + json.dumps(line))

            self.step_i += 1
            if self.step_i % cfg["eval_every"] == 0 or self.step_i == cfg["steps"]:
                self._eval()
            if self.step_i % cfg.get("ckpt_every", 100) == 0:
                self._save()
        self._save()

    def _cap_c_tokens(self, records, feats, keep):
        cfg = self.cfg
        c = cfg["c_task"]
        base_prompts = cfg["mixture"].get(c, 0)
        over = cfg.get("c_oversample", 1)
        c_tok_all = sum(f["n_gen_tokens"] for f, r in zip(feats, records)
                        if r["task"] == c)
        budget = c_tok_all / max(over, 1)
        kept = [(i, feats[i]["n_gen_tokens"]) for i, r in enumerate(records)
                if r["task"] == c and keep[i]]
        self.rng.shuffle(kept)
        tok = 0
        for i, nt in kept:
            if tok + nt > budget * 1.05:
                keep[i] = False
            else:
                tok += nt
        return keep

    def _eval(self):
        cfg = self.cfg
        from .rollout import eval_detail
        res = {"step": self.step_i}
        for task, n in cfg["eval_sizes"].items():
            recs = eval_detail(self.llm, self.fmt, task, n, seed=cfg["seed"])
            res[task] = sum(r["success"] for r in recs) / max(1, len(recs))
            fs = [compute_features(r) for r in recs]
            m = len(fs) or 1
            # behavioral readouts move before pass@1 does in sparse regimes
            res[task + "_beh"] = {
                "err": round(sum(f["err_event"] for f in fs) / m, 3),
                "errrec": round(sum(f["err_recovery"] for f in fs) / m, 3),
                "replan": round(sum(f["replan"] for f in fs) / m, 3),
                "verify": round(sum(f["verify_cnt"] for f in fs) / m, 3),
                "len": round(sum(f["len"] for f in fs) / m, 2),
            }
        res["t"] = time.time()
        self.f_eval.write(json.dumps(res) + "\n")
        self.f_eval.flush()
        print(f"[eval] {json.dumps(res)}")

    def _save(self):
        path = os.path.join(self.run_dir, "last_ckpt")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        print(f"[ckpt] saved to {path}")

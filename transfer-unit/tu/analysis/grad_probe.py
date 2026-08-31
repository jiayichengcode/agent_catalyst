"""Gradient-geometry probe: measure the quantities the catalysis theory says
should predict speed-up.

For a given model checkpoint (base or mid-training) and a pair of datasets
(B target, C candidate catalyst), compute:
  cos(g_B, g_C)        cosine of the mean gradients (shared-direction overlap)
  |g_C| / |g_B|        relative gradient magnitude
  cos_within_B         mean pairwise cosine among B minibatch gradients
                       (self-collinearity: high => Adam's second moment
                        saturates in few directions => slow after the first
                        descent; the "activation energy" of the reaction)
  cos_within_C         same for C
  eff_rank_B/C         participation ratio of per-batch gradient directions

Theory prediction (inverted-U): speed-up is maximal for intermediate
cos(g_B,g_C) — near 0 the catalyst shares no parameters (inert), near 1 it
adds no new directions (mere budget).

Usage:
  python -m tu.analysis.grad_probe --b-data data/donor_codegen.jsonl \
     --c-data data/donor_mathhard.jsonl --c-select clean --n-batches 8
"""
from __future__ import annotations

import argparse
import json
import math
import random

import torch


def _open_maybe_gz(path):
    """Open .jsonl or .jsonl.gz transparently."""
    import gzip
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    import os
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return open(path)


def load(path, sel, rng, n, max_tok=2048):
    rows = []
    with _open_maybe_gz(path) as f:
        for line in f:
            d = json.loads(line)
            ft = d["feat"]
            if sel == "b_success" and ft["success"] != 1:
                continue
            if sel == "clean" and ft["success"] != 1:
                continue
            if sel == "errrec" and ft["err_recovery"] != 1:
                continue
            if len(d["ids"]) > max_tok:
                continue
            rows.append(d)
    rng.shuffle(rows)
    return rows[:n]


def flat_grad(model, batch, pad, device, keys):
    model.zero_grad(set_to_none=True)
    maxlen = max(len(d["ids"]) for d in batch)
    inp = torch.full((len(batch), maxlen), pad, dtype=torch.long)
    lm = torch.zeros((len(batch), maxlen))
    att = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, d in enumerate(batch):
        L = len(d["ids"])
        inp[i, :L] = torch.tensor(d["ids"])
        lm[i, :L] = torch.tensor(d["loss_mask"], dtype=torch.float)
        att[i, :L] = 1
    inp, lm, att = inp.to(device), lm.to(device), att.to(device)
    out = model(input_ids=inp, attention_mask=att, use_cache=False)
    logits = out.logits[:, :-1]
    tgt = inp[:, 1:]
    w = lm[:, 1:]
    B, L, V = logits.shape
    fl = logits.reshape(B * L, V)
    ft = tgt.reshape(B * L)
    fw = w.reshape(B * L)
    tot = fw.sum().clamp(min=1)
    loss = fl.new_zeros((), dtype=torch.float32)
    for s0 in range(0, B * L, 4096):
        sl = slice(s0, min(s0 + 4096, B * L))
        if not torch.any(fw[sl] != 0):
            continue
        lp = torch.log_softmax(fl[sl].float(), dim=-1).gather(
            -1, ft[sl].unsqueeze(-1)).squeeze(-1)
        loss = loss + -(lp * fw[sl]).sum()
    (loss / tot).backward()
    g = torch.cat([model.get_parameter(k).grad.detach().flatten().float()
                   for k in keys if model.get_parameter(k).grad is not None])
    return g.cpu()


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def mean_pairwise_cos(gs):
    if len(gs) < 2:
        return float("nan")
    vals = [cos(gs[i], gs[j]) for i in range(len(gs)) for j in range(i + 1, len(gs))]
    return sum(vals) / len(vals)


def eff_rank(gs):
    """Participation ratio of the Gram spectrum: how many distinct directions."""
    n = len(gs)
    G = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = torch.dot(gs[i], gs[j])
    ev = torch.linalg.eigvalsh(G).clamp(min=0)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 ** 2 / (s2 + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B")
    ap.add_argument("--b-data", required=True)
    ap.add_argument("--c-data", required=True)
    ap.add_argument("--c-select", default="clean")
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--layers", default="model.layers.20,model.layers.30",
                    help="probe a subset of layers to bound memory")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model-ckpt", help="vllm-named .pt of a mid-training ckpt")
    ap.add_argument("--out", help="append results as jsonl")
    ap.add_argument("--joint-rank", action="store_true",
                    help="also report eff_rank of B and C gradients pooled")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
    import sys as _sys
    _sys.modules.pop("transformers.models.qwen3_5.modeling_qwen3_5", None)
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    if args.model_ckpt:
        sd = torch.load(args.model_ckpt, map_location="cpu")
        from ..training.grpo import build_key_remap
        rm, _ = build_key_remap([n for n, _ in model.named_parameters()],
                                list(sd.keys()))
        inv = {v: k for k, v in rm.items()}
        loaded = 0
        with torch.no_grad():
            for ck, v in sd.items():
                tn = inv.get(ck)
                if tn is not None:
                    model.get_parameter(tn).copy_(v.to(torch.bfloat16).cuda())
                    loaded += 1
        print(f"[probe] loaded {loaded} tensors from {args.model_ckpt}")
    pad = tok.pad_token_id or 0
    prefixes = tuple(args.layers.split(","))
    keys = [n for n, p in model.named_parameters()
            if any(n.startswith(pf + ".") for pf in prefixes)]
    print(f"[probe] {len(keys)} params in {prefixes}")

    nb, bs = args.n_batches, args.batch
    b_rows = load(args.b_data, "b_success", rng, nb * bs)
    c_rows = load(args.c_data, args.c_select, rng, nb * bs)
    gB = [flat_grad(model, b_rows[i * bs:(i + 1) * bs], pad, "cuda:0", keys)
          for i in range(nb) if len(b_rows) >= (i + 1) * bs]
    gC = [flat_grad(model, c_rows[i * bs:(i + 1) * bs], pad, "cuda:0", keys)
          for i in range(nb) if len(c_rows) >= (i + 1) * bs]
    mB = torch.stack(gB).mean(0)
    mC = torch.stack(gC).mean(0)
    res = {
        "tag": args.tag, "b_data": args.b_data, "c_data": args.c_data,
        "c_select": args.c_select, "n_batches_B": len(gB), "n_batches_C": len(gC),
        "cos_BC": round(cos(mB, mC), 4),
        "norm_ratio_C_over_B": round(float(mC.norm() / (mB.norm() + 1e-12)), 4),
        "cos_within_B": round(mean_pairwise_cos(gB), 4),
        "cos_within_C": round(mean_pairwise_cos(gC), 4),
        "eff_rank_B": round(eff_rank(gB), 3),
        "eff_rank_C": round(eff_rank(gC), 3),
    }
    if args.joint_rank:
        res["eff_rank_joint"] = round(eff_rank(gB + gC), 3)
        res["rank_gain"] = round(res["eff_rank_joint"] - res["eff_rank_B"], 3)
    print("[probe]", json.dumps(res))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

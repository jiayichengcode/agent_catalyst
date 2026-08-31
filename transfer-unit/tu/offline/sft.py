"""SFT catalysis experiment (plan §3 A5 / E6): mix catalyst-trace data with
target-task (B) demonstration data in ONE likelihood loss and measure whether
B is learned faster per B-token.

Conditions (--c-select): none | errrec | clean | random | errrec_spanmask
Ordering (--order): mixed (interleaved shuffle) | seq (all C first, then B)

Readouts: held-out B CE loss every eval-every steps (catalysis = faster B-loss
drop at fixed B-token budget); checkpoint for pass@1 eval via
run.py --eval-only --model <ckpt>.

Usage:
  python -m tu.offline.sft --b-data data/donor_sql.jsonl \
      --c-data data/donor_pyfix.jsonl --c-select errrec --order mixed \
      --run-name sftcat_errrec --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
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


def load_records(path, sel=None, max_tokens=None, rng=None):
    rows = []
    with _open_maybe_gz(path) as f:
        for line in f:
            d = json.loads(line)
            ft = d["feat"]
            if sel == "b_success" and ft["success"] != 1:
                continue
            if sel == "errrec" and ft["err_recovery"] != 1:
                continue
            if sel == "clean" and not (ft["success"] == 1 and ft["err_event"] == 0):
                continue
            if sel == "fail" and ft["success"] != 0:
                continue
            rows.append(d)
    if rng:
        rng.shuffle(rows)
    if max_tokens is not None:
        out, tok = [], 0
        for d in rows:
            n = sum(d["loss_mask"])
            if tok + n > max_tokens:
                break
            out.append(d)
            tok += n
        return out, tok
    return rows, sum(sum(d["loss_mask"]) for d in rows)


def spanmask_c(rec):
    """Zero the loss on S_err_react spans (assistant turns that follow an
    error observation)."""
    lm = list(rec["loss_mask"])
    for a in rec["assistant_turns"]:
        if a.get("follows_error_obs"):
            for j in range(a["start"], min(a["end"], len(lm))):
                lm[j] = 0
    rec = dict(rec)
    rec["loss_mask"] = lm
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B")
    ap.add_argument("--b-data", required=True)
    ap.add_argument("--c-data")
    ap.add_argument("--c-select", default="none",
                    choices=["none", "errrec", "clean", "random", "fail",
                             "b_success", "errrec_spanmask"])
    ap.add_argument("--order", default="mixed", choices=["mixed", "seq"])
    ap.add_argument("--b-tokens", type=int, default=400_000)
    ap.add_argument("--b-repeat", type=int, default=1)
    ap.add_argument("--c-skip-b", action="store_true",
                    help="exclude episodes already used as B data from the C pool")
    ap.add_argument("--c-tokens", type=int, default=400_000)
    ap.add_argument("--b-val-episodes", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--micro-padded", type=int, default=4096)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=0,
                    help="save a vllm-ready ckpt every N optimizer steps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir",
                    default="/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst/runs")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "sft_log.jsonl"), "a")

    b_all, _ = load_records(args.b_data, sel="b_success", rng=rng)
    b_val = b_all[:args.b_val_episodes]
    b_pool = b_all[args.b_val_episodes:]
    b_train, b_tok = [], 0
    for d in b_pool:
        n = sum(d["loss_mask"])
        if b_tok + n > args.b_tokens:
            break
        b_train.append(d)
        b_tok += n
    items = [("B", d) for d in b_train] * args.b_repeat
    c_tok = 0
    if args.c_select != "none":
        sel = {"errrec": "errrec", "clean": "clean", "random": None,
               "fail": "fail", "b_success": "b_success",
               "errrec_spanmask": "errrec"}[args.c_select]
        c_rows, _ = load_records(args.c_data, sel=sel, rng=rng)
        if args.c_skip_b:
            used = {d["instance_id"] for d in b_train} | {d["instance_id"] for d in b_val}
            c_rows = [d for d in c_rows if d["instance_id"] not in used]
        c_train = []
        for d in c_rows:
            n = sum(d["loss_mask"])
            if c_tok + n > args.c_tokens:
                break
            if args.c_select == "errrec_spanmask":
                d = spanmask_c(d)
            c_train.append(d)
            c_tok += n
        items += [("C", d) for d in c_train]
    print(f"[sft] B train {len(b_train)} eps / {b_tok} tok; "
          f"C {args.c_select}: {c_tok} tok; order={args.order}")
    if args.order == "mixed":
        rng.shuffle(items)
    else:
        items.sort(key=lambda x: 0 if x[0] == "C" else 1)

    from transformers import AutoTokenizer
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: False
    import sys as _sys
    _sys.modules.pop("transformers.models.qwen3_5.modeling_qwen3_5", None)
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    pad = tok.pad_token_id or 0

    remap = None
    if args.save_every > 0:
        import json as _json
        from ..training.grpo import build_key_remap
        idx = os.path.join(args.model, "model.safetensors.index.json")
        ckpt_keys = list(_json.load(open(idx))["weight_map"].keys())
        remap, _missed = build_key_remap(
            [k for k, _ in model.named_parameters()], ckpt_keys)
        os.makedirs(os.path.join(run_dir, "ckpts"), exist_ok=True)

    def save_step_ckpt(step_no):
        sd = {}
        params = dict(model.named_parameters())
        for k, tgt in remap.items():
            sd[tgt] = params[k].detach().to(torch.bfloat16).cpu()
        torch.save(sd, os.path.join(run_dir, "ckpts", f"step{step_no:05d}.pt"))

    def ce_loss(batch, train=True):
        maxlen = max(len(d["ids"]) for _s, d in batch)
        B = len(batch)
        inp = torch.full((B, maxlen), pad, dtype=torch.long)
        lm = torch.zeros((B, maxlen))
        att = torch.zeros((B, maxlen), dtype=torch.long)
        for bi, (_s, d) in enumerate(batch):
            L = len(d["ids"])
            inp[bi, :L] = torch.tensor(d["ids"])
            lm[bi, :L] = torch.tensor(d["loss_mask"], dtype=torch.float)
            att[bi, :L] = 1
        inp, lm, att = inp.to("cuda:0"), lm.to("cuda:0"), att.to("cuda:0")
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            out = model(input_ids=inp, attention_mask=att, use_cache=False)
            logits = out.logits[:, :-1]
            tgt = inp[:, 1:]
            m = lm[:, 1:]
            Bx, L, V = logits.shape
            fl = logits.reshape(Bx * L, V)
            ft = tgt.reshape(Bx * L)
            w = m.reshape(Bx * L)
            tot = w.sum().clamp(min=1)
            loss = fl.new_zeros((), dtype=torch.float32)
            for s0 in range(0, Bx * L, 4096):
                sl = slice(s0, min(s0 + 4096, Bx * L))
                if not torch.any(w[sl] != 0):
                    continue
                lp = torch.log_softmax(fl[sl].float(), dim=-1).gather(
                    -1, ft[sl].unsqueeze(-1)).squeeze(-1)
                loss = loss + -(lp * w[sl]).sum()
            return loss / tot, int(tot.item())

    def val_b_loss():
        model.eval()
        tot_l, tot_n = 0.0, 0
        batch, cur = [], 0
        for d in b_val:
            if batch and max(cur, len(d["ids"])) * (len(batch) + 1) > args.micro_padded:
                l, n = ce_loss(batch, train=False)
                tot_l += float(l) * n
                tot_n += n
                batch, cur = [], 0
            batch.append(("B", d))
            cur = max(cur, len(d["ids"]))
        if batch:
            l, n = ce_loss(batch, train=False)
            tot_l += float(l) * n
            tot_n += n
        model.train()
        return tot_l / max(1, tot_n)

    # micro-batches by padded budget
    micros, batch, cur = [], [], 0
    for it in items:
        L = len(it[1]["ids"])
        if batch and max(cur, L) * (len(batch) + 1) > args.micro_padded:
            micros.append(batch)
            batch, cur = [], 0
        batch.append(it)
        cur = max(cur, L)
    if batch:
        micros.append(batch)

    b_tok_seen = 0
    step = 0
    vl = val_b_loss()
    print(f"[sft step 0] val_B_loss {vl:.4f}")
    logf.write(json.dumps({"step": 0, "b_tok_seen": 0, "val_b_loss": vl}) + "\n")
    opt.zero_grad(set_to_none=True)
    for mi, mb in enumerate(micros):
        loss, ntok = ce_loss(mb, train=True)
        (loss / args.accum).backward()
        b_tok_seen += sum(sum(d["loss_mask"]) for s, d in mb if s == "B")
        if (mi + 1) % args.accum == 0 or mi == len(micros) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.eval_every == 0:
                vl = val_b_loss()
                print(f"[sft step {step}] micros {mi+1}/{len(micros)} "
                      f"b_tok_seen {b_tok_seen} val_B_loss {vl:.4f}")
                logf.write(json.dumps({"step": step, "b_tok_seen": b_tok_seen,
                                       "val_b_loss": vl}) + "\n")
                logf.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_step_ckpt(step)
                logf.write(json.dumps({"ckpt": step,
                                       "b_tok_seen": b_tok_seen}) + "\n")
                logf.flush()
    if args.save_every > 0:
        save_step_ckpt(step + 1)
        logf.write(json.dumps({"ckpt": step + 1, "b_tok_seen": b_tok_seen}) + "\n")
    vl = val_b_loss()
    logf.write(json.dumps({"step": step, "b_tok_seen": b_tok_seen,
                           "val_b_loss": vl, "final": True}) + "\n")
    logf.flush()
    ck = os.path.join(run_dir, "ckpt")
    model.save_pretrained(ck, safe_serialization=True)
    tok.save_pretrained(ck)
    print(f"[sft] final val_B_loss {vl:.4f}; saved {ck}")


if __name__ == "__main__":
    main()

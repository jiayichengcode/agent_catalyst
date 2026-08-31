"""Export an SFT ckpt (saved by Qwen3_5ForCausalLM) into a vllm-loadable
directory: original multimodal config + original weights with the SFT text
weights spliced in under their original names.

Usage:
  python -m tu.offline.export_ckpt --sft runs/sftcat_bonly/ckpt \
      --base /mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B \
      --out runs/sftcat_bonly/ckpt_vllm
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

from ..training.grpo import build_key_remap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sft_sd = {}
    for shard in sorted(glob.glob(os.path.join(args.sft, "*.safetensors"))):
        sft_sd.update(load_file(shard))
    base_sd = {}
    for shard in sorted(glob.glob(os.path.join(args.base, "*.safetensors"))):
        base_sd.update(load_file(shard))
    remap, missed = build_key_remap(list(sft_sd.keys()), list(base_sd.keys()))
    print(f"[export] sft keys {len(sft_sd)}, mapped {len(remap)}, missed {len(missed)}")
    if missed:
        print("  e.g. missed:", missed[:5])
    n_replaced = 0
    for k, v in sft_sd.items():
        tgt = remap.get(k)
        if tgt is None:
            continue
        assert base_sd[tgt].shape == v.shape, (k, tgt, v.shape, base_sd[tgt].shape)
        base_sd[tgt] = v.to(torch.bfloat16)
        n_replaced += 1
    print(f"[export] replaced {n_replaced}/{len(base_sd)} tensors")
    save_file(base_sd, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})
    for f in os.listdir(args.base):
        if f.endswith((".json", ".txt")) and "safetensors.index" not in f:
            shutil.copy(os.path.join(args.base, f), os.path.join(args.out, f))
    print(f"[export] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Generate generic self-anchor data: the BASE model's own responses to a
fixed set of generic instructions, stored in donor format (ids + loss_mask on
the response tokens). By construction this data carries no new information —
its gradient is a restoring force that activates only when training drifts
the model away from its own distribution (buffer-solution catalyst).

Usage: python -m tu.offline.gen_anchor --out data/donor_anchorG.jsonl --n-per 6
"""
from __future__ import annotations

import argparse
import json

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about autumn rain.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "List three tips for writing readable code.",
    "What is the difference between a list and a tuple in Python?",
    "Describe how photosynthesis works, briefly.",
    "Translate 'good morning, my friend' into French and Spanish.",
    "Give a polite reply declining a meeting invitation.",
    "What are the primary colors and how do they mix?",
    "Explain the concept of compound interest with a small example.",
    "Write a two-line product description for a thermos bottle.",
    "How do vaccines work, in simple terms?",
    "Name three famous bridges and where they are.",
    "What is a healthy breakfast? Give one example.",
    "Explain recursion to a beginner in three sentences.",
    "Write a short thank-you note to a teacher.",
    "What causes seasons on Earth?",
    "Give two synonyms and one antonym for 'happy'.",
    "Briefly compare cats and dogs as pets.",
    "What does HTTP stand for and what does it do?",
    "Suggest a simple weekend plan for relaxing.",
    "Explain what a database index is, briefly.",
    "Write one sentence using the word 'serendipity'.",
    "What is the capital of Japan and one fact about it?",
    "How should you store fresh basil?",
    "Explain the rule of thirds in photography.",
    "What's the difference between weather and climate?",
    "Give a beginner tip for learning guitar.",
    "Describe the taste of dark chocolate in two sentences.",
    "What is version control and why is it useful?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B")
    ap.add_argument("--n-per", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=280)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from ..training.rollout import ChatFormat
    llm = LLM(model=args.model, max_model_len=2048, gpu_memory_utilization=0.85,
              enforce_eager=True, trust_remote_code=True, seed=args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fmt = ChatFormat(tok)

    prompts, metas = [], []
    for pi, p in enumerate(PROMPTS):
        ids = fmt.prompt_ids("You are a helpful assistant.", p)
        for g in range(args.n_per):
            prompts.append(TokensPrompt(prompt_token_ids=list(ids)))
            metas.append((pi, g, ids))
    sps = [SamplingParams(temperature=args.temp, top_p=0.95,
                          max_tokens=args.max_tokens,
                          stop_token_ids=fmt.stop_token_ids,
                          seed=args.seed + 977 * i)
           for i in range(len(prompts))]
    outs = llm.generate(prompts, sps, use_tqdm=False)
    n = 0
    with open(args.out, "w") as f:
        for (pi, g, ids), out in zip(metas, outs):
            gen = list(out.outputs[0].token_ids)
            if len(gen) < 8:
                continue
            full = list(ids) + gen
            mask = [0] * len(ids) + [1] * len(gen)
            f.write(json.dumps({
                "task": "anchorG", "instance_id": f"anchorG/{pi}/{g}",
                "ids": full, "loss_mask": mask, "assistant_turns": [],
                "feat": {"success": 1, "err_event": 0, "err_recovery": 0,
                         "err_giveup": 0, "verify_cnt": 0, "len": 1,
                         "n_gen_tokens": len(gen)},
            }) + "\n")
            n += 1
    print(f"wrote {n} anchor records -> {args.out}")


if __name__ == "__main__":
    main()

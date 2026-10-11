#!/usr/bin/env python
"""Domain-adapt Qwen3.5 to one OOD corpus with LoRA.

This is the analogue of the vision experiment's per-dataset LoRA-CLIP: the
backbone is adapted to the target domain first, and the SAE arms are then
trained and scored on the *adapted* model's residual stream. It is what makes
the `base` vs `lora` axis of the eval matrix mean anything -- without it every
arm reads the same activations.

Following CLIP-LoRA's convention (`CLIP_LoRA/run_utils.py`: `--params q k v`,
LoRA on the attention query/key/value matrices at every layer), adapters go on
the q/k/v projections only -- no output projection, no MLP. Qwen3.5 is a hybrid
stack, so "q/k/v" means `q_proj/k_proj/v_proj` in the full-attention layers and
the fused `in_proj_qkv`/`in_proj_z` in the linear-attention layers. Rank is
raised from CLIP-LoRA's r=2 to r=8 because the backbone here is 2B parameters
rather than 86M and r=2 barely moves the residual stream.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.data import CachedTokenWindows, get_corpus

QKV_TARGETS = ["q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--cache_dir", default="out/token_cache")
    p.add_argument("--out_dir", default="out/lora")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--targets", nargs="+", default=QKV_TARGETS)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--total_tokens", type=int, default=2_000_000)
    p.add_argument("--seqs_per_step", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--grad_checkpoint", dest="grad_checkpoint",
                   action="store_true", default=True,
                   help="Trade speed for a smaller activation footprint. On by "
                        "default because the dev box shares its GPU; turn off "
                        "with --no_grad_checkpoint on a dedicated GPU, where it "
                        "costs roughly 3x wall time for nothing.")
    p.add_argument("--no_grad_checkpoint", dest="grad_checkpoint",
                   action="store_false")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out_dir) / args.corpus
    out.mkdir(parents=True, exist_ok=True)

    corpus = get_corpus(args.corpus)
    tok = AutoTokenizer.from_pretrained(args.model)
    train = CachedTokenWindows(args.cache_dir, args.corpus, "train", seed=args.seed)
    evl = CachedTokenWindows(args.cache_dir, args.corpus, "eval", seed=args.seed)
    print(f"[lora] {corpus.name} ({corpus.domain}) | {train} | {evl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.config.use_cache = False

    present = {n.split(".")[-1] for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)}
    targets = [t for t in args.targets if t in present]
    missing = [t for t in args.targets if t not in present]
    if not targets:
        raise SystemExit(f"none of {args.targets} exist in {args.model}; has {sorted(present)}")
    if missing:
        print(f"[lora] note: {missing} absent from this architecture, skipped")

    peft_cfg = LoraConfig(r=args.r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout, bias="none",
                          task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, peft_cfg)
    if args.grad_checkpoint:
        # Trades speed for a much smaller activation footprint; tokens-per-step
        # is held fixed by grad_accum either way, so this changes wall time and
        # memory but not the optimisation being run.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] targets={targets} r={args.r} -> {trainable:,} trainable "
          f"/ {total:,} ({100 * trainable / total:.3f}%)")

    @torch.no_grad()
    def eval_ce(tag):
        model.eval()
        evl.reset()
        tot, n = 0.0, 0
        for _ in range(args.eval_batches):
            ids = evl.next_batch(args.seqs_per_step).to(args.device)
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            tot += loss.item(); n += 1
        ce = tot / n
        print(f"[lora] {tag}: eval CE {ce:.4f} (ppl {math.exp(ce):.2f})")
        model.train()
        return ce

    ce_before = eval_ce("before")

    tokens_per_step = args.seqs_per_step * train.seq_len * args.grad_accum
    n_steps = max(1, args.total_tokens // tokens_per_step)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))

    print(f"[lora] {n_steps} steps x {tokens_per_step} tokens = "
          f"{n_steps * tokens_per_step:,} tokens")
    model.train()
    t0 = time.time()
    history = []
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            ids = train.next_batch(args.seqs_per_step).to(args.device)
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss / args.grad_accum
            loss.backward()
            acc += loss.item()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        history.append({"step": step, "train_ce": acc})
        if step % 25 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}/{n_steps} train CE {acc:.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    ce_after = eval_ce("after")
    model.save_pretrained(out / "adapter")
    meta = dict(corpus=args.corpus, model=args.model, targets=targets, r=args.r,
                lora_alpha=args.lora_alpha, lr=args.lr, steps=n_steps,
                tokens=n_steps * tokens_per_step, seed=args.seed,
                eval_ce_before=ce_before, eval_ce_after=ce_after,
                eval_ppl_before=math.exp(ce_before), eval_ppl_after=math.exp(ce_after),
                trainable_params=trainable, wall_seconds=time.time() - t0)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    json.dump(history, open(out / "history.json", "w"), indent=1)
    print(f"[lora] saved {out/'adapter'} | CE {ce_before:.4f} -> {ce_after:.4f}")


if __name__ == "__main__":
    main()

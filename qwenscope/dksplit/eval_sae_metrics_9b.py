#!/usr/bin/env python
"""SAE metrics for each 9B SAE arm, on the three distributions that matter.

    lora_dksplit    LoRA model, held-out DKSplit prompts   -- the splice target
    lora_corpus     LoRA model, general corpus             -- the adapted model
    instruct_corpus instruct model, general corpus         -- RETENTION: the
                    distribution the released SAE already handled (FVE 0.711),
                    which a fine-tuned SAE should not lose

Metrics are computed with no hook registered, so the SAE cannot influence the
activations it is scored on. `act_rms` is reported per distribution because the
whole problem here is a 63x scale shift.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import ResidCapture, resolve_layer
from qsae.data import CachedTokenWindows
from qsae.sae import load_local_sae, load_qwenscope_sae
from eval_dksplit import ADAPTER, BASE, load_benchmark, render
from prompts import PROMPTS

SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"


def build(with_lora, device):
    tok = AutoTokenizer.from_pretrained(ADAPTER, token=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                             device_map=device, token=False)
    if with_lora:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ADAPTER, token=False).merge_and_unload()
    m.eval()
    return m, tok


@torch.no_grad()
def acts_corpus(model, windows, layer, device, n_batches, seqs=4):
    out = []
    for _ in range(n_batches):
        ids = windows.next_batch(seqs).to(device)
        with ResidCapture(model, layer) as cap:
            a = cap.run(input_ids=ids, use_cache=False)
        out.append(a[:, 1:].reshape(-1, a.shape[-1]).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def acts_dksplit(model, tok, prefixes, layer, chat_mode, device, batch=16):
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prefixes), batch):
        texts = [render(tok, PROMPTS["advanced"], p, chat_mode)
                 for p in prefixes[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with ResidCapture(model, layer) as cap:
            a = cap.run(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"], use_cache=False)
        out.append(a[enc["attention_mask"].bool()].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def metrics(sae, acts, device, chunk=8192):
    """FVE, cosine, L0, dead fraction, and MSE.

    Two MSE variants are reported because raw MSE is not comparable across
    these distributions: the LoRA model's activations have ~63x the RMS of the
    instruct model's, so its raw MSE is inflated by scale alone.
    `mse_norm` divides by mean ||x||^2, giving a scale-free error that can be
    read across rows.
    """
    err = var = cos = 0.0
    sq = 0.0
    n = 0
    ever = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    l0 = 0.0
    for i in range(0, acts.shape[0], chunk):
        x = acts[i:i + chunk].to(device)
        x_hat, f, _ = sae(x)
        err += (x - x_hat).pow(2).sum(-1).sum().item()
        var += (x - sae.b_dec).pow(2).sum(-1).sum().item()
        cos += F.cosine_similarity(x, x_hat, dim=-1).sum().item()
        sq += x.pow(2).sum(-1).sum().item()
        act = f > 1e-8
        l0 += act.float().sum().item()
        ever |= act.any(0)
        n += x.shape[0]
    d_in = acts.shape[-1]
    return dict(n_tokens=n, act_rms=acts.pow(2).mean().sqrt().item(),
                fve=1 - err / max(var, 1e-12), cosine=cos / max(n, 1),
                l0=l0 / max(n, 1),
                dead_frac=1 - ever.float().mean().item(),
                mse=err / max(n, 1) / d_in,          # per-token, per-dimension
                mse_sum=err / max(n, 1),             # per-token, summed over dims
                mse_norm=err / max(sq, 1e-12))       # scale-free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--sae_repo", default=SAE_REPO)
    ap.add_argument("--sae_dir", default="dksplit/out/sae")
    ap.add_argument("--arms", nargs="+", default=["masked", "ftsae", "scratchsae"])
    ap.add_argument("--encode_variant", default="raw")
    ap.add_argument("--cache_dir", default="out/token_cache")
    ap.add_argument("--corpus", default="fineweb")
    ap.add_argument("--corpus_batches", type=int, default=12)
    ap.add_argument("--n_dksplit", type=int, default=400)
    ap.add_argument("--train_half", type=int, default=2500)
    ap.add_argument("--chat_mode", default="no_think")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="dksplit/out/sae_metrics_9b.json")
    args = ap.parse_args()

    df = load_benchmark("5000")
    held_out = list(df["prefix"].iloc[args.train_half:].head(args.n_dksplit))

    # Collect activations once per (model, distribution), on CPU.
    pools = {}
    for with_lora, name in ((True, "lora"), (False, "instruct")):
        model, tok = build(with_lora, args.device)
        abs_layer = resolve_layer(model, args.layer)
        w = CachedTokenWindows(args.cache_dir, args.corpus, "eval")
        pools[f"{name}_corpus"] = acts_corpus(model, w, args.layer, args.device,
                                              args.corpus_batches)
        if with_lora:
            pools["lora_dksplit"] = acts_dksplit(model, tok, held_out, args.layer,
                                                 args.chat_mode, args.device)
        print(f"[acts] {name}: corpus {pools[f'{name}_corpus'].shape[0]} tokens"
              + (f", dksplit {pools['lora_dksplit'].shape[0]} tokens" if with_lora else ""),
              flush=True)
        del model
        torch.cuda.empty_cache()

    saes = {"gsae": None}
    for arm in args.arms:
        p = Path(args.sae_dir) / arm / f"sae_9b_layer{args.layer}_{arm}.pt"
        if p.exists():
            saes[arm] = str(p)
        else:
            print(f"[skip] {arm}: missing {p}")

    report = {"layer": args.layer, "distributions": {}, "rows": []}
    for dist, acts in pools.items():
        report["distributions"][dist] = dict(n_tokens=acts.shape[0],
                                             act_rms=acts.pow(2).mean().sqrt().item())

    for label, path in saes.items():
        sae = (load_qwenscope_sae(args.sae_repo, args.layer, device=args.device,
                                  encode_variant=args.encode_variant)[0]
               if path is None else
               load_local_sae(path, device=args.device,
                              encode_variant=args.encode_variant))
        for dist, acts in pools.items():
            m = metrics(sae, acts, args.device)
            report["rows"].append(dict(sae=label, distribution=dist, **m))
            print(f"  {label:11s} {dist:16s} fve={m['fve']:+.4f} "
                  f"cos={m['cosine']:.4f} l0={m['l0']:.1f} "
                  f"dead={m['dead_frac']:.4f} mse={m['mse']:.4g} "
                  f"mse_norm={m['mse_norm']:.4f} rms={m['act_rms']:.2f}", flush=True)
        del sae
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("\n=== FVE: SAE x distribution ===")
    dists = list(pools)
    print(f"{'sae':12s}" + "".join(f"{d:>18s}" for d in dists))
    for label in saes:
        row = f"{label:12s}"
        for d in dists:
            r = next((x for x in report["rows"]
                      if x["sae"] == label and x["distribution"] == d), None)
            row += f"{r['fve']:18.4f}" if r else f"{'-':>18s}"
        print(row)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

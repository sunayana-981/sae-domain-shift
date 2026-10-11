#!/usr/bin/env python
"""How well does the 9B Qwen-Scope SAE reconstruct, and on which layer?

Two questions, one pass:

1. WHICH LAYER. The SAE ships for all 32 layers; splice at the one that
   reconstructs best on the activations that actually matter here -- DKSplit
   prompts, not a generic corpus.

2. HOW BAD IS THE MODEL MISMATCH. The SAE's config names its base model as
   `Qwen/Qwen3.5-9B-Base`, but DKSplit runs on `Qwen/Qwen3.5-9B` (instruct)
   plus a LoRA adapter, and Qwen-Scope publishes no instruct-variant 9B SAE.
   That is two stacked distribution shifts the SAE never saw, so FVE is
   measured on all three models and reported side by side rather than assumed
   comparable.

Also checks the encoder convention, as for the 2B: the card does not say
whether the encoder reads x or x - b_dec.
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

from qsae.activations import ResidCapture, decoder_layers, resolve_layer
from qsae.sae import ENCODE_VARIANTS, load_qwenscope_sae
from eval_dksplit import ADAPTER, BASE, load_benchmark, render
from prompts import PROMPTS

BASE_VARIANT = "Qwen/Qwen3.5-9B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"


@torch.no_grad()
def capture(model, tok, prefixes, layer, chat_mode, device, batch=8):
    """Residual-stream activations over rendered DKSplit prompts."""
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(prefixes), batch):
        texts = [render(tok, PROMPTS["advanced"], p, chat_mode)
                 for p in prefixes[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with ResidCapture(model, layer) as cap:
            acts = cap.run(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"], use_cache=False)
        mask = enc["attention_mask"].bool()
        outs.append(acts[mask].float().cpu())   # real tokens only, no padding
    return torch.cat(outs)


@torch.no_grad()
def fidelity(sae, acts):
    x_hat, feats, _ = sae(acts)
    err = (acts - x_hat).pow(2).sum(-1).sum().item()
    var = (acts - sae.b_dec).pow(2).sum(-1).sum().item()
    return dict(fve=1 - err / max(var, 1e-12),
                cosine=F.cosine_similarity(acts, x_hat, dim=-1).mean().item(),
                l0=(feats > 1e-8).float().sum(-1).mean().item())


def build(which, device):
    tok = AutoTokenizer.from_pretrained(ADAPTER, token=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mid = BASE_VARIANT if which == "base_variant" else BASE
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16,
                                             device_map=device, token=False)
    if which == "lora":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ADAPTER, token=False).merge_and_unload()
    m.eval()
    return m, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["base_variant", "instruct", "lora"],
                   choices=["base_variant", "instruct", "lora"])
    p.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24, 27, 30])
    p.add_argument("--sae_repo", default=SAE_REPO)
    p.add_argument("--n_prompts", type=int, default=64)
    p.add_argument("--chat_mode", default="no_think")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="dksplit/out/probe_9b_sae.json")
    args = p.parse_args()

    prefixes = list(load_benchmark("5000").head(args.n_prompts)["prefix"])
    report = {"sae_repo": args.sae_repo, "n_prompts": len(prefixes),
              "chat_mode": args.chat_mode, "models": {}}

    for which in args.models:
        print(f"\n{'='*66}\nMODEL: {which}", flush=True)
        model, tok = build(which, args.device)
        n_layers = len(decoder_layers(model))
        print(f"  {n_layers} decoder layers", flush=True)
        report["models"][which] = {}

        for layer in args.layers:
            abs_l = resolve_layer(model, layer)
            acts = capture(model, tok, prefixes, layer, args.chat_mode,
                           args.device).to(args.device)
            row = {"abs_layer": abs_l, "n_tokens": acts.shape[0],
                   "act_rms": acts.pow(2).mean().sqrt().item()}
            for variant in ENCODE_VARIANTS:
                sae, _ = load_qwenscope_sae(args.sae_repo, abs_l,
                                            device=args.device,
                                            encode_variant=variant)
                row[variant] = fidelity(sae, acts)
                del sae
                torch.cuda.empty_cache()
            best = max(ENCODE_VARIANTS, key=lambda v: row[v]["fve"])
            row["best_variant"] = best
            print(f"  layer {abs_l:2d}  raw fve={row['raw']['fve']:+.4f} "
                  f"cos={row['raw']['cosine']:.4f} l0={row['raw']['l0']:.0f}  |  "
                  f"centered fve={row['centered']['fve']:+.4f}  -> {best}", flush=True)
            report["models"][which][str(abs_l)] = row
            del acts
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*66}\nFVE by model x layer (raw encoding)")
    hdr = "layer  " + "".join(f"{m:>16s}" for m in args.models)
    print(hdr)
    for layer in args.layers:
        cells = ""
        for m in args.models:
            ks = report["models"].get(m, {})
            k = next((kk for kk in ks if ks[kk]["abs_layer"] == resolve_layer_static(layer, 32)), None)
            cells += f"{ks[k]['raw']['fve']:16.4f}" if k else f"{'-':>16s}"
        print(f"{resolve_layer_static(layer,32):5d}  {cells}")
    print(f"\nwrote {out}")


def resolve_layer_static(layer, n):
    return layer if layer >= 0 else n + layer


if __name__ == "__main__":
    main()

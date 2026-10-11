#!/usr/bin/env python
"""Pin down the Qwen-Scope encoder convention before anything is trained.

The model card gives the four tensors but not whether the encoder sees `x` or
the centred `x - b_dec`. Both are one line of code and only one reconstructs,
so this measures FVE / cosine / L0 for each on the reference corpus and prints
the winner. Its answer is what `--encode_variant` is set to for every run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import LMActivationStore, decoder_layers, resolve_layer
from qsae.data import CachedTokenWindows, get_corpus
from qsae.sae import ENCODE_VARIANTS, load_qwenscope_sae


@torch.no_grad()
def score(sae, acts):
    x_hat, feats, _ = sae(acts)
    err = (acts - x_hat).pow(2).sum(-1).sum().item()
    var = (acts - sae.b_dec).pow(2).sum(-1).sum().item()
    var_mean = (acts - acts.mean(0, keepdim=True)).pow(2).sum(-1).sum().item()
    return dict(
        fve_bdec=1 - err / max(var, 1e-12),
        fve_mean=1 - err / max(var_mean, 1e-12),
        cosine=F.cosine_similarity(acts, x_hat, dim=-1).mean().item(),
        l0=(feats > 1e-8).float().sum(-1).mean().item(),
        mse=err / acts.shape[0] / acts.shape[1],
        act_rms=acts.pow(2).mean().sqrt().item(),
        recon_rms=x_hat.pow(2).mean().sqrt().item(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--sae_repo", default="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50")
    p.add_argument("--layers", type=int, nargs="+", default=[20])
    p.add_argument("--corpus", default="fineweb")
    p.add_argument("--cache_dir", default="out/token_cache")
    p.add_argument("--n_batches", type=int, default=4)
    p.add_argument("--tokens_per_batch", type=int, default=4096)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="out/probe/sae_convention.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    print(f"[probe] {args.model}: {len(decoder_layers(model))} decoder layers")

    windows = CachedTokenWindows(args.cache_dir, args.corpus, "train")
    print(f"[probe] {windows}")
    report = {"model": args.model, "sae_repo": args.sae_repo,
              "corpus": args.corpus, "layers": {}}

    for layer in args.layers:
        windows.reset()
        store = LMActivationStore(model, windows, layer,
                                  tokens_per_batch=args.tokens_per_batch,
                                  device=args.device)
        acts = torch.cat([store.get_batch_activations() for _ in range(args.n_batches)])
        print(f"\n[probe] layer {layer} (abs {resolve_layer(model, layer)}) — "
              f"{acts.shape[0]} tokens, act RMS {acts.pow(2).mean().sqrt():.4f}")

        per_variant = {}
        for variant in ENCODE_VARIANTS:
            sae, raw_cfg = load_qwenscope_sae(args.sae_repo, resolve_layer(model, layer),
                                              device=args.device, encode_variant=variant)
            m = score(sae, acts)
            per_variant[variant] = m
            print(f"  {variant:9s} fve(b_dec)={m['fve_bdec']:+.4f} "
                  f"fve(mean)={m['fve_mean']:+.4f} cos={m['cosine']:.4f} "
                  f"l0={m['l0']:.1f} recon_rms={m['recon_rms']:.4f}")
            del sae
            torch.cuda.empty_cache()

        sae, raw_cfg = load_qwenscope_sae(args.sae_repo, resolve_layer(model, layer),
                                          device=args.device)
        norms = sae.decoder_norms()
        print(f"  decoder col norms: mean={norms.mean():.4f} std={norms.std():.4f} "
              f"min={norms.min():.4f} max={norms.max():.4f}")
        print(f"  b_dec norm={sae.b_dec.norm():.4f}  k={sae.k}  d_sae={sae.d_sae}")

        best = max(ENCODE_VARIANTS, key=lambda v: per_variant[v]["fve_bdec"])
        print(f"  -> best encode_variant: {best}")
        report["layers"][str(layer)] = {
            "abs_layer": resolve_layer(model, layer),
            "variants": per_variant,
            "best_variant": best,
            "decoder_norm_mean": norms.mean().item(),
            "decoder_norm_std": norms.std().item(),
            "decoder_unit_norm": bool(abs(norms.mean().item() - 1.0) < 0.05
                                      and norms.std().item() < 0.05),
            "b_dec_norm": sae.b_dec.norm().item(),
            "sae_config": raw_cfg,
        }
        del sae, store, acts
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[probe] wrote {out}")


if __name__ == "__main__":
    main()
